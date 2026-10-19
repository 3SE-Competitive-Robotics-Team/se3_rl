"""编译真实 MCU 补偿/控制/CAN 函数，与训练、sim2sim 和 MJCF 虚功对照。"""

from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

from se3_runtime._serialleg_v1 import (
    _ACTIVE_UPPER,
    closedchain_passive_position,
    knee_gas_spring_compensation_torque,
)
from se3_shared.fourbar import knee_gas_spring_compensation_torque_torch
from se3_shared.robot import JointGroup


def function(source: str, name: str) -> str:
    """提取实际 C 函数体，避免测试副本掩盖索引或符号错误。"""
    match = re.search(r"(?:static )?(?:void|float|int)\s+" + name + r"\([^;{}]*\)\s*\{", source)
    assert match, name
    end = match.end()
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end]


def harness(root: Path) -> str:
    core = (root / "Custom/Tasks/Src/task_control_core.c").read_text(encoding="utf-8")
    driver = (root / "Custom/Drivers/MIT_Motor/dm8009_drv.c").read_text(encoding="utf-8")
    output = (root / "Custom/Tasks/Src/task_control_output.c").read_text(encoding="utf-8")
    general = (root / "Custom/Tasks/Inc/task_general.h").read_text(encoding="utf-8")
    limits = (root / "Custom/Drivers/MIT_Motor/dm8009_drv.h").read_text(encoding="gb18030")
    defines = "\n".join(re.findall(r"^#define (?:LF|LB|RF|RB|LEFT|RIGHT)\s+\d+", general, re.M))
    defines += "\n" + "\n".join(re.findall(r"^#define NX_POLICY_JOINT_.*", core, re.M))
    defines += "\n" + "\n".join(
        re.findall(
            r"^#define (?:P_MIN|P_MAX|V_MIN|V_MAX|KP_MIN|KP_MAX|KD_MIN|KD_MAX|T_MIN|T_MAX)\s+.*",
            limits,
            re.M,
        )
    )
    send = re.search(r"DM_Mit_control\(-hipMotorTorque[^;]+;", output)
    assert send
    return (
        r"""
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "knee_spring_compensation.h"
#include "task_nx_comm.h"
_Static_assert(sizeof(nx_comm_target_payload_t) == 36, "TARGET");
#define PI 3.14159265358979323846f
#define MIT_MODE 0
#define LIMIT_MIN_MAX(x,a,b) do { if ((x)<(a)) (x)=(a); if ((x)>(b)) (x)=(b); } while (0)
#define Limiter(v,m) ((v)<-(m)?-(m):((v)>(m)?(m):(v)))
"""
        + defines
        + r"""
typedef struct { float joint_pos[4], wheel_vel[2], knee_spring_force; uint8_t knee_spring_enabled; } nx_policy_target_t;
struct { struct { float joint_angle[4], wheel_vel[2]; } motor; } estimateData;
struct { float output[2]; } control_state;
float hipMotorTorque[4], velocity[4], wheelTorque[2], wheelMotorTorque[2];
int WheelTorqueInt[2], mit_hfdcan;
struct { int CanDead; struct { unsigned TimeStamp; } Super; struct { int state; } para; } HipMotor[4];
float can_torque[4];
unsigned xTaskGetTickCount(void) { return 0; }
void ClearError(int a, int b) { (void)a; (void)b; }
void enable_motor_mode(int a,int b) { (void)a; (void)b; }
void DWT_Delay(double a) { (void)a; }
float pid_update(int *p,float v) { (void)p; return v; }
static float nx_policy_joint_velocity(uint8_t j) { return velocity[j]; }
static float nx_policy_wrap_angle_pi(float a) { return a - 2.0f*PI*floorf((a+PI)/(2.0f*PI)); }
static void nx_policy_outputs_clear(void) { memset(hipMotorTorque,0,sizeof(hipMotorTorque)); }
void fdcanx_send_data(int *h, uint16_t id,uint8_t *data,int n) {
    (void)h; (void)n;
    int raw = ((data[6]&15)<<8)|data[7];
    can_torque[id-1] = raw*(T_MAX-T_MIN)/4095.0f+T_MIN;
}
"""
        + "\n".join(function(driver, n) for n in ("float_to_uint", "mit_ctrl", "DM_Mit_control"))
        + "\n"
        + function(core, "nx_policy_output_update")
        + r"""
void evaluate(float *q, float force, int enabled, float pd, float *out) {
    nx_policy_target_t t = {0};
    const int map[4] = {LF,LB,RF,RB};
    for (int j=0;j<4;++j) {
        estimateData.motor.joint_angle[map[j]]=q[j];
        t.joint_pos[map[j]]=q[j]+pd/NX_POLICY_JOINT_KP;
    }
    t.knee_spring_force=force; t.knee_spring_enabled=enabled;
    knee_spring_compensation(q,force,enabled,out);
    nx_policy_output_update(&t);
"""
        + send.group()
        + r"""
    for (int j=0;j<4;++j) {
        out[4+j]=hipMotorTorque[map[j]];
        out[8+j]=can_torque[map[j]];
    }
}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware-root", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    args = parser.parse_args()
    root = args.firmware_root.resolve()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        source, library = Path(tmp) / "check.c", Path(tmp) / "check.dll"
        source.write_text(harness(root), encoding="utf-8")
        subprocess.run(
            [
                args.cc,
                "-std=c11",
                "-O2",
                "-shared",
                str(source),
                "-o",
                str(library),
                "-I" + str(root / "Custom/Controller"),
                "-I" + str(root / "Custom/Tasks/Inc"),
                "-lm",
            ],
            check=True,
        )
        lib = ctypes.CDLL(str(library))
        array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        lib.evaluate.argtypes = [array, ctypes.c_float, ctypes.c_int, ctypes.c_float, array]
        lib.evaluate.restype = None

        def evaluate(
            q: np.ndarray, force: float = 300.0, enabled: int = 1, pd: float = 0.0
        ) -> np.ndarray:
            out = np.zeros(12, np.float32)
            lib.evaluate(np.asarray(q, dtype=np.float32), force, enabled, pd, out)
            return out.astype(np.float64)

        rng = np.random.default_rng(42)
        alpha = np.linspace(0.0, _ACTIVE_UPPER, 10001)
        base = rng.uniform(-0.5, 0.5, (len(alpha), 2))
        q = np.column_stack(
            (base[:, 0], base[:, 0] - alpha, base[:, 1], base[:, 1] + alpha[::-1])
        ).astype(np.float32)
        max_runtime = max_training = max_can = 0.0
        for force in (270.0, 300.0, 330.0):
            actual = np.array([evaluate(row, force) for row in q])
            expected = np.array([knee_gas_spring_compensation_torque(row, force) for row in q])
            training = knee_gas_spring_compensation_torque_torch(torch.from_numpy(q), force).numpy()
            max_runtime = max(max_runtime, float(np.max(np.abs(actual[:, :4] - expected))))
            max_training = max(max_training, float(np.max(np.abs(actual[:, :4] - training))))
            max_can = max(max_can, float(np.max(np.abs(actual[:, 8:] - expected))))
            np.testing.assert_allclose(actual[:, 4:8], actual[:, :4], atol=2e-5, rtol=0)
            assert np.all(np.sign(actual[:, 8:]) == np.sign(expected))
        assert max_runtime < 0.001, max_runtime
        assert max_training < 0.02, max_training
        assert max_can < 108 / 4095 + 0.001, max_can
        for enabled, force in ((0, 300.0), (0, 0.0)):
            np.testing.assert_array_equal(evaluate(q[100], force, enabled)[:8], 0)
        for force in (float("nan"), -1.0, float("inf")):
            np.testing.assert_array_equal(evaluate(q[100], force)[:8], 0)
        for pd in (-60.0, -30.0, 30.0, 60.0):
            out = evaluate(q[100], pd=pd)
            np.testing.assert_allclose(out[4:8], np.clip(out[:4] + pd, -40, 40), atol=3e-5, rtol=0)
        for a in (-0.2, _ACTIVE_UPPER + 0.2):
            row = np.array([0, -a, 0, a], np.float32)
            np.testing.assert_allclose(
                evaluate(row)[:4], knee_gas_spring_compensation_torque(row, 300), atol=0.001, rtol=0
            )

        # 独立物理校验：MJCF tendon 长度有限差分求广义力，不复用补偿雅可比。
        mjcf = (
            Path(__file__).resolve().parents[1]
            / "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
        )
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        data = mujoco.MjData(model)
        names = (*JointGroup.POLICY_LEG_NAMES, *JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES)
        addresses = [model.joint(n).qposadr[0] for n in names]

        def spring_length(row: np.ndarray) -> float:
            data.qpos[:] = model.qpos0
            data.qpos[addresses] = np.concatenate((row, closedchain_passive_position(row)))
            mujoco.mj_forward(model, data)
            return float(
                sum(data.ten_length[model.tendon(n).id] for n in ("l_knee_spring", "r_knee_spring"))
            )

        max_residual = 0.0
        for a in np.linspace(0.001, _ACTIVE_UPPER - 0.001, 101):
            row = np.array([0.2, 0.2 - a, -0.1, -0.1 + (_ACTIVE_UPPER - a)])
            physical = np.zeros(4)
            for j in range(4):
                delta = np.eye(4)[j] * 1e-6
                physical[j] = 300 * (spring_length(row + delta) - spring_length(row - delta)) / 2e-6
            compensation = evaluate(row)[:4]
            assert np.all(compensation * physical < 0), (row, compensation, physical)
            max_residual = max(max_residual, float(np.max(np.abs(compensation + physical))))
        assert max_residual < 0.001, max_residual
        print(f"30003 个姿态/推力组合：C vs sim2sim 最大误差 {max_runtime:.8f} N·m")
        print(f"C vs 训练 float32 最大误差 {max_training:.8f} N·m")
        print(f"真实 CAN 编码后最大误差 {max_can:.8f} N·m（12 bit 量化）")
        print(f"MJCF 弹簧广义力 + C 补偿最大残差 {max_residual:.8f} N·m；四电机方向均相反")
        print("关闭、无效推力、角度边界、PD 叠加后限矩检查通过")
        # Windows 上卸载 DLL 后临时目录才能删除。
        if hasattr(ctypes, "windll"):
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            ctypes.windll.kernel32.FreeLibrary(lib._handle)


if __name__ == "__main__":
    main()
