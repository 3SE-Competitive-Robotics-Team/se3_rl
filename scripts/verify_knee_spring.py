"""气弹簧 + 四连杆机构运动学/静力学解算（验证脚本）。

目的：独立于 MuJoCo 内部 tendon 受力，重新解算 SerialLeg 闭链四连杆的
几何与气弹簧等效力矩，用来核对 MJCF 里的气弹簧建模是否正确。

机构拓扑（共轴四连杆，左腿为例，全部绕 y 轴在 x-z 平面内运动）：
- O：lf0_Joint 与 l_drive_bar_Joint 的共轴铰点（两电机共轴链轮传动）
- 前主动杆（thigh, lf0_Link）：O → K(膝铰 lf1_Joint)，由 lf0 驱动
- 小腿上段（lf1_Link 被动）：K → C(lf_coupler_closure)
- 后驱动杆（l_drive_bar_Link）：O → P(l_coupler_Joint)，由 l_drive_bar 驱动
- 连杆（l_coupler_Link 被动）：P → D(l_coupler_end) ，connect 约束 D == C

给定两个主动杆角 (lf0, l_drive_bar)，被动角 (lf1, l_coupler) 由闭链求解。
气弹簧挂在膝关节两侧：site l_spring_p1（在 thigh 上） / l_spring_p2（在小腿上）。

等效力矩用虚功原理：tau_i = -F_spring * d(L_spring)/d(theta_i)
其中 theta_i 取主动杆角（保持另一个主动杆角不变，被动角随闭链调整）。
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

_MJCF = Path(__file__).resolve().parent.parent / (
    "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
)

# 气弹簧恒力（来自 MJCF actuator biasprm="300 0 0"）
SPRING_FORCE = 300.0


def _jadr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def _site_xz(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    p = data.site_xpos[sid]
    return np.array([p[0], p[2]])


def _body_xz(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    p = data.xpos[bid]
    return np.array([p[0], p[2]])


class LegModel:
    """左腿闭链解算器：输入两主动杆角，输出被动角、气弹簧长度、几何量。"""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_MJCF))
        self.data = mujoco.MjData(self.model)
        # 关节 qpos 地址
        self.adr_lf0 = _jadr(self.model, "lf0_Joint")
        self.adr_lf1 = _jadr(self.model, "lf1_Joint")
        self.adr_drive = _jadr(self.model, "l_drive_bar_Joint")
        self.adr_coupler = _jadr(self.model, "l_coupler_Joint")
        # 气弹簧 tendon id
        self.ten_spring = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, "l_knee_spring")
        # base 固定在原点直立
        self.data.qpos[0:7] = [0, 0, 0.22, 1, 0, 0, 0]
        # 被动角初值（取 standing keyframe）
        self._passive_guess = np.array([-0.73636135, 0.785409053])

    def _fk(self, lf0: float, lf1: float, drive: float, coupler: float) -> None:
        q = self.data.qpos
        q[self.adr_lf0] = lf0
        q[self.adr_lf1] = lf1
        q[self.adr_drive] = drive
        q[self.adr_coupler] = coupler
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_tendon(self.model, self.data)

    def _loop_residual(self, lf0: float, drive: float, passive: np.ndarray) -> np.ndarray:
        """connect 约束残差：D(l_coupler_end) - C(lf_coupler_closure)，x-z 两维。"""
        self._fk(lf0, passive[0], drive, passive[1])
        d = _site_xz(self.data, self.model, "l_coupler_end")
        c = _site_xz(self.data, self.model, "lf_coupler_closure")
        return d - c

    def solve(self, lf0: float, drive: float) -> dict[str, float]:
        """给定两主动杆角，Newton 求闭链被动角，返回几何与受力量。"""
        passive = self._passive_guess.copy()
        for _ in range(100):
            r = self._loop_residual(lf0, drive, passive)
            if np.linalg.norm(r) < 1e-10:
                break
            # 数值雅可比
            jac = np.zeros((2, 2))
            eps = 1e-7
            for k in range(2):
                pp = passive.copy()
                pp[k] += eps
                rp = self._loop_residual(lf0, drive, pp)
                jac[:, k] = (rp - r) / eps
            passive = passive - np.linalg.solve(jac, r)
        else:
            raise RuntimeError(
                f"闭链求解未收敛: lf0={lf0}, drive={drive}, |r|={np.linalg.norm(r):.2e}"
            )
        # 收敛后记住，作为下一次初值（连续 sweep 加速）
        self._passive_guess = passive.copy()
        self._fk(lf0, passive[0], drive, passive[1])

        # 气弹簧长度（MuJoCo tendon 长度 = 两 site 距离）
        l_spring = float(self.data.ten_length[self.ten_spring])
        # 腿长：O(lf0_Link 原点) 到 轮轴(l_wheel_Link 原点)
        o = _body_xz(self.data, self.model, "lf0_Link")
        wheel = _body_xz(self.data, self.model, "l_wheel_Link")
        leg_len = float(np.linalg.norm(wheel - o))
        # 主动杆夹角：两主动杆角之差（与 tendon l_active_rod_angle 一致）
        rod_gap = lf0 - drive
        return {
            "lf0": lf0,
            "drive": drive,
            "lf1": float(passive[0]),
            "coupler": float(passive[1]),
            "l_spring": l_spring,
            "leg_len": leg_len,
            "rod_gap": rod_gap,
        }

    def spring_torque(self, lf0: float, drive: float, h: float = 1e-5) -> dict[str, float]:
        """虚功法求气弹簧在两主动杆上的等效力矩。

        tau_i = -F * dL_spring/dtheta_i （F 为恒定张力，拉短 tendon 为正）。
        分别对 lf0、drive 取偏导，另一主动杆角固定，被动角随闭链重解。
        """
        l0 = self.solve(lf0, drive)["l_spring"]
        # d L / d lf0
        lp = self.solve(lf0 + h, drive)["l_spring"]
        lm = self.solve(lf0 - h, drive)["l_spring"]
        dL_dlf0 = (lp - lm) / (2 * h)
        # d L / d drive
        lp = self.solve(lf0, drive + h)["l_spring"]
        lm = self.solve(lf0, drive - h)["l_spring"]
        dL_ddrive = (lp - lm) / (2 * h)
        return {
            "dL_dlf0": dL_dlf0,
            "dL_ddrive": dL_ddrive,
            "tau_lf0": -SPRING_FORCE * dL_dlf0,
            "tau_drive": -SPRING_FORCE * dL_ddrive,
            "l_spring": l0,
        }


def main() -> None:
    leg = LegModel()

    print("=" * 96)
    print("气弹簧 + 四连杆 运动学/静力学解算（左腿，F_spring = 300 N）")
    print("=" * 96)

    # 先验证 control=0 参考点：lf0=0, drive=0
    g0 = leg.solve(0.0, 0.0)
    print("\n[control=0 参考点校验] lf0=0, l_drive_bar=0")
    print(f"  腿长 (O->轮轴)        = {g0['leg_len'] * 1000:.2f} mm   (期望 ≈ 330.51 mm)")
    print(f"  气弹簧长度 l_spring    = {g0['l_spring'] * 1000:.2f} mm")
    print(f"  被动膝角 lf1           = {math.degrees(g0['lf1']):.3f} deg")
    print(f"  被动连杆角 l_coupler   = {math.degrees(g0['coupler']):.3f} deg")
    print(f"  主动杆夹角 (lf0-drive) = {math.degrees(g0['rod_gap']):.3f} deg")

    # 沿 standing 协调运动方向 sweep：control=0 -> standing(lf0=0.2, drive=-0.5403) 线性外推
    # 比例：lf0 = 0.270 * s, drive = -0.730 * s，s = 主动杆夹角(rod_gap)
    print("\n[沿协调运动 sweep] s = 主动杆夹角差 rad；lf0=0.270*s, drive=-0.730*s")
    header = (
        f"{'s(rad)':>7} {'lf0(rad)':>9} {'drive(rad)':>10} "
        f"{'leg(mm)':>8} {'spring(mm)':>10} "
        f"{'dL/dlf0':>9} {'dL/ddrv':>9} "
        f"{'tau_lf0(Nm)':>11} {'tau_drv(Nm)':>11} {'tau_sum':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in np.linspace(0.0, 1.7, 18):
        lf0 = 0.270 * s
        drive = -0.730 * s
        geo = leg.solve(lf0, drive)
        tq = leg.spring_torque(lf0, drive)
        print(
            f"{s:7.3f} {lf0:9.4f} {drive:10.4f} "
            f"{geo['leg_len'] * 1000:8.2f} {geo['l_spring'] * 1000:10.2f} "
            f"{tq['dL_dlf0'] * 1000:9.3f} {tq['dL_ddrive'] * 1000:9.3f} "
            f"{tq['tau_lf0']:11.3f} {tq['tau_drive']:11.3f} "
            f"{tq['tau_lf0'] + tq['tau_drive']:8.3f}"
        )

    print("\n说明：")
    print("  dL/dθ 单位 mm/rad；tau = -300 * dL/dθ（N·m）。")
    print("  tau_lf0 = 气弹簧在前主动杆(lf0)上的等效力矩；tau_drv = 在后驱动杆上的等效力矩。")


if __name__ == "__main__":
    main()
