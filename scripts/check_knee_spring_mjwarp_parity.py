"""气弹簧 mjwarp/native MuJoCo 一致性对照。

训练端物理由 MuJoCo Warp 求解，sim2sim 端由原生 MuJoCo 求解。气弹簧建模为
spatial tendon + 恒力 actuator，两个引擎对 tendon 几何与广义力的实现是独立的，
本脚本在同一批姿态下逐位对比：

- tendon 长度（弹簧几何）
- 弹簧 actuator 力（应恒等于 biasprm[0]）
- qfrc_actuator（弹簧经 tendon 力臂分配到各关节的广义力矩）

原始 MJCF 只有气弹簧两个 actuator（电机 actuator 由 MJLab 运行时注入），因此
qfrc_actuator 即弹簧的全部贡献。

用法：
    uv run python scripts/check_knee_spring_mjwarp_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_closed_chain_v3_train_obb_trim.xml"
)

sys.path.insert(0, str(REPO_ROOT / "src"))

from se3_shared import JointGroup, policy_to_closedchain_passive_pos_np  # noqa: E402

# 与 docs/plan/knee_spring_modeling.md 的实测表一致的主动杆夹角扫描
_ALPHA_SWEEP = np.linspace(0.0, 1.51, 9)
_LENGTH_ATOL_M = 1e-5
_FORCE_ATOL_N = 1e-3
_QFRC_ATOL_NM = 1e-3


def _qpos_for_alpha(model: mujoco.MjModel, alpha: float) -> np.ndarray:
    """构造 policy_pos=(0, -α, 0, α) 对应的完整 qpos，底盘悬空避免接触干扰。"""
    policy_pos = np.array([[0.0, -alpha, 0.0, alpha]], dtype=np.float64)
    passive_pos = policy_to_closedchain_passive_pos_np(policy_pos)[0]

    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[2] = 1.0  # 自由基座 z
    qpos[3] = 1.0  # 四元数 w

    joint_values = dict(zip(JointGroup.POLICY_LEG_NAMES, policy_pos[0], strict=True))
    joint_values.update(
        dict(zip(JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES, passive_pos, strict=True))
    )
    for name, value in joint_values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[model.jnt_qposadr[joint_id]] = float(value)
    return qpos


def main() -> None:
    import mujoco_warp as mjwarp
    import warp as wp

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)

    spring_actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in JointGroup.KNEE_SPRING_ACTUATOR_NAMES
    ]
    if any(aid < 0 for aid in spring_actuator_ids):
        raise SystemExit(f"MJCF 缺少气弹簧 actuator: {JointGroup.KNEE_SPRING_ACTUATOR_NAMES}")
    expected_force = np.asarray(
        [model.actuator_biasprm[aid, 0] for aid in spring_actuator_ids],
        dtype=np.float64,
    )

    wp.init()
    max_len_diff = 0.0
    max_force_diff = 0.0
    max_qfrc_diff = 0.0

    for alpha in _ALPHA_SWEEP:
        qpos = _qpos_for_alpha(model, alpha)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        native_len = data.ten_length.copy()
        native_actuator_force = data.actuator_force.copy()
        native_qfrc = data.qfrc_actuator.copy()

        with wp.ScopedDevice(wp.get_device()):
            warp_model = mjwarp.put_model(model)
            warp_data = mjwarp.put_data(model, data)
            mjwarp.forward(warp_model, warp_data)
            warp_len = warp_data.ten_length.numpy()[0]
            warp_actuator_force = warp_data.actuator_force.numpy()[0]
            warp_qfrc = warp_data.qfrc_actuator.numpy()[0]

        len_diff = float(np.max(np.abs(warp_len - native_len)))
        force_diff = float(np.max(np.abs(warp_actuator_force - native_actuator_force)))
        qfrc_diff = float(np.max(np.abs(warp_qfrc - native_qfrc)))
        max_len_diff = max(max_len_diff, len_diff)
        max_force_diff = max(max_force_diff, force_diff)
        max_qfrc_diff = max(max_qfrc_diff, qfrc_diff)

        spring_force = native_actuator_force[spring_actuator_ids]
        if not np.allclose(spring_force, expected_force, atol=_FORCE_ATOL_N):
            raise SystemExit(
                f"alpha={alpha:.3f}: 原生弹簧力 {spring_force} 偏离 biasprm {expected_force}"
            )
        print(
            f"alpha={alpha:.3f}  |dL|={len_diff:.3e} m  "
            f"|dF|={force_diff:.3e} N  |dqfrc|={qfrc_diff:.3e} N·m"
        )

    print(
        f"\nmax diffs: length={max_len_diff:.3e} m, "
        f"force={max_force_diff:.3e} N, qfrc={max_qfrc_diff:.3e} N·m"
    )
    failures = []
    if max_len_diff > _LENGTH_ATOL_M:
        failures.append(f"tendon 长度偏差 {max_len_diff:.3e} > {_LENGTH_ATOL_M:.1e}")
    if max_force_diff > _FORCE_ATOL_N:
        failures.append(f"actuator 力偏差 {max_force_diff:.3e} > {_FORCE_ATOL_N:.1e}")
    if max_qfrc_diff > _QFRC_ATOL_NM:
        failures.append(f"广义力矩偏差 {max_qfrc_diff:.3e} > {_QFRC_ATOL_NM:.1e}")
    if failures:
        raise SystemExit("mjwarp/native 气弹簧不一致: " + "; ".join(failures))
    print("PASS: mjwarp 与原生 MuJoCo 的气弹簧几何与广义力一致")


if __name__ == "__main__":
    main()
