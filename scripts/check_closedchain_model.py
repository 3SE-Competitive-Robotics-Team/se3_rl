"""闭链 SerialLeg MJCF 的最小验收检查。

用法：
    uv run python scripts/check_closedchain_model.py
    uv run python scripts/check_closedchain_model.py --model assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from se3_shared import RobotConfig

DEFAULT_MODEL = Path("assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml")
# SW 总装坐标经 *_xi 坐标系导出后，MuJoCo 局部轴为 [SW_Z, SW_X, SW_Y]。
# 训练模型额外把同侧 P1/P2 投影到同一 MuJoCo y 平面，避免气弹簧产生侧向力。
ZERO_POSE_SPRING_BASE_POS = {
    "l_spring_p1": np.array([0.00705, 0.17382702, -0.04678497], dtype=np.float64),
    "l_spring_p2": np.array([-0.22025, 0.17382702, -0.04678497], dtype=np.float64),
    "r_spring_p1": np.array([0.00705, -0.17382702, -0.04678497], dtype=np.float64),
    "r_spring_p2": np.array([-0.22025, -0.17382702, -0.04678497], dtype=np.float64),
}
SPRING_SITE_BODIES = {
    "l_spring_p1": "lf0_Link",
    "l_spring_p2": "lf1_Link",
    "r_spring_p1": "rf0_Link",
    "r_spring_p2": "rf1_Link",
}
BODY_PARENTS = {
    "lf1_Link": "lf0_Link",
    "rf1_Link": "rf0_Link",
    "l_coupler_Link": "l_drive_bar_Link",
    "r_coupler_Link": "r_drive_bar_Link",
}
JOINT_AXES = {
    "lf0_Joint": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "lf1_Joint": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "l_drive_bar_Joint": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "l_coupler_Joint": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "rf0_Joint": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "rf1_Joint": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "r_drive_bar_Joint": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "r_coupler_Joint": np.array([0.0, -1.0, 0.0], dtype=np.float64),
}
ZERO_POSE_SITE_PAIRS = (
    ("lf_thigh_end", "lf_calf_closure"),
    ("rf_thigh_end", "rf_calf_closure"),
    ("l_coupler_end", "lf_coupler_closure"),
    ("r_coupler_end", "rf_coupler_closure"),
)
DEFAULT_POSE_SITE_PAIRS = (
    ("l_coupler_end", "lf_coupler_closure"),
    ("r_coupler_end", "rf_coupler_closure"),
)
CONNECT_EQUALITY_NAMES = ("l_coupler_to_lf_calf", "r_coupler_to_rf_calf")


def _names(model: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj, idx) or f"<unnamed:{idx}>" for idx in range(count)]


def _set_default_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    cfg = RobotConfig()
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    if model.nq >= 7:
        data.qpos[2] = float(cfg.default_base_height)
    for joint_name, value in cfg.default_model_joint_pos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        data.qpos[model.jnt_qposadr[jid]] = float(value)


def _set_zero_joint_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """把所有 hinge 关节置零，用于校验 CAD 默认装配位。"""
    mujoco.mj_resetData(model, data)
    if model.nq >= 7:
        data.qpos[2] = 0.3
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[model.jnt_qposadr[jid]] = 0.0


def _site_delta(data: mujoco.MjData, a: str, b: str) -> np.ndarray:
    return np.asarray(data.site(a).xpos - data.site(b).xpos, dtype=np.float64)


def _connect_force_norms(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[tuple[str, float, list[float]]]:
    """读取默认姿态下 connect equality 的约束力，单位等价于 N。"""
    out: list[tuple[str, float, list[float]]] = []
    for eq_id, name in enumerate(CONNECT_EQUALITY_NAMES):
        rows = [
            row
            for row in range(data.nefc)
            if int(data.efc_type[row]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            and int(data.efc_id[row]) == eq_id
        ]
        if not rows:
            continue
        force = np.asarray(data.efc_force[rows], dtype=np.float64)
        out.append((name, float(np.linalg.norm(force)), force.tolist()))
    return out


def _print_model_summary(model: mujoco.MjModel) -> None:
    print(
        "model:",
        f"njnt={model.njnt}",
        f"nq={model.nq}",
        f"nv={model.nv}",
        f"nu={model.nu}",
        f"neq={model.neq}",
        f"ntendon={model.ntendon}",
    )
    print("joints:")
    for jid, name in enumerate(_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)):
        print(
            f"  {jid:02d} {name:24s}"
            f" qpos={int(model.jnt_qposadr[jid]):02d}"
            f" qvel={int(model.jnt_dofadr[jid]):02d}"
        )
    print("actuators:")
    for aid, name in enumerate(_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)):
        trnid = model.actuator_trnid[aid]
        bias_force = float(model.actuator_biasprm[aid, 0])
        print(
            f"  {aid:02d} {name:24s}"
            f" trnid={tuple(int(x) for x in trnid)}"
            f" bias_force={bias_force:.3f}"
        )


def _print_tendon_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    print("tendons:")
    for tid, name in enumerate(_names(model, mujoco.mjtObj.mjOBJ_TENDON, model.ntendon)):
        limits = model.tendon_range[tid]
        print(
            f"  {tid:02d} {name:24s}"
            f" length={float(data.ten_length[tid]):.6f}"
            f" range=({float(limits[0]):.6f}, {float(limits[1]):.6f})"
        )


def _check_gas_springs(model: mujoco.MjModel, expected_force: float) -> None:
    """检查闭链 MJCF 中左右气弹簧常力是否保持一致。"""
    for name in ("l_knee_gas_spring", "r_knee_gas_spring"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        force = float(model.actuator_biasprm[aid, 0])
        if not np.isclose(force, expected_force, atol=1.0e-6):
            raise SystemExit(f"{name} 常力应为 {expected_force:.1f} N，实际 {force:.6f} N")


def _check_active_rod_angle_tendons(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """检查主动杆夹角 tendon 限位和默认站姿夹角。"""
    cfg = RobotConfig()
    expected_limits = np.asarray(cfg.active_rod_angle_limits, dtype=np.float64)
    for name, expected_angle in zip(
        ("l_active_rod_angle", "r_active_rod_angle"),
        cfg.default_active_rod_angles,
        strict=True,
    ):
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        if tid < 0:
            continue
        limits = np.asarray(model.tendon_range[tid], dtype=np.float64)
        length = float(data.ten_length[tid])
        if not np.allclose(limits, expected_limits, atol=1.0e-6):
            raise SystemExit(f"{name} 限位应为 {tuple(expected_limits)}，实际 {tuple(limits)}")
        if not np.isclose(length, expected_angle, atol=1.0e-4):
            raise SystemExit(f"{name} 默认夹角应为 {expected_angle:.6f}，实际 {length:.6f}")


def _check_joint_axes(model: mujoco.MjModel) -> None:
    """检查闭链关节轴方向是否符合 SW 读取到的 zhou 基准。"""
    for joint_name, expected_axis in JOINT_AXES.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        axis = np.asarray(model.jnt_axis[jid], dtype=np.float64)
        if not np.allclose(axis, expected_axis, atol=1.0e-8):
            raise SystemExit(
                f"{joint_name} 轴向应为 {expected_axis.tolist()}，实际 {axis.tolist()}"
            )


def _check_zero_pose_geometry(model: mujoco.MjModel) -> None:
    """校验 q=0 时闭链闭合，且气弹簧挂点固连在正确连杆。"""
    data = mujoco.MjData(model)
    _set_zero_joint_pose(model, data)
    mujoco.mj_forward(model, data)

    for body_name, parent_name in BODY_PARENTS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
        if bid < 0 or parent_id < 0:
            continue
        actual_id = int(model.body_parentid[bid])
        if actual_id != parent_id:
            actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, actual_id)
            raise SystemExit(f"{body_name} 应挂在 {parent_name} 下，实际挂在 {actual} 下")

    for site_name, body_name in SPRING_SITE_BODIES.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if sid < 0 or bid < 0:
            continue
        if int(model.site_bodyid[sid]) != bid:
            actual = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.site_bodyid[sid]),
            )
            raise SystemExit(f"{site_name} 应固连在 {body_name}，实际固连在 {actual}")

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_pos = np.asarray(data.xpos[base_id], dtype=np.float64)
    for site_name, expected_pos in ZERO_POSE_SPRING_BASE_POS.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            continue
        rel = np.asarray(data.site_xpos[sid], dtype=np.float64) - base_pos
        actual_pos = rel
        if not np.allclose(actual_pos, expected_pos, atol=5.0e-4):
            raise SystemExit(
                f"{site_name} q=0 base 坐标应为 {expected_pos.tolist()}，实际 {actual_pos.tolist()}"
            )

    for prefix in ("l", "r"):
        p1 = f"{prefix}_spring_p1"
        p2 = f"{prefix}_spring_p2"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, p1) >= 0:
            spring_y_delta = _site_delta(data, p2, p1)[1]
            if abs(float(spring_y_delta)) > 1.0e-6:
                raise SystemExit(f"{prefix} 气弹簧 q=0 y 偏移应为 0，实际 {spring_y_delta:.9f}")
            spring_z_delta = _site_delta(data, p2, p1)[2]
            if abs(float(spring_z_delta)) > 1.0e-6:
                raise SystemExit(f"{prefix} 气弹簧 q=0 z 偏移应为 0，实际 {spring_z_delta:.9f}")

    for a, b in ZERO_POSE_SITE_PAIRS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a) >= 0:
            closure_delta = _site_delta(data, a, b)
            if float(np.linalg.norm(closure_delta)) > 1.0e-6:
                raise SystemExit(f"{a}/{b} q=0 未闭合，误差 {closure_delta.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-residual", type=float, default=1.0e-6)
    parser.add_argument("--max-default-connect-force", type=float, default=50.0)
    parser.add_argument("--spring-force", type=float, default=300.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    _set_default_pose(model, data)
    mujoco.mj_forward(model, data)

    _print_model_summary(model)
    _print_tendon_state(model, data)
    _check_gas_springs(model, float(args.spring_force))
    _check_joint_axes(model)
    _check_active_rod_angle_tendons(model, data)
    _check_zero_pose_geometry(model)

    residuals: list[float] = []
    if model.neq:
        residual = np.asarray(data.efc_pos[: model.neq * 3], dtype=np.float64)
        residuals.extend(float(abs(v)) for v in residual)
        print("equality_residual:", residual.tolist())

    for a, b in DEFAULT_POSE_SITE_PAIRS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a) >= 0:
            delta = _site_delta(data, a, b)
            residuals.extend(float(abs(v)) for v in delta)
            print(f"{a}_to_{b}_delta:", delta.tolist())

    force_norms = _connect_force_norms(model, data)
    for name, norm, force in force_norms:
        print(f"{name}_force:", force, f"norm={norm:.3f} N")
    max_force = max((norm for _, norm, _ in force_norms), default=0.0)
    if max_force > float(args.max_default_connect_force):
        raise SystemExit(f"默认站姿 connect 约束力过大: {max_force:.3f} N")

    for joint_name in ("lf1_Joint", "l_coupler_Joint", "rf1_Joint", "r_coupler_Joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid >= 0:
            print(f"{joint_name}: {float(data.qpos[model.jnt_qposadr[jid]]):.8f}")

    for body_name in ("l_wheel_Link", "r_wheel_Link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            print(f"{body_name}_pos:", data.xpos[bid].tolist())

    max_residual = max(residuals, default=0.0)
    print(f"max_residual={max_residual:.3e}")
    if max_residual > float(args.max_residual):
        raise SystemExit(f"闭链默认站姿 residual 超限: {max_residual:.3e}")


if __name__ == "__main__":
    main()
