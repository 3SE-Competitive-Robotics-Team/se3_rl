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

DEFAULT_MODEL = Path("assets/robots/serialleg/mjcf/serialleg_closed_chain_v2_spring.xml")


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


def _site_delta(data: mujoco.MjData, a: str, b: str) -> np.ndarray:
    return np.asarray(data.site(a).xpos - data.site(b).xpos, dtype=np.float64)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-residual", type=float, default=1.0e-6)
    parser.add_argument("--spring-force", type=float, default=300.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    _set_default_pose(model, data)
    mujoco.mj_forward(model, data)

    _print_model_summary(model)
    _print_tendon_state(model, data)
    _check_gas_springs(model, float(args.spring_force))
    _check_active_rod_angle_tendons(model, data)

    residuals: list[float] = []
    if model.neq:
        residual = np.asarray(data.efc_pos[: model.neq * 3], dtype=np.float64)
        residuals.extend(float(abs(v)) for v in residual)
        print("equality_residual:", residual.tolist())

    for prefix in ("l", "r"):
        end = f"{prefix}_fourbar_end"
        target = f"{prefix}_fourbar_target"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end) >= 0:
            delta = _site_delta(data, end, target)
            residuals.extend(float(abs(v)) for v in delta)
            print(f"{prefix}_fourbar_delta:", delta.tolist())

    for joint_name in ("lf1_Joint", "rf1_Joint"):
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
