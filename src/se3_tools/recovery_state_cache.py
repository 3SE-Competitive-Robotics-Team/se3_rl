"""生成倒地自启训练用的稳定初态缓存。

脚本只负责离线采样物理上已经落稳的 root / joint 状态。训练端读取
``assets/recovery_states/serialleg_flat_v1.npz`` 后，可避免每次 reset 都从
不真实的随机悬空姿态开始探索。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from se3_shared import RobotConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MJCF = (
    _PROJECT_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_fidelity_cylinder_wheels.xml"
)
_DEFAULT_OUTPUT = _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_flat_v1.npz"
_JOINT_QPOS_SLICE = slice(7, 17)
_JOINT_QVEL_SLICE = slice(6, 16)
_CTRL_LEG_COLUMNS = np.asarray([0, 1, 5, 6], dtype=np.int64)
_CTRL_LEG_QPOS = np.asarray([7, 8, 12, 13], dtype=np.int64)
_CTRL_LEG_QVEL = np.asarray([6, 7, 11, 12], dtype=np.int64)


def _quat_wxyz_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """把 XYZ 欧拉角转换成 MuJoCo 使用的 wxyz 四元数。"""
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _sample_pose(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """采样典型倒地姿态；返回四元数和姿态标签。"""
    pose_type = int(rng.integers(1, 5))
    yaw = float(rng.uniform(-np.pi, np.pi))
    coupled = float(rng.uniform(-0.35, 0.35))
    if pose_type == 1:  # 左侧翻
        roll, pitch = float(rng.uniform(1.35, np.pi)), coupled
    elif pose_type == 2:  # 右侧翻
        roll, pitch = -float(rng.uniform(1.35, np.pi)), coupled
    elif pose_type == 3:  # 前趴
        roll, pitch = coupled, float(rng.uniform(1.35, np.pi))
    else:  # 后仰
        roll, pitch = coupled, -float(rng.uniform(1.35, np.pi))
    return _quat_wxyz_from_euler(roll, pitch, yaw), pose_type


def _reset_random_fallen(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    default_joint_pos: np.ndarray,
) -> int:
    """把机器人放到随机倒地姿态并从空中释放。"""
    mujoco.mj_resetData(model, data)
    quat, pose_type = _sample_pose(rng)
    data.qpos[0:3] = np.asarray(
        [rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.35, 0.55)]
    )
    data.qpos[3:7] = quat
    data.qvel[0:6] = rng.uniform(-0.4, 0.4, size=6)

    joint_offset = rng.uniform(-0.35, 0.35, size=10)
    data.qpos[_JOINT_QPOS_SLICE] = default_joint_pos + joint_offset
    data.qpos[9] = 0.0
    data.qpos[14] = 0.0
    data.qvel[_JOINT_QVEL_SLICE] = rng.uniform(-0.8, 0.8, size=10)
    mujoco.mj_forward(model, data)
    return pose_type


def _settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int,
    default_joint_pos: np.ndarray,
    leg_kp: float,
    leg_kd: float,
    max_torque: float,
) -> None:
    """弱 PD 扶住腿部关节后落地并等待速度衰减。"""
    data.ctrl[:] = 0.0
    for _ in range(int(steps)):
        data.qfrc_applied[:] = 0.0
        leg_error = data.qpos[_CTRL_LEG_QPOS] - default_joint_pos[_CTRL_LEG_COLUMNS]
        leg_vel = data.qvel[_CTRL_LEG_QVEL]
        torque = -float(leg_kp) * leg_error - float(leg_kd) * leg_vel
        data.qfrc_applied[_CTRL_LEG_QVEL] = np.clip(torque, -float(max_torque), float(max_torque))
        mujoco.mj_step(model, data)


def _is_stable_sample(
    data: mujoco.MjData,
    min_height: float,
    max_speed: float,
    max_leg_abs: float,
    max_leg_speed: float,
) -> bool:
    """过滤穿模、爆速和 NaN 样本。"""
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return False
    if float(data.qpos[2]) < min_height:
        return False
    if float(np.linalg.norm(data.qvel[0:6])) > max_speed:
        return False
    if float(np.max(np.abs(data.qpos[_CTRL_LEG_QPOS]))) > max_leg_abs:
        return False
    return float(np.max(np.abs(data.qvel[_CTRL_LEG_QVEL]))) <= max_leg_speed


def generate_cache(
    *,
    mjcf_path: Path,
    output_path: Path,
    num_states: int,
    seed: int,
    settle_steps: int,
    max_attempts: int,
    min_height: float,
    max_speed: float,
    max_leg_abs: float,
    max_leg_speed: float,
    leg_kp: float,
    leg_kd: float,
    max_torque: float,
) -> None:
    """生成并保存倒地稳定初态缓存。"""
    robot_cfg = RobotConfig()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    model.opt.timestep = robot_cfg.sim_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    default_joint_pos = np.asarray(robot_cfg.default_dof_pos, dtype=np.float64)
    full_default = np.zeros(10, dtype=np.float64)
    full_default[[0, 1, 2, 5, 6, 7]] = default_joint_pos

    root_pos: list[np.ndarray] = []
    root_quat: list[np.ndarray] = []
    root_lin_vel: list[np.ndarray] = []
    root_ang_vel: list[np.ndarray] = []
    joint_pos: list[np.ndarray] = []
    joint_vel: list[np.ndarray] = []
    pose_type: list[int] = []
    settle_used: list[int] = []

    attempts = 0
    while len(root_pos) < num_states and attempts < max_attempts:
        attempts += 1
        sampled_pose_type = _reset_random_fallen(model, data, rng, full_default)
        _settle(
            model,
            data,
            settle_steps,
            full_default,
            leg_kp=leg_kp,
            leg_kd=leg_kd,
            max_torque=max_torque,
        )
        if not _is_stable_sample(
            data,
            min_height=min_height,
            max_speed=max_speed,
            max_leg_abs=max_leg_abs,
            max_leg_speed=max_leg_speed,
        ):
            continue
        root_pos.append(np.asarray(data.qpos[0:3], dtype=np.float32).copy())
        root_quat.append(np.asarray(data.qpos[3:7], dtype=np.float32).copy())
        root_lin_vel.append(np.asarray(data.qvel[0:3], dtype=np.float32).copy())
        root_ang_vel.append(np.asarray(data.qvel[3:6], dtype=np.float32).copy())
        joint_pos.append(np.asarray(data.qpos[_JOINT_QPOS_SLICE], dtype=np.float32).copy())
        joint_vel.append(np.asarray(data.qvel[_JOINT_QVEL_SLICE], dtype=np.float32).copy())
        pose_type.append(sampled_pose_type)
        settle_used.append(settle_steps)

    if len(root_pos) < num_states:
        raise RuntimeError(
            f"只生成 {len(root_pos)}/{num_states} 个可用倒地初态，"
            f"请提高 --max-attempts 或放宽 --max-speed。"
        )

    if len(root_pos) == 0:
        raise RuntimeError("没有生成任何可用倒地初态，请放宽过滤条件或检查 MJCF 接触。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        root_pos=np.stack(root_pos),
        root_quat=np.stack(root_quat),
        root_lin_vel=np.stack(root_lin_vel),
        root_ang_vel=np.stack(root_ang_vel),
        joint_pos=np.stack(joint_pos),
        joint_vel=np.stack(joint_vel),
        pose_type=np.asarray(pose_type, dtype=np.int64),
        settle_steps=np.asarray(settle_used, dtype=np.int64),
    )
    print(
        f"saved {len(root_pos)} recovery states to {output_path} "
        f"(attempts={attempts}, accept_rate={len(root_pos) / attempts:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成倒地自启训练初态缓存")
    parser.add_argument("--mjcf", type=Path, default=_DEFAULT_MJCF)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--num-states", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=80000)
    parser.add_argument("--min-height", type=float, default=0.04)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--max-leg-abs", type=float, default=2.5)
    parser.add_argument("--max-leg-speed", type=float, default=3.0)
    parser.add_argument("--leg-kp", type=float, default=20.0)
    parser.add_argument("--leg-kd", type=float, default=1.0)
    parser.add_argument("--max-torque", type=float, default=40.0)
    args = parser.parse_args()
    generate_cache(
        mjcf_path=args.mjcf,
        output_path=args.output,
        num_states=args.num_states,
        seed=args.seed,
        settle_steps=args.settle_steps,
        max_attempts=args.max_attempts,
        min_height=args.min_height,
        max_speed=args.max_speed,
        max_leg_abs=args.max_leg_abs,
        max_leg_speed=args.max_leg_speed,
        leg_kp=args.leg_kp,
        leg_kd=args.leg_kd,
        max_torque=args.max_torque,
    )


if __name__ == "__main__":
    main()
