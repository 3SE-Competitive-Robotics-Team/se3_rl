"""生成倒地自启训练用的稳定初态缓存。

v2 缓存使用台阶任务的高保真 collision MJCF，保存 joint_names 和 split，训练端按名称
重排关节列，避免不同 MJCF 的 qpos 顺序变化污染 reset 状态。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from se3_shared import JointGroup, RobotConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MJCF_DIR = _PROJECT_ROOT / "assets" / "robots" / "serialleg" / "mjcf"
_DEFAULT_MJCF = _MJCF_DIR / "serialleg_fourbar_surrogate_stair_visualbase_coacd_train.xml"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_stair_v2.npz"
_FORMAT_VERSION = 2
_POSE_TYPE_NAMES = ("standing", "left_side", "right_side", "prone", "supine")
_POSE_NAME_TO_ID = {name: idx for idx, name in enumerate(_POSE_TYPE_NAMES)}
_FALLEN_POSE_NAMES = ("left_side", "right_side", "prone", "supine")


def _quat_wxyz_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """把 XYZ 欧拉角转换成 MuJoCo 使用的 wxyz 四元数。"""
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _sample_pose(rng: np.random.Generator, pose_name: str) -> tuple[np.ndarray, int]:
    """采样一个典型倒地姿态；返回四元数和姿态标签。"""
    yaw = float(rng.uniform(-np.pi, np.pi))
    coupled = float(rng.uniform(-0.35, 0.35))
    if pose_name == "left_side":
        roll, pitch = float(rng.uniform(1.35, np.pi)), coupled
    elif pose_name == "right_side":
        roll, pitch = -float(rng.uniform(1.35, np.pi)), coupled
    elif pose_name == "prone":
        roll, pitch = coupled, float(rng.uniform(1.35, np.pi))
    elif pose_name == "supine":
        roll, pitch = coupled, -float(rng.uniform(1.35, np.pi))
    else:
        raise ValueError(f"未知倒地姿态: {pose_name}")
    return _quat_wxyz_from_euler(roll, pitch, yaw), _POSE_NAME_TO_ID[pose_name]


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    """返回 MuJoCo 对象名称，缺名时给出稳定占位。"""
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return str(name) if name else f"{obj_type.name}_{obj_id}"


def _one_dof_joints(model: mujoco.MjModel) -> tuple[tuple[int, str, int, int], ...]:
    """返回非 freejoint 的一维关节信息: joint_id/name/qpos_addr/qvel_addr。"""
    joints: list[tuple[int, str, int, int]] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            name = _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(f"recovery cache 暂不支持非一维关节: {name}")
        joints.append(
            (
                joint_id,
                _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
        )
    return tuple(joints)


def _actuator_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """返回 actuator 名称。"""
    return tuple(_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu))


def _source_hash(path: Path) -> str:
    """计算 MJCF 文件哈希，用于 cache 和模型版本追溯。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_joint_pos(joint_names: Sequence[str]) -> np.ndarray:
    """按当前 MJCF 关节顺序生成默认关节角。"""
    robot_cfg = RobotConfig()
    default_by_name = robot_cfg.default_model_joint_pos
    return np.asarray([default_by_name.get(name, 0.0) for name in joint_names], dtype=np.float64)


def _limited_joint_value(
    model: mujoco.MjModel,
    joint_id: int,
    value: float,
) -> float:
    """把关节值裁剪到 MJCF 限位内。"""
    if int(model.jnt_limited[joint_id]) == 0:
        return float(value)
    lower, upper = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    return float(np.clip(value, lower, upper))


def _set_joint_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints: Sequence[tuple[int, str, int, int]],
    rng: np.random.Generator,
    default_joint_pos: np.ndarray,
    joint_offset_range: float,
    joint_vel_range: tuple[float, float],
    wheel_joint_vel_range: tuple[float, float],
) -> None:
    """初始化所有一维关节；轮子角度置零，速度可单独采样。"""
    wheel_names = set(JointGroup.WHEEL_NAMES)
    for column, (joint_id, name, qpos_addr, qvel_addr) in enumerate(joints):
        if name in wheel_names:
            data.qpos[qpos_addr] = 0.0
            data.qvel[qvel_addr] = rng.uniform(*wheel_joint_vel_range)
            continue
        value = default_joint_pos[column] + rng.uniform(-joint_offset_range, joint_offset_range)
        data.qpos[qpos_addr] = _limited_joint_value(model, joint_id, value)
        data.qvel[qvel_addr] = rng.uniform(*joint_vel_range)


def _reset_random_fallen(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    joints: Sequence[tuple[int, str, int, int]],
    default_joint_pos: np.ndarray,
    pose_names: Sequence[str],
    drop_height_range: tuple[float, float],
    joint_offset_range: float,
    root_lin_vel_range: tuple[float, float],
    root_ang_vel_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    wheel_joint_vel_range: tuple[float, float],
) -> int:
    """把机器人放到随机倒地姿态并从空中释放。"""
    mujoco.mj_resetData(model, data)
    pose_name = str(rng.choice(tuple(pose_names)))
    quat, pose_type = _sample_pose(rng, pose_name)
    data.qpos[0:3] = np.asarray(
        [
            rng.uniform(-0.08, 0.08),
            rng.uniform(-0.08, 0.08),
            rng.uniform(*drop_height_range),
        ],
        dtype=np.float64,
    )
    data.qpos[3:7] = quat
    data.qvel[0:3] = rng.uniform(*root_lin_vel_range, size=3)
    data.qvel[3:6] = rng.uniform(*root_ang_vel_range, size=3)
    _set_joint_state(
        model,
        data,
        joints,
        rng,
        default_joint_pos,
        joint_offset_range,
        joint_vel_range,
        wheel_joint_vel_range,
    )
    mujoco.mj_forward(model, data)
    return pose_type


def _settle_zero_torque(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    """零力矩释放并等待自然落稳。"""
    if model.nu > 0:
        data.ctrl[:] = 0.0
    for _ in range(int(steps)):
        data.qfrc_applied[:] = 0.0
        if model.nu > 0:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)


def _settle_weak_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints: Sequence[tuple[int, str, int, int]],
    steps: int,
    default_joint_pos: np.ndarray,
    leg_kp: float,
    leg_kd: float,
    max_torque: float,
) -> None:
    """诊断用弱 PD settle；正式 v2 cache 不建议使用。"""
    wheel_names = set(JointGroup.WHEEL_NAMES)
    if model.nu > 0:
        data.ctrl[:] = 0.0
    for _ in range(int(steps)):
        data.qfrc_applied[:] = 0.0
        for column, (_, name, qpos_addr, qvel_addr) in enumerate(joints):
            if name in wheel_names:
                continue
            torque = -float(leg_kp) * (
                float(data.qpos[qpos_addr]) - float(default_joint_pos[column])
            ) - float(leg_kd) * float(data.qvel[qvel_addr])
            data.qfrc_applied[qvel_addr] = np.clip(torque, -float(max_torque), float(max_torque))
        mujoco.mj_step(model, data)


def _collect_joint_pos(
    data: mujoco.MjData,
    joints: Sequence[tuple[int, str, int, int]],
) -> np.ndarray:
    """按 joint_names 顺序读取关节位置。"""
    return np.asarray([data.qpos[qpos_addr] for _, _, qpos_addr, _ in joints], dtype=np.float32)


def _collect_joint_vel(
    data: mujoco.MjData,
    joints: Sequence[tuple[int, str, int, int]],
) -> np.ndarray:
    """按 joint_names 顺序读取关节速度。"""
    return np.asarray([data.qvel[qvel_addr] for _, _, _, qvel_addr in joints], dtype=np.float32)


def _is_stable_sample(
    data: mujoco.MjData,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    min_height: float,
    max_root_speed: float,
    max_joint_abs: float,
    max_joint_speed: float,
) -> bool:
    """过滤穿模、爆速和 NaN 样本。"""
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return False
    if not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all():
        return False
    if float(data.qpos[2]) < min_height:
        return False
    if float(np.linalg.norm(data.qvel[0:6])) > max_root_speed:
        return False
    if float(np.max(np.abs(joint_pos))) > max_joint_abs:
        return False
    return float(np.max(np.abs(joint_vel))) <= max_joint_speed


def _split_labels(
    pose_type: np.ndarray,
    train_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """按姿态类别内部划分 train/eval，避免某类姿态只落在单一 split。"""
    split = np.full(pose_type.shape[0], "train", dtype="<U5")
    ratio = min(max(float(train_ratio), 0.0), 1.0)
    for pose_id in np.unique(pose_type):
        indices = np.flatnonzero(pose_type == pose_id)
        rng.shuffle(indices)
        train_count = round(len(indices) * ratio)
        split[indices[train_count:]] = "eval"
    return split


def _parse_pose_names(raw: str) -> tuple[str, ...]:
    """解析逗号分隔的姿态类别列表。"""
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise ValueError("--pose-types 不能为空")
    unsupported = [name for name in names if name not in _FALLEN_POSE_NAMES]
    if unsupported:
        raise ValueError(f"不支持的 pose type: {unsupported}; 可选 {_FALLEN_POSE_NAMES}")
    return names


def generate_cache(
    *,
    mjcf_path: Path,
    output_path: Path,
    num_states: int,
    seed: int,
    settle_seconds: float,
    settle_steps: int | None,
    settle_mode: str,
    max_attempts: int,
    min_height: float,
    max_root_speed: float,
    max_joint_abs: float,
    max_joint_speed: float,
    leg_kp: float,
    leg_kd: float,
    max_torque: float,
    train_split_ratio: float,
    pose_names: Sequence[str],
    drop_height_range: tuple[float, float],
    joint_offset_range: float,
    root_lin_vel_range: tuple[float, float],
    root_ang_vel_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    wheel_joint_vel_range: tuple[float, float],
) -> None:
    """生成并保存 v2 倒地稳定初态缓存。"""
    robot_cfg = RobotConfig()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    model.opt.timestep = robot_cfg.sim_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    joints = _one_dof_joints(model)
    joint_names = tuple(name for _, name, _, _ in joints)
    actuator_names = _actuator_names(model)
    default_joint_pos = _default_joint_pos(joint_names)
    resolved_settle_steps = (
        int(settle_steps)
        if settle_steps is not None
        else max(1, round(float(settle_seconds) / float(model.opt.timestep)))
    )

    root_pos: list[np.ndarray] = []
    root_quat: list[np.ndarray] = []
    root_lin_vel: list[np.ndarray] = []
    root_ang_vel: list[np.ndarray] = []
    joint_pos: list[np.ndarray] = []
    joint_vel: list[np.ndarray] = []
    ctrl: list[np.ndarray] = []
    pose_type: list[int] = []
    settle_used: list[int] = []
    contact_count: list[int] = []
    root_speed_norm: list[float] = []
    joint_speed_norm: list[float] = []

    attempts = 0
    while len(root_pos) < num_states and attempts < max_attempts:
        attempts += 1
        sampled_pose_type = _reset_random_fallen(
            model,
            data,
            rng,
            joints,
            default_joint_pos,
            pose_names,
            drop_height_range,
            joint_offset_range,
            root_lin_vel_range,
            root_ang_vel_range,
            joint_vel_range,
            wheel_joint_vel_range,
        )
        if settle_mode == "zero-torque":
            _settle_zero_torque(model, data, resolved_settle_steps)
        elif settle_mode == "weak-pd":
            _settle_weak_pd(
                model,
                data,
                joints,
                resolved_settle_steps,
                default_joint_pos,
                leg_kp=leg_kp,
                leg_kd=leg_kd,
                max_torque=max_torque,
            )
        else:
            raise ValueError(f"未知 settle_mode: {settle_mode}")

        sample_joint_pos = _collect_joint_pos(data, joints)
        sample_joint_vel = _collect_joint_vel(data, joints)
        if not _is_stable_sample(
            data,
            sample_joint_pos,
            sample_joint_vel,
            min_height=min_height,
            max_root_speed=max_root_speed,
            max_joint_abs=max_joint_abs,
            max_joint_speed=max_joint_speed,
        ):
            continue
        root_pos.append(np.asarray(data.qpos[0:3], dtype=np.float32).copy())
        root_quat.append(np.asarray(data.qpos[3:7], dtype=np.float32).copy())
        root_lin_vel.append(np.asarray(data.qvel[0:3], dtype=np.float32).copy())
        root_ang_vel.append(np.asarray(data.qvel[3:6], dtype=np.float32).copy())
        joint_pos.append(sample_joint_pos.copy())
        joint_vel.append(sample_joint_vel.copy())
        ctrl.append(np.asarray(data.ctrl, dtype=np.float32).copy())
        pose_type.append(sampled_pose_type)
        settle_used.append(resolved_settle_steps)
        contact_count.append(int(data.ncon))
        root_speed_norm.append(float(np.linalg.norm(data.qvel[0:6])))
        joint_speed_norm.append(float(np.linalg.norm(sample_joint_vel)))

    if len(root_pos) < num_states:
        raise RuntimeError(
            f"只生成 {len(root_pos)}/{num_states} 个可用倒地初态，"
            f"请提高 --max-attempts 或放宽速度/高度过滤。"
        )

    pose_type_array = np.asarray(pose_type, dtype=np.int64)
    split = _split_labels(pose_type_array, train_split_ratio, rng)
    metadata = {
        "format_version": _FORMAT_VERSION,
        "source_mjcf": str(mjcf_path),
        "source_mjcf_sha256": _source_hash(mjcf_path),
        "settle_mode": settle_mode,
        "settle_seconds": float(resolved_settle_steps) * float(model.opt.timestep),
        "sim_dt": float(model.opt.timestep),
        "num_states": int(num_states),
        "pose_type_names": _POSE_TYPE_NAMES,
        "train_split_ratio": float(train_split_ratio),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.asarray(_FORMAT_VERSION, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
        source_mjcf=np.asarray(str(mjcf_path), dtype=np.str_),
        source_mjcf_sha256=np.asarray(metadata["source_mjcf_sha256"], dtype=np.str_),
        joint_names=np.asarray(joint_names, dtype=np.str_),
        actuator_names=np.asarray(actuator_names, dtype=np.str_),
        pose_type_names=np.asarray(_POSE_TYPE_NAMES, dtype=np.str_),
        split=split,
        root_pos=np.stack(root_pos),
        root_quat=np.stack(root_quat),
        root_lin_vel=np.stack(root_lin_vel),
        root_ang_vel=np.stack(root_ang_vel),
        joint_pos=np.stack(joint_pos),
        joint_vel=np.stack(joint_vel),
        ctrl=(
            np.stack(ctrl)
            if ctrl and ctrl[0].size > 0
            else np.zeros((len(root_pos), 0), dtype=np.float32)
        ),
        pose_type=pose_type_array,
        settle_steps=np.asarray(settle_used, dtype=np.int64),
        contact_count=np.asarray(contact_count, dtype=np.int64),
        root_speed_norm=np.asarray(root_speed_norm, dtype=np.float32),
        joint_speed_norm=np.asarray(joint_speed_norm, dtype=np.float32),
    )
    counts = {
        _POSE_TYPE_NAMES[int(pose_id)]: int((pose_type_array == pose_id).sum())
        for pose_id in np.unique(pose_type_array)
    }
    print(
        f"saved {len(root_pos)} recovery states to {output_path} "
        f"(attempts={attempts}, accept_rate={len(root_pos) / attempts:.3f}, "
        f"settle_mode={settle_mode}, split=train:{int((split == 'train').sum())} "
        f"eval:{int((split == 'eval').sum())}, pose_counts={counts})"
    )


def _range_arg(value: str) -> tuple[float, float]:
    """解析 low,high 形式的浮点范围参数。"""
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("范围参数必须是 low,high")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成倒地自启训练 v2 初态缓存")
    parser.add_argument("--mjcf", type=Path, default=_DEFAULT_MJCF)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--num-states", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-mode", choices=("zero-torque", "weak-pd"), default="zero-torque")
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=160000)
    parser.add_argument("--min-height", type=float, default=0.04)
    parser.add_argument("--max-root-speed", type=float, default=0.35)
    parser.add_argument(
        "--max-speed", type=float, default=None, help="兼容旧参数，等同 max-root-speed"
    )
    parser.add_argument("--max-joint-abs", type=float, default=3.2)
    parser.add_argument(
        "--max-leg-abs", type=float, default=None, help="兼容旧参数，等同 max-joint-abs"
    )
    parser.add_argument("--max-joint-speed", type=float, default=1.0)
    parser.add_argument(
        "--max-leg-speed", type=float, default=None, help="兼容旧参数，等同 max-joint-speed"
    )
    parser.add_argument("--leg-kp", type=float, default=20.0)
    parser.add_argument("--leg-kd", type=float, default=1.0)
    parser.add_argument("--max-torque", type=float, default=40.0)
    parser.add_argument("--train-split-ratio", type=float, default=0.5)
    parser.add_argument("--pose-types", type=str, default="left_side,right_side,prone,supine")
    parser.add_argument("--drop-height-range", type=_range_arg, default=(0.45, 0.55))
    parser.add_argument("--joint-offset-range", type=float, default=0.35)
    parser.add_argument("--root-lin-vel-range", type=_range_arg, default=(-0.4, 0.4))
    parser.add_argument("--root-ang-vel-range", type=_range_arg, default=(-0.4, 0.4))
    parser.add_argument("--joint-vel-range", type=_range_arg, default=(-0.8, 0.8))
    parser.add_argument("--wheel-joint-vel-range", type=_range_arg, default=(-0.8, 0.8))
    args = parser.parse_args()

    max_root_speed = args.max_root_speed if args.max_speed is None else float(args.max_speed)
    max_joint_abs = args.max_joint_abs if args.max_leg_abs is None else float(args.max_leg_abs)
    max_joint_speed = (
        args.max_joint_speed if args.max_leg_speed is None else float(args.max_leg_speed)
    )
    generate_cache(
        mjcf_path=args.mjcf.resolve(),
        output_path=args.output,
        num_states=args.num_states,
        seed=args.seed,
        settle_seconds=args.settle_seconds,
        settle_steps=args.settle_steps,
        settle_mode=args.settle_mode,
        max_attempts=args.max_attempts,
        min_height=args.min_height,
        max_root_speed=max_root_speed,
        max_joint_abs=max_joint_abs,
        max_joint_speed=max_joint_speed,
        leg_kp=args.leg_kp,
        leg_kd=args.leg_kd,
        max_torque=args.max_torque,
        train_split_ratio=args.train_split_ratio,
        pose_names=_parse_pose_names(args.pose_types),
        drop_height_range=args.drop_height_range,
        joint_offset_range=args.joint_offset_range,
        root_lin_vel_range=args.root_lin_vel_range,
        root_ang_vel_range=args.root_ang_vel_range,
        joint_vel_range=args.joint_vel_range,
        wheel_joint_vel_range=args.wheel_joint_vel_range,
    )


if __name__ == "__main__":
    main()
