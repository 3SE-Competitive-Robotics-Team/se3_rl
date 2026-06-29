"""用 MJLab/MJWarp 批量生成倒地自启 v2 初态缓存。"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene.scene import Scene
from mjlab.sim import Simulation
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_shared.fourbar import (
    policy_to_closedchain_passive_pos_torch,
    policy_to_closedchain_passive_vel_torch,
)
from se3_tools.recovery_state_cache import (
    _DEFAULT_MJCF,
    _DEFAULT_OUTPUT,
    _FALLEN_POSE_NAMES,
    _FORMAT_VERSION,
    _POSE_NAME_TO_ID,
    _POSE_TYPE_NAMES,
    _range_arg,
    _source_hash,
    _split_labels,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

_SHARED_ROBOT = SharedRobotConfig()
_FORMAT_VERSION_V3 = 3
_DEFAULT_V3_OUTPUT = _DEFAULT_OUTPUT.with_name("serialleg_stair_v3_preview.npz")
_FINAL_POSE_TYPE_NAMES = (
    "upright",
    "inverted",
    "side_x_pos",
    "side_x_neg",
    "side_y_pos",
    "side_y_neg",
)
_FINAL_POSE_NAME_TO_ID = {name: idx for idx, name in enumerate(_FINAL_POSE_TYPE_NAMES)}


def _cache_robot_cfg(mjcf_path: Path) -> EntityCfg:
    """构造无 actuator 的 recovery cache 专用机器人实体。"""
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(mjcf_path)),
        articulation=EntityArticulationInfoCfg(actuators=()),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, _SHARED_ROBOT.default_base_height),
            joint_pos=_SHARED_ROBOT.default_model_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


def _default_device() -> str:
    """选择默认仿真设备。"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_pose_names(raw: str) -> tuple[str, ...]:
    """解析逗号分隔的倒地姿态类别。"""
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise ValueError("--pose-types 不能为空")
    unsupported = [name for name in names if name not in _FALLEN_POSE_NAMES]
    if unsupported:
        raise ValueError(f"不支持的 pose type: {unsupported}; 可选 {_FALLEN_POSE_NAMES}")
    return names


def _parse_final_pose_names(raw: str) -> tuple[str, ...]:
    """解析逗号分隔的 settle 后最终姿态类别。"""
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise ValueError("--final-pose-types 不能为空")
    unsupported = [name for name in names if name not in _FINAL_POSE_NAME_TO_ID]
    if unsupported:
        raise ValueError(f"不支持的 final pose type: {unsupported}; 可选 {_FINAL_POSE_TYPE_NAMES}")
    return names


def _balanced_quota(num_states: int, pose_names: Sequence[str]) -> dict[int, int]:
    """按最终姿态类别生成尽量均匀的目标数量。"""
    if not pose_names:
        return {}
    base = int(num_states) // len(pose_names)
    remainder = int(num_states) % len(pose_names)
    quota: dict[int, int] = {}
    for index, name in enumerate(pose_names):
        quota[_FINAL_POSE_NAME_TO_ID[str(name)]] = base + (1 if index < remainder else 0)
    return quota


def _counts_by_name(counts: dict[int, int], names: Sequence[str]) -> dict[str, int]:
    """把类别 id 计数转换成可读名称。"""
    return {str(names[int(pose_id)]): int(value) for pose_id, value in sorted(counts.items())}


def _rand_uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """在指定设备上采样均匀分布。"""
    return torch.empty(shape, device=device).uniform_(float(low), float(high), generator=generator)


def _sample_pose_batch(
    num_envs: int,
    pose_names: Sequence[str],
    *,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """批量采样倒地姿态四元数和姿态标签。"""
    pose_name_ids = torch.tensor(
        [_POSE_NAME_TO_ID[name] for name in pose_names],
        device=device,
        dtype=torch.long,
    )
    choice = torch.randint(
        len(pose_names),
        (num_envs,),
        device=device,
        generator=generator,
    )
    pose_type = pose_name_ids[choice]
    yaw = _rand_uniform(-torch.pi, torch.pi, (num_envs,), device=device, generator=generator)
    coupled = _rand_uniform(-0.35, 0.35, (num_envs,), device=device, generator=generator)
    roll = torch.zeros(num_envs, device=device)
    pitch = torch.zeros(num_envs, device=device)

    left_id = _POSE_NAME_TO_ID["left_side"]
    right_id = _POSE_NAME_TO_ID["right_side"]
    prone_id = _POSE_NAME_TO_ID["prone"]
    supine_id = _POSE_NAME_TO_ID["supine"]

    left_mask = pose_type == left_id
    if left_mask.any():
        roll[left_mask] = _rand_uniform(
            1.35,
            torch.pi,
            (int(left_mask.sum().item()),),
            device=device,
            generator=generator,
        )
        pitch[left_mask] = coupled[left_mask]

    right_mask = pose_type == right_id
    if right_mask.any():
        roll[right_mask] = -_rand_uniform(
            1.35,
            torch.pi,
            (int(right_mask.sum().item()),),
            device=device,
            generator=generator,
        )
        pitch[right_mask] = coupled[right_mask]

    prone_mask = pose_type == prone_id
    if prone_mask.any():
        roll[prone_mask] = coupled[prone_mask]
        pitch[prone_mask] = _rand_uniform(
            1.35,
            torch.pi,
            (int(prone_mask.sum().item()),),
            device=device,
            generator=generator,
        )

    supine_mask = pose_type == supine_id
    if supine_mask.any():
        roll[supine_mask] = coupled[supine_mask]
        pitch[supine_mask] = -_rand_uniform(
            1.35,
            torch.pi,
            (int(supine_mask.sum().item()),),
            device=device,
            generator=generator,
        )

    return quat_from_euler_xyz(roll, pitch, yaw), pose_type


def _build_sim(
    num_envs: int,
    device: str,
    seed: int,
    *,
    mjcf_path: Path,
) -> tuple[Scene, Simulation]:
    """创建只用于 settle 的 MJLab 场景和仿真器。"""
    cfg = flat_env_cfg(play=True)
    cfg.scene.entities["robot"] = _cache_robot_cfg(mjcf_path)
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.sensors = ()
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = int(num_envs)
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.seed = int(seed)
    scene = Scene(cfg.scene, device=device)
    sim = Simulation(
        num_envs=scene.num_envs,
        cfg=cfg.sim,
        spec=scene.spec,
        variant_info=scene.collect_variant_info(),
        device=device,
    )
    scene.initialize(
        mj_model=sim.mj_model,
        model=sim.model,
        data=sim.data,
    )
    return scene, sim


def _joint_limits(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """读取并修正关节限位，未限位关节不参与裁剪。"""
    lower = robot.data.joint_pos_limits[0, :, 0].clone()
    upper = robot.data.joint_pos_limits[0, :, 1].clone()
    return lower, upper


def _sample_initial_state(
    *,
    scene: Scene,
    sim: Simulation,
    pose_names: Sequence[str],
    drop_height_range: tuple[float, float],
    joint_offset_range: float,
    root_lin_vel_range: tuple[float, float],
    root_ang_vel_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    wheel_joint_vel_range: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    """批量写入随机倒地初始状态，返回姿态标签。"""
    robot = scene["robot"]
    device = scene.device
    num_envs = scene.num_envs
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    root_quat, pose_type = _sample_pose_batch(
        num_envs,
        pose_names,
        device=device,
        generator=generator,
    )

    root_pos = scene.env_origins[env_ids].clone()
    root_pos[:, 0] += _rand_uniform(-0.08, 0.08, (num_envs,), device=device, generator=generator)
    root_pos[:, 1] += _rand_uniform(-0.08, 0.08, (num_envs,), device=device, generator=generator)
    root_pos[:, 2] += _rand_uniform(
        drop_height_range[0],
        drop_height_range[1],
        (num_envs,),
        device=device,
        generator=generator,
    )

    default_joint_pos = robot.data.default_joint_pos
    joint_pos = default_joint_pos.clone()
    joint_vel = torch.empty_like(robot.data.default_joint_vel)
    joint_vel.uniform_(joint_vel_range[0], joint_vel_range[1], generator=generator)
    joint_names = tuple(str(name) for name in robot.joint_names)
    wheel_names = set(JointGroup.WHEEL_NAMES)
    policy_leg_names = set(JointGroup.POLICY_LEG_NAMES)
    passive_leg_names = set(JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES)
    is_closedchain = policy_leg_names.issubset(joint_names) and passive_leg_names.issubset(
        joint_names
    )
    wheel_ids = [index for index, name in enumerate(joint_names) if name in wheel_names]
    if wheel_ids:
        wheel_id_tensor = torch.tensor(wheel_ids, device=device, dtype=torch.long)
        joint_pos[:, wheel_id_tensor] = 0.0
        joint_vel[:, wheel_id_tensor] = _rand_uniform(
            wheel_joint_vel_range[0],
            wheel_joint_vel_range[1],
            (num_envs, len(wheel_ids)),
            device=device,
            generator=generator,
        )
    if is_closedchain:
        policy_leg_ids = [joint_names.index(name) for name in JointGroup.POLICY_LEG_NAMES]
        passive_leg_ids = [
            joint_names.index(name) for name in JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES
        ]
        policy_leg_tensor = torch.tensor(policy_leg_ids, device=device, dtype=torch.long)
        passive_leg_tensor = torch.tensor(passive_leg_ids, device=device, dtype=torch.long)
        policy_pos = joint_pos[:, policy_leg_tensor].clone()
        policy_pos += _rand_uniform(
            -joint_offset_range,
            joint_offset_range,
            (num_envs, len(policy_leg_ids)),
            device=device,
            generator=generator,
        )
        policy_vel = joint_vel[:, policy_leg_tensor].clone()
        joint_pos[:, policy_leg_tensor] = policy_pos
        joint_vel[:, policy_leg_tensor] = policy_vel
        joint_pos[:, passive_leg_tensor] = policy_to_closedchain_passive_pos_torch(policy_pos)
        joint_vel[:, passive_leg_tensor] = policy_to_closedchain_passive_vel_torch(
            policy_pos,
            policy_vel,
        )
    else:
        non_wheel_ids = [index for index, name in enumerate(joint_names) if name not in wheel_names]
        if non_wheel_ids:
            non_wheel_id_tensor = torch.tensor(non_wheel_ids, device=device, dtype=torch.long)
            joint_pos[:, non_wheel_id_tensor] += _rand_uniform(
                -joint_offset_range,
                joint_offset_range,
                (num_envs, len(non_wheel_ids)),
                device=device,
                generator=generator,
            )
    lower, upper = _joint_limits(robot)
    joint_pos = torch.minimum(torch.maximum(joint_pos, lower), upper)
    if is_closedchain:
        policy_leg_ids = [joint_names.index(name) for name in JointGroup.POLICY_LEG_NAMES]
        passive_leg_ids = [
            joint_names.index(name) for name in JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES
        ]
        policy_leg_tensor = torch.tensor(policy_leg_ids, device=device, dtype=torch.long)
        passive_leg_tensor = torch.tensor(passive_leg_ids, device=device, dtype=torch.long)
        policy_pos = joint_pos[:, policy_leg_tensor].clone()
        policy_vel = joint_vel[:, policy_leg_tensor].clone()
        joint_pos[:, passive_leg_tensor] = policy_to_closedchain_passive_pos_torch(policy_pos)
        joint_vel[:, passive_leg_tensor] = policy_to_closedchain_passive_vel_torch(
            policy_pos,
            policy_vel,
        )

    root_vel = torch.cat(
        [
            _rand_uniform(
                root_lin_vel_range[0],
                root_lin_vel_range[1],
                (num_envs, 3),
                device=device,
                generator=generator,
            ),
            _rand_uniform(
                root_ang_vel_range[0],
                root_ang_vel_range[1],
                (num_envs, 3),
                device=device,
                generator=generator,
            ),
        ],
        dim=1,
    )

    sim.reset(env_ids)
    robot.reset(env_ids)
    robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    sim.data.ctrl[:] = 0.0
    sim.data.qfrc_applied[:] = 0.0
    sim.forward()
    return pose_type


def _settle_zero_torque(sim: Simulation, settle_steps: int) -> None:
    """零力矩批量 settle。"""
    for _ in range(int(settle_steps)):
        sim.data.ctrl[:] = 0.0
        sim.data.qfrc_applied[:] = 0.0
        sim.step()
    sim.data.ctrl[:] = 0.0
    sim.data.qfrc_applied[:] = 0.0
    sim.forward()


def _contact_count(sim: Simulation, num_envs: int) -> torch.Tensor:
    """按 env 统计当前活跃接触数量。"""
    contact_dim = sim.data.contact.dim
    active = contact_dim > 0
    if not active.any():
        return torch.zeros(num_envs, device=contact_dim.device, dtype=torch.long)
    world_ids = sim.data.contact.worldid[active].long()
    return torch.bincount(world_ids, minlength=num_envs)[:num_envs]


def _classify_final_pose(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """按机身坐标系重力主轴给 settle 后姿态分桶。"""
    pg = projected_gravity_b
    axis = torch.argmax(torch.abs(pg), dim=1)
    pose_type = torch.empty(pg.shape[0], device=pg.device, dtype=torch.long)

    z_mask = axis == 2
    pose_type[z_mask & (pg[:, 2] < 0.0)] = _FINAL_POSE_NAME_TO_ID["upright"]
    pose_type[z_mask & (pg[:, 2] >= 0.0)] = _FINAL_POSE_NAME_TO_ID["inverted"]

    x_mask = axis == 0
    pose_type[x_mask & (pg[:, 0] >= 0.0)] = _FINAL_POSE_NAME_TO_ID["side_x_pos"]
    pose_type[x_mask & (pg[:, 0] < 0.0)] = _FINAL_POSE_NAME_TO_ID["side_x_neg"]

    y_mask = axis == 1
    pose_type[y_mask & (pg[:, 1] >= 0.0)] = _FINAL_POSE_NAME_TO_ID["side_y_pos"]
    pose_type[y_mask & (pg[:, 1] < 0.0)] = _FINAL_POSE_NAME_TO_ID["side_y_neg"]
    return pose_type


def _joint_or_zero(
    joint_pos: torch.Tensor,
    joint_names: Sequence[str],
    name: str,
) -> torch.Tensor:
    """读取关节列；缺失时返回零列以兼容不同 MJCF。"""
    try:
        index = tuple(joint_names).index(name)
    except ValueError:
        return torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    return joint_pos[:, index]


def _bucketize(values: torch.Tensor, edges: Sequence[float]) -> torch.Tensor:
    """把连续值映射到离散分桶。"""
    edge_tensor = torch.as_tensor(tuple(edges), device=values.device, dtype=values.dtype)
    return torch.bucketize(values.contiguous(), edge_tensor).to(torch.long)


def _diversity_bucket(
    *,
    root_pos_rel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_names: Sequence[str],
    final_pose_type: torch.Tensor,
) -> torch.Tensor:
    """构造粗粒度形态桶，用于限制同构样本过度重复。"""
    lf0 = _joint_or_zero(joint_pos, joint_names, "lf0_Joint")
    rf0 = _joint_or_zero(joint_pos, joint_names, "rf0_Joint")
    lf1_abs = torch.abs(_joint_or_zero(joint_pos, joint_names, "lf1_Joint"))
    rf1_abs = torch.abs(_joint_or_zero(joint_pos, joint_names, "rf1_Joint"))
    height_bin = _bucketize(root_pos_rel[:, 2], (0.07, 0.10, 0.13, 0.16, 0.20, 0.24))
    lf0_bin = _bucketize(lf0, (-0.8, -0.4, 0.0, 0.4, 0.8))
    rf0_bin = _bucketize(rf0, (-0.8, -0.4, 0.0, 0.4, 0.8))
    lf1_bin = _bucketize(lf1_abs, (0.25, 0.65, 1.05, 1.45))
    rf1_bin = _bucketize(rf1_abs, (0.25, 0.65, 1.05, 1.45))
    return (
        final_pose_type * 100000
        + height_bin * 10000
        + lf0_bin * 1000
        + rf0_bin * 100
        + lf1_bin * 10
        + rf1_bin
    )


def _accepted_mask(
    *,
    scene: Scene,
    root_pos_rel: torch.Tensor,
    root_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    min_height: float,
    max_root_speed: float,
    max_joint_abs: float,
    max_joint_speed: float,
) -> torch.Tensor:
    """筛掉非有限值、穿地和仍在高速运动的样本。"""
    finite = (
        torch.isfinite(root_pos_rel).all(dim=1)
        & torch.isfinite(root_vel_w).all(dim=1)
        & torch.isfinite(joint_pos).all(dim=1)
        & torch.isfinite(joint_vel).all(dim=1)
    )
    root_speed = torch.linalg.norm(root_vel_w, dim=1)
    joint_abs = torch.max(torch.abs(joint_pos), dim=1).values
    joint_speed = torch.max(torch.abs(joint_vel), dim=1).values
    return (
        finite
        & (root_pos_rel[:, 2] >= float(min_height))
        & (root_speed <= float(max_root_speed))
        & (joint_abs <= float(max_joint_abs))
        & (joint_speed <= float(max_joint_speed))
    )


def _append_samples(store: dict[str, list[np.ndarray]], key: str, value: torch.Tensor) -> None:
    """把 GPU/CPU tensor 转成 numpy 并追加到缓存列表。"""
    store[key].append(value.detach().cpu().numpy())


def _choose_samples(
    *,
    mask: torch.Tensor,
    final_pose_type: torch.Tensor,
    diversity_bucket: torch.Tensor,
    remaining: int,
    final_pose_quota: dict[int, int],
    final_pose_counts: dict[int, int],
    diversity_counts: dict[int, int],
    max_per_diversity_bin: int,
) -> torch.Tensor:
    """在基础物理过滤后执行最终姿态配额和形态多样性筛选。"""
    candidate_ids = mask.nonzero(as_tuple=False).squeeze(-1)
    if candidate_ids.numel() == 0 or remaining <= 0:
        return torch.empty(0, device=mask.device, dtype=torch.long)
    if not final_pose_quota and max_per_diversity_bin <= 0:
        return candidate_ids[:remaining]

    chosen: list[int] = []
    for env_id in candidate_ids.detach().cpu().tolist():
        pose_id = int(final_pose_type[env_id].item())
        if final_pose_quota:
            if pose_id not in final_pose_quota:
                continue
            if final_pose_counts.get(pose_id, 0) >= final_pose_quota[pose_id]:
                continue

        diversity_id = int(diversity_bucket[env_id].item())
        if max_per_diversity_bin > 0 and diversity_counts.get(diversity_id, 0) >= int(
            max_per_diversity_bin
        ):
            continue

        chosen.append(int(env_id))
        if final_pose_quota:
            final_pose_counts[pose_id] = final_pose_counts.get(pose_id, 0) + 1
        if max_per_diversity_bin > 0:
            diversity_counts[diversity_id] = diversity_counts.get(diversity_id, 0) + 1
        if len(chosen) >= remaining:
            break

    if not chosen:
        return torch.empty(0, device=mask.device, dtype=torch.long)
    return torch.as_tensor(chosen, device=mask.device, dtype=torch.long)


def _sample_settle_steps(
    *,
    fixed_settle_steps: int | None,
    settle_seconds: float,
    settle_seconds_range: tuple[float, float] | None,
    timestep: float,
    generator: torch.Generator,
    device: str,
) -> int:
    """按固定步数或每批随机时长确定 settle 步数。"""
    if fixed_settle_steps is not None:
        return int(fixed_settle_steps)
    if settle_seconds_range is not None:
        seconds = float(
            torch.empty((), device=device)
            .uniform_(
                float(settle_seconds_range[0]),
                float(settle_seconds_range[1]),
                generator=generator,
            )
            .item()
        )
    else:
        seconds = float(settle_seconds)
    return max(1, round(seconds / float(timestep)))


def generate_cache_batched(
    *,
    mjcf_path: Path,
    output_path: Path,
    num_states: int,
    num_envs: int,
    seed: int,
    device: str,
    format_version: int,
    settle_seconds: float,
    settle_steps: int | None,
    settle_seconds_range: tuple[float, float] | None,
    max_batches: int,
    min_height: float,
    max_root_speed: float,
    max_joint_abs: float,
    max_joint_speed: float,
    train_split_ratio: float,
    pose_names: Sequence[str],
    balance_final_pose: bool,
    final_pose_names: Sequence[str],
    max_per_diversity_bin: int,
    drop_height_range: tuple[float, float],
    joint_offset_range: float,
    root_lin_vel_range: tuple[float, float],
    root_ang_vel_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    wheel_joint_vel_range: tuple[float, float],
) -> None:
    """用 MJLab 批量仿真生成并保存 recovery cache。"""
    if device.startswith("cuda"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(int(seed))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    scene, sim = _build_sim(num_envs=num_envs, device=device, seed=seed, mjcf_path=mjcf_path)
    robot = scene["robot"]
    sim_dt = float(sim.mj_model.opt.timestep)
    if format_version not in {_FORMAT_VERSION, _FORMAT_VERSION_V3}:
        raise ValueError(f"不支持的 format_version={format_version}; 当前支持 2 或 3")
    if balance_final_pose and format_version < 3:
        raise ValueError("--balance-final-pose 只能配合 --format-version 3 使用")
    if settle_seconds_range is not None and settle_seconds_range[1] < settle_seconds_range[0]:
        raise ValueError("--settle-seconds-range 必须满足 low <= high")
    final_pose_quota = (
        _balanced_quota(int(num_states), final_pose_names) if balance_final_pose else {}
    )
    final_pose_counts = {pose_id: 0 for pose_id in final_pose_quota}
    diversity_counts: dict[int, int] = {}
    joint_names = tuple(str(name) for name in robot.joint_names)
    actuator_names = tuple(
        name for name in (sim.mj_model.actuator(i).name for i in range(sim.mj_model.nu)) if name
    )
    pose_type_names = _FINAL_POSE_TYPE_NAMES if format_version >= 3 else _POSE_TYPE_NAMES
    source_mjcf = mjcf_path
    store: dict[str, list[np.ndarray]] = {
        "root_pos": [],
        "root_quat": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "joint_pos": [],
        "joint_vel": [],
        "pose_type": [],
        "initial_pose_type": [],
        "final_pose_type": [],
        "final_projected_gravity": [],
        "diversity_bucket": [],
        "settle_steps": [],
        "contact_count": [],
        "root_speed_norm": [],
        "joint_speed_norm": [],
    }

    accepted_total = 0
    attempted_total = 0
    start = time.perf_counter()
    for batch_index in range(1, int(max_batches) + 1):
        initial_pose_type = _sample_initial_state(
            scene=scene,
            sim=sim,
            pose_names=pose_names,
            drop_height_range=drop_height_range,
            joint_offset_range=joint_offset_range,
            root_lin_vel_range=root_lin_vel_range,
            root_ang_vel_range=root_ang_vel_range,
            joint_vel_range=joint_vel_range,
            wheel_joint_vel_range=wheel_joint_vel_range,
            generator=generator,
        )
        batch_settle_steps = _sample_settle_steps(
            fixed_settle_steps=settle_steps,
            settle_seconds=settle_seconds,
            settle_seconds_range=settle_seconds_range,
            timestep=sim_dt,
            generator=generator,
            device=device,
        )
        _settle_zero_torque(sim, batch_settle_steps)

        root_pose = robot.data.root_link_pose_w
        root_vel = robot.data.root_link_vel_w
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        root_pos_rel = root_pose[:, 0:3] - scene.env_origins
        final_projected_gravity = robot.data.projected_gravity_b
        final_pose_type = _classify_final_pose(final_projected_gravity)
        diversity = _diversity_bucket(
            root_pos_rel=root_pos_rel,
            joint_pos=joint_pos,
            joint_names=joint_names,
            final_pose_type=final_pose_type,
        )
        mask = _accepted_mask(
            scene=scene,
            root_pos_rel=root_pos_rel,
            root_vel_w=root_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            min_height=min_height,
            max_root_speed=max_root_speed,
            max_joint_abs=max_joint_abs,
            max_joint_speed=max_joint_speed,
        )
        accepted = int(mask.sum().item())
        attempted_total += int(num_envs)
        chosen = _choose_samples(
            mask=mask,
            final_pose_type=final_pose_type,
            diversity_bucket=diversity,
            remaining=int(num_states) - accepted_total,
            final_pose_quota=final_pose_quota,
            final_pose_counts=final_pose_counts,
            diversity_counts=diversity_counts,
            max_per_diversity_bin=max_per_diversity_bin,
        )
        if chosen.numel() > 0:
            _append_samples(store, "root_pos", root_pos_rel[chosen].to(torch.float32))
            _append_samples(store, "root_quat", root_pose[chosen, 3:7].to(torch.float32))
            _append_samples(store, "root_lin_vel", root_vel[chosen, 0:3].to(torch.float32))
            _append_samples(store, "root_ang_vel", root_vel[chosen, 3:6].to(torch.float32))
            _append_samples(store, "joint_pos", joint_pos[chosen].to(torch.float32))
            _append_samples(store, "joint_vel", joint_vel[chosen].to(torch.float32))
            if format_version >= 3:
                _append_samples(store, "pose_type", final_pose_type[chosen].to(torch.long))
            else:
                _append_samples(store, "pose_type", initial_pose_type[chosen].to(torch.long))
            _append_samples(store, "initial_pose_type", initial_pose_type[chosen].to(torch.long))
            _append_samples(store, "final_pose_type", final_pose_type[chosen].to(torch.long))
            _append_samples(
                store,
                "final_projected_gravity",
                final_projected_gravity[chosen].to(torch.float32),
            )
            _append_samples(store, "diversity_bucket", diversity[chosen].to(torch.long))
            settle_tensor = torch.full(
                (chosen.numel(),),
                int(batch_settle_steps),
                device=device,
                dtype=torch.long,
            )
            _append_samples(store, "settle_steps", settle_tensor)
            _append_samples(store, "contact_count", _contact_count(sim, num_envs)[chosen])
            _append_samples(store, "root_speed_norm", torch.linalg.norm(root_vel[chosen], dim=1))
            _append_samples(store, "joint_speed_norm", torch.linalg.norm(joint_vel[chosen], dim=1))
            accepted_total += int(chosen.numel())

        elapsed = max(time.perf_counter() - start, 1.0e-6)
        count_suffix = ""
        if final_pose_quota:
            count_suffix = f" final={_counts_by_name(final_pose_counts, _FINAL_POSE_TYPE_NAMES)}"
        print(
            f"batch={batch_index} candidates={accepted} selected={int(chosen.numel())} "
            f"total={accepted_total}/{num_states} attempted={attempted_total} "
            f"settle_steps={batch_settle_steps} speed={attempted_total / elapsed:.1f} envs/s"
            f"{count_suffix}",
            flush=True,
        )
        if accepted_total >= num_states:
            break

    if accepted_total < num_states:
        raise RuntimeError(
            f"只生成 {accepted_total}/{num_states} 个可用倒地初态，"
            f"请提高 --max-batches 或放宽速度/高度过滤。"
        )

    arrays = {key: np.concatenate(values, axis=0) for key, values in store.items()}
    pose_type_array = arrays["pose_type"].astype(np.int64)
    split_rng = np.random.default_rng(int(seed) + 1009)
    split = _split_labels(pose_type_array, train_split_ratio, split_rng)
    settle_seconds_value = (
        None
        if settle_seconds_range is not None and settle_steps is None
        else float(arrays["settle_steps"][0]) * sim_dt
    )
    metadata = {
        "format_version": int(format_version),
        "generator": "mjlab_batched",
        "source_mjcf": str(source_mjcf),
        "source_mjcf_sha256": _source_hash(source_mjcf),
        "settle_mode": "zero-torque",
        "settle_seconds": settle_seconds_value,
        "settle_seconds_range": (
            tuple(float(v) for v in settle_seconds_range)
            if settle_seconds_range is not None
            else None
        ),
        "sim_dt": sim_dt,
        "num_states": int(num_states),
        "num_envs": int(num_envs),
        "device": str(device),
        "pose_type_names": pose_type_names,
        "pose_type_semantics": "final_pose_type" if format_version >= 3 else "initial_pose_type",
        "initial_pose_type_names": _POSE_TYPE_NAMES,
        "final_pose_type_names": _FINAL_POSE_TYPE_NAMES,
        "train_split_ratio": float(train_split_ratio),
        "attempted_states": int(attempted_total),
        "balance_final_pose": bool(balance_final_pose),
        "final_pose_target_names": tuple(str(name) for name in final_pose_names),
        "final_pose_quota": _counts_by_name(final_pose_quota, _FINAL_POSE_TYPE_NAMES),
        "max_per_diversity_bin": int(max_per_diversity_bin),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "format_version": np.asarray(format_version, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.str_),
        "source_mjcf": np.asarray(str(source_mjcf), dtype=np.str_),
        "source_mjcf_sha256": np.asarray(metadata["source_mjcf_sha256"], dtype=np.str_),
        "joint_names": np.asarray(joint_names, dtype=np.str_),
        "actuator_names": np.asarray(actuator_names, dtype=np.str_),
        "pose_type_names": np.asarray(pose_type_names, dtype=np.str_),
        "initial_pose_type_names": np.asarray(_POSE_TYPE_NAMES, dtype=np.str_),
        "final_pose_type_names": np.asarray(_FINAL_POSE_TYPE_NAMES, dtype=np.str_),
        "split": split,
        "root_pos": arrays["root_pos"].astype(np.float32),
        "root_quat": arrays["root_quat"].astype(np.float32),
        "root_lin_vel": arrays["root_lin_vel"].astype(np.float32),
        "root_ang_vel": arrays["root_ang_vel"].astype(np.float32),
        "joint_pos": arrays["joint_pos"].astype(np.float32),
        "joint_vel": arrays["joint_vel"].astype(np.float32),
        "ctrl": np.zeros((num_states, 0), dtype=np.float32),
        "pose_type": pose_type_array,
        "initial_pose_type": arrays["initial_pose_type"].astype(np.int64),
        "final_pose_type": arrays["final_pose_type"].astype(np.int64),
        "final_projected_gravity": arrays["final_projected_gravity"].astype(np.float32),
        "diversity_bucket": arrays["diversity_bucket"].astype(np.int64),
        "settle_steps": arrays["settle_steps"].astype(np.int64),
        "contact_count": arrays["contact_count"].astype(np.int64),
        "root_speed_norm": arrays["root_speed_norm"].astype(np.float32),
        "joint_speed_norm": arrays["joint_speed_norm"].astype(np.float32),
    }
    np.savez_compressed(output_path, **save_kwargs)
    counts = {
        pose_type_names[int(pose_id)]: int((pose_type_array == pose_id).sum())
        for pose_id in np.unique(pose_type_array)
    }
    print(
        f"saved {num_states} recovery states to {output_path} "
        f"(attempted={attempted_total}, accept_rate={num_states / attempted_total:.3f}, "
        f"settle_mode=zero-torque, split=train:{int((split == 'train').sum())} "
        f"eval:{int((split == 'eval').sum())}, pose_counts={counts})"
    )


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="用 MJLab/MJWarp 批量生成 recovery 初态缓存")
    parser.add_argument("--mjcf", type=Path, default=_DEFAULT_MJCF)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-states", type=int, default=40000)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--format-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--settle-seconds-range", type=_range_arg, default=None)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--min-height", type=float, default=0.04)
    parser.add_argument("--max-root-speed", type=float, default=0.35)
    parser.add_argument("--max-joint-abs", type=float, default=3.2)
    parser.add_argument("--max-joint-speed", type=float, default=1.0)
    parser.add_argument("--train-split-ratio", type=float, default=0.5)
    parser.add_argument("--pose-types", type=str, default="left_side,right_side,prone,supine")
    parser.add_argument("--balance-final-pose", action="store_true")
    parser.add_argument(
        "--final-pose-types",
        type=str,
        default="upright,inverted,side_x_pos,side_x_neg,side_y_pos,side_y_neg",
    )
    parser.add_argument("--max-per-diversity-bin", type=int, default=0)
    parser.add_argument("--drop-height-range", type=_range_arg, default=(0.45, 0.55))
    parser.add_argument("--joint-offset-range", type=float, default=0.35)
    parser.add_argument("--root-lin-vel-range", type=_range_arg, default=(-0.4, 0.4))
    parser.add_argument("--root-ang-vel-range", type=_range_arg, default=(-0.4, 0.4))
    parser.add_argument("--joint-vel-range", type=_range_arg, default=(-0.8, 0.8))
    parser.add_argument("--wheel-joint-vel-range", type=_range_arg, default=(-0.8, 0.8))
    args = parser.parse_args()
    output_path = (
        args.output
        if args.output is not None
        else (_DEFAULT_OUTPUT if args.format_version == 2 else _DEFAULT_V3_OUTPUT)
    )

    generate_cache_batched(
        mjcf_path=args.mjcf,
        output_path=output_path,
        num_states=args.num_states,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        format_version=args.format_version,
        settle_seconds=args.settle_seconds,
        settle_steps=args.settle_steps,
        settle_seconds_range=args.settle_seconds_range,
        max_batches=args.max_batches,
        min_height=args.min_height,
        max_root_speed=args.max_root_speed,
        max_joint_abs=args.max_joint_abs,
        max_joint_speed=args.max_joint_speed,
        train_split_ratio=args.train_split_ratio,
        pose_names=_parse_pose_names(args.pose_types),
        balance_final_pose=args.balance_final_pose,
        final_pose_names=_parse_final_pose_names(args.final_pose_types),
        max_per_diversity_bin=args.max_per_diversity_bin,
        drop_height_range=args.drop_height_range,
        joint_offset_range=args.joint_offset_range,
        root_lin_vel_range=args.root_lin_vel_range,
        root_ang_vel_range=args.root_ang_vel_range,
        joint_vel_range=args.joint_vel_range,
        wheel_joint_vel_range=args.wheel_joint_vel_range,
    )


if __name__ == "__main__":
    main()
