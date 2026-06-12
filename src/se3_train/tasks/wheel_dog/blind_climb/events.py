"""WheelDog 盲爬任务事件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from se3_train.mdp import events as shared_events
from se3_train.tasks.wheel_dog.robot_cfg import (
    DOG_BASE_HEIGHT,
    DOG_LEG_JOINT_IDS,
    DOG_WHEEL_JOINT_IDS,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

randomize_friction = shared_events.randomize_friction
randomize_restitution = shared_events.randomize_restitution
randomize_base_mass = shared_events.randomize_base_mass
randomize_inertia = shared_events.randomize_inertia
randomize_com = shared_events.randomize_com
randomize_pd_gains = shared_events.randomize_pd_gains


def reset_root_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    base_height: float = DOG_BASE_HEIGHT,
    x_noise: float = 0.06,
    y_noise: float = 0.08,
    yaw_range: tuple[float, float] = (-0.12, 0.12),
) -> None:
    """重置 base 到左平台中心附近，并让机身朝向坑坡方向。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    num_ids = len(env_ids)

    pos = root_states[:, 0:3].clone()
    pos[:, 2] = float(base_height)
    pos[:, 0] += sample_uniform(
        torch.tensor(-float(x_noise), device=env.device),
        torch.tensor(float(x_noise), device=env.device),
        (num_ids,),
        env.device,
    )
    pos[:, 1] += sample_uniform(
        torch.tensor(-float(y_noise), device=env.device),
        torch.tensor(float(y_noise), device=env.device),
        (num_ids,),
        env.device,
    )
    pos[:, 0:3] += env.scene.env_origins[env_ids]

    yaw = sample_uniform(
        torch.tensor(float(yaw_range[0]), device=env.device),
        torch.tensor(float(yaw_range[1]), device=env.device),
        (num_ids,),
        env.device,
    )
    roll = torch.zeros(num_ids, device=env.device)
    pitch = torch.zeros(num_ids, device=env.device)
    quat_delta = quat_from_euler_xyz(roll, pitch, yaw)
    new_quat = quat_mul(root_states[:, 3:7], quat_delta)

    vel = torch.zeros(num_ids, 6, device=env.device)
    asset.write_root_link_pose_to_sim(torch.cat([pos, new_quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def reset_joints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    leg_pos_noise: float = 0.03,
) -> None:
    """重置关节到默认微屈姿态，腿部加小扰动，轮子清零。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    leg_ids = list(DOG_LEG_JOINT_IDS)
    wheel_ids = list(DOG_WHEEL_JOINT_IDS)
    noise = sample_uniform(
        torch.tensor(-float(leg_pos_noise), device=env.device),
        torch.tensor(float(leg_pos_noise), device=env.device),
        (len(env_ids), len(leg_ids)),
        env.device,
    )
    joint_pos[:, leg_ids] += noise
    joint_pos[:, wheel_ids] = 0.0

    soft_limits = asset.data.soft_joint_pos_limits
    if soft_limits is not None:
        joint_pos[:, leg_ids] = torch.clamp(
            joint_pos[:, leg_ids],
            soft_limits[env_ids[:, None], leg_ids, 0],
            soft_limits[env_ids[:, None], leg_ids, 1],
        )

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def push_robots(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """按课程当前范围随机设置根速度扰动。"""
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]
    active_range = getattr(env, "_wheel_dog_push_velocity_range", velocity_range)
    range_list = [
        active_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


__all__ = [
    "push_robots",
    "randomize_base_mass",
    "randomize_com",
    "randomize_friction",
    "randomize_inertia",
    "randomize_pd_gains",
    "randomize_restitution",
    "reset_joints",
    "reset_root_state",
]
