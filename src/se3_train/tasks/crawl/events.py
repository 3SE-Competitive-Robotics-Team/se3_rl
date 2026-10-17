"""趴地移动任务的事件函数：域随机化沿用 Flat，reset 改为趴姿出生。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from se3_shared.torch_constants import device_constant
from se3_train.mdp.events import (
    _apply_policy_leg_reset,
    push_robots,
    randomize_base_mass,
    randomize_com,
    randomize_default_dof_pos,
    randomize_friction,
    randomize_inertia,
    randomize_knee_spring_force,
    randomize_motor_passive_params,
    randomize_pd_gains,
    randomize_restitution,
    reset_root_state_full,
)
from se3_train.mdp.joint_indices import policy_leg_joint_ids, tensor_ids

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# 名义趴姿（policy 语义，左腿；右腿镜像取反）。2026-10-17 在导轮 MJCF 上扫 lf0 × 主动杆夹角得到：
# 机身水平、导轮着地时主轮恰好同时着地的姿态是一条连续族，主轮只能在机身后方，轮心 x ∈ [-0.32, -0.11] m，
# 腿部碰撞体离地约 3 cm。取主轮在 x ≈ -0.225 m、主动杆夹角 0.767（限位 [0, 1.51] 的正中）的一点：
# 整机质心（x ≈ -0.03）落在导轮（x ∈ [-0.020, +0.024]）与主轮之间，静止即稳定，夹角两侧都留足余量。
CRAWL_FRONT_JOINT_POS = 0.611
CRAWL_ACTIVE_ROD_ANGLE = 0.767
# 支撑点半径，与 serialleg_closed_chain_v3_guide_wheels.xml 的碰撞 geom 一致。
_MAIN_WHEEL_RADIUS_M = 0.060
_GUIDE_WHEEL_RADIUS_M = 0.010
_MAIN_WHEEL_BODIES = ("l_wheel_Link", "r_wheel_Link")
_GUIDE_WHEEL_BODIES = tuple(f"{side}_guide_wheel_{i}_Link" for side in "lr" for i in range(3))


def _support_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """缓存主轮 + 导轮 body 索引，顺序为 2 个主轮在前、6 个导轮在后。"""
    attr_name = f"_crawl_support_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list):
        return cached
    robot = env.scene[asset_cfg.name]
    names = _MAIN_WHEEL_BODIES + _GUIDE_WHEEL_BODIES
    body_ids, body_names = robot.find_bodies(names, preserve_order=True)
    if len(body_ids) != len(names):
        raise RuntimeError(f"趴地任务需要带导轮的 MJCF，实际只找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def reset_crawl_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_joint_offset_range: float = 0.10,
    active_rod_angle_offset_range: float = 0.05,
    support_clearance: float = 0.002,
) -> None:
    """把腿放到名义趴姿附近，并把机身高度校正到最低支撑点离地 support_clearance。

    左右腿各自独立加均匀噪声（lf0 ±front_joint_offset_range、主动杆夹角 ±active_rod_angle_offset_range），
    夹角越界由 _apply_policy_leg_reset 夹到 MJCF tendon 限位内，被动关节按解析闭链补齐。
    机身姿态由先执行的 reset_root_state_full 给出（只采 yaw），这里只改高度：
    取主轮底与导轮底的最低点，升降机身使其离地 support_clearance，可抬可降。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if len(env_ids) == 0:
        return
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    def _noise(half_range: float) -> torch.Tensor:
        return sample_uniform(
            device_constant(-float(half_range), device=env.device),
            device_constant(float(half_range), device=env.device),
            (n,),
            env.device,
        )

    left_front = CRAWL_FRONT_JOINT_POS + _noise(front_joint_offset_range)
    left_active = CRAWL_ACTIVE_ROD_ANGLE + _noise(active_rod_angle_offset_range)
    right_front = CRAWL_FRONT_JOINT_POS + _noise(front_joint_offset_range)
    right_active = CRAWL_ACTIVE_ROD_ANGLE + _noise(active_rod_angle_offset_range)
    # policy 顺序 lf0 / l_drive_bar / rf0 / r_drive_bar；左夹角 = lf0 - l_drive_bar，右侧镜像取反。
    policy_pos = torch.stack(
        (
            left_front,
            left_front - left_active,
            -right_front,
            -(right_front - right_active),
        ),
        dim=1,
    )
    leg_ids = tensor_ids(policy_leg_joint_ids(asset), device=env.device)
    _apply_policy_leg_reset(
        asset, joint_pos, joint_vel, leg_ids, policy_pos, torch.zeros_like(policy_pos)
    )
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    env.sim.forward()
    body_ids = _support_body_ids(env, asset_cfg)
    body_z = asset.data.body_link_pos_w[env_ids][:, body_ids, 2]
    radius = torch.tensor(
        (_MAIN_WHEEL_RADIUS_M,) * len(_MAIN_WHEEL_BODIES)
        + (_GUIDE_WHEEL_RADIUS_M,) * len(_GUIDE_WHEEL_BODIES),
        device=env.device,
        dtype=body_z.dtype,
    )
    ground_z = env.scene.env_origins[env_ids, 2].unsqueeze(1)
    lowest = (body_z - radius - ground_z).min(dim=1).values
    pos = asset.data.root_link_pos_w[env_ids].clone()
    quat = asset.data.root_link_quat_w[env_ids].clone()
    pos[:, 2] += float(support_clearance) - lowest
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    env.sim.forward()


__all__ = [
    "CRAWL_ACTIVE_ROD_ANGLE",
    "CRAWL_FRONT_JOINT_POS",
    "push_robots",
    "randomize_base_mass",
    "randomize_com",
    "randomize_default_dof_pos",
    "randomize_friction",
    "randomize_inertia",
    "randomize_knee_spring_force",
    "randomize_motor_passive_params",
    "randomize_pd_gains",
    "randomize_restitution",
    "reset_crawl_pose",
    "reset_root_state_full",
]
