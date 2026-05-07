"""SE3 轮腿机器人的域随机化事件。

与原始 Isaac Gym 实现配置保持一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from se3_shared import JointGroup

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_full(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """重置 base 到默认站立状态,yaw 随机,xy 小偏移。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    n = len(env_ids)
    pos = root_states[:, 0:3].clone()
    pos[:, 0] += sample_uniform(
        torch.tensor(-0.1, device=env.device),
        torch.tensor(0.1, device=env.device),
        (n,),
        env.device,
    )
    pos[:, 1] += sample_uniform(
        torch.tensor(-0.1, device=env.device),
        torch.tensor(0.1, device=env.device),
        (n,),
        env.device,
    )
    pos[:, 0:3] += env.scene.env_origins[env_ids]

    # 仅随机化 yaw,保持直立。
    yaw = sample_uniform(
        torch.tensor(-torch.pi, device=env.device),
        torch.tensor(torch.pi, device=env.device),
        (n,),
        env.device,
    )
    roll = torch.zeros(n, device=env.device)
    pitch = torch.zeros(n, device=env.device)
    quat_delta = quat_from_euler_xyz(roll, pitch, yaw)
    default_quat = root_states[:, 3:7]
    new_quat = quat_mul(default_quat, quat_delta)

    vel = torch.zeros(n, 6, device=env.device)

    asset.write_root_link_pose_to_sim(torch.cat([pos, new_quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def reset_joints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """重置关节位置到默认站立姿态(default_joint_pos)附近小范围随机。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    joint_pos[:, JointGroup.WHEELS] = 0.0

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def push_robots(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机速度推动。"""
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]

    range_list = [
        velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def randomize_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化几何体摩擦系数。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    friction = sample_uniform(
        torch.tensor(friction_range[0], device=env.device),
        torch.tensor(friction_range[1], device=env.device),
        (n, 1),
        env.device,
    )

    # 写入该实体的所有几何体。
    geom_ids = asset_cfg.geom_ids
    if isinstance(geom_ids, slice):
        env.sim.model.geom_friction[env_ids, :, 0] = friction
    else:
        for gid in geom_ids:
            env.sim.model.geom_friction[env_ids, gid, 0] = friction.squeeze(-1)


def randomize_restitution(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    restitution_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化几何体恢复系数。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    _ = sample_uniform(
        torch.tensor(restitution_range[0], device=env.device),
        torch.tensor(restitution_range[1], device=env.device),
        (n, 1),
        env.device,
    )

    geom_ids = asset_cfg.geom_ids
    if isinstance(geom_ids, slice):
        env.sim.model.geom_margin[env_ids, :] = 0.0
        # MuJoCo 没有完全相同的逐几何体恢复系数;
        # 我们使用 solref/solimp 来设置接触属性。
        # 这是随机化范围的占位符。
    else:
        for gid in geom_ids:
            env.sim.model.geom_margin[env_ids, gid] = 0.0


def randomize_base_mass(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化基座连杆附加质量。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    default_mass = env.sim.get_default_field("body_mass")
    base_body_idx = 0  # base_link 是第 0 个 body。

    added_mass = sample_uniform(
        torch.tensor(mass_range[0], device=env.device),
        torch.tensor(mass_range[1], device=env.device),
        (n,),
        env.device,
    )

    env.sim.model.body_mass[env_ids, base_body_idx] = default_mass[base_body_idx] + added_mass


def randomize_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    inertia_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化基座连杆惯性。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    n = len(env_ids)
    default_inertia = env.sim.get_default_field("body_inertia")
    base_body_idx = 0

    scale = sample_uniform(
        torch.tensor(inertia_range[0], device=env.device),
        torch.tensor(inertia_range[1], device=env.device),
        (n, 3),
        env.device,
    )

    env.sim.model.body_inertia[env_ids, base_body_idx] = default_inertia[base_body_idx] * scale


def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    com_range: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化基座连杆质心偏移。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    n = len(env_ids)
    default_ipos = env.sim.get_default_field("body_ipos")
    base_body_idx = 0

    offset = sample_uniform(
        torch.tensor(-com_range, device=env.device),
        torch.tensor(com_range, device=env.device),
        (n, 3),
        env.device,
    )

    env.sim.model.body_ipos[env_ids, base_body_idx] = default_ipos[base_body_idx] + offset


def randomize_pd_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化 PD 增益(缩放默认增益)。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    kp_scale = sample_uniform(
        torch.tensor(kp_range[0], device=env.device),
        torch.tensor(kp_range[1], device=env.device),
        (n, 1),
        env.device,
    )
    kd_scale = sample_uniform(
        torch.tensor(kd_range[0], device=env.device),
        torch.tensor(kd_range[1], device=env.device),
        (n, 1),
        env.device,
    )

    # 随机化执行器增益(gainprm 和 biasprm)。
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")

    actuator_ids = asset_cfg.actuator_ids
    if isinstance(actuator_ids, slice):
        env.sim.model.actuator_gainprm[env_ids, :, 0] = default_gainprm[:, 0] * kp_scale
        env.sim.model.actuator_biasprm[env_ids, :, 1] = default_biasprm[:, 1] * kp_scale
        env.sim.model.actuator_biasprm[env_ids, :, 2] = default_biasprm[:, 2] * kd_scale
    else:
        for aid in actuator_ids:
            env.sim.model.actuator_gainprm[env_ids, aid, 0] = default_gainprm[
                aid, 0
            ] * kp_scale.squeeze(-1)
            env.sim.model.actuator_biasprm[env_ids, aid, 1] = default_biasprm[
                aid, 1
            ] * kp_scale.squeeze(-1)
            env.sim.model.actuator_biasprm[env_ids, aid, 2] = default_biasprm[
                aid, 2
            ] * kd_scale.squeeze(-1)


def randomize_default_dof_pos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化默认关节位置。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)

    offset = sample_uniform(
        torch.tensor(offset_range[0], device=env.device),
        torch.tensor(offset_range[1], device=env.device),
        (n, asset.num_joints),
        env.device,
    )

    default_joint_pos = asset.data.default_joint_pos.clone()
    default_joint_pos[env_ids] += offset

    # 裁剪到关节限制范围内。
    soft_limits = asset.data.soft_joint_pos_limits
    if soft_limits is not None:
        default_joint_pos[env_ids] = torch.clamp(
            default_joint_pos[env_ids],
            soft_limits[env_ids, :, 0],
            soft_limits[env_ids, :, 1],
        )

    asset.data.default_joint_pos[env_ids] = default_joint_pos[env_ids]
