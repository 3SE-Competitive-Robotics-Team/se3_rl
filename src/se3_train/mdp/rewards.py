"""SE3 轮腿机器人的奖励函数。

所有奖励函数接收 env 并返回 [num_envs] 的张量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _upright_factor(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """计算直立因子:clamp(-pg_z, 0, 0.7) / 0.7。"""
    return torch.clamp(-projected_gravity_z, 0.0, 0.7) / 0.7


def _recovery_penalty_gate(
    env: ManagerBasedRlEnv, projected_gravity_z: torch.Tensor
) -> torch.Tensor:
    """在宽限期内使用 1.0,否则使用直立因子。"""
    # 宽限期 0.75s,与参考实现对齐:0.75 / (0.005 * 4) = 37 控制步。
    grace_steps = 37
    in_grace = env.episode_length_buf < grace_steps
    upright = _upright_factor(projected_gravity_z)
    return torch.where(in_grace, torch.ones_like(upright), upright)


def tracking_lin_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float) -> torch.Tensor:
    """x 方向速度跟踪的 exp(-error^2/sigma),直立门控。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_x = robot.data.root_link_lin_vel_b[:, 0]
    error = lin_vel_x - cmd[:, 0]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return torch.exp(-(error**2) / sigma) * gate


def tracking_ang_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float) -> torch.Tensor:
    """偏航角速度跟踪的 exp(-error^2/sigma),直立门控。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_z = robot.data.root_link_ang_vel_b[:, 2]
    error = ang_vel_z - cmd[:, 1]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return torch.exp(-(error**2) / sigma) * gate


def upward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """直立因子:倒地=0,直立=1。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    return _upright_factor(pg_z)


def lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座 z 方向速度的平方,直立门控。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return robot.data.root_link_lin_vel_b[:, 2] ** 2 * gate


def ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    """横滚/俯仰角速度平方和,直立门控。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    ang_vel = robot.data.root_link_ang_vel_b
    return (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2) * gate


def base_height(env: ManagerBasedRlEnv, target: float) -> torch.Tensor:
    """exp(-error²/0.05),直立门控。正权重奖励接近目标高度。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    height = robot.data.root_link_pos_w[:, 2]
    error = torch.square(height - target)
    return torch.exp(-error / 0.05) * gate


def leg_torques(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部执行器力矩平方和（position actuator 索引 0,1,2,3）。"""
    robot = env.scene[asset_cfg.name]
    leg_act_ids = [0, 1, 2, 3]
    torques = robot.data.actuator_force[:, leg_act_ids]
    return torch.sum(torques**2, dim=1)


def leg_dof_acc(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部关节加速度平方和(排除轮子)。"""
    robot = env.scene[asset_cfg.name]
    acc = robot.data.joint_acc
    leg_ids = [0, 1, 3, 4]
    return torch.sum(acc[:, leg_ids] ** 2, dim=1)


def leg_power(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部关节 |力矩 * 速度| 之和。"""
    robot = env.scene[asset_cfg.name]
    leg_joint_ids = [0, 1, 3, 4]
    leg_act_ids = [0, 1, 2, 3]
    torques = robot.data.actuator_force[:, leg_act_ids]
    vel = robot.data.joint_vel[:, leg_joint_ids]
    return torch.sum(torch.abs(torques * vel), dim=1)


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """当前动作与上一动作差值的平方和。"""
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def stand_still(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """命令 < 0.1 时 |关节位置 - 默认位置| 之和,直立门控。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("velocity_height")
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    # 线速度和偏航速度的联合范数 < 0.1 才视为站立,与参考实现对齐。
    standing = torch.linalg.norm(cmd[:, :2], dim=1) < 0.1

    default_pos = robot.data.default_joint_pos
    leg_ids = [0, 1, 3, 4]
    deviation = torch.sum(
        torch.abs(robot.data.joint_pos[:, leg_ids] - default_pos[:, leg_ids]), dim=1
    )
    return deviation * gate * standing.float()


def joint_pos_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """关节偏差(L2 范数),站立时放大 5 倍,直立门控。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("velocity_height")
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    default_pos = robot.data.default_joint_pos
    leg_ids = [0, 1, 3, 4]
    deviation = torch.linalg.norm(robot.data.joint_pos[:, leg_ids] - default_pos[:, leg_ids], dim=1)

    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    body_vel = torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
    stand_still_scale = 5.0
    command_threshold = 0.1
    velocity_threshold = 0.5
    reward = torch.where(
        torch.logical_or(cmd_mag > command_threshold, body_vel > velocity_threshold),
        deviation,
        stand_still_scale * deviation,
    )
    return reward * gate


def joint_mirror(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """左右关节位置差的平均平方,直立门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    # 左腿:索引 0, 1。右腿:索引 3, 4。
    # 2 对镜像:(lf0, rf0) 和 (lf1, rf1)。
    diff = robot.data.joint_pos[:, [0, 1]] - robot.data.joint_pos[:, [3, 4]]
    num_pairs = 2
    return torch.sum(diff**2, dim=1) / num_pairs * gate


def collision(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """受惩罚的身体接触计数,恢复惩罚门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = torch.norm(data.force, dim=-1)  # [B, N]
    contact_count = (force_mag > 0.1).float().sum(dim=1)
    return contact_count * gate


def contact_forces(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """轮子接触力超过阈值的部分,除以 100 归一化,恢复门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = torch.norm(data.force, dim=-1)  # [B, N]
    excess = torch.clamp(force_mag - threshold, min=0.0) / 100.0
    return torch.sum(excess, dim=1) * gate


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    force_threshold: float,
    cmd_threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """静止时轮子接触,直立门控。"""
    cmd = env.command_manager.get_command("velocity_height")
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    stationary = torch.abs(cmd[:, 0]) < cmd_threshold

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = torch.norm(data.force, dim=-1)  # [B, N]
    has_contact = (force_mag > force_threshold).float()
    return torch.sum(has_contact, dim=1) * gate * stationary.float()


def dof_pos_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """关节限位违规惩罚(仅腿部关节:0,1,3,4)。"""
    robot = env.scene[asset_cfg.name]
    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
        return torch.zeros(env.num_envs, device=env.device)

    leg_ids = [0, 1, 3, 4]
    pos = robot.data.joint_pos[:, leg_ids]
    limits = soft_limits[:, leg_ids]

    out_of_limits = -(pos - limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (pos - limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
