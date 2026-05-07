"""SE3 轮腿机器人的奖励函数。

所有奖励函数接收 env 并返回 [num_envs] 的张量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

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
    grace_steps = 37
    upright = _upright_factor(projected_gravity_z)
    in_grace = env.episode_length_buf < grace_steps
    return torch.where(in_grace, torch.ones_like(upright), upright)


def tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float = 2.0,
) -> torch.Tensor:
    """x 方向速度跟踪,将 v_z 折入同一 exp 核消除目标冲突。

    reward = exp(-(error_x² + vz_weight·v_z²) / sigma)
    低速时 sigma 收紧(adaptive),直立门控。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    lin_vel = robot.data.root_link_lin_vel_b
    error_x = lin_vel[:, 0] - cmd[:, 0]
    vz = lin_vel[:, 2]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    cmd_mag = torch.abs(cmd[:, 0])
    sigma = torch.where(cmd_mag < 0.2, sigma_stand, sigma_move)
    return torch.exp(-(error_x**2 + vz_weight * vz**2) / sigma) * gate


def tracking_ang_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float) -> torch.Tensor:
    """偏航角速度跟踪的 exp(-error^2/sigma),直立门控。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_z = robot.data.root_link_ang_vel_b[:, 2]
    error = ang_vel_z - cmd[:, 1]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return torch.exp(-(error**2) / sigma) * gate


def tracking_orientation(env: ManagerBasedRlEnv, command_name: str, sigma: float) -> torch.Tensor:
    """pitch/roll 姿态跟踪奖励。

    使用 projected_gravity 提取当前 pitch/roll,与指令中的目标 pitch/roll 计算误差。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg = robot.data.projected_gravity_b

    # 从 projected_gravity 估计 pitch 和 roll。
    # pg = R^T @ [0,0,-1], 在 body frame 中:
    #   pitch ≈ asin(pg_x)  (前倾为正)
    #   roll  ≈ asin(-pg_y) (右倾为正)
    current_pitch = torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0))
    current_roll = torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0))

    target_pitch = cmd[:, 2]
    target_roll = cmd[:, 3]

    pitch_error = current_pitch - target_pitch
    roll_error = current_roll - target_roll
    error_sq = pitch_error**2 + roll_error**2

    gate = _upright_factor(pg[:, 2])
    return torch.exp(-error_sq / sigma) * gate


def tracking_height(
    env: ManagerBasedRlEnv, command_name: str, sigma: float, height_sensor_name: str
) -> torch.Tensor:
    """高度跟踪奖励: exp(-error²/sigma),直立门控。

    使用 TerrainHeightSensor 获取机体离地面的垂直距离（clearance），
    适用于平地和崎岖地形。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    target_height = cmd[:, 4]
    error = torch.square(height - target_height)
    return torch.exp(-error / sigma) * gate


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


def angular_momentum(env: ManagerBasedRlEnv) -> torch.Tensor:
    """全身角动量范数平方,直立门控。

    使用 MuJoCo subtree_angmom[root_body] 获取整个机器人子树的角动量。
    对两轮倒立摆尤为重要——腿部激进动作产生的角动量脉冲会直接导致失稳。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    root_body_id = robot.data.indexing.root_body_id
    angmom = env.sim.data.subtree_angmom[:, root_body_id]
    return torch.sum(angmom**2, dim=-1) * gate


def leg_torques(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部执行器力矩平方和（position actuator 索引 0,1,2,3）。"""
    robot = env.scene[asset_cfg.name]
    leg_act_ids = [0, 1, 2, 3]
    torques = robot.data.actuator_force[:, leg_act_ids]
    return torch.sum(torques**2, dim=1)


def wheel_torques(
    env: ManagerBasedRlEnv,
    max_torque: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """轮子执行器力矩超出额定值的平方和。

    max_torque: 轮子电机额定最大力矩 (N·m)。
    """
    robot = env.scene[asset_cfg.name]
    wheel_act_ids = [4, 5]
    torques = robot.data.actuator_force[:, wheel_act_ids]
    excess = torch.clamp(torch.abs(torques) - max_torque, min=0.0)
    return torch.sum(excess**2, dim=1)


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
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """站立时关节偏差平方和,高度自适应 sigma。

    当 cmd_height 偏离 default_height 时,惩罚自动放松,
    避免高度指令与姿态惩罚的结构性矛盾。
    衰减因子: exp(-height_tolerance * (cmd_h - default_h)²)
    """
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    leg_ids = [0, 1, 3, 4]
    diff = robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
    reward = torch.sum(diff**2, dim=1)

    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    vel_scale = (cmd_norm <= command_threshold).float()

    height_deviation = cmd[:, 4] - default_height
    height_scale = torch.exp(-height_tolerance * height_deviation**2)

    return reward * vel_scale * height_scale * gate


def joint_mirror(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """左右关节位置差的平均平方,直立门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

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

    force_mag = torch.norm(data.force, dim=-1)
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

    force_mag = torch.norm(data.force, dim=-1)
    excess = torch.clamp(force_mag - threshold, min=0.0) / 100.0
    return torch.sum(excess, dim=1) * gate


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    force_threshold: float,
    cmd_threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """静止时轮子接触,直立门控。"""
    cmd = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    stationary = torch.abs(cmd[:, 0]) < cmd_threshold

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = torch.norm(data.force, dim=-1)
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
