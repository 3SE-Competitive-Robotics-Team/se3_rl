"""WheelDog 盲爬任务奖励。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.rewards import is_alive
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.tasks.wheel_dog.robot_cfg import (
    DOG_ABAD_JOINT_IDS,
    DOG_BASE_HEIGHT,
    DOG_HIP_JOINT_IDS,
    DOG_KNEE_JOINT_IDS,
    DOG_LEG_JOINT_IDS,
    DOG_WHEEL_JOINT_IDS,
)

from . import terrain_progress

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _upright_factor(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """计算直立门控，完全直立时为 1。"""
    return torch.clamp(-projected_gravity_z, 0.0, 0.7) / 0.7


def tracking_lin_vel_xy(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """XY 线速度指数跟踪奖励，参考 M20 的 track_lin_vel_xy_exp。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2]), dim=1)
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    return torch.exp(-err / (float(std) ** 2)) * gate


def forward_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_velocity: float = 1.8,
    max_backward_velocity: float = 1.0,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """按指令方向奖励世界系速度，反向运动会得到负值。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    direction = torch.where(cmd_x < 0.0, -torch.ones_like(cmd_x), torch.ones_like(cmd_x))
    speed_along_cmd = robot.data.root_link_lin_vel_w[:, 0] * direction
    speed_along_cmd = torch.nan_to_num(speed_along_cmd, nan=0.0, posinf=0.0, neginf=0.0)
    active = torch.linalg.norm(cmd[:, :2], dim=1) > float(command_threshold)
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    facility = terrain_progress.is_facility_terrain(env)
    return (
        torch.clamp(
            speed_along_cmd,
            min=-float(max_backward_velocity),
            max=float(max_velocity),
        )
        * active.float()
        * gate
        * facility.float()
    )


def progress_forward(
    env: ManagerBasedRlEnv,
    command_name: str,
    success_distance: float = terrain_progress.FINAL_SUCCESS_DISTANCE,
    max_progress_ratio: float = 1.2,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励 episode 内沿设施方向取得的相对进度。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    progress_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress_x = torch.nan_to_num(progress_x, nan=0.0, posinf=0.0, neginf=0.0)
    active = torch.linalg.norm(cmd[:, :2], dim=1) > float(command_threshold)
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    target_progress = terrain_progress.current_success_distance(
        env,
        final_success_distance=success_distance,
    )
    normalized = progress_x / torch.clamp(target_progress, min=1.0e-6)
    corridor = terrain_progress.corridor_gate(env)
    facility = terrain_progress.is_facility_terrain(env)
    return (
        torch.clamp(normalized, min=0.0, max=float(max_progress_ratio))
        * active.float()
        * gate
        * corridor
        * facility.float()
    )


def obstacle_lift(
    env: ManagerBasedRlEnv,
    command_name: str,
    before_high_edge: float = 0.25,
    after_high_edge: float = 0.35,
    max_vertical_velocity: float = 0.7,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """坑边窗口内奖励向上的 base 速度，帮助跨越反坡高边。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    progress_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress_x = torch.nan_to_num(progress_x, nan=0.0, posinf=0.0, neginf=0.0)
    vz = torch.nan_to_num(
        robot.data.root_link_lin_vel_w[:, 2],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    active = torch.linalg.norm(cmd[:, :2], dim=1) > float(command_threshold)
    start_progress, end_progress = terrain_progress.obstacle_window(
        env,
        before_high_edge=before_high_edge,
        after_high_edge=after_high_edge,
    )
    in_window = (progress_x > start_progress) & (progress_x < end_progress)
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    corridor = terrain_progress.corridor_gate(env)
    facility = terrain_progress.is_facility_terrain(env)
    return (
        torch.clamp(vz, min=0.0, max=float(max_vertical_velocity))
        * active.float()
        * in_window.float()
        * gate
        * corridor
        * facility.float()
    )


def success_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    success_distance: float = terrain_progress.FINAL_SUCCESS_DISTANCE,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """到达右侧安全区域后的稀疏完成奖励。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    progress_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress_x = torch.nan_to_num(progress_x, nan=0.0, posinf=0.0, neginf=0.0)
    active = torch.linalg.norm(cmd[:, :2], dim=1) > float(command_threshold)
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    target_progress = terrain_progress.current_success_distance(
        env,
        final_success_distance=success_distance,
    )
    in_corridor = terrain_progress.within_corridor(env)
    facility = terrain_progress.is_facility_terrain(env)
    return (
        (progress_x > target_progress).float()
        * in_corridor.float()
        * active.float()
        * gate
        * facility.float()
    )


def lateral_corridor(
    env: ManagerBasedRlEnv,
    soft_half_width: float = terrain_progress.CORRIDOR_SOFT_HALF_WIDTH,
    hard_half_width: float = terrain_progress.CORRIDOR_HARD_HALF_WIDTH,
) -> torch.Tensor:
    """惩罚偏离中心通道，防止沿全局地面绕开坑坡。"""
    abs_y = torch.abs(terrain_progress.lateral_offset(env))
    soft = float(soft_half_width)
    hard = max(float(hard_half_width), soft + 1.0e-6)
    excess = torch.clamp(abs_y - soft, min=0.0) / (hard - soft)
    facility = terrain_progress.is_facility_terrain(env)
    return torch.clamp(torch.square(excess), max=4.0) * facility.float()


def run_stuck(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_speed: float = 0.25,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """移动指令下惩罚长期低于最小前向速度。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    direction = torch.where(cmd_x < 0.0, -torch.ones_like(cmd_x), torch.ones_like(cmd_x))
    speed_along_cmd = robot.data.root_link_lin_vel_w[:, 0] * direction
    speed_along_cmd = torch.nan_to_num(speed_along_cmd, nan=0.0, posinf=0.0, neginf=0.0)
    active = torch.linalg.norm(cmd[:, :2], dim=1) > float(command_threshold)
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    facility = terrain_progress.is_facility_terrain(env)
    deficit = (float(min_speed) - speed_along_cmd) / max(float(min_speed), 1.0e-6)
    return torch.clamp(deficit, min=0.0, max=1.0) * active.float() * gate * facility.float()


def tracking_ang_vel_z(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """yaw 角速度指数跟踪奖励。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.square(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    return torch.exp(-err / (float(std) ** 2)) * gate


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """基座 z 方向速度平方惩罚。"""
    robot = env.scene[asset_cfg.name]
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    return torch.square(robot.data.root_link_lin_vel_b[:, 2]) * gate


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """roll/pitch 角速度平方惩罚。"""
    robot = env.scene[asset_cfg.name]
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    ang_vel = robot.data.root_link_ang_vel_b
    return (torch.square(ang_vel[:, 0]) + torch.square(ang_vel[:, 1])) * gate


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """机身 pitch/roll 倾斜惩罚。"""
    robot = env.scene[asset_cfg.name]
    pg = robot.data.projected_gravity_b
    return torch.square(pg[:, 0]) + torch.square(pg[:, 1])


def upward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """直立奖励，给恢复阶段持续梯度。"""
    robot = env.scene[asset_cfg.name]
    return _upright_factor(robot.data.projected_gravity_b[:, 2])


def base_height_l2(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float = DOG_BASE_HEIGHT,
) -> torch.Tensor:
    """base_link 高度平方误差惩罚。"""
    sensor: TerrainHeightSensor = env.scene[sensor_name]
    height = sensor.data.heights[:, 0]
    return torch.square(height - float(target_height))


def bad_tilt(
    env: ManagerBasedRlEnv,
    soft_limit_deg: float = 18.0,
    hard_limit_deg: float = 45.0,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """超过软倾角后的 barrier 惩罚。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    soft = torch.deg2rad(torch.tensor(float(soft_limit_deg), device=env.device))
    hard = torch.deg2rad(torch.tensor(float(hard_limit_deg), device=env.device))
    excess = torch.clamp((tilt - soft) / torch.clamp(hard - soft, min=1.0e-6), min=0.0)
    return torch.clamp(torch.square(excess), max=float(max_penalty))


def joint_torques_l2(
    env: ManagerBasedRlEnv,
    joint_ids: tuple[int, ...] = DOG_LEG_JOINT_IDS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """指定关节执行器力矩平方和。"""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.actuator_force[:, list(joint_ids)]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRlEnv,
    joint_ids: tuple[int, ...] = DOG_LEG_JOINT_IDS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """指定关节加速度平方和。"""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_acc[:, list(joint_ids)]), dim=1)


def joint_power(
    env: ManagerBasedRlEnv,
    joint_ids: tuple[int, ...] = DOG_LEG_JOINT_IDS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """指定关节 |力矩 * 速度| 之和。"""
    robot = env.scene[asset_cfg.name]
    ids = list(joint_ids)
    return torch.sum(
        torch.abs(robot.data.actuator_force[:, ids] * robot.data.joint_vel[:, ids]), dim=1
    )


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """动作变化率平方惩罚。"""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """原始动作平方惩罚，抑制策略输出远超控制契约的目标。"""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def joint_pos_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    joint_ids: tuple[int, ...],
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """关节偏离默认姿态惩罚，静止时更严格。"""
    robot = env.scene[asset_cfg.name]
    ids = list(joint_ids)
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_speed = torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
    diff = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    penalty = torch.linalg.norm(diff, dim=1)
    stand = (cmd_norm < command_threshold) & (body_speed < velocity_threshold)
    return torch.where(stand, float(stand_still_scale) * penalty, penalty)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """零速度指令下的全腿关节偏离惩罚。"""
    robot = env.scene[asset_cfg.name]
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    ids = list(DOG_LEG_JOINT_IDS)
    diff = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    return torch.sum(torch.square(diff), dim=1) * (cmd_norm < command_threshold).float()


def dof_pos_limits(
    env: ManagerBasedRlEnv,
    joint_ids: tuple[int, ...] = DOG_LEG_JOINT_IDS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """关节软限位违规惩罚。"""
    robot = env.scene[asset_cfg.name]
    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
        return torch.zeros(env.num_envs, device=env.device)
    ids = list(joint_ids)
    pos = robot.data.joint_pos[:, ids]
    limits = soft_limits[:, ids]
    out = -(pos - limits[:, :, 0]).clip(max=0.0)
    out += (pos - limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out, dim=1)


def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """非轮子 body 接触地面的计数惩罚。"""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force = finite_contact_force_norm(sensor.data.force)
    return (force > float(force_threshold)).float().sum(dim=1)


def contact_forces(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 120.0,
) -> torch.Tensor:
    """轮子接触力超过阈值的惩罚。"""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force = finite_contact_force_norm(sensor.data.force)
    excess = torch.clamp(force - float(threshold), min=0.0) / 100.0
    return torch.sum(excess, dim=1)


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """静止指令下奖励轮子接触地面。"""
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force = finite_contact_force_norm(sensor.data.force)
    contacts = (force > float(force_threshold)).float().sum(dim=1)
    return contacts * (cmd_norm < command_threshold).float()


__all__ = [
    "DOG_ABAD_JOINT_IDS",
    "DOG_HIP_JOINT_IDS",
    "DOG_KNEE_JOINT_IDS",
    "DOG_LEG_JOINT_IDS",
    "DOG_WHEEL_JOINT_IDS",
    "action_l2",
    "action_rate_l2",
    "ang_vel_xy_l2",
    "bad_tilt",
    "base_height_l2",
    "contact_forces",
    "dof_pos_limits",
    "feet_contact_without_cmd",
    "flat_orientation_l2",
    "forward_velocity",
    "is_alive",
    "joint_acc_l2",
    "joint_pos_penalty",
    "joint_power",
    "joint_torques_l2",
    "lateral_corridor",
    "lin_vel_z_l2",
    "obstacle_lift",
    "progress_forward",
    "run_stuck",
    "stand_still",
    "success_progress",
    "tracking_ang_vel_z",
    "tracking_lin_vel_xy",
    "undesired_contacts",
    "upward",
]
