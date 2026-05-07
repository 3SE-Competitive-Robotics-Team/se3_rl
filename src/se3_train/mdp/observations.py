"""SE3 轮腿机器人的观测函数（27D 关节空间）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_LEG_IDS = [0, 1, 3, 4]


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return robot.data.root_link_ang_vel_b * 0.25


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/高度指令,缩放 (2.0, 0.25, 5.0)。"""
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor([2.0, 0.25, 5.0], device=cmd.device)
    return cmd * scale


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部关节位置（相对默认位姿），4D。"""
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, _LEG_IDS] - robot.data.default_joint_pos[:, _LEG_IDS]


def leg_joint_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部关节速度,缩放 0.25,4D。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, _LEG_IDS] * 0.25


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节位置（MJCF 已修正轴方向,无需手动取反）。"""
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, [2, 5]]


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,缩放 0.05（MJCF 已修正轴方向）。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, [2, 5]] * 0.05


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一步的 6 个动作。"""
    return env.action_manager.action


# --- Critic 特权观测 ---


def base_lin_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的线速度,3D（特权信息,actor 不可见）。"""
    robot = env.scene["robot"]
    return robot.data.root_link_lin_vel_b


def wheel_contact_force_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """轮子地面接触力标量,2D（特权信息）。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return torch.norm(data.force, dim=-1)
