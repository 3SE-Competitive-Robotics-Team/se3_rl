"""SE3 轮腿机器人的观测函数。

观测空间:
- actor: 31D (原 27D + pitch/roll/height cmd 扩展)
- critic: actor + 特权信息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import JointGroup, ObservationConfig

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 模块级观测配置，作为缩放系数的单一来源
_OBS_CFG = ObservationConfig()


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return robot.data.root_link_ang_vel_b * _OBS_CFG.ang_vel_scale


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/姿态/高度指令,缩放 (2.0, 0.25, 5.0, 5.0, 5.0)。

    5 维: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    """
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor(list(_OBS_CFG.command_scale), device=cmd.device)
    return cmd * scale


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部关节位置（相对默认位姿），4D。"""
    robot = env.scene["robot"]
    return (
        robot.data.joint_pos[:, JointGroup.LEGS] - robot.data.default_joint_pos[:, JointGroup.LEGS]
    )


def leg_joint_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部关节速度,缩放 0.25,4D。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, JointGroup.LEGS] * _OBS_CFG.leg_vel_scale


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节位置（MJCF 已修正轴方向,无需手动取反）。"""
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, JointGroup.WHEELS]


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,缩放 0.05（MJCF 已修正轴方向）。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, JointGroup.WHEELS] * _OBS_CFG.wheel_vel_scale


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
