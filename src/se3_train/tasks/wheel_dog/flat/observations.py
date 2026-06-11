"""WheelDog 平地任务观测项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.tasks.wheel_dog.robot_cfg import DOG_LEG_JOINT_IDS, DOG_WHEEL_JOINT_IDS

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系角速度，缩放 0.25。"""
    robot = env.scene["robot"]
    return robot.data.root_link_ang_vel_b * 0.25


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的重力投影。"""
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度指令观测，3D: [vx, vy, yaw_rate]。"""
    cmd = env.command_manager.get_command("base_velocity")
    scale = torch.tensor((2.0, 2.0, 0.25), device=cmd.device, dtype=cmd.dtype)
    return cmd * scale


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """12 个腿部关节相对默认姿态的位置。"""
    robot = env.scene["robot"]
    ids = list(DOG_LEG_JOINT_IDS)
    return robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


def joint_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """16 个受控关节速度，缩放 0.05。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel * 0.05


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """4 个轮子关节速度，缩放 0.05。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, list(DOG_WHEEL_JOINT_IDS)] * 0.05


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一控制步的 16 维动作。"""
    return env.action_manager.action


def base_lin_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系线速度，critic 特权观测。"""
    robot = env.scene["robot"]
    return robot.data.root_link_lin_vel_b


def wheel_contact_force_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """四个轮子的地面接触力范数，critic 特权观测。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return finite_contact_force_norm(sensor.data.force)


def base_height_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """base_link 离地高度，critic 特权观测。"""
    from mjlab.sensor import TerrainHeightSensor

    sensor: TerrainHeightSensor = env.scene[sensor_name]
    return sensor.data.heights


__all__ = [
    "base_ang_vel_obs",
    "base_height_obs",
    "base_lin_vel_obs",
    "commands_obs",
    "joint_vel_obs",
    "last_actions_obs",
    "leg_joint_pos_obs",
    "projected_gravity_obs",
    "wheel_contact_force_obs",
    "wheel_vel_obs",
]
