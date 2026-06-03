"""42 维跳跃任务专属 actor 观测。

布局为 32 维主线 jump 观测后追加 10 维运行状态：
base_lin_vel(3) + base_height(1) + wheel_contact_forces(2) + wheel_height(1)
+ leg_contact_forces(3)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.observations import base_height_obs, base_lin_vel_obs, wheel_contact_force_obs

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def base_lin_vel_actor_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的线速度，3D。"""
    return base_lin_vel_obs(env)


def base_height_actor_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """底盘离地高度，1D。"""
    return base_height_obs(env, sensor_name=sensor_name)


def wheel_contact_force_actor_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """左右轮地面接触力模长，2D。"""
    return wheel_contact_force_obs(env, sensor_name=sensor_name)


def wheel_height_actor_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """左轮离地高度，1D。"""
    return base_height_obs(env, sensor_name=sensor_name)


def leg_contact_force_actor_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """腿部触地力摘要，3D：[最大值, 均值, 非零接触比例]。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, 3, device=env.device)

    force_norm = finite_contact_force_norm(force)
    max_force = force_norm.flatten(start_dim=1).max(dim=1).values
    mean_force = force_norm.flatten(start_dim=1).mean(dim=1)
    contact_ratio = (force_norm.flatten(start_dim=1) > 1.0).float().mean(dim=1)
    return torch.stack((max_force, mean_force, contact_ratio), dim=1)
