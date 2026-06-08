"""台阶训练奖励和诊断项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def stair_climb_height(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    max_gain: float = 0.35,
) -> torch.Tensor:
    """奖励机器人相对指令站高的净爬升高度。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    base_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    climb = torch.clamp(base_z - cmd[:, 4], min=0.0, max=float(max_gain))
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/stair_climb_height": climb.mean().item(),
                "Stair/stair_base_z": base_z.mean().item(),
                "Stair/stair_command_height": cmd[:, 4].mean().item(),
            }
        )
    return climb


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    sigma: float = 0.25,
) -> torch.Tensor:
    """台阶前进速度跟踪奖励，保留向前梯度。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    error = torch.square(robot.data.root_link_lin_vel_b[:, 0] - cmd[:, 0])
    reward = torch.exp(-error / float(sigma))
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["Stair/forward_speed_error"] = torch.sqrt(error).mean().item()
    return reward


def stair_contact_diagnostics(
    env: ManagerBasedRlEnv,
    wheel_sensor_name: str = "wheel_sensor",
    leg_sensor_name: str = "leg_contact_sensor",
    collision_sensor_name: str = "collision_sensor",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """记录台阶训练关心的接触诊断，返回零奖励。"""
    wheel_contact = _contact_any(env, wheel_sensor_name, force_threshold)
    leg_contact = _contact_any(env, leg_sensor_name, force_threshold)
    collision_contact = _contact_any(env, collision_sensor_name, force_threshold)
    nonwheel = leg_contact | collision_contact
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/wheel_contact_rate": wheel_contact.float().mean().item(),
                "Stair/nonwheel_contact_rate": nonwheel.float().mean().item(),
                "Stair/leg_contact_rate": leg_contact.float().mean().item(),
                "Stair/base_collision_rate": collision_contact.float().mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


def ctbc_active_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 CTBC 当前是否激活，用于后续门控奖励。"""
    state = getattr(env, "stair_climb_state", None)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return state.active_mask()


def _contact_any(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force = torch.linalg.norm(data.force, dim=-1).reshape(env.num_envs, -1)
    return (force > float(force_threshold)).any(dim=1)
