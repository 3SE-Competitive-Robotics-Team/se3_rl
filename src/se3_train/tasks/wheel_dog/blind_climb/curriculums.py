"""WheelDog 盲爬任务课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    """按全局 step 阶段推进 XY 速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    step = int(env.common_step_counter)
    active = velocity_stages[0]
    for stage in velocity_stages:
        if step >= int(stage["step"]):
            active = stage
    term.cfg.lin_vel_x_range = tuple(active["lin_vel_x_range"])
    term.cfg.lin_vel_y_range = tuple(active["lin_vel_y_range"])
    term.cfg.ang_vel_yaw_range = tuple(active.get("ang_vel_yaw_range", (0.0, 0.0)))
    return {
        "cmd_x_limit": torch.tensor(abs(term.cfg.lin_vel_x_range[1]), device=env.device),
        "cmd_y_limit": torch.tensor(abs(term.cfg.lin_vel_y_range[1]), device=env.device),
        "cmd_stage_step": torch.tensor(float(active["step"]), device=env.device),
    }


def terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    success_distance: float = 1.8,
    min_progress_ratio: float = 0.25,
) -> dict[str, torch.Tensor]:
    """按盲爬完成度推进地形难度。"""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter), device=env.device),
            "progress": torch.tensor(0.0, device=env.device),
            "terrain_level": torch.tensor(0.0, device=env.device),
        }
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter), device=env.device),
            "progress": torch.tensor(0.0, device=env.device),
            "terrain_level": torch.tensor(0.0, device=env.device),
        }
    if env.common_step_counter == 0:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter), device=env.device),
            "progress": torch.tensor(0.0, device=env.device),
            "terrain_level": terrain.terrain_levels[env_ids].float().mean(),
            "terrain_type": terrain.terrain_types[env_ids].float().mean(),
            "progress_x": torch.tensor(0.0, device=env.device),
        }

    robot = env.scene["robot"]
    origin = env.scene.env_origins[env_ids]
    progress_x = robot.data.root_link_pos_w[env_ids, 0] - origin[:, 0]
    cmd = env.command_manager.get_command(command_name)[env_ids]
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    target_progress = torch.clamp(
        cmd_speed * float(env.max_episode_length_s) * float(min_progress_ratio),
        min=0.20,
    )
    move_up = progress_x > float(success_distance)
    move_down = (progress_x < target_progress) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    return {
        "step_counter": torch.tensor(float(env.common_step_counter), device=env.device),
        "progress": torch.tensor(
            min(1.0, max(0.0, float(env.common_step_counter) / max(env.max_episode_length, 1))),
            device=env.device,
        ),
        "terrain_level": terrain.terrain_levels[env_ids].float().mean(),
        "terrain_type": terrain.terrain_types[env_ids].float().mean(),
        "progress_x": progress_x.mean(),
    }


def push_disturbance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    push_stages: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    """按全局 step 阶段推进外部速度扰动范围。"""
    del env_ids
    step = int(env.common_step_counter)
    active = push_stages[0]
    for stage in push_stages:
        if step >= int(stage["step"]):
            active = stage
    env._wheel_dog_push_velocity_range = active["velocity_range"]
    vx = env._wheel_dog_push_velocity_range.get("x", (0.0, 0.0))
    vy = env._wheel_dog_push_velocity_range.get("y", (0.0, 0.0))
    return {
        "push_x_limit": torch.tensor(abs(vx[1]), device=env.device),
        "push_y_limit": torch.tensor(abs(vy[1]), device=env.device),
        "push_stage_step": torch.tensor(float(active["step"]), device=env.device),
    }


__all__ = ["commands_vel", "push_disturbance", "terrain_levels"]
