"""WheelDog 盲爬任务课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import terrain_progress

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
    success_distance: float = terrain_progress.FINAL_SUCCESS_DISTANCE,
    min_progress_ratio: float = 0.25,
    healthy_height_margin: float = -0.02,
    healthy_tilt_limit_deg: float = 35.0,
    top_sample_min_level: int = 35,
    high_level_threshold: int = 35,
    target_level_threshold: int = 39,
    success_half_width: float = terrain_progress.CORRIDOR_SUCCESS_HALF_WIDTH,
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
    progress_x = torch.nan_to_num(progress_x, nan=0.0, posinf=0.0, neginf=0.0)
    cmd = env.command_manager.get_command(command_name)[env_ids]
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    episode_target = cmd_speed * float(env.max_episode_length_s) * float(min_progress_ratio)
    current_success = terrain_progress.current_success_distance(
        env,
        final_success_distance=success_distance,
        env_ids=env_ids,
    )
    facility_target = current_success * float(min_progress_ratio)
    target_progress = torch.clamp(torch.minimum(episode_target, facility_target), min=0.15)
    robot = env.scene["robot"]
    rel_height = robot.data.root_link_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[env_ids, 2], -1.0, 1.0))
    healthy = (rel_height > float(healthy_height_margin)) & (
        tilt < torch.deg2rad(torch.tensor(float(healthy_tilt_limit_deg), device=env.device))
    )
    lateral_offset = terrain_progress.lateral_offset(env, env_ids=env_ids)
    in_corridor = torch.abs(lateral_offset) <= float(success_half_width)
    healthy = healthy & in_corridor
    move_up = (progress_x > current_success) & healthy
    move_down = (progress_x < target_progress) & ~move_up
    num_rows = int(terrain.terrain_origins.shape[0])
    top_sample_min_level = max(0, min(int(top_sample_min_level), num_rows - 1))
    new_levels = terrain.terrain_levels[env_ids] + move_up.long() - move_down.long()
    new_levels = torch.clamp(new_levels, 0, num_rows - 1)
    top_mask = move_up & (terrain.terrain_levels[env_ids] >= num_rows - 1)
    if top_mask.any():
        new_levels[top_mask] = torch.randint(
            top_sample_min_level,
            num_rows,
            (int(top_mask.sum().item()),),
            device=env.device,
            dtype=new_levels.dtype,
        )
    terrain.terrain_levels[env_ids] = new_levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    new_levels, num_rows = terrain_progress.current_terrain_levels(env, env_ids=env_ids)
    difficulty = terrain_progress.current_difficulty(env, env_ids=env_ids)
    all_levels, _ = terrain_progress.current_terrain_levels(env)
    all_difficulty = terrain_progress.current_difficulty(env)
    target_level_threshold = max(0, min(int(target_level_threshold), num_rows - 1))
    high_level_threshold = max(0, min(int(high_level_threshold), num_rows - 1))
    return {
        "step_counter": torch.tensor(float(env.common_step_counter), device=env.device),
        "progress": torch.tensor(
            min(1.0, max(0.0, float(env.common_step_counter) / max(env.max_episode_length, 1))),
            device=env.device,
        ),
        "terrain_level": new_levels.float().mean(),
        "terrain_type": terrain.terrain_types[env_ids].float().mean(),
        "terrain_difficulty": difficulty.mean(),
        "max_terrain_level": torch.tensor(float(num_rows - 1), device=env.device),
        "progress_x": progress_x.mean(),
        "success_progress_x": current_success.mean(),
        "target_progress_x": target_progress.mean(),
        "healthy_ratio": healthy.float().mean(),
        "in_corridor_ratio": in_corridor.float().mean(),
        "lateral_offset_abs": torch.abs(lateral_offset).mean(),
        "move_up_ratio": move_up.float().mean(),
        "move_down_ratio": move_down.float().mean(),
        "global_terrain_level": all_levels.float().mean(),
        "global_terrain_difficulty": all_difficulty.mean(),
        "high_level_ratio": (all_levels >= high_level_threshold).float().mean(),
        "target_level_ratio": (all_levels >= target_level_threshold).float().mean(),
        "top_sample_min_level": torch.tensor(float(top_sample_min_level), device=env.device),
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
