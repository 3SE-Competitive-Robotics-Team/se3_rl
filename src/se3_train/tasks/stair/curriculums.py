"""台阶任务使用的课程函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.curriculums import commands_height, commands_vel, push_disturbance
from se3_train.tasks.stair.rewards import stair_wheel_support_rise
from se3_train.tasks.stair.terrain_curriculum import (
    DEFAULT_BUCKET_WEIGHT_STAGES,
    DEFAULT_LEVEL_BUCKETS,
    DEFAULT_LEVEL_MAX_STAGES,
    sample_levels,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height


def stair_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    asset_name: str = "robot",
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    move_up_distance_ratio: float = 0.10,
    move_down_distance_ratio: float = 0.06,
    move_up_min_steps: float = 2.0,
    hold_height_tolerance_m: float = 0.02,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    contact_force_threshold_n: float = 1.0,
    wheel_radius_m: float = 0.060,
    wheel_clearance_tol_m: float = 0.035,
    support_duration_s: float = 0.10,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = ("inv_pyramid_stairs",),
    walking_phase_iterations: int = 0,
    flat_terrain_type_name: str = "flat",
    steps_per_policy_iter: int = 64,
    fixed_iteration: int | None = None,
    max_level_stages: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_MAX_STAGES,
    level_buckets: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_BUCKETS,
    bucket_weight_stages: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = DEFAULT_BUCKET_WEIGHT_STAGES,
) -> dict[str, torch.Tensor]:
    """按训练迭代数开放最高 row，并在低/中/高 bucket 中持续采样。"""
    terrain = env.scene.terrain
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
        return _zero_log(env)

    ids = _env_ids_tensor(env, env_ids)
    if ids.numel() == 0:
        return _zero_log(env)

    if fixed_iteration is None:
        iteration = int(getattr(env, "common_step_counter", 0)) // max(
            1, int(steps_per_policy_iter)
        )
    else:
        iteration = max(0, int(fixed_iteration))
    if iteration < max(0, int(walking_phase_iterations)):
        _set_walking_phase_terrain(
            env,
            terrain,
            ids,
            flat_terrain_type_name=flat_terrain_type_name,
        )
        logs = _zero_log(env)
        logs["walking_phase"] = torch.tensor(1.0, device=env.device)
        logs["iteration"] = torch.tensor(float(iteration), device=env.device)
        return logs

    newly_entered = _restore_training_terrain_types(env, terrain, ids)
    origins = env.scene.env_origins[ids]
    robot = env.scene[asset_name]
    root_pos = robot.data.root_link_pos_w[ids]
    valid_episode = (env.episode_length_buf[ids] > 0) & (~newly_entered)
    stair_mask = _terrain_type_mask(terrain, ids, terrain_type_names, env.device)

    del standing_height, upright_threshold, support_duration_s, hold_height_tolerance_m
    height_gain_all = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
        use_episode_max=True,
    )
    current_height_gain_all = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
        use_episode_max=False,
    )
    height_gain = torch.nan_to_num(
        height_gain_all[ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    current_height_gain = torch.nan_to_num(
        current_height_gain_all[ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    distance = torch.nan_to_num(
        torch.norm(root_pos[:, :2] - origins[:, :2], dim=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    terrain_size_x = float(
        getattr(getattr(terrain.cfg, "terrain_generator", None), "size", (8.0, 8.0))[0]
    )
    move_up_distance = terrain_size_x * float(move_up_distance_ratio)
    move_down_distance = terrain_size_x * float(move_down_distance_ratio)
    target_level = terrain.terrain_levels[ids].clone()
    sampled_logs = {
        "max_allowed_level": torch.tensor(0.0, device=env.device),
        "sampled_max_level": torch.tensor(0.0, device=env.device),
        "bucket_low_rate": torch.tensor(0.0, device=env.device),
        "bucket_mid_rate": torch.tensor(0.0, device=env.device),
        "bucket_high_rate": torch.tensor(0.0, device=env.device),
    }
    if torch.any(stair_mask):
        sampled_levels, sampled_logs = sample_levels(
            env,
            terrain,
            int(stair_mask.sum().item()),
            iteration,
            max_level_stages=max_level_stages,
            level_buckets=level_buckets,
            bucket_weight_stages=bucket_weight_stages,
        )
        target_level[stair_mask] = sampled_levels
    target_levels_float = target_level.float()
    step_height = _step_height_for_levels(terrain, target_levels_float, step_height_range)
    move_up_height = step_height * float(move_up_min_steps)
    state = getattr(env, "stair_climb_state", None)
    if state is not None:
        support_duration_all = state.max_wheel_supported_both_duration()
    else:
        support_duration_all = torch.zeros(env.num_envs, device=env.device)
    support_duration = torch.nan_to_num(
        support_duration_all[ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    move_up = stair_mask & (terrain.terrain_levels[ids] < target_level)
    move_down = stair_mask & (terrain.terrain_levels[ids] > target_level)
    _set_env_levels(terrain, ids, target_level=target_level, active_mask=stair_mask)

    levels = terrain.terrain_levels[ids].float()
    valid_stair = stair_mask & valid_episode
    level_max = (
        torch.max(levels[stair_mask])
        if torch.any(stair_mask)
        else torch.tensor(0.0, device=env.device)
    )
    return {
        "level_mean": _masked_mean(levels, stair_mask),
        "level_max": level_max,
        "move_up_rate": _masked_mean(move_up.float(), valid_stair),
        "move_down_rate": _masked_mean(move_down.float(), valid_stair),
        "height_gain_mean": _masked_mean(height_gain, valid_stair),
        "current_height_gain_mean": _masked_mean(current_height_gain, valid_stair),
        "support_drop_mean": _masked_mean(
            torch.clamp(height_gain - current_height_gain, min=0.0),
            valid_stair,
        ),
        "support_duration_mean": _masked_mean(support_duration, valid_stair),
        "distance_mean": _masked_mean(distance, valid_stair),
        "move_up_distance": torch.tensor(float(move_up_distance), device=env.device),
        "move_down_distance": torch.tensor(float(move_down_distance), device=env.device),
        "move_up_height_mean": _masked_mean(move_up_height, valid_stair),
        "target_level": _masked_mean(target_level.float(), stair_mask),
        "target_level_max_allowed": sampled_logs["max_allowed_level"],
        "target_level_sampled_max": sampled_logs["sampled_max_level"],
        "bucket_low_rate": sampled_logs["bucket_low_rate"],
        "bucket_mid_rate": sampled_logs["bucket_mid_rate"],
        "bucket_high_rate": sampled_logs["bucket_high_rate"],
        "stair_env_rate": torch.mean(stair_mask.float()),
        "walking_phase": torch.tensor(0.0, device=env.device),
        "iteration": torch.tensor(float(iteration), device=env.device),
    }


def _set_walking_phase_terrain(
    env: ManagerBasedRlEnv,
    terrain,
    ids: torch.Tensor,
    *,
    flat_terrain_type_name: str,
) -> None:
    original_types = getattr(env, "_stair_training_terrain_types", None)
    if not isinstance(original_types, torch.Tensor):
        env._stair_training_terrain_types = terrain.terrain_types.clone()
        env._stair_training_phase_started = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    flat_type = _terrain_type_index(terrain, flat_terrain_type_name)
    terrain.terrain_levels[ids] = 0
    terrain.terrain_types[ids] = flat_type
    terrain.env_origins[ids] = terrain.terrain_origins[0, flat_type]
    env._stair_training_phase_started[ids] = False


def _restore_training_terrain_types(
    env: ManagerBasedRlEnv,
    terrain,
    ids: torch.Tensor,
) -> torch.Tensor:
    original_types = getattr(env, "_stair_training_terrain_types", None)
    phase_started = getattr(env, "_stair_training_phase_started", None)
    if not isinstance(original_types, torch.Tensor) or not isinstance(phase_started, torch.Tensor):
        return torch.zeros(ids.shape, device=env.device, dtype=torch.bool)

    newly_entered = ~phase_started[ids]
    terrain.terrain_types[ids] = original_types[ids]
    terrain.env_origins[ids] = terrain.terrain_origins[
        terrain.terrain_levels[ids],
        terrain.terrain_types[ids],
    ]
    phase_started[ids] = True
    return newly_entered


def _terrain_type_index(terrain, terrain_type_name: str) -> int:
    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_names = tuple((getattr(generator_cfg, "sub_terrains", {}) or {}).keys())
    if terrain_type_name not in terrain_names:
        raise ValueError(f"未知地形类型: {terrain_type_name}")
    return terrain_names.index(terrain_type_name)


def _step_height_for_levels(
    terrain,
    levels: torch.Tensor,
    step_height_range: tuple[float, float],
) -> torch.Tensor:
    """按 terrain row 估算当前台阶高度。"""
    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    num_rows = max(1, int(getattr(generator_cfg, "num_rows", 10)) - 1)
    min_height, max_height = (float(step_height_range[0]), float(step_height_range[1]))
    alpha = torch.clamp(levels, min=0.0, max=float(num_rows)) / float(num_rows)
    return min_height + alpha * (max_height - min_height)


def _set_env_levels(
    terrain,
    ids: torch.Tensor,
    *,
    target_level: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    """把本次 reset 的台阶环境放到由 iter 决定的地形 row。"""
    selected = ids[active_mask]
    if selected.numel() == 0:
        return
    terrain.terrain_levels[selected] = target_level[active_mask]
    terrain.env_origins[selected] = terrain.terrain_origins[
        terrain.terrain_levels[selected],
        terrain.terrain_types[selected],
    ]


def _env_ids_tensor(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)


def _terrain_type_mask(
    terrain,
    env_ids: torch.Tensor,
    terrain_type_names: tuple[str, ...],
    device: torch.device | str,
) -> torch.Tensor:
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return torch.zeros(env_ids.shape, device=device, dtype=torch.bool)

    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    selected = {str(name) for name in terrain_type_names}
    env_terrain_types = terrain_types.to(device=device)[env_ids]
    mask = torch.zeros(env_ids.shape, device=device, dtype=torch.bool)
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if str(terrain_name) in selected:
            mask |= env_terrain_types == terrain_index
    return mask


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return torch.tensor(0.0, device=value.device)
    selected = torch.nan_to_num(
        value[mask].float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return selected.mean()


def _zero_log(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, device=env.device)
    return {
        "level_mean": zero,
        "level_max": zero,
        "move_up_rate": zero,
        "move_down_rate": zero,
        "height_gain_mean": zero,
        "current_height_gain_mean": zero,
        "support_drop_mean": zero,
        "support_duration_mean": zero,
        "distance_mean": zero,
        "move_up_distance": zero,
        "move_down_distance": zero,
        "move_up_height_mean": zero,
        "target_level": zero,
        "target_level_max_allowed": zero,
        "target_level_sampled_max": zero,
        "bucket_low_rate": zero,
        "bucket_mid_rate": zero,
        "bucket_high_rate": zero,
        "stair_env_rate": zero,
        "walking_phase": zero,
        "iteration": zero,
    }


__all__ = [
    "commands_height",
    "commands_vel",
    "push_disturbance",
    "stair_terrain_levels",
]
