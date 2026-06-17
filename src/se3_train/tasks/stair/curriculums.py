"""台阶任务使用的课程函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.curriculums import commands_height, commands_vel, push_disturbance

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height


def stair_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    asset_name: str = "robot",
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    command_name: str = "velocity_height",
    move_up_distance_ratio: float = 0.35,
    move_down_distance_ratio: float = 0.12,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = ("inv_pyramid_stairs",),
    walking_phase_iterations: int = 0,
    flat_terrain_type_name: str = "flat",
    steps_per_policy_iter: int = 64,
    fixed_iteration: int | None = None,
    success_threshold: float = 0.8,
    low_stair_ratio: float = 0.20,
    flat_ratio: float = 0.05,
    min_success_samples: int = 128,
    initial_target_level: int = 0,
) -> dict[str, torch.Tensor]:
    """全局台阶难度课程：达标后升一级，不回退。"""
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

    newly_entered = _mark_stair_phase_started(env, ids)
    target_level = _ensure_mastery_state(
        env,
        terrain,
        initial_target_level=initial_target_level,
    )
    robot = env.scene[asset_name]
    origins = env.scene.env_origins[ids]
    root_pos = robot.data.root_link_pos_w[ids]
    pg_z = robot.data.projected_gravity_b[ids, 2]
    valid_episode = (env.episode_length_buf[ids] > 0) & (~newly_entered)
    stair_mask = _terrain_type_mask(terrain, ids, terrain_type_names, env.device)

    height_gain = (root_pos[:, 2] - origins[:, 2]) - float(standing_height)
    target_direction = _target_direction_w(env, ids, command_name=command_name)
    offset_xy = root_pos[:, :2] - origins[:, :2]
    distance = torch.sum(offset_xy * target_direction, dim=1)
    terrain_size_x = float(
        getattr(getattr(terrain.cfg, "terrain_generator", None), "size", (8.0, 8.0))[0]
    )
    move_up_distance = terrain_size_x * float(move_up_distance_ratio)
    move_down_distance = terrain_size_x * float(move_down_distance_ratio)

    upright = pg_z < float(upright_threshold)
    target_episode = (
        stair_mask
        & valid_episode
        & (terrain.terrain_levels[ids].to(device=env.device) == int(target_level))
    )
    success = target_episode & upright & (distance > move_up_distance)
    failure = target_episode & ((~upright) | (distance < move_down_distance))
    current_success_count = int(success.sum().item())
    current_failure_count = int(failure.sum().item())
    current_target_count = int(target_episode.sum().item())
    window_success_count = _mastery_window_count(env, "success") + current_success_count
    window_failure_count = _mastery_window_count(env, "failure") + current_failure_count
    window_target_count = _mastery_window_count(env, "target") + current_target_count
    env._stair_mastery_success_count = window_success_count
    env._stair_mastery_failure_count = window_failure_count
    env._stair_mastery_target_count = window_target_count
    success_rate = window_success_count / max(1, window_target_count)
    failure_rate = window_failure_count / max(1, window_target_count)

    upgraded = False
    evaluated = False
    num_rows = int(terrain.terrain_origins.shape[0])
    max_target_level = max(0, num_rows - 1)
    if window_target_count >= max(1, int(min_success_samples)):
        evaluated = True
        if success_rate >= float(success_threshold) and target_level < max_target_level:
            target_level += 1
            env._stair_mastery_target_level = target_level
            env._stair_mastery_upgrade_count = _mastery_upgrade_count(env) + 1
            upgraded = True
        _reset_mastery_window(env)

    assignment = _assign_mastery_terrain(
        env,
        terrain,
        ids,
        target_level=target_level,
        terrain_type_names=terrain_type_names,
        flat_terrain_type_name=flat_terrain_type_name,
        low_stair_ratio=low_stair_ratio,
        flat_ratio=flat_ratio,
    )

    levels = terrain.terrain_levels[ids].float()
    assigned_stair = _terrain_type_mask(terrain, ids, terrain_type_names, env.device)
    level_max = (
        torch.max(levels[assigned_stair])
        if torch.any(assigned_stair)
        else torch.tensor(0.0, device=env.device)
    )
    return {
        "level_mean": _masked_mean(levels, assigned_stair),
        "level_max": level_max,
        "move_up_rate": torch.tensor(float(success_rate), device=env.device),
        "move_down_rate": torch.tensor(0.0, device=env.device),
        "height_gain_mean": _masked_mean(height_gain, target_episode),
        "distance_mean": _masked_mean(distance, target_episode),
        "stair_env_rate": torch.mean(assigned_stair.float()),
        "walking_phase": torch.tensor(0.0, device=env.device),
        "iteration": torch.tensor(float(iteration), device=env.device),
        "target_level": torch.tensor(float(target_level), device=env.device),
        "target_success_rate": torch.tensor(float(success_rate), device=env.device),
        "target_failure_rate": torch.tensor(float(failure_rate), device=env.device),
        "target_sample_count": torch.tensor(float(window_target_count), device=env.device),
        "target_window_evaluated": torch.tensor(float(evaluated), device=env.device),
        "upgraded": torch.tensor(float(upgraded), device=env.device),
        "upgrade_count": torch.tensor(float(_mastery_upgrade_count(env)), device=env.device),
        "target_sample_rate": assignment["target_rate"],
        "low_stair_sample_rate": assignment["low_stair_rate"],
        "flat_sample_rate": assignment["flat_rate"],
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


def _mark_stair_phase_started(env: ManagerBasedRlEnv, ids: torch.Tensor) -> torch.Tensor:
    phase_started = getattr(env, "_stair_training_phase_started", None)
    if not isinstance(phase_started, torch.Tensor):
        return torch.zeros(ids.shape, device=env.device, dtype=torch.bool)

    newly_entered = ~phase_started[ids]
    phase_started[ids] = True
    return newly_entered


def _terrain_type_index(terrain, terrain_type_name: str) -> int:
    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_names = tuple((getattr(generator_cfg, "sub_terrains", {}) or {}).keys())
    if terrain_type_name not in terrain_names:
        raise ValueError(f"未知地形类型: {terrain_type_name}")
    return terrain_names.index(terrain_type_name)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _target_direction_w(
    env: ManagerBasedRlEnv,
    ids: torch.Tensor,
    *,
    command_name: str,
) -> torch.Tensor:
    try:
        term = env.command_manager.get_term(command_name)
        direction = getattr(term, "target_direction_w", None)
    except Exception:
        direction = None
    if isinstance(direction, torch.Tensor) and direction.shape == (env.num_envs, 2):
        result = direction.to(device=env.device, dtype=torch.float32)[ids]
    else:
        robot = env.scene["robot"]
        yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w[ids])
        result = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    return result / torch.linalg.norm(result, dim=1, keepdim=True).clamp_min(1.0e-6)


def _assign_mastery_terrain(
    env: ManagerBasedRlEnv,
    terrain,
    ids: torch.Tensor,
    *,
    target_level: int,
    terrain_type_names: tuple[str, ...],
    flat_terrain_type_name: str,
    low_stair_ratio: float,
    flat_ratio: float,
) -> dict[str, torch.Tensor]:
    device = env.device
    num_rows = int(terrain.terrain_origins.shape[0])
    target_level = max(0, min(int(target_level), num_rows - 1))
    flat_type = _terrain_type_index(terrain, flat_terrain_type_name)
    stair_type = _terrain_type_index(terrain, terrain_type_names[0])
    flat_ratio = min(max(float(flat_ratio), 0.0), 0.95)
    low_stair_ratio = min(max(float(low_stair_ratio), 0.0), 1.0 - flat_ratio)

    n = int(ids.numel())
    if n == 0:
        zero = torch.tensor(0.0, device=device)
        return {"target_rate": zero, "low_stair_rate": zero, "flat_rate": zero}

    random_values = torch.rand(n, device=device)
    flat_mask = random_values < flat_ratio
    nonflat_mask = ~flat_mask
    low_prob_inside_stairs = low_stair_ratio / max(1.0e-6, 1.0 - flat_ratio)
    low_mask = (
        nonflat_mask
        & (target_level > 0)
        & (torch.rand(n, device=device) < low_prob_inside_stairs)
    )
    target_mask = nonflat_mask & (~low_mask)

    selected_types = torch.full((n,), stair_type, device=device, dtype=terrain.terrain_types.dtype)
    selected_types[flat_mask] = flat_type
    selected_levels = torch.full(
        (n,),
        target_level,
        device=device,
        dtype=terrain.terrain_levels.dtype,
    )
    selected_levels[flat_mask] = 0
    if target_level > 0 and torch.any(low_mask):
        selected_levels[low_mask] = torch.randint(
            0,
            target_level,
            (int(low_mask.sum().item()),),
            device=device,
            dtype=terrain.terrain_levels.dtype,
        )

    terrain.terrain_types[ids] = selected_types
    terrain.terrain_levels[ids] = selected_levels
    terrain.env_origins[ids] = terrain.terrain_origins[
        terrain.terrain_levels[ids],
        terrain.terrain_types[ids],
    ]
    return {
        "target_rate": target_mask.float().mean(),
        "low_stair_rate": low_mask.float().mean(),
        "flat_rate": flat_mask.float().mean(),
    }


def _ensure_mastery_state(
    env: ManagerBasedRlEnv,
    terrain,
    *,
    initial_target_level: int,
) -> int:
    current = getattr(env, "_stair_mastery_target_level", None)
    num_rows = int(terrain.terrain_origins.shape[0])
    if current is None:
        current = max(0, min(int(initial_target_level), max(0, num_rows - 1)))
        env._stair_mastery_target_level = current
        env._stair_mastery_upgrade_count = 0
        _reset_mastery_window(env)
    return int(current)


def _mastery_upgrade_count(env: ManagerBasedRlEnv) -> int:
    return int(getattr(env, "_stair_mastery_upgrade_count", 0))


def _mastery_window_count(env: ManagerBasedRlEnv, name: str) -> int:
    return int(getattr(env, f"_stair_mastery_{name}_count", 0))


def _reset_mastery_window(env: ManagerBasedRlEnv) -> None:
    env._stair_mastery_success_count = 0
    env._stair_mastery_failure_count = 0
    env._stair_mastery_target_count = 0


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
    finite_mask = mask & torch.isfinite(value)
    if not torch.any(finite_mask):
        return torch.tensor(0.0, device=value.device)
    return value[finite_mask].float().mean()


def _zero_log(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, device=env.device)
    return {
        "level_mean": zero,
        "level_max": zero,
        "move_up_rate": zero,
        "move_down_rate": zero,
        "height_gain_mean": zero,
        "distance_mean": zero,
        "stair_env_rate": zero,
        "walking_phase": zero,
        "iteration": zero,
        "target_level": zero,
        "target_success_rate": zero,
        "target_failure_rate": zero,
        "target_sample_count": zero,
        "target_window_evaluated": zero,
        "upgraded": zero,
        "upgrade_count": zero,
        "target_sample_rate": zero,
        "low_stair_sample_rate": zero,
        "flat_sample_rate": zero,
    }


__all__ = [
    "commands_height",
    "commands_vel",
    "push_disturbance",
    "stair_terrain_levels",
]
