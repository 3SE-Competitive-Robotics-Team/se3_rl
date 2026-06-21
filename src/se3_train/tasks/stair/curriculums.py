"""台阶任务使用的课程函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.curriculums import commands_height, commands_vel, push_disturbance

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def stair_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    asset_name: str = "robot",
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = ("inv_pyramid_stairs",),
    walking_phase_iterations: int = 0,
    flat_terrain_type_name: str = "flat",
    steps_per_policy_iter: int = 64,
    fixed_iteration: int | None = None,
    step_height_range: tuple[float, float] = (0.04, 0.22),
    success_steps: int = 3,
    height_sensor_name: str = "wheel_height_sensor",
    success_threshold: float = 0.7,
    low_stair_ratio: float = 0.20,
    flat_ratio: float = 0.05,
    mastery_window_iterations: int = 16,
    mastery_evaluation_interval_iterations: int = 4,
    min_success_samples: int = 4096,
    initial_target_level: int = 0,
) -> dict[str, torch.Tensor]:
    """按终止 episode 的三阶爬升结果推进全局台阶难度，不回退。"""
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
        iteration=iteration,
    )
    robot = env.scene[asset_name]
    origins = env.scene.env_origins[ids]
    pg_z = robot.data.projected_gravity_b[ids, 2]
    valid_episode = (env.episode_length_buf[ids] > 0) & (~newly_entered)
    stair_mask = _terrain_type_mask(terrain, ids, terrain_type_names, env.device)

    terrain_levels = terrain.terrain_levels[ids].to(device=env.device)
    step_height = _step_height_for_levels(terrain, terrain_levels, step_height_range)
    wheel_ground_z = _wheel_ground_height(env, robot, ids, height_sensor_name)
    height_gain = wheel_ground_z - origins[:, 2]
    steps_climbed = torch.clamp(height_gain / step_height.clamp_min(1.0e-6), min=0.0)
    radial_distance = torch.linalg.vector_norm(
        robot.data.root_link_pos_w[ids, :2] - origins[:, :2], dim=1
    )
    upright = pg_z < float(upright_threshold)
    target_episode = stair_mask & valid_episode & (terrain_levels == int(target_level))
    success = target_episode & upright & (steps_climbed >= max(1, int(success_steps)))
    current_success_count = int(success.sum().item())
    current_target_count = int(target_episode.sum().item())
    _record_mastery_outcomes(
        env,
        iteration=iteration,
        success_count=current_success_count,
        target_count=current_target_count,
    )

    level_start_iteration = _mastery_level_start_iteration(env)
    completed_iteration = iteration - 1
    window_success_count, window_target_count, window_elapsed = _mastery_window_counts(
        env,
        level_start_iteration=level_start_iteration,
        completed_iteration=completed_iteration,
        window_iterations=mastery_window_iterations,
    )
    window_failure_count = max(0, window_target_count - window_success_count)
    success_rate = window_success_count / max(1, window_target_count)
    failure_rate = window_failure_count / max(1, window_target_count)

    upgraded = False
    evaluated = False
    num_rows = int(terrain.terrain_origins.shape[0])
    max_target_level = max(0, num_rows - 1)
    evaluation_due = (
        window_elapsed >= max(1, int(mastery_window_iterations))
        and completed_iteration - _mastery_last_evaluation_iteration(env)
        >= max(1, int(mastery_evaluation_interval_iterations))
        and window_target_count >= max(1, int(min_success_samples))
    )
    if evaluation_due:
        evaluated = True
        passed = success_rate >= float(success_threshold)
        _set_last_mastery_evaluation(
            env,
            iteration=completed_iteration,
            target_level=target_level,
            success_rate=success_rate,
            sample_count=window_target_count,
            passed=passed,
        )
        if passed and target_level < max_target_level:
            target_level += 1
            env._stair_mastery_target_level = target_level
            env._stair_mastery_upgrade_count = _mastery_upgrade_count(env) + 1
            _reset_mastery_window(env, level_start_iteration=iteration)
            upgraded = True

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
        "move_down_rate": torch.tensor(float(failure_rate), device=env.device),
        "height_gain_mean": _masked_mean(height_gain, target_episode),
        "distance_mean": _masked_mean(radial_distance, target_episode),
        "stair_env_rate": torch.mean(assigned_stair.float()),
        "walking_phase": torch.tensor(0.0, device=env.device),
        "iteration": torch.tensor(float(iteration), device=env.device),
        "target_level": torch.tensor(float(target_level), device=env.device),
        "target_success_rate": torch.tensor(float(success_rate), device=env.device),
        "target_failure_rate": torch.tensor(float(failure_rate), device=env.device),
        "target_sample_count": torch.tensor(float(window_target_count), device=env.device),
        "target_window_evaluated": torch.tensor(float(evaluated), device=env.device),
        "target_window_elapsed_iterations": torch.tensor(float(window_elapsed), device=env.device),
        "target_required_steps": torch.tensor(float(success_steps), device=env.device),
        "target_steps_mean": _masked_mean(steps_climbed, target_episode),
        "target_step_height_mean": _masked_mean(step_height, target_episode),
        "target_last_evaluation_success_rate": torch.tensor(
            _mastery_last_evaluation_value(env, "success_rate"), device=env.device
        ),
        "target_last_evaluation_sample_count": torch.tensor(
            _mastery_last_evaluation_value(env, "sample_count"), device=env.device
        ),
        "target_last_evaluation_passed": torch.tensor(
            _mastery_last_evaluation_value(env, "passed"), device=env.device
        ),
        "target_last_evaluation_level": torch.tensor(
            _mastery_last_evaluation_value(env, "target_level"), device=env.device
        ),
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


def _step_height_for_levels(
    terrain,
    terrain_levels: torch.Tensor,
    step_height_range: tuple[float, float],
) -> torch.Tensor:
    """按 terrain row 复用倒金字塔的单阶高度插值。"""
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    num_rows = max(1, int(getattr(terrain_generator, "num_rows", 10)) - 1)
    height_min, height_max = (float(value) for value in step_height_range)
    difficulty = torch.clamp(terrain_levels.float(), min=0.0, max=float(num_rows))
    return height_min + difficulty / float(num_rows) * (height_max - height_min)


def _wheel_ground_height(
    env: ManagerBasedRlEnv, robot, ids: torch.Tensor, sensor_name: str
) -> torch.Tensor:
    """返回两侧轮下方较低地面的世界 z，避免 body-height 指令污染爬阶判据。"""
    cached_ids = getattr(env, "_stair_mastery_wheel_body_ids", None)
    if not isinstance(cached_ids, list) or len(cached_ids) != 2:
        cached_ids, body_names = robot.find_bodies(
            ("l_wheel_Link", "r_wheel_Link"), preserve_order=True
        )
        if len(cached_ids) != 2:
            raise RuntimeError(f"台阶课程必须找到左右轮 body，实际找到: {body_names}")
        env._stair_mastery_wheel_body_ids = cached_ids

    sensor = env.scene[sensor_name]
    clearances = sensor.data.heights.reshape(env.num_envs, -1)[ids]
    if clearances.shape[1] != 2:
        raise RuntimeError(
            f"{sensor_name} 必须按左右轮返回两个 frame，高度 shape={tuple(clearances.shape)}"
        )
    clearances = torch.where(
        torch.isfinite(clearances), clearances, torch.full_like(clearances, float("inf"))
    )
    wheel_z = robot.data.body_link_pos_w[ids][:, cached_ids, 2]
    return (wheel_z - clearances).min(dim=1).values


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
        nonflat_mask & (target_level > 0) & (torch.rand(n, device=device) < low_prob_inside_stairs)
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
    iteration: int,
) -> int:
    current = getattr(env, "_stair_mastery_target_level", None)
    num_rows = int(terrain.terrain_origins.shape[0])
    if current is None:
        current = max(0, min(int(initial_target_level), max(0, num_rows - 1)))
        env._stair_mastery_target_level = current
        env._stair_mastery_upgrade_count = 0
        _reset_mastery_window(env, level_start_iteration=iteration)
    return int(current)


def _mastery_upgrade_count(env: ManagerBasedRlEnv) -> int:
    return int(getattr(env, "_stair_mastery_upgrade_count", 0))


def _record_mastery_outcomes(
    env: ManagerBasedRlEnv,
    *,
    iteration: int,
    success_count: int,
    target_count: int,
) -> None:
    buckets = getattr(env, "_stair_mastery_buckets", None)
    if not isinstance(buckets, dict):
        buckets = {}
        env._stair_mastery_buckets = buckets
    bucket = buckets.setdefault(int(iteration), [0, 0])
    bucket[0] += int(success_count)
    bucket[1] += int(target_count)


def _mastery_window_counts(
    env: ManagerBasedRlEnv,
    *,
    level_start_iteration: int,
    completed_iteration: int,
    window_iterations: int,
) -> tuple[int, int, int]:
    if completed_iteration < level_start_iteration:
        return 0, 0, 0

    window_iterations = max(1, int(window_iterations))
    window_start = max(level_start_iteration, completed_iteration - window_iterations + 1)
    buckets = getattr(env, "_stair_mastery_buckets", {})
    if not isinstance(buckets, dict):
        return 0, 0, 0

    for stale_iteration in tuple(buckets):
        if stale_iteration < window_start:
            del buckets[stale_iteration]

    success_count = 0
    target_count = 0
    for bucket_iteration, counts in buckets.items():
        if window_start <= bucket_iteration <= completed_iteration:
            success_count += int(counts[0])
            target_count += int(counts[1])
    return success_count, target_count, completed_iteration - level_start_iteration + 1


def _mastery_level_start_iteration(env: ManagerBasedRlEnv) -> int:
    return int(getattr(env, "_stair_mastery_level_start_iteration", 0))


def _mastery_last_evaluation_iteration(env: ManagerBasedRlEnv) -> int:
    return int(getattr(env, "_stair_mastery_last_evaluation_iteration", -1))


def _set_last_mastery_evaluation(
    env: ManagerBasedRlEnv,
    *,
    iteration: int,
    target_level: int,
    success_rate: float,
    sample_count: int,
    passed: bool,
) -> None:
    env._stair_mastery_last_evaluation_iteration = int(iteration)
    env._stair_mastery_last_evaluation = {
        "success_rate": float(success_rate),
        "sample_count": float(sample_count),
        "passed": float(passed),
        "target_level": float(target_level),
    }


def _mastery_last_evaluation_value(env: ManagerBasedRlEnv, name: str) -> float:
    values = getattr(env, "_stair_mastery_last_evaluation", None)
    if not isinstance(values, dict):
        return 0.0
    return float(values.get(name, 0.0))


def _reset_mastery_window(env: ManagerBasedRlEnv, *, level_start_iteration: int) -> None:
    env._stair_mastery_buckets = {}
    env._stair_mastery_level_start_iteration = int(level_start_iteration)
    env._stair_mastery_last_evaluation_iteration = int(level_start_iteration) - 1


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
        "target_window_elapsed_iterations": zero,
        "target_required_steps": zero,
        "target_steps_mean": zero,
        "target_step_height_mean": zero,
        "target_last_evaluation_success_rate": zero,
        "target_last_evaluation_sample_count": zero,
        "target_last_evaluation_passed": zero,
        "target_last_evaluation_level": zero,
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
