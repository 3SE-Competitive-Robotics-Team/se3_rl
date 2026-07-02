"""台阶任务使用的事件函数。"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from se3_train.mdp import events as mdp_events
from se3_train.mdp.height_default_cache import update_policy_default_from_height_cache
from se3_train.tasks.flat.events import *  # noqa: F403
from se3_train.tasks.stair.state import StairClimbState
from se3_train.tasks.stair.terrain_curriculum import (
    DEFAULT_BUCKET_WEIGHT_STAGES,
    DEFAULT_LEVEL_BUCKETS,
    DEFAULT_LEVEL_MAX_STAGES,
    sample_levels,
)

_TASK_MODE_STAIR = 0
_TASK_MODE_RECOVERY = 1
_TASK_MODE_FLAT = 2
_TASK_MODE_SHARED = _TASK_MODE_RECOVERY


def _ensure_long_buffer(env, name: str) -> torch.Tensor:
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        values = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        setattr(env, name, values)
    return values


def _ensure_bool_buffer(env, name: str) -> torch.Tensor:
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        values = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, name, values)
    return values


def _active_iteration(env, steps_per_policy_iter: int) -> int:
    state = getattr(env, "stair_climb_state", None)
    if state is not None:
        return int(state.local_iteration)
    return int(getattr(env, "common_step_counter", 0)) // max(1, int(steps_per_policy_iter))


def _terrain_type_index(env, terrain_type_name: str) -> int:
    terrain = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        raise RuntimeError("台阶 task mixture 需要 generator terrain")
    terrain_names = tuple(terrain_generator.sub_terrains)
    if terrain_type_name not in terrain_names:
        raise ValueError(f"未知地形类型: {terrain_type_name}")
    return terrain_names.index(terrain_type_name)


def _set_terrain(env, env_ids: torch.Tensor, terrain_type_name: str, terrain_level: int) -> None:
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return
    terrain_type = _terrain_type_index(env, terrain_type_name)
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    level = max(0, min(int(terrain_level), max_level))
    terrain.terrain_levels[env_ids] = level
    terrain.terrain_types[env_ids] = terrain_type
    terrain.env_origins[env_ids] = terrain.terrain_origins[level, terrain_type]


def _set_terrain_levels(
    env,
    env_ids: torch.Tensor,
    terrain_type_name: str,
    terrain_levels: torch.Tensor,
) -> None:
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None or env_ids.numel() == 0:
        return
    terrain_type = _terrain_type_index(env, terrain_type_name)
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    levels = torch.clamp(
        terrain_levels.to(device=env.device, dtype=torch.long),
        min=0,
        max=max_level,
    )
    terrain.terrain_levels[env_ids] = levels
    terrain.terrain_types[env_ids] = terrain_type
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain_type]


def _sample_uniform(
    env,
    env_ids: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    low, high = float(value_range[0]), float(value_range[1])
    if low == high:
        return torch.full((env_ids.numel(),), low, device=env.device)
    return low + (high - low) * torch.rand(env_ids.numel(), device=env.device)


def _normalized_task_mode_probs(
    stair_prob: float,
    recovery_prob: float,
    flat_prob: float,
    device: torch.device,
    *,
    merge_shared: bool = False,
) -> torch.Tensor:
    if merge_shared:
        shared_prob = max(float(recovery_prob), 0.0) + max(float(flat_prob), 0.0)
        probs = torch.tensor(
            [
                max(float(stair_prob), 0.0),
                shared_prob,
                0.0,
            ],
            device=device,
            dtype=torch.float32,
        )
        total = float(probs.sum().item())
        if total <= 1.0e-6:
            probs[0] = 1.0
            return probs
        return probs / total

    probs = torch.tensor(
        [
            max(float(stair_prob), 0.0),
            max(float(recovery_prob), 0.0),
            max(float(flat_prob), 0.0),
        ],
        device=device,
        dtype=torch.float32,
    )
    total = float(probs.sum().item())
    if total <= 1.0e-6:
        probs[0] = 1.0
        return probs
    return probs / total


def _task_mode_counts_from_probs(probs: torch.Tensor, total_count: int) -> torch.Tensor:
    if total_count <= 0:
        return torch.zeros(3, device=probs.device, dtype=torch.long)
    raw_counts = probs * float(total_count)
    counts = torch.floor(raw_counts).to(dtype=torch.long)
    remainder = int(total_count - counts.sum().item())
    if remainder > 0:
        fractions = raw_counts - torch.floor(raw_counts)
        order = torch.argsort(fractions, descending=True)
        counts[order[:remainder]] += 1
    return counts


def _sample_balanced_task_modes(
    env,
    env_ids: torch.Tensor,
    task_mode: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """按当前全局占用补偿 reset 采样，避免短 episode mode 被时间占比稀释。"""
    num_new = int(env_ids.numel())
    if num_new <= 0:
        return torch.empty(0, device=env.device, dtype=torch.long)

    reset_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    reset_mask[env_ids] = True
    remaining_modes = task_mode[~reset_mask]
    remaining_counts = torch.stack(
        [
            (remaining_modes == _TASK_MODE_STAIR).sum(),
            (remaining_modes == _TASK_MODE_RECOVERY).sum(),
            (remaining_modes == _TASK_MODE_FLAT).sum(),
        ]
    ).to(device=env.device, dtype=torch.long)
    target_counts = _task_mode_counts_from_probs(probs, env.num_envs)
    needed_counts = torch.clamp(target_counts - remaining_counts, min=0)
    needed_total = int(needed_counts.sum().item())
    if needed_total <= 0:
        counts = _task_mode_counts_from_probs(probs, num_new)
    elif needed_total > num_new:
        counts = _task_mode_counts_from_probs(needed_counts.float() / float(needed_total), num_new)
    else:
        counts = needed_counts
        leftover = num_new - needed_total
        if leftover > 0:
            counts = counts + _task_mode_counts_from_probs(probs, leftover)

    mode_values = torch.tensor(
        [_TASK_MODE_STAIR, _TASK_MODE_RECOVERY, _TASK_MODE_FLAT],
        device=env.device,
        dtype=torch.long,
    )
    local_mode = torch.repeat_interleave(mode_values, counts)
    if local_mode.numel() < num_new:
        padding = torch.full(
            (num_new - local_mode.numel(),),
            _TASK_MODE_STAIR,
            device=env.device,
            dtype=torch.long,
        )
        local_mode = torch.cat((local_mode, padding), dim=0)
    elif local_mode.numel() > num_new:
        local_mode = local_mode[:num_new]
    return local_mode[torch.randperm(num_new, device=env.device)]


def _sync_shared_mode_masks(env, env_ids: torch.Tensor, local_mode: torch.Tensor) -> None:
    shared = local_mode == _TASK_MODE_SHARED
    recovery_mask = _ensure_bool_buffer(env, "_stair_recovery_mode_mask")
    shared_mask = _ensure_bool_buffer(env, "_stair_shared_mode_mask")
    flat_zero_mask = _ensure_bool_buffer(env, "_stair_flat_zero_command_mask")
    recovery_mask[env_ids] = shared
    shared_mask[env_ids] = shared
    flat_zero_mask[env_ids] = False

    recovery_active = _ensure_bool_buffer(env, "_recovery_reset_mask")
    recovery_episode = _ensure_bool_buffer(env, "_recovery_episode_mask")
    recovery_active[env_ids] = shared
    recovery_episode[env_ids] = shared


def _shared_mode_ids(env, env_ids: torch.Tensor, task_mode: torch.Tensor) -> torch.Tensor:
    shared_mask = getattr(env, "_stair_shared_mode_mask", None)
    if isinstance(shared_mask, torch.Tensor) and shared_mask.shape[0] == env.num_envs:
        return env_ids[shared_mask[env_ids].to(device=env.device, dtype=torch.bool)]
    return env_ids[task_mode[env_ids] == _TASK_MODE_SHARED]


def _apply_command_ranges(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    lin_vel_x_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float],
    height_range: tuple[float, float],
) -> None:
    if env_ids.numel() == 0 or not hasattr(env, "command_manager"):
        return
    cmd = env.command_manager.get_command(command_name)
    cmd[env_ids, 0] = _sample_uniform(env, env_ids, lin_vel_x_range)
    if cmd.shape[1] > 1:
        cmd[env_ids, 1] = _sample_uniform(env, env_ids, ang_vel_yaw_range)
    if cmd.shape[1] > 2:
        cmd[env_ids, 2:4] = 0.0
    if cmd.shape[1] > 4:
        cmd[env_ids, 4] = _sample_uniform(env, env_ids, height_range)
    if cmd.shape[1] >= 8:
        cmd[env_ids, 5] = 0.0
        cmd[env_ids, 7] = 0.0
    update_policy_default_from_height_cache(env, command_name, env_ids=env_ids, command=cmd)


def sample_stair_task_mode(
    env,
    env_ids: torch.Tensor | None,
    stair_prob: float = 0.70,
    recovery_prob: float = 0.30,
    flat_prob: float = 0.0,
    stair_terrain_type_name: str = "forward_stairs",
    recovery_terrain_type_name: str = "flat",
    flat_terrain_type_name: str = "flat",
    max_level_stages: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_MAX_STAGES,
    level_buckets: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_BUCKETS,
    bucket_weight_stages: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = DEFAULT_BUCKET_WEIGHT_STAGES,
    steps_per_policy_iter: int = 64,
    balance_occupancy: bool = True,
) -> None:
    """reset 前采样 stair/recovery rehearsal mode，并同步地形 origin。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    probs = _normalized_task_mode_probs(
        stair_prob,
        recovery_prob,
        flat_prob,
        env.device,
        merge_shared=True,
    )
    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    if balance_occupancy:
        local_mode = _sample_balanced_task_modes(env, env_ids, task_mode, probs)
    else:
        stair_p = float(probs[0].item())
        r = torch.rand(env_ids.numel(), device=env.device)
        local_mode = torch.full(
            (env_ids.numel(),), _TASK_MODE_STAIR, device=env.device, dtype=torch.long
        )
        local_mode[r >= stair_p] = _TASK_MODE_SHARED

    task_mode[env_ids] = local_mode
    _sync_shared_mode_masks(env, env_ids, local_mode)

    iteration = _active_iteration(env, steps_per_policy_iter)
    stair_ids = env_ids[local_mode == _TASK_MODE_STAIR]
    shared_ids = env_ids[local_mode == _TASK_MODE_SHARED]
    if stair_ids.numel() > 0:
        terrain_levels, _ = sample_levels(
            env,
            env.scene.terrain,
            stair_ids.numel(),
            iteration,
            max_level_stages=max_level_stages,
            level_buckets=level_buckets,
            bucket_weight_stages=bucket_weight_stages,
        )
        _set_terrain_levels(env, stair_ids, stair_terrain_type_name, terrain_levels)
    _set_terrain(env, shared_ids, recovery_terrain_type_name, 0)


def _active_stage(
    iteration: int,
    stages: tuple[dict, ...] | list[dict] | None,
) -> dict:
    active: dict = {}
    for stage in sorted(tuple(stages or ()), key=lambda item: int(item.get("iteration", 0))):
        if iteration >= int(stage.get("iteration", 0)):
            active = stage
        else:
            break
    return active


def sample_stair_shared_task_mode(
    env,
    env_ids: torch.Tensor | None,
    stair_prob: float = 0.85,
    shared_prob: float = 0.15,
    mixture_stages: tuple[dict, ...] | list[dict] | None = None,
    stair_terrain_type_name: str = "forward_stairs",
    shared_terrain_type_name: str = "flat",
    max_level_stages: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_MAX_STAGES,
    level_buckets: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_BUCKETS,
    bucket_weight_stages: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = DEFAULT_BUCKET_WEIGHT_STAGES,
    steps_per_policy_iter: int = 64,
    balance_occupancy: bool = True,
) -> None:
    """采样台阶和 shared rehearsal 模式，并把环境放到对应地形。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    iteration = _active_iteration(env, steps_per_policy_iter)
    stage = _active_stage(iteration, mixture_stages)
    stair_prob = float(stage.get("stair_prob", stair_prob))
    shared_prob = float(stage.get("shared_prob", shared_prob))

    probs = _normalized_task_mode_probs(
        stair_prob,
        shared_prob,
        0.0,
        env.device,
        merge_shared=True,
    )
    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    if balance_occupancy:
        local_mode = _sample_balanced_task_modes(env, env_ids, task_mode, probs)
    else:
        stair_p = float(probs[0].item())
        local_mode = torch.full(
            (env_ids.numel(),), _TASK_MODE_STAIR, device=env.device, dtype=torch.long
        )
        local_mode[torch.rand(env_ids.numel(), device=env.device) >= stair_p] = _TASK_MODE_SHARED

    task_mode[env_ids] = local_mode
    _sync_shared_mode_masks(env, env_ids, local_mode)

    shared_mask = local_mode == _TASK_MODE_SHARED
    stair_ids = env_ids[local_mode == _TASK_MODE_STAIR]
    shared_ids = env_ids[shared_mask]
    if stair_ids.numel() > 0:
        terrain_levels, _ = sample_levels(
            env,
            env.scene.terrain,
            stair_ids.numel(),
            iteration,
            max_level_stages=max_level_stages,
            level_buckets=level_buckets,
            bucket_weight_stages=bucket_weight_stages,
        )
        _set_terrain_levels(env, stair_ids, stair_terrain_type_name, terrain_levels)
    _set_terrain(env, shared_ids, shared_terrain_type_name, 0)

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        log["Stair/task_mode_stair_target"] = float(probs[0].item())
        log["Stair/task_mode_shared_target"] = float(probs[1].item())
        log["Stair/task_mode_shared_rate"] = float(shared_mask.float().mean().item())


def apply_stair_task_mode_commands(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
    recovery_lin_vel_x_range: tuple[float, float] = (0.0, 0.0),
    recovery_ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0),
    recovery_height_range: tuple[float, float] = (0.24, 0.30),
    flat_lin_vel_x_range: tuple[float, float] = (-1.5, 1.5),
    flat_ang_vel_yaw_range: tuple[float, float] = (-1.5, 1.5),
    flat_height_range: tuple[float, float] = (0.20, 0.32),
    flat_zero_command_prob: float = 0.30,
    shared_lin_vel_x_range: tuple[float, float] | None = None,
    shared_ang_vel_yaw_range: tuple[float, float] | None = None,
    shared_height_range: tuple[float, float] | None = None,
) -> None:
    """reset 后按 recovery mode 重采样倒地自起指令。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    flat_zero_mask = _ensure_bool_buffer(env, "_stair_flat_zero_command_mask")
    flat_zero_mask[env_ids] = False
    shared_ids = _shared_mode_ids(env, env_ids, task_mode)
    flat_ids = env_ids[task_mode[env_ids] == _TASK_MODE_FLAT]
    _apply_command_ranges(
        env,
        shared_ids,
        command_name,
        shared_lin_vel_x_range or recovery_lin_vel_x_range,
        shared_ang_vel_yaw_range or recovery_ang_vel_yaw_range,
        shared_height_range or recovery_height_range,
    )
    _apply_command_ranges(
        env,
        flat_ids,
        command_name,
        flat_lin_vel_x_range,
        flat_ang_vel_yaw_range,
        flat_height_range,
    )
    zero_prob = max(0.0, min(1.0, float(flat_zero_command_prob)))
    if flat_ids.numel() > 0 and zero_prob > 0.0:
        zero_ids = flat_ids[torch.rand(flat_ids.numel(), device=env.device) < zero_prob]
        flat_zero_mask[zero_ids] = True
        _apply_flat_zero_commands(env, zero_ids, command_name)


def apply_stair_shared_rehearsal_commands(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
    lin_vel_x_range: tuple[float, float] = (-1.89, 1.89),
    ang_vel_yaw_range: tuple[float, float] = (-9.41, 9.41),
    height_range: tuple[float, float] = (0.195, 0.390),
) -> None:
    """给 shared rehearsal 环境写入 recovery-discovery 最难指令范围。"""
    apply_stair_task_mode_commands(
        env,
        env_ids,
        command_name=command_name,
        shared_lin_vel_x_range=lin_vel_x_range,
        shared_ang_vel_yaw_range=ang_vel_yaw_range,
        shared_height_range=height_range,
        flat_zero_command_prob=0.0,
    )


def _clamp_command_ranges(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    lin_vel_x_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float],
    height_range: tuple[float, float],
) -> None:
    """把已重采样的指令夹回指定 mode 的合法范围。"""
    if env_ids.numel() == 0 or not hasattr(env, "command_manager"):
        return
    cmd = env.command_manager.get_command(command_name)
    cmd[env_ids, 0] = torch.clamp(
        cmd[env_ids, 0],
        min=float(lin_vel_x_range[0]),
        max=float(lin_vel_x_range[1]),
    )
    if cmd.shape[1] > 1:
        cmd[env_ids, 1] = torch.clamp(
            cmd[env_ids, 1],
            min=float(ang_vel_yaw_range[0]),
            max=float(ang_vel_yaw_range[1]),
        )
    if cmd.shape[1] > 2:
        cmd[env_ids, 2:4] = 0.0
    if cmd.shape[1] > 4:
        cmd[env_ids, 4] = torch.clamp(
            cmd[env_ids, 4],
            min=float(height_range[0]),
            max=float(height_range[1]),
        )
    if cmd.shape[1] >= 8:
        cmd[env_ids, 5] = 0.0
        cmd[env_ids, 7] = 0.0
    update_policy_default_from_height_cache(env, command_name, env_ids=env_ids, command=cmd)


def _apply_flat_zero_commands(env, env_ids: torch.Tensor, command_name: str) -> None:
    """保持一部分 flat rehearsal env 使用零速指令。"""
    if env_ids.numel() == 0 or not hasattr(env, "command_manager"):
        return
    cmd = env.command_manager.get_command(command_name)
    cmd[env_ids, 0] = 0.0
    if cmd.shape[1] > 1:
        cmd[env_ids, 1] = 0.0
    if cmd.shape[1] > 2:
        cmd[env_ids, 2:4] = 0.0
    if cmd.shape[1] >= 8:
        cmd[env_ids, 5] = 0.0
        cmd[env_ids, 7] = 0.0
    update_policy_default_from_height_cache(env, command_name, env_ids=env_ids, command=cmd)


def enforce_recovery_active_commands(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
    recovery_lin_vel_x_range: tuple[float, float] = (0.0, 0.0),
    recovery_ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0),
    recovery_height_range: tuple[float, float] = (0.24, 0.30),
    flat_lin_vel_x_range: tuple[float, float] = (-1.5, 1.5),
    flat_ang_vel_yaw_range: tuple[float, float] = (-1.5, 1.5),
    flat_height_range: tuple[float, float] = (0.20, 0.32),
    flat_zero_command_prob: float = 0.30,
    shared_lin_vel_x_range: tuple[float, float] | None = None,
    shared_ang_vel_yaw_range: tuple[float, float] | None = None,
    shared_height_range: tuple[float, float] | None = None,
) -> None:
    """每步把 recovery active 指令限制在 recovery discovery 后期课程范围内。"""
    del env_ids, flat_zero_command_prob
    active = getattr(env, "_recovery_reset_mask", None)
    if isinstance(active, torch.Tensor) and active.shape[0] == env.num_envs:
        _clamp_command_ranges(
            env,
            active.to(device=env.device, dtype=torch.bool).nonzero().flatten(),
            command_name,
            shared_lin_vel_x_range or recovery_lin_vel_x_range,
            shared_ang_vel_yaw_range or recovery_ang_vel_yaw_range,
            shared_height_range or recovery_height_range,
        )

    task_mode = getattr(env, "_stair_task_mode", None)
    if isinstance(task_mode, torch.Tensor) and task_mode.shape[0] == env.num_envs:
        flat_ids = (
            (task_mode.to(device=env.device, dtype=torch.long) == _TASK_MODE_FLAT)
            .nonzero(as_tuple=False)
            .flatten()
        )
        _clamp_command_ranges(
            env,
            flat_ids,
            command_name,
            flat_lin_vel_x_range,
            flat_ang_vel_yaw_range,
            flat_height_range,
        )
        flat_zero_mask = getattr(env, "_stair_flat_zero_command_mask", None)
        if isinstance(flat_zero_mask, torch.Tensor) and flat_zero_mask.shape[0] == env.num_envs:
            zero_ids = flat_ids[flat_zero_mask[flat_ids].to(device=env.device, dtype=torch.bool)]
            _apply_flat_zero_commands(env, zero_ids, command_name)


def enforce_shared_rehearsal_commands(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
    lin_vel_x_range: tuple[float, float] = (-1.89, 1.89),
    ang_vel_yaw_range: tuple[float, float] = (-9.41, 9.41),
    height_range: tuple[float, float] = (0.195, 0.390),
) -> None:
    """把 shared rehearsal 指令持续夹到 recovery-discovery 最难范围。"""
    enforce_recovery_active_commands(
        env,
        env_ids,
        command_name=command_name,
        shared_lin_vel_x_range=lin_vel_x_range,
        shared_ang_vel_yaw_range=ang_vel_yaw_range,
        shared_height_range=height_range,
        flat_zero_command_prob=0.0,
    )


def reset_root_state_stair_shared(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg,
    shared_state_cache_path: str | None,
    shared_state_cache_split: str = "train",
    shared_recovery_grace_steps: int = 400,
    steps_per_policy_iter: int = 64,
) -> None:
    """台阶环境按直立状态 reset，shared 环境按 recovery-discovery 最难采样 reset。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    shared = task_mode[env_ids].to(device=env.device, dtype=torch.long) == _TASK_MODE_SHARED
    stair_ids = env_ids[~shared]
    shared_ids = env_ids[shared]

    if stair_ids.numel() > 0:
        mdp_events.reset_root_state_full(
            env,
            stair_ids,
            asset_cfg=asset_cfg,
            recovery_prob=0.0,
            recovery_grace_steps=shared_recovery_grace_steps,
            recovery_command_height=None,
            recovery_zero_velocity_command=False,
        )
    if shared_ids.numel() > 0:
        mdp_events.reset_root_state_recovery_discovery_mixed(
            env,
            shared_ids,
            asset_cfg=asset_cfg,
            pos_xy_range=(-0.15, 0.15),
            height_offset_range=(0.0, 0.02),
            yaw_range=(-math.pi, math.pi),
            roll_jitter_range=(-math.radians(5.0), math.radians(5.0)),
            pitch_jitter_range=(-math.radians(5.0), math.radians(5.0)),
            lin_vel_range=(0.0, 0.0),
            ang_vel_range=(0.0, 0.0),
            clearance_range=(0.001, 0.005),
            pose_weights=(0.08, 0.17, 0.17, 0.29, 0.29),
            recovery_command_height=None,
            recovery_state_cache_path=shared_state_cache_path,
            recovery_state_cache_split=shared_state_cache_split,
            recovery_grace_steps=shared_recovery_grace_steps,
            standard_recovery_zero_velocity_command=False,
            source_curriculum_stages=[
                {
                    "iteration": 0,
                    "cache_ratio": 0.70,
                    "near_upright_ratio": 0.05,
                }
            ],
            standard_curriculum_stages=[
                {
                    "iteration": 0,
                    "roll_jitter_range": (-math.radians(20.0), math.radians(20.0)),
                    "pitch_jitter_range": (-math.radians(20.0), math.radians(20.0)),
                    "lin_vel_range": (-0.08, 0.08),
                    "ang_vel_range": (-0.30, 0.30),
                }
            ],
            use_iterations=True,
            steps_per_policy_iter=steps_per_policy_iter,
        )

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        total = max(1, int(env_ids.numel()))
        log["Reset/stair_shared_reset_rate"] = float(shared_ids.numel()) / float(total)


def reset_joints_stair_shared(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg,
    shared_joint_offset_range: float = 0.25,
    shared_joint_vel_range: tuple[float, float] = (-0.50, 0.50),
    shared_joint_randomization_prob: float = 0.75,
    **kwargs,
) -> None:
    """台阶模式使用默认关节，shared 模式使用 discovery 最难关节随机。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    shared = task_mode[env_ids].to(device=env.device, dtype=torch.long) == _TASK_MODE_SHARED
    stair_ids = env_ids[~shared]
    shared_ids = env_ids[shared]
    if stair_ids.numel() > 0:
        mdp_events.reset_joints(env, stair_ids, asset_cfg=asset_cfg, **kwargs)
    if shared_ids.numel() > 0:
        shared_kwargs = dict(kwargs)
        shared_kwargs.update(
            {
                "joint_offset_range": float(shared_joint_offset_range),
                "joint_vel_range": shared_joint_vel_range,
                "joint_randomization_prob": float(shared_joint_randomization_prob),
            }
        )
        mdp_events.reset_joints(env, shared_ids, asset_cfg=asset_cfg, **shared_kwargs)


def init_stair_climb_state(
    env,
    env_ids: torch.Tensor | None,
    trigger_mode: str = "pitch",
    contact_window: int = 3,
    force_threshold_n: float = 10.0,
    pitch_threshold_rad: float = math.radians(6.0),
    pitch_threshold_deg: float | None = None,
    pitch_window: int = 3,
    ff_amplitude_rad: float = 1.70,
    coordinate_mode: str = "body_polar",
    leg_length_m: float = 0.24,
    swing_angle_rad: float = math.radians(10.0),
    swing_angle_deg: float | None = None,
    ff_x_m: float = 0.02,
    ff_lift_m: float = 0.02,
    ff_duration_s: float | None = None,
    ff_period_s: float = 0.6,
    ff_rise_ratio: float = 0.35,
    ff_hold_ratio: float = 0.0,
    ff_wheel_action: float = 0.0,
    ff_start_iter: int = 0,
    ann_start_iter: int = 900,
    ann_end_iter: int = 1800,
    phantom_trigger_iter: int = 0,
    allow_bilateral_trigger: bool = False,
    profile_path: Path | str | None = None,
) -> None:
    """startup 事件：在 env 上挂载 CTBC 状态机。"""
    del env_ids
    if hasattr(env, "stair_climb_state"):
        return
    control_dt = float(env.physics_dt) * int(env.cfg.decimation)
    env.stair_climb_state = StairClimbState(
        num_envs=env.num_envs,
        device=env.device,
        trigger_mode=trigger_mode,
        contact_window=contact_window,
        force_threshold_n=force_threshold_n,
        pitch_threshold_rad=pitch_threshold_rad,
        pitch_threshold_deg=pitch_threshold_deg,
        pitch_window=pitch_window,
        ff_amplitude_rad=ff_amplitude_rad,
        coordinate_mode=coordinate_mode,
        leg_length_m=leg_length_m,
        swing_angle_rad=swing_angle_rad,
        swing_angle_deg=swing_angle_deg,
        ff_x_m=ff_x_m,
        ff_lift_m=ff_lift_m,
        ff_duration_s=ff_duration_s,
        ff_period_s=ff_period_s,
        ff_rise_ratio=ff_rise_ratio,
        ff_hold_ratio=ff_hold_ratio,
        ff_wheel_action=ff_wheel_action,
        control_dt=control_dt,
        ff_start_iter=ff_start_iter,
        ann_start_iter=ann_start_iter,
        ann_end_iter=ann_end_iter,
        phantom_trigger_iter=phantom_trigger_iter,
        allow_bilateral_trigger=allow_bilateral_trigger,
        profile_path=profile_path,
    )


def set_fixed_stair_terrain(
    env,
    env_ids: torch.Tensor | None,
    terrain_level: int,
    terrain_type_name: str = "forward_stairs",
) -> None:
    """将 viewer 环境固定到指定台阶 row 和地形类型。"""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    num_rows = int(terrain.terrain_origins.shape[0])
    level = max(0, min(int(terrain_level), num_rows - 1))
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        return
    terrain_names = tuple(terrain_generator.sub_terrains)
    if terrain_type_name not in terrain_names:
        raise ValueError(f"未知台阶地形类型: {terrain_type_name}")
    terrain_type = terrain_names.index(terrain_type_name)

    terrain.terrain_levels[env_ids] = level
    terrain.terrain_types[env_ids] = terrain_type
    terrain.env_origins[env_ids] = terrain.terrain_origins[level, terrain_type]


def set_train_view_iteration(
    env,
    env_ids: torch.Tensor | None,
    iteration: int,
    steps_per_policy_iter: int = 64,
) -> None:
    """Set the local TrainView curriculum counter to the mirrored checkpoint iter."""
    del env_ids
    iteration = max(0, int(iteration))
    env.common_step_counter = iteration * max(1, int(steps_per_policy_iter))
    state = getattr(env, "stair_climb_state", None)
    if state is not None:
        state.set_fixed_iteration(iteration)


def reset_stair_climb_state(env, env_ids: torch.Tensor | None) -> None:
    """reset 事件：清空指定 env 的 CTBC 状态机。"""
    state = getattr(env, "stair_climb_state", None)
    if state is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    state.reset(env_ids)


def step_stair_climb_state(
    env,
    env_ids: torch.Tensor | None,
    sensor_name: str = "wheel_sensor",
    riser_sensor_name: str | None = None,
    riser_normal_z_max: float = 0.5,
    num_steps_per_env: int = 64,
) -> None:
    """interval 事件：每控制步更新 CTBC 状态机。"""
    del env_ids
    from mjlab.sensor import ContactSensor

    state = getattr(env, "stair_climb_state", None)
    if state is None:
        return

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        wheel_xy = torch.zeros(env.num_envs, 2, device=env.device)
    else:
        force_xy = data.force[..., :2]
        wheel_xy = torch.norm(force_xy, dim=-1)

    if riser_sensor_name:
        riser_sensor: ContactSensor = env.scene[riser_sensor_name]
        riser_data = riser_sensor.data
        if riser_data.force is not None and riser_data.normal is not None:
            force = riser_data.force.reshape(env.num_envs, 2, -1, 3)
            normal = riser_data.normal.reshape(env.num_envs, 2, -1, 3)
            force_xy = torch.norm(force[..., :2], dim=-1)
            valid = torch.abs(normal[..., 2]) <= float(riser_normal_z_max)
            if riser_data.found is not None:
                found = riser_data.found.reshape(env.num_envs, 2, -1) > 0
                valid = valid & found
            wheel_xy = torch.where(valid, force_xy, torch.zeros_like(force_xy)).sum(dim=-1)

    inactive = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    task_mode = getattr(env, "_stair_task_mode", None)
    if isinstance(task_mode, torch.Tensor) and task_mode.shape[0] == env.num_envs:
        inactive |= task_mode.to(device=env.device, dtype=torch.long) != _TASK_MODE_STAIR
    recovery_active = getattr(env, "_recovery_reset_mask", None)
    if isinstance(recovery_active, torch.Tensor) and recovery_active.shape[0] == env.num_envs:
        inactive |= recovery_active.to(device=env.device, dtype=torch.bool)
    inactive_ids = inactive.nonzero().flatten()
    if inactive_ids.numel() > 0:
        wheel_xy[inactive_ids] = 0.0
        state.reset(inactive_ids)

    robot = env.scene["robot"]
    projected_gravity = torch.nan_to_num(
        robot.data.projected_gravity_b,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pitch_rad = torch.atan2(projected_gravity[:, 0], -projected_gravity[:, 2])
    if inactive_ids.numel() > 0:
        pitch_rad[inactive_ids] = 0.0
    state.step(wheel_xy, pitch_rad=pitch_rad)
    iteration = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    state.update_iter(iteration)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(state.diag())


__all__ = [
    "apply_stair_shared_rehearsal_commands",
    "apply_stair_task_mode_commands",
    "enforce_recovery_active_commands",
    "enforce_shared_rehearsal_commands",
    "init_stair_climb_state",
    "reset_joints_stair_shared",
    "reset_root_state_stair_shared",
    "reset_stair_climb_state",
    "sample_stair_shared_task_mode",
    "sample_stair_task_mode",
    "set_fixed_stair_terrain",
    "set_train_view_iteration",
    "step_stair_climb_state",
]
