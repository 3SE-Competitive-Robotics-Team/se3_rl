"""台阶任务使用的事件函数。"""

from __future__ import annotations

import torch

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
    stair_terrain_type_name: str = "inv_pyramid_stairs",
    recovery_terrain_type_name: str = "flat",
    max_level_stages: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_MAX_STAGES,
    level_buckets: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_BUCKETS,
    bucket_weight_stages: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = DEFAULT_BUCKET_WEIGHT_STAGES,
    steps_per_policy_iter: int = 64,
) -> None:
    """reset 前采样 stair/recovery rehearsal mode，并同步地形 origin。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    total = max(float(stair_prob) + float(recovery_prob), 1.0e-6)
    stair_p = float(stair_prob) / total

    r = torch.rand(env_ids.numel(), device=env.device)
    local_mode = torch.full(
        (env_ids.numel(),), _TASK_MODE_STAIR, device=env.device, dtype=torch.long
    )
    local_mode[r >= stair_p] = _TASK_MODE_RECOVERY

    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    recovery_mask = _ensure_bool_buffer(env, "_stair_recovery_mode_mask")
    task_mode[env_ids] = local_mode
    recovery_mask[env_ids] = local_mode == _TASK_MODE_RECOVERY

    iteration = _active_iteration(env, steps_per_policy_iter)
    stair_ids = env_ids[local_mode == _TASK_MODE_STAIR]
    recovery_ids = env_ids[local_mode == _TASK_MODE_RECOVERY]
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
    _set_terrain(env, recovery_ids, recovery_terrain_type_name, 0)


def apply_stair_task_mode_commands(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
    recovery_lin_vel_x_range: tuple[float, float] = (-1.5, 1.5),
    recovery_ang_vel_yaw_range: tuple[float, float] = (-1.0, 1.0),
    recovery_height_range: tuple[float, float] = (0.195, 0.390),
) -> None:
    """reset 后按 recovery mode 重采样倒地自起指令。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    task_mode = _ensure_long_buffer(env, "_stair_task_mode")
    recovery_ids = env_ids[task_mode[env_ids] == _TASK_MODE_RECOVERY]
    _apply_command_ranges(
        env,
        recovery_ids,
        command_name,
        recovery_lin_vel_x_range,
        recovery_ang_vel_yaw_range,
        recovery_height_range,
    )


def enforce_recovery_active_commands(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
    recovery_lin_vel_x_range: tuple[float, float] = (-1.5, 1.5),
    recovery_ang_vel_yaw_range: tuple[float, float] = (-1.0, 1.0),
    recovery_height_range: tuple[float, float] = (0.195, 0.390),
) -> None:
    """每步把 recovery active 指令限制在 recovery_finetune 最难课程范围内。"""
    del env_ids
    active = getattr(env, "_recovery_reset_mask", None)
    if not isinstance(active, torch.Tensor) or active.shape[0] != env.num_envs:
        return
    active = active.to(device=env.device, dtype=torch.bool)
    if not active.any() or not hasattr(env, "command_manager"):
        return
    cmd = env.command_manager.get_command(command_name)
    cmd[active, 0] = torch.clamp(
        cmd[active, 0],
        min=float(recovery_lin_vel_x_range[0]),
        max=float(recovery_lin_vel_x_range[1]),
    )
    if cmd.shape[1] > 1:
        cmd[active, 1] = torch.clamp(
            cmd[active, 1],
            min=float(recovery_ang_vel_yaw_range[0]),
            max=float(recovery_ang_vel_yaw_range[1]),
        )
    if cmd.shape[1] > 2:
        cmd[active, 2:4] = 0.0
    if cmd.shape[1] > 4:
        low, high = float(recovery_height_range[0]), float(recovery_height_range[1])
        cmd[active, 4] = torch.clamp(cmd[active, 4], min=low, max=high)
    if cmd.shape[1] >= 8:
        cmd[active, 5] = 0.0
        cmd[active, 7] = 0.0
    update_policy_default_from_height_cache(
        env,
        command_name,
        env_ids=active.nonzero().flatten(),
        command=cmd,
    )


def init_stair_climb_state(
    env,
    env_ids: torch.Tensor | None,
    contact_window: int = 3,
    force_threshold_n: float = 30.0,
    ff_amplitude_rad: float = 1.2,
    ff_x_m: float = 0.025,
    ff_lift_m: float = 0.085,
    ff_period_s: float = 0.6,
    ff_rise_ratio: float = 0.25,
    ff_hold_ratio: float = 0.45,
    ff_wheel_action: float = 0.0,
    ff_start_iter: int = 0,
    ann_start_iter: int = 200,
    ann_end_iter: int = 800,
    phantom_trigger_iter: int = 0,
    allow_bilateral_trigger: bool = True,
) -> None:
    """startup 事件：在 env 上挂载 CTBC 状态机。"""
    del env_ids
    if hasattr(env, "stair_climb_state"):
        return
    control_dt = float(env.physics_dt) * int(env.cfg.decimation)
    env.stair_climb_state = StairClimbState(
        num_envs=env.num_envs,
        device=env.device,
        contact_window=contact_window,
        force_threshold_n=force_threshold_n,
        ff_amplitude_rad=ff_amplitude_rad,
        ff_x_m=ff_x_m,
        ff_lift_m=ff_lift_m,
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
    )


def set_fixed_stair_terrain(
    env,
    env_ids: torch.Tensor | None,
    terrain_level: int,
    terrain_type_name: str = "inv_pyramid_stairs",
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

    recovery_active = getattr(env, "_recovery_reset_mask", None)
    if isinstance(recovery_active, torch.Tensor) and recovery_active.shape[0] == env.num_envs:
        active_ids = recovery_active.to(device=env.device, dtype=torch.bool).nonzero().flatten()
        if active_ids.numel() > 0:
            wheel_xy[active_ids] = 0.0
            state.reset(active_ids)

    state.step(wheel_xy)
    iteration = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    state.update_iter(iteration)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(state.diag())


__all__ = [
    "apply_stair_task_mode_commands",
    "enforce_recovery_active_commands",
    "init_stair_climb_state",
    "reset_stair_climb_state",
    "sample_stair_task_mode",
    "set_fixed_stair_terrain",
    "set_train_view_iteration",
    "step_stair_climb_state",
]
