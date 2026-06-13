"""台阶任务使用的事件函数。"""

from __future__ import annotations

import torch

from se3_train.tasks.flat.events import *  # noqa: F403
from se3_train.tasks.stair.state import StairClimbState


def init_stair_climb_state(
    env,
    env_ids: torch.Tensor | None,
    contact_window: int = 3,
    force_threshold_n: float = 30.0,
    ff_amplitude_rad: float = 1.2,
    ff_period_s: float = 0.6,
    ff_start_iter: int = 0,
    ann_start_iter: int = 200,
    ann_end_iter: int = 800,
    phantom_trigger_iter: int = 0,
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
        ff_period_s=ff_period_s,
        control_dt=control_dt,
        ff_start_iter=ff_start_iter,
        ann_start_iter=ann_start_iter,
        ann_end_iter=ann_end_iter,
        phantom_trigger_iter=phantom_trigger_iter,
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
    steps_per_policy_iter: int = 128,
) -> None:
    """Set the local TrainView curriculum counter to the mirrored checkpoint iter."""
    del env_ids
    iteration = max(0, int(iteration))
    env.common_step_counter = iteration * max(1, int(steps_per_policy_iter))
    state = getattr(env, "stair_climb_state", None)
    if state is not None:
        state.update_iter(iteration)


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
    num_steps_per_env: int = 128,
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

    state.step(wheel_xy)
    iteration = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    state.update_iter(iteration)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(state.diag())


__all__ = [
    "init_stair_climb_state",
    "reset_stair_climb_state",
    "set_fixed_stair_terrain",
    "set_train_view_iteration",
    "step_stair_climb_state",
]
