"""CTBC 台阶任务奖励和诊断函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import recovery_state
from se3_train.mdp.joint_indices import wheel_joint_ids
from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_STAIR_TERRAIN_TYPES = ("inv_pyramid_stairs",)
_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height
_WHEEL_RADIUS_M = 0.060
_WHEEL_SUPPORT_CLEARANCE_TOL_M = 0.035
_WHEEL_SUPPORT_FORCE_THRESHOLD_N = 1.0
_TASK_MODE_STAIR = 0
_TASK_MODE_RECOVERY = 1


def _get_stair_state(env: ManagerBasedRlEnv):
    return getattr(env, "stair_climb_state", None)


def _finite(value: torch.Tensor) -> torch.Tensor:
    """将异常状态的非有限值折成零，避免污染整批奖励。"""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """计算 mask 内均值；空 mask 返回 0。"""
    if mask.any():
        return _finite(values[mask]).float().mean().item()
    return 0.0


def _upright_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    pg_z = torch.nan_to_num(robot.data.projected_gravity_b[:, 2], nan=1.0, posinf=1.0, neginf=1.0)
    return torch.clamp(-pg_z, 0.0, 0.7) / 0.7


def _terrain_type_mask(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...],
) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if terrain_types is None or terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    if not sub_terrains:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    selected = set(terrain_type_names)
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if terrain_name in selected:
            mask |= terrain_types.to(device=env.device) == terrain_index
    recovery_active = recovery_state.recovery_active_mask(env)
    if recovery_active.shape[0] == env.num_envs:
        mask &= ~recovery_active
    return mask


def _task_mode_mask(env: ManagerBasedRlEnv, modes: tuple[int, ...]) -> torch.Tensor:
    """按 reset 采样的任务 mode 做 per-env 奖励门控。"""
    mode = getattr(env, "_stair_task_mode", None)
    if not isinstance(mode, torch.Tensor) or mode.shape[0] != env.num_envs:
        default_value = _TASK_MODE_STAIR in modes
        return torch.full((env.num_envs,), default_value, device=env.device, dtype=torch.bool)
    mode = mode.to(device=env.device, dtype=torch.long)
    selected = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for value in modes:
        selected |= mode == int(value)
    return selected


def _wheel_body_ids(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> list[int]:
    attr_name = f"_stair_wheel_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _wheel_terrain_measurements(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _get_stair_state(env)
    if state is None:
        zeros = torch.zeros(env.num_envs, 2, device=env.device)
        return zeros, zeros

    robot = env.scene[asset_cfg.name]
    sensor = env.scene[height_sensor_name]
    heights = _finite(sensor.data.heights)
    if heights.ndim == 1:
        heights = heights.unsqueeze(-1)
    if heights.shape[1] < 2:
        heights = heights.expand(-1, 2)
    heights = heights[:, :2]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos_w = _finite(robot.data.body_link_pos_w[:, body_ids, :])
    terrain_z = wheel_pos_w[:, :, 2] - heights
    return state.wheel_terrain_rise(terrain_z), heights


def _wheel_contact_force(env: ManagerBasedRlEnv, contact_sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[contact_sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    force = _finite(data.force)
    force_mag = torch.linalg.vector_norm(force, dim=-1)
    if force_mag.ndim == 3:
        force_mag = force_mag.amax(dim=-1)
    if force_mag.ndim == 1:
        force_mag = force_mag.unsqueeze(-1)
    if force_mag.shape[1] < 2:
        force_mag = force_mag.expand(-1, 2)
    return force_mag[:, :2]


def stair_wheel_support_rise(
    env: ManagerBasedRlEnv,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    support_mode: str = "both",
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
    require_contact_support: bool = True,
    use_episode_max: bool = False,
) -> torch.Tensor:
    """按轮端真实接触支撑的地形抬升量估计上阶进度。"""
    rise, wheel_heights = _wheel_terrain_measurements(env, height_sensor_name, asset_cfg)
    if require_contact_support:
        wheel_force = _wheel_contact_force(env, contact_sensor_name)
        wheel_contact = wheel_force >= float(contact_force_threshold_n)
        near_support_height = wheel_heights <= (
            float(wheel_radius_m) + float(wheel_clearance_tol_m)
        )
        support_mask = wheel_contact & near_support_height
        rise = torch.where(support_mask, rise, torch.zeros_like(rise))
        rise = torch.clamp(rise, min=0.0)
        state = _get_stair_state(env)
        if state is not None:
            state.record_wheel_supported_rise(
                rise,
                step_index=int(getattr(env, "common_step_counter", 0)),
            )
            if use_episode_max:
                if support_mode == "both":
                    support_rise = state.max_wheel_supported_both_rise()
                    terrain_mask = _terrain_type_mask(env, terrain_type_names)
                    support_rise = torch.where(
                        terrain_mask,
                        support_rise,
                        torch.zeros_like(support_rise),
                    )
                    return _finite(support_rise)
                rise = state.max_wheel_supported_rise()
    if support_mode == "any":
        support_rise = torch.max(rise, dim=1).values
    elif support_mode == "both":
        support_rise = torch.min(rise, dim=1).values
    else:
        raise ValueError(f"未知轮端支撑模式: {support_mode}")
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    support_rise = torch.where(terrain_mask, support_rise, torch.zeros_like(support_rise))
    return _finite(support_rise)


def _ctbc_trigger_weight(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    weight = state.ctbc_trigger_weight()
    recovery_active = recovery_state.recovery_active_mask(env)
    if recovery_active.shape[0] == env.num_envs:
        weight = torch.where(recovery_active, torch.zeros_like(weight), weight)
    return weight


def _ctbc_active_side_mask(env: ManagerBasedRlEnv, width: int) -> torch.Tensor:
    """返回当前 CTBC 相位要求摆动的轮侧 mask。"""
    state = _get_stair_state(env)
    mask = torch.zeros(env.num_envs, max(0, int(width)), device=env.device, dtype=torch.bool)
    if state is None or width <= 0:
        return mask
    active = state.ff_phase >= 0
    copy_width = min(mask.shape[1], active.shape[1])
    mask[:, :copy_width] = active[:, :copy_width]
    return mask


def _local_iteration(env: ManagerBasedRlEnv, steps_per_policy_iter: int = 64) -> int:
    state = _get_stair_state(env)
    if state is not None:
        return int(state.local_iteration)
    return int(getattr(env, "common_step_counter", 0)) // max(1, int(steps_per_policy_iter))


def _stair_phase_gate(
    env: ManagerBasedRlEnv,
    walking_phase_iterations: int,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    active = _local_iteration(env, steps_per_policy_iter) >= max(0, int(walking_phase_iterations))
    return torch.full((env.num_envs,), float(active), device=env.device)


def stair_phase_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    radial_velocity_blend: float = 0.75,
    radial_min_distance: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    gate = _stair_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    gate = gate * _task_mode_mask(env, (_TASK_MODE_STAIR,)).float()
    gate = gate * _terrain_type_mask(env, terrain_type_names).float()
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    return (
        stair_forward_progress(
            env,
            command_name=command_name,
            sigma=sigma,
            radial_velocity_blend=radial_velocity_blend,
            radial_min_distance=radial_min_distance,
            asset_cfg=asset_cfg,
        )
        * gate
    )


def stair_steps_climbed(
    env: ManagerBasedRlEnv,
    step_height: float | None = None,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    step_depth: float = 0.30,
    start_x_offset: float = 0.0,
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """按左右轮真实支撑地形抬升量估计每个 env 当前越过的台阶级数。"""
    del step_depth, start_x_offset, standing_height
    support_rise = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
    )

    if step_height is None:
        terrain_level = stair_terrain_level(env)
        terrain_generator = getattr(
            getattr(getattr(env.scene, "terrain", None), "cfg", None),
            "terrain_generator",
            None,
        )
        num_rows = max(1, int(getattr(terrain_generator, "num_rows", 10)) - 1)
        step_height_tensor = float(step_height_range[0]) + (
            torch.clamp(terrain_level, min=0.0, max=float(num_rows))
            / float(num_rows)
            * (float(step_height_range[1]) - float(step_height_range[0]))
        )
    else:
        step_height_tensor = torch.full_like(support_rise, float(step_height))

    step_height_tensor = _finite(step_height_tensor)
    steps = torch.clamp(support_rise / torch.clamp(step_height_tensor, min=1.0e-6), min=0.0)
    return _finite(steps * _upright_gate(env))


def stair_max_x_progress(
    env: ManagerBasedRlEnv,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """保留旧指标名；实际记录左右轮共同支撑地形的抬升量。"""
    gain = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
    )
    return _finite(gain * _upright_gate(env))


def stair_height_gain(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """兼容旧目标任务的高度增益指标，实际使用轮端支撑地形抬升。"""
    del command_name, standing_height
    gain = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
    )
    gain = _finite(gain * _upright_gate(env))
    return _finite(torch.clamp(gain, min=0.0))


def stair_climb_progress(
    env: ManagerBasedRlEnv,
    max_height_gain: float = 1.0,
    max_radial_progress: float = 4.0,
    radial_weight: float = 0.25,
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """奖励左右轮真实支撑地形的新增抬升量。"""
    del max_radial_progress, radial_weight, standing_height
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    height_gain = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
    )
    radial_progress = torch.zeros_like(height_gain)
    height_delta, radial_delta = state.climb_progress_delta(
        height_gain,
        radial_progress,
        max_height_gain=max_height_gain,
        max_radial_progress=0.0,
    )
    progress_delta = height_delta + radial_delta
    reward = progress_delta / max(float(env.step_dt), 1.0e-6) * _upright_gate(env)
    return _finite(reward)


def stair_support_height(
    env: ManagerBasedRlEnv,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    max_steps: float = 3.0,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """按当前双轮真实支撑高度持续奖励，避免只奖励瞬时新增高度。"""
    current_rise = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
        use_episode_max=False,
    )
    step_height = torch.clamp(_step_height_for_envs(env, step_height_range), min=1.0e-6)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    steps = torch.clamp(current_rise / step_height, min=0.0, max=float(max_steps))
    return _finite(steps * terrain_mask.float() * _upright_gate(env))


def stair_terrain_level(env: ManagerBasedRlEnv) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return torch.zeros(env.num_envs, device=env.device)
    for attr in ("terrain_levels", "env_terrain_level", "level"):
        value = getattr(terrain, attr, None)
        if isinstance(value, torch.Tensor):
            return _finite(value.to(device=env.device).float())
    return torch.zeros(env.num_envs, device=env.device)


def _step_height_for_envs(
    env: ManagerBasedRlEnv,
    step_height_range: tuple[float, float],
) -> torch.Tensor:
    """按当前 terrain level 估算每个 env 的台阶高度。"""
    terrain_generator = getattr(
        getattr(getattr(env.scene, "terrain", None), "cfg", None),
        "terrain_generator",
        None,
    )
    num_rows = max(1, int(getattr(terrain_generator, "num_rows", 10)) - 1)
    terrain_level = stair_terrain_level(env)
    min_height, max_height = (float(step_height_range[0]), float(step_height_range[1]))
    alpha = torch.clamp(terrain_level, min=0.0, max=float(num_rows)) / float(num_rows)
    return _finite(min_height + alpha * (max_height - min_height))


def stair_support_descent(
    env: ManagerBasedRlEnv,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    drop_tolerance_steps: float = 0.35,
    activation_steps: float = 0.70,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """惩罚已经双轮支撑到上层后又掉回低台阶。"""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    current_rise = stair_wheel_support_rise(
        env,
        height_sensor_name=height_sensor_name,
        contact_sensor_name=contact_sensor_name,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
        support_mode="both",
        contact_force_threshold_n=contact_force_threshold_n,
        wheel_radius_m=wheel_radius_m,
        wheel_clearance_tol_m=wheel_clearance_tol_m,
    )
    max_rise = _finite(state.max_wheel_supported_both_rise())
    step_height = torch.clamp(_step_height_for_envs(env, step_height_range), min=1.0e-6)
    reached_upper = max_rise >= step_height * float(activation_steps)
    tolerated_drop = step_height * float(drop_tolerance_steps)
    drop_steps = torch.clamp((max_rise - current_rise - tolerated_drop) / step_height, min=0.0)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return _finite(drop_steps * reached_upper.float() * terrain_mask.float() * _upright_gate(env))


def stair_feet_clearance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    h_min: float = 0.03,
    h_max: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """摆动相轮子离地高度奖励，仅在 CTBC 触发时计入。"""
    del asset_cfg
    if _get_stair_state(env) is None:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene[sensor_name]
    wheel_heights = torch.nan_to_num(sensor.data.heights, nan=0.0)
    if wheel_heights.ndim == 1:
        wheel_heights = wheel_heights.unsqueeze(-1)
    in_range = ((wheel_heights > h_min) & (wheel_heights < h_max)).float()
    active = _ctbc_active_side_mask(env, in_range.shape[-1]).float()
    return (in_range * active).sum(dim=-1) * _ctbc_trigger_weight(env)


def stair_feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """摆动相轮子空中时间奖励，仅在 CTBC 触发时计入。"""
    del asset_cfg
    if _get_stair_state(env) is None:
        return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = torch.norm(torch.nan_to_num(data.force, nan=0.0), dim=-1)
    in_air = (force_mag < 1.0).float()
    active = _ctbc_active_side_mask(env, in_air.shape[-1]).float()
    air_time = torch.clamp(in_air * active * float(env.step_dt), max=0.5)
    return air_time.sum(dim=-1) * _ctbc_trigger_weight(env)


def stair_contact_number(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """摆动侧必须离地，支撑侧必须接触；双侧摆动时不给正向支撑奖励。"""
    del asset_cfg
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = torch.norm(torch.nan_to_num(data.force, nan=0.0), dim=-1)
    in_contact = force_mag > 1.0
    ff_active = _ctbc_active_side_mask(env, in_contact.shape[-1])
    swing_match = (~in_contact) & ff_active
    swing_mismatch = in_contact & ff_active
    support_match = in_contact & ~ff_active
    support_mismatch = (~in_contact) & ~ff_active
    has_support_side = (~ff_active).any(dim=-1)
    support_reward = support_match.float().sum(dim=-1)
    support_penalty = support_mismatch.float().sum(dim=-1)
    swing_reward = swing_match.float().sum(dim=-1)
    swing_penalty = swing_mismatch.float().sum(dim=-1)
    reward = swing_reward + support_reward - 1.3 * (swing_penalty + support_penalty)
    reward = torch.where(has_support_side, reward, -2.0 * torch.ones_like(reward))
    return reward * _ctbc_trigger_weight(env)


def stair_wheel_swing_zero_vel(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """摆动相轮子角速度零速奖励，仅在 CTBC 触发时计入。"""
    del sensor_name
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    ff_active = _ctbc_active_side_mask(env, wheel_vel.shape[-1]).float()
    reward = torch.exp(-(ff_active * wheel_vel**2).sum(dim=-1))
    return reward * _ctbc_trigger_weight(env)


def stair_wheel_fore_aft_offset_penalty(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    allowed_offset_m: float = 0.05,
    scale_m: float = 0.04,
    max_penalty: float = 4.0,
    ctbc_active_scale: float = 0.35,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    """惩罚左右轮在机身前后方向的错位。

    SerialLeg 缺少可主动调节轮距的横向自由度，因此不沿用 Tron1 的足端间距
    惩罚；这里只有 base 坐标系 x 方向的左右轮前后错位。CTBC 主动摆轮期间会
    暂时制造小幅前后偏移，所以只在超过容忍带后扣分，并在 CTBC 触发时降权。
    """
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    gate = _stair_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)

    contact_force = _wheel_contact_force(env, contact_sensor_name)
    both_contact = torch.all(contact_force >= float(contact_force_threshold_n), dim=1)
    active = (gate > 0.0) & terrain_mask & both_contact

    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos_w = _finite(robot.data.body_link_pos_w[:, body_ids, :])
    delta_w = wheel_pos_w[:, 0, :] - wheel_pos_w[:, 1, :]
    delta_b = quat_apply_inverse(robot.data.root_link_quat_w, delta_w)
    fore_aft_offset = torch.abs(_finite(delta_b[:, 0]))

    excess = torch.clamp(fore_aft_offset - float(allowed_offset_m), min=0.0)
    penalty = (excess / max(float(scale_m), 1.0e-6)) ** 2
    penalty = torch.clamp(penalty, max=float(max_penalty))

    ctbc_weight = _ctbc_trigger_weight(env)
    ctbc_scale = 1.0 - (1.0 - float(ctbc_active_scale)) * ctbc_weight
    penalty = penalty * ctbc_scale * active.float() * _upright_gate(env)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/diag_wheel_fore_aft_offset_m": _masked_mean(
                    fore_aft_offset,
                    active,
                ),
                "Stair/diag_wheel_fore_aft_penalty": _masked_mean(penalty, active),
                "Stair/diag_wheel_fore_aft_active_rate": active.float().mean().item(),
            }
        )

    return _finite(penalty)


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    radial_velocity_blend: float = 0.75,
    radial_min_distance: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """台阶场景向外爬升速度跟踪，避免车身歪后沿自身 x 轴跑下台阶。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_vel = _finite(robot.data.root_link_lin_vel_b[:, 0])
    command = _finite(cmd[:, 0])
    body_score = torch.exp(-((lin_vel - command) ** 2) / sigma)

    origins = getattr(env.scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor):
        return _finite(body_score * _upright_gate(env))

    root_pos = _finite(robot.data.root_link_pos_w[:, :2])
    root_vel = _finite(robot.data.root_link_lin_vel_w[:, :2])
    radial = root_pos - origins[:, :2].to(device=env.device)
    radial_distance = torch.linalg.vector_norm(radial, dim=1)
    radial_dir = radial / torch.clamp(radial_distance, min=1.0e-6).unsqueeze(-1)
    radial_vel = torch.sum(root_vel * radial_dir, dim=1)
    radial_command = torch.clamp(command, min=0.0)
    radial_score = torch.exp(-((radial_vel - radial_command) ** 2) / sigma)

    blend = min(max(float(radial_velocity_blend), 0.0), 1.0)
    score = (1.0 - blend) * body_score + blend * radial_score
    near_center = radial_distance < float(radial_min_distance)
    score = torch.where(near_center, body_score, score)
    return _finite(score * _upright_gate(env))


def _radial_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    radial_min_distance: float,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    origins = getattr(env.scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor):
        return _finite(robot.data.root_link_lin_vel_b[:, 0])

    root_pos = _finite(robot.data.root_link_pos_w[:, :2])
    root_vel = _finite(robot.data.root_link_lin_vel_w[:, :2])
    radial = root_pos - origins[:, :2].to(device=env.device)
    radial_distance = torch.linalg.vector_norm(radial, dim=1)
    radial_dir = radial / torch.clamp(radial_distance, min=1.0e-6).unsqueeze(-1)
    radial_vel = torch.sum(root_vel * radial_dir, dim=1)
    body_vx = _finite(robot.data.root_link_lin_vel_b[:, 0])
    return torch.where(radial_distance < float(radial_min_distance), body_vx, radial_vel)


def stair_radial_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    speed_scale: float = 0.30,
    command_threshold: float = 0.2,
    radial_min_distance: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """奖励沿台阶径向向外前进，后退时给负值。"""
    cmd = env.command_manager.get_command(command_name)
    commanded_forward = cmd[:, 0] > float(command_threshold)
    radial_vel = _radial_velocity(env, asset_cfg, radial_min_distance)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    scaled = torch.clamp(radial_vel / max(float(speed_scale), 1.0e-6), min=-1.0, max=1.0)
    return _finite(scaled * commanded_forward.float() * terrain_mask.float() * _upright_gate(env))


def stair_radial_retreat(
    env: ManagerBasedRlEnv,
    command_name: str,
    deadband_mps: float = 0.03,
    speed_scale: float = 0.30,
    command_threshold: float = 0.2,
    radial_min_distance: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """惩罚沿台阶径向退回坑底。"""
    cmd = env.command_manager.get_command(command_name)
    commanded_forward = cmd[:, 0] > float(command_threshold)
    radial_vel = _radial_velocity(env, asset_cfg, radial_min_distance)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    retreat = torch.clamp(
        (-radial_vel - float(deadband_mps)) / max(float(speed_scale), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    return _finite(retreat * commanded_forward.float() * terrain_mask.float() * _upright_gate(env))


def stair_riser_stall(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_duration_s: float = 0.25,
    command_threshold: float = 0.2,
    speed_threshold: float = 0.15,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """惩罚轮子持续顶住台阶立面但机体没有继续前进。"""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    commanded_forward = command[:, 0] > float(command_threshold)
    stalled = torch.abs(robot.data.root_link_lin_vel_b[:, 0]) < float(speed_threshold)
    riser_contact = state.riser_stall_active(min_duration_s)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return (commanded_forward & stalled & riser_contact & terrain_mask).float() * _upright_gate(env)


def stair_commanded_stall(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.2,
    forward_speed_threshold: float = 0.15,
    vertical_speed_threshold: float = 0.04,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """惩罚有前进指令但既不前进也不爬升的台阶停滞。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    commanded_forward = command[:, 0] > float(command_threshold)
    slow_forward = robot.data.root_link_lin_vel_b[:, 0] < float(forward_speed_threshold)
    slow_vertical = torch.abs(robot.data.root_link_lin_vel_w[:, 2]) < float(
        vertical_speed_threshold
    )
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return (
        commanded_forward & slow_forward & slow_vertical & terrain_mask
    ).float() * _upright_gate(env)


def leg_torques_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from se3_train.mdp.rewards import leg_torques

    result = leg_torques(env, asset_cfg=asset_cfg)
    state = _get_stair_state(env)
    if state is None:
        return result * _upright_gate(env)
    return result * (1.0 - _ctbc_trigger_weight(env)) * _upright_gate(env)


def leg_power_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from se3_train.mdp.rewards import leg_power

    result = leg_power(env, asset_cfg=asset_cfg)
    state = _get_stair_state(env)
    if state is None:
        return result * _upright_gate(env)
    return result * (1.0 - _ctbc_trigger_weight(env)) * _upright_gate(env)


def stand_still_no_ctbc(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = _DEFAULT_STANDING_HEIGHT,
    height_tolerance: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from se3_train.mdp.rewards import stand_still

    result = stand_still(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        default_height=default_height,
        height_tolerance=height_tolerance,
        asset_cfg=asset_cfg,
    )
    if _get_stair_state(env) is None:
        return result
    return result * (1.0 - _ctbc_trigger_weight(env))


def action_rate_no_ctbc(env: ManagerBasedRlEnv) -> torch.Tensor:
    from se3_train.mdp.rewards import action_rate

    result = action_rate(env)
    if _get_stair_state(env) is None:
        return result * _upright_gate(env)
    scale = 1.0 - 0.8 * _ctbc_trigger_weight(env)
    return result * scale * _upright_gate(env)


def contact_forces_no_ctbc(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from se3_train.mdp.rewards import contact_forces

    result = contact_forces(
        env,
        threshold=threshold,
        sensor_name=sensor_name,
        asset_cfg=asset_cfg,
        use_recovery_gate=False,
    )
    if _get_stair_state(env) is None:
        return result
    return result * (1.0 - 0.5 * _ctbc_trigger_weight(env))


def recovery_stagnation_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    max_steps: int = 256,
    min_delta: float = 0.02,
    height_scale: float = 0.08,
) -> torch.Tensor:
    """非终止型 recovery 停滞惩罚；只施加梯度压力，不触发 timeout/reset。"""
    active = recovery_state.recovery_active_mask(env)
    robot = env.scene["robot"]
    pg_z = torch.nan_to_num(robot.data.projected_gravity_b[:, 2], nan=1.0, posinf=1.0, neginf=1.0)
    upright_score = torch.clamp((-pg_z + 1.0) * 0.5, 0.0, 1.0)

    cmd = env.command_manager.get_command(command_name)
    target_height = cmd[:, 4]
    sensor = env.scene[height_sensor_name]
    height = _finite(sensor.data.heights[:, 0])
    height_score = torch.exp(
        -torch.square(height - target_height) / max(float(height_scale), 1.0e-6)
    )
    score = 0.7 * upright_score + 0.3 * height_score

    best = recovery_state.ensure_float_buffer(env, "_stair_recovery_stagnation_best")
    count = recovery_state.ensure_long_buffer(env, "_stair_recovery_stagnation_count")
    first_step = env.episode_length_buf <= 1
    reset_mask = first_step | ~active
    best[reset_mask] = score[reset_mask].detach()
    count[reset_mask] = 0

    improved = active & (score > best + float(min_delta))
    best[:] = torch.maximum(best, score.detach())
    count[active & ~improved] += 1
    count[improved] = 0

    penalty = torch.clamp(count.float() / max(1, int(max_steps)), 0.0, 1.0) * active.float()
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Recovery/stagnation_steps_nonterminal": _masked_mean(count.float(), active),
                "Recovery/stagnation_penalty_nonterminal": _masked_mean(penalty, active),
                "Recovery/stagnation_score": _masked_mean(score, active),
            }
        )
    return _finite(penalty)


def stair_diagnostics(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
) -> torch.Tensor:
    """把台阶关键指标写入训练日志，不直接改变奖励。"""
    support_params = {
        "height_sensor_name": height_sensor_name,
        "contact_sensor_name": contact_sensor_name,
        "terrain_type_names": terrain_type_names,
        "contact_force_threshold_n": contact_force_threshold_n,
        "wheel_radius_m": wheel_radius_m,
        "wheel_clearance_tol_m": wheel_clearance_tol_m,
    }
    steps = stair_steps_climbed(env, **support_params)
    height_gain = stair_height_gain(
        env,
        command_name=command_name,
        **support_params,
    )
    terrain_level = stair_terrain_level(env)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    recovery_mode = _task_mode_mask(env, (_TASK_MODE_RECOVERY,))
    robot = env.scene["robot"]
    body_vx = _finite(robot.data.root_link_lin_vel_b[:, 0])
    origins = getattr(env.scene, "env_origins", None)
    if isinstance(origins, torch.Tensor):
        root_pos = _finite(robot.data.root_link_pos_w[:, :2])
        root_vel = _finite(robot.data.root_link_lin_vel_w[:, :2])
        radial = root_pos - origins[:, :2].to(device=env.device)
        radial_distance = torch.linalg.vector_norm(radial, dim=1)
        radial_dir = radial / torch.clamp(radial_distance, min=1.0e-6).unsqueeze(-1)
        radial_vx = torch.sum(root_vel * radial_dir, dim=1)
    else:
        radial_vx = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        wheel_support_rise = stair_wheel_support_rise(
            env,
            **support_params,
            support_mode="both",
        )
        state = _get_stair_state(env)
        max_wheel_support_rise = (
            _finite(state.max_wheel_supported_both_rise())
            if state is not None
            else torch.zeros(env.num_envs, device=env.device)
        )
        support_drop = torch.clamp(max_wheel_support_rise - wheel_support_rise, min=0.0)
        raw_wheel_rise = stair_wheel_support_rise(
            env,
            height_sensor_name=height_sensor_name,
            terrain_type_names=terrain_type_names,
            support_mode="both",
            require_contact_support=False,
        )
        max_support_duration = (
            state.max_wheel_supported_both_duration().mean().item() if state is not None else 0.0
        )
        action_term = env.action_manager.get_term("delayed_action")
        ctbc_delta = getattr(action_term, "ctbc_action_delta", None)
        if not isinstance(ctbc_delta, torch.Tensor):
            ctbc_delta = torch.zeros(env.num_envs, 6, device=env.device)
        ctbc_delta = _finite(ctbc_delta)
        env.extras["log"].update(
            {
                "Stair/obs_steps_climbed": steps.mean().item(),
                "Stair/obs_height_gain": height_gain.mean().item(),
                "Stair/obs_x_progress": stair_max_x_progress(env, **support_params).mean().item(),
                "Stair/obs_wheel_support_rise": wheel_support_rise.mean().item(),
                "Stair/obs_wheel_support_rise_max": max_wheel_support_rise.mean().item(),
                "Stair/diag_wheel_support_drop": support_drop.mean().item(),
                "Stair/obs_wheel_terrain_rise_raw": raw_wheel_rise.mean().item(),
                "Stair/diag_wheel_support_both_duration_s": max_support_duration,
                "Stair/obs_terrain_level": terrain_level.mean().item(),
                "Stair/diag_stair_env_rate": terrain_mask.float().mean().item(),
                "Stair/diag_task_mode_recovery_rate": recovery_mode.float().mean().item(),
                "Stair/diag_body_vx": body_vx.mean().item(),
                "Stair/diag_radial_vx": radial_vx.mean().item(),
                "Stair/diag_radial_retreat_rate": (radial_vx < -0.05).float().mean().item(),
                "Stair/diag_ctbc_delta_abs_mean": torch.abs(ctbc_delta).mean().item(),
                "Stair/diag_ctbc_leg_delta_abs_mean": torch.abs(ctbc_delta[:, :4]).mean().item(),
                "Stair/diag_ctbc_wheel_delta_abs_mean": torch.abs(ctbc_delta[:, 4:6]).mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


__all__ = [
    *_FLAT_REWARD_ALL,
    "action_rate_no_ctbc",
    "contact_forces_no_ctbc",
    "leg_power_no_ctbc",
    "leg_torques_no_ctbc",
    "recovery_stagnation_penalty",
    "stair_climb_progress",
    "stair_commanded_stall",
    "stair_contact_number",
    "stair_diagnostics",
    "stair_feet_air_time",
    "stair_feet_clearance",
    "stair_forward_progress",
    "stair_height_gain",
    "stair_max_x_progress",
    "stair_phase_forward_progress",
    "stair_radial_retreat",
    "stair_radial_velocity",
    "stair_riser_stall",
    "stair_steps_climbed",
    "stair_support_descent",
    "stair_support_height",
    "stair_terrain_level",
    "stair_wheel_support_rise",
    "stair_wheel_fore_aft_offset_penalty",
    "stair_wheel_swing_zero_vel",
    "stand_still_no_ctbc",
]
