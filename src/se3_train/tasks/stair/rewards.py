"""CTBC 台阶任务奖励与进度判定函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import recovery_state

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_STAIR_TERRAIN_TYPES = ("forward_stairs",)
_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height
_WHEEL_RADIUS_M = 0.060
_WHEEL_SUPPORT_CLEARANCE_TOL_M = 0.035
_WHEEL_SUPPORT_FORCE_THRESHOLD_N = 1.0
_TASK_MODE_STAIR = 0


def _get_stair_state(env: ManagerBasedRlEnv):
    return getattr(env, "stair_climb_state", None)


def _finite(value: torch.Tensor) -> torch.Tensor:
    """将异常状态的非有限值折成零，避免污染整批奖励。"""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


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


def _signed_x_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回相对 terrain origin 的世界系 +x 进度，和 sim2sim 直线台阶方向对齐。"""
    robot = env.scene[asset_cfg.name]
    origins = getattr(env.scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    root_x = _finite(robot.data.root_link_pos_w[:, 0])
    origin_x = origins[:, 0].to(device=env.device)
    return _finite(root_x - origin_x)


def _lateral_y_offset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回相对 terrain origin 的世界系 y 偏移，用于排除侧向绕台阶。"""
    robot = env.scene[asset_cfg.name]
    origins = getattr(env.scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    root_y = _finite(robot.data.root_link_pos_w[:, 1])
    origin_y = origins[:, 1].to(device=env.device)
    return _finite(root_y - origin_y)


def _wheel_riser_contact_mask(
    env: ManagerBasedRlEnv,
    sensor_name: str | None,
    force_threshold_n: float,
    normal_z_max: float,
) -> torch.Tensor:
    """返回左右轮是否正在接触台阶立面；立面接触不能当作上表面支撑。"""
    mask = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
    if not sensor_name:
        return mask
    try:
        sensor: ContactSensor = env.scene[sensor_name]
    except KeyError:
        return mask
    data = sensor.data
    if data.force is None or data.normal is None:
        return mask

    force = _finite(data.force).reshape(env.num_envs, 2, -1, 3)
    normal = _finite(data.normal).reshape(env.num_envs, 2, -1, 3)
    force_mag = torch.linalg.vector_norm(force, dim=-1)
    valid = (force_mag >= float(force_threshold_n)) & (
        torch.abs(normal[..., 2]) <= float(normal_z_max)
    )
    if data.found is not None:
        found = data.found.reshape(env.num_envs, 2, -1) > 0
        valid &= found
    return valid.any(dim=-1)


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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
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
        riser_contact = _wheel_riser_contact_mask(
            env,
            riser_sensor_name,
            riser_contact_force_threshold_n,
            riser_normal_z_max,
        )
        support_mask &= ~riser_contact
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


def stair_area_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算楼梯本体长宽范围内的当前纵向进度。"""
    terrain = env.scene.terrain
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", {}) or {}
    stair_cfg = next(
        (sub_terrains[name] for name in terrain_type_names if name in sub_terrains),
        None,
    )
    longitudinal_position = _signed_x_progress(env, asset_cfg)
    lateral_position = _lateral_y_offset(env, asset_cfg)
    base_valid = _terrain_type_mask(env, terrain_type_names) & _task_mode_mask(
        env, (_TASK_MODE_STAIR,)
    )
    if stair_cfg is None:
        invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return torch.zeros_like(longitudinal_position), invalid, base_valid, lateral_position

    stair_start = float(stair_cfg.stair_start_x)
    stair_length = float(stair_cfg.step_depth) * max(1, int(stair_cfg.step_count))
    stair_half_width = max(0.0, float(stair_cfg.half_width))
    valid = (
        base_valid
        & (longitudinal_position >= stair_start)
        & (longitudinal_position <= stair_start + stair_length)
        & (torch.abs(lateral_position) <= stair_half_width)
    )
    progress = torch.where(valid, longitudinal_position - stair_start, 0.0)
    return _finite(progress), valid, base_valid, _finite(lateral_position)


def stair_success_components(
    env: ManagerBasedRlEnv,
    min_success_steps: float = 1.0,
    forward_progress_m: float | None = None,
    step_depth_m: float = 0.50,
    forward_progress_step_fraction: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> dict[str, torch.Tensor]:
    """仅按楼梯本体宽度范围内的最大纵向进度判断是否成功。"""
    current_progress, current_valid, base_valid, lateral_position = stair_area_progress(
        env,
        asset_cfg,
        terrain_type_names,
    )

    if forward_progress_m is None:
        forward_target = (
            float(step_depth_m) * float(min_success_steps) * float(forward_progress_step_fraction)
        )
    else:
        forward_target = float(forward_progress_m)
    state = _get_stair_state(env)
    max_progress = (
        _finite(state.max_stair_longitudinal_progress()) if state is not None else current_progress
    )
    target = max(0.0, forward_target)
    current_success = current_valid & (current_progress >= target)
    success = base_valid & (max_progress >= target)

    return {
        "success": success,
        "candidate": current_success,
        "current_success": current_success,
        "current_progress": current_progress,
        "max_progress": max_progress,
        "lateral_position": lateral_position,
        "forward_target": torch.full((env.num_envs,), target, device=env.device),
        "current_valid": current_valid,
        "valid": base_valid,
    }


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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
) -> torch.Tensor:
    """奖励楼梯本体范围内新增的最大纵向进度。"""
    del standing_height
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    del (
        max_height_gain,
        height_sensor_name,
        contact_sensor_name,
        contact_force_threshold_n,
        wheel_radius_m,
        wheel_clearance_tol_m,
        riser_sensor_name,
        riser_contact_force_threshold_n,
        riser_normal_z_max,
    )
    current_progress, _, _, _ = stair_area_progress(
        env,
        asset_cfg=asset_cfg,
        terrain_type_names=terrain_type_names,
    )
    previous_max = state.max_stair_longitudinal_progress()
    current_max = state.record_stair_longitudinal_progress(current_progress)
    progress_delta = torch.clamp(current_max - previous_max, min=0.0)
    progress_delta = torch.clamp(progress_delta, max=max(0.0, float(max_radial_progress)))
    reward = (
        progress_delta / max(float(env.step_dt), 1.0e-6) * float(radial_weight) * _upright_gate(env)
    )
    return _finite(reward)


def stair_support_height(
    env: ManagerBasedRlEnv,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    max_steps: float = 3.0,
    target_steps: float = 1.0,
    success_height_tolerance_m: float = 0.015,
    shaping_power: float = 2.0,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
) -> torch.Tensor:
    """按目标台阶高度给持续奖励，压低低高度支撑的局部最优。"""
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
        riser_sensor_name=riser_sensor_name,
        riser_contact_force_threshold_n=riser_contact_force_threshold_n,
        riser_normal_z_max=riser_normal_z_max,
        use_episode_max=False,
    )
    step_height = torch.clamp(_step_height_for_envs(env, step_height_range), min=1.0e-6)
    target_rise = torch.clamp(
        step_height * float(target_steps) - float(success_height_tolerance_m),
        min=1.0e-6,
    )
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    progress = torch.clamp(current_rise / target_rise, min=0.0)
    shaped_below_target = torch.pow(
        torch.clamp(progress, min=0.0, max=1.0),
        max(1.0, float(shaping_power)),
    )
    above_target = torch.clamp(progress - 1.0, min=0.0, max=max(0.0, float(max_steps) - 1.0))
    steps = torch.clamp(shaped_below_target + above_target, min=0.0, max=float(max_steps))
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


__all__ = [
    "stair_area_progress",
    "stair_climb_progress",
    "stair_success_components",
    "stair_support_height",
    "stair_wheel_support_rise",
]
