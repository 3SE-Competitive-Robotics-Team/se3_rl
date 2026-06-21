"""CTBC 台阶任务奖励和诊断函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import rewards as base_rewards
from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_STAIR_TERRAIN_TYPES = ("inv_pyramid_stairs",)
_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height


def _get_stair_state(env: ManagerBasedRlEnv):
    return getattr(env, "stair_climb_state", None)


def _upright_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
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
    return mask


def _finite_mean_item(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    finite_mask = torch.isfinite(value)
    if mask is not None:
        finite_mask = finite_mask & mask
    if not torch.any(finite_mask):
        return 0.0
    return value[finite_mask].float().mean().item()


def _ctbc_trigger_weight(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    return state.ctbc_trigger_weight()


def _ctbc_trigger_active(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    return state.contact_triggered().float()


def _wheel_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    attr_name = f"_stair_ctbc_wheel_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _ctbc_side_outcome_components(
    env: ManagerBasedRlEnv,
    *,
    command_name: str,
    terrain_type_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-side CTBC active mask, height gain, and target-direction progress gain."""
    state = _get_stair_state(env)
    if state is None:
        zero = torch.zeros(env.num_envs, 2, device=env.device)
        return zero.bool(), zero, zero

    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos_w = torch.nan_to_num(robot.data.body_link_pos_w[:, body_ids, :], nan=0.0)
    wheel_z = wheel_pos_w[:, :, 2]
    direction = _target_direction_w(env, command_name)
    wheel_progress = torch.sum(wheel_pos_w[:, :, :2] * direction.unsqueeze(1), dim=-1)

    active = state.ff_phase >= 0
    shape = (env.num_envs, 2)
    start_z_name = "_stair_ctbc_side_start_z"
    start_progress_name = "_stair_ctbc_side_start_progress"
    initialized_name = "_stair_ctbc_side_initialized"
    start_z = getattr(env, start_z_name, None)
    start_progress = getattr(env, start_progress_name, None)
    initialized = getattr(env, initialized_name, None)
    if (
        not isinstance(start_z, torch.Tensor)
        or start_z.shape != shape
        or start_z.device != env.device
    ):
        start_z = torch.zeros(shape, device=env.device)
        setattr(env, start_z_name, start_z)
    if (
        not isinstance(start_progress, torch.Tensor)
        or start_progress.shape != shape
        or start_progress.device != env.device
    ):
        start_progress = torch.zeros(shape, device=env.device)
        setattr(env, start_progress_name, start_progress)
    if (
        not isinstance(initialized, torch.Tensor)
        or initialized.shape != shape
        or initialized.device != env.device
    ):
        initialized = torch.zeros(shape, device=env.device, dtype=torch.bool)
        setattr(env, initialized_name, initialized)

    initialized[~active] = False
    new_active = active & (~initialized)
    start_z[new_active] = wheel_z[new_active]
    start_progress[new_active] = wheel_progress[new_active]
    initialized[new_active] = True

    height_gain = torch.clamp(wheel_z - start_z, min=0.0)
    progress_gain = torch.clamp(wheel_progress - start_progress, min=0.0)
    terrain_mask = _terrain_type_mask(env, terrain_type_names).unsqueeze(1)
    active = active & terrain_mask
    return active, height_gain, progress_gain


def _ctbc_side_terrain_contact(
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    """Return per-side wheel contact with terrain, including side faces and top surfaces."""
    try:
        sensor: ContactSensor = env.scene[sensor_name]
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    force = torch.nan_to_num(data.force, nan=0.0).reshape(env.num_envs, 2, -1, 3)
    valid = torch.ones(force.shape[:-1], dtype=torch.bool, device=env.device)
    if data.found is not None:
        found = data.found.reshape(env.num_envs, 2, -1) > 0
        valid = valid & found
    force_contact = torch.norm(force, dim=-1) > float(force_threshold)
    return (valid & force_contact).any(dim=-1)


def _ctbc_side_time_weight(
    env: ManagerBasedRlEnv,
    *,
    reward_window_s: float,
    decay_power: float,
) -> torch.Tensor:
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    elapsed = torch.clamp(state.ff_phase.float(), min=0.0) * float(state.control_dt)
    window = max(float(reward_window_s), float(state.control_dt))
    linear = torch.clamp(1.0 - elapsed / window, min=0.0, max=1.0)
    return linear ** max(float(decay_power), 0.0)


def _local_iteration(env: ManagerBasedRlEnv, steps_per_policy_iter: int = 64) -> int:
    state = _get_stair_state(env)
    if state is not None:
        return int(state.local_iteration)
    return int(getattr(env, "common_step_counter", 0)) // max(1, int(steps_per_policy_iter))


def _walking_phase_gate(
    env: ManagerBasedRlEnv,
    walking_phase_iterations: int,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    active = _local_iteration(env, steps_per_policy_iter) < max(0, int(walking_phase_iterations))
    return torch.full((env.num_envs,), float(active), device=env.device)


def _stair_phase_gate(
    env: ManagerBasedRlEnv,
    walking_phase_iterations: int,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    active = _local_iteration(env, steps_per_policy_iter) >= max(0, int(walking_phase_iterations))
    return torch.full((env.num_envs,), float(active), device=env.device)


def _terrain_size_x(env: ManagerBasedRlEnv) -> float:
    terrain = getattr(env.scene, "terrain", None)
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    return float(getattr(terrain_generator, "size", (8.0, 8.0))[0])


def _radial_progress_components(
    env: ManagerBasedRlEnv,
    *,
    move_up_distance_ratio: float,
    terrain_type_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    root_xy = robot.data.root_link_pos_w[:, :2]
    origin_xy = (
        env.scene.env_origins[:, :2]
        if env.scene.env_origins is not None
        else torch.zeros_like(root_xy)
    )
    offset_xy = root_xy - origin_xy
    distance = torch.norm(offset_xy, dim=1)
    move_up_distance = max(_terrain_size_x(env) * float(move_up_distance_ratio), 1.0e-6)
    progress = distance / move_up_distance
    terrain_mask = _terrain_type_mask(env, terrain_type_names)

    radial_dir = offset_xy / torch.clamp(distance, min=1.0e-6).unsqueeze(1)
    radial_speed = torch.sum(robot.data.root_link_lin_vel_w[:, :2] * radial_dir, dim=1)
    radial_speed = torch.where(distance > 1.0e-5, radial_speed, torch.zeros_like(radial_speed))
    return progress, radial_speed, distance, terrain_mask


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _command_term(env: ManagerBasedRlEnv, command_name: str):
    try:
        return env.command_manager.get_term(command_name)
    except Exception:
        return None


def _target_direction_w(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    term = _command_term(env, command_name)
    direction = getattr(term, "target_direction_w", None)
    if isinstance(direction, torch.Tensor) and direction.shape == (env.num_envs, 2):
        direction = direction.to(device=env.device, dtype=torch.float32)
    else:
        robot = env.scene["robot"]
        quat = robot.data.root_link_quat_w
        yaw = _yaw_from_quat_wxyz(quat)
        direction = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    norm = torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    return direction / norm


def _target_yaw_w(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    term = _command_term(env, command_name)
    target_yaw = getattr(term, "target_yaw", None)
    if isinstance(target_yaw, torch.Tensor) and target_yaw.shape == (env.num_envs,):
        return target_yaw.to(device=env.device, dtype=torch.float32)
    direction = _target_direction_w(env, command_name)
    return torch.atan2(direction[:, 1], direction[:, 0])


def _target_progress_components(
    env: ManagerBasedRlEnv,
    *,
    command_name: str,
    move_up_distance_ratio: float,
    terrain_type_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    root_xy = robot.data.root_link_pos_w[:, :2]
    origin_xy = (
        env.scene.env_origins[:, :2]
        if env.scene.env_origins is not None
        else torch.zeros_like(root_xy)
    )
    offset_xy = root_xy - origin_xy
    direction = _target_direction_w(env, command_name)
    tangent = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)

    target_distance = torch.sum(offset_xy * direction, dim=1)
    lateral_distance = torch.sum(offset_xy * tangent, dim=1)
    move_up_distance = max(_terrain_size_x(env) * float(move_up_distance_ratio), 1.0e-6)
    progress = target_distance / move_up_distance

    vel_xy = robot.data.root_link_lin_vel_w[:, :2]
    target_speed = torch.sum(vel_xy * direction, dim=1)
    tangent_speed = torch.sum(vel_xy * tangent, dim=1)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return progress, target_speed, tangent_speed, lateral_distance, terrain_mask


def _stair_active_mask(
    env: ManagerBasedRlEnv,
    *,
    command_name: str,
    command_threshold: float,
    upright_threshold: float,
    terrain_type_names: tuple[str, ...],
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return (
        terrain_mask
        & _curriculum_upright_mask(env, upright_threshold)
        & (command[:, 0] > float(command_threshold))
    )


def _curriculum_upright_mask(env: ManagerBasedRlEnv, upright_threshold: float) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b[:, 2] < float(upright_threshold)


def flat_phase_tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float = 2.0,
    use_upright_gate: bool = True,
    tracking_upright_full_cos: float = 0.7,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    gate = _walking_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    reward = base_rewards.tracking_lin_vel(
        env,
        command_name=command_name,
        sigma_move=sigma_move,
        sigma_stand=sigma_stand,
        vz_weight=vz_weight,
        use_upright_gate=use_upright_gate,
        tracking_upright_full_cos=tracking_upright_full_cos,
    )
    return reward * gate


def stair_phase_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    move_up_distance_ratio: float = 0.35,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    gate = _stair_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    return (
        stair_forward_progress(
            env,
            command_name=command_name,
            sigma=sigma,
            move_up_distance_ratio=move_up_distance_ratio,
            terrain_type_names=terrain_type_names,
            asset_cfg=asset_cfg,
        )
        * gate
    )


def flat_phase_wheel_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    gate = _walking_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    penalty = base_rewards.flat_wheel_contact_penalty(
        env,
        command_name=command_name,
        sensor_name=sensor_name,
        force_threshold=force_threshold,
    )
    return penalty * gate


def flat_phase_leg_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    gate = _walking_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    penalty = base_rewards.flat_leg_contact_penalty(
        env,
        command_name=command_name,
        sensor_name=sensor_name,
        force_threshold=force_threshold,
    )
    return penalty * gate


def stair_steps_climbed(
    env: ManagerBasedRlEnv,
    step_height: float | None = None,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    step_depth: float = 0.30,
    start_x_offset: float = 0.0,
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """每个 env 当前越过的台阶级数估算。"""
    del step_depth, start_x_offset
    robot = env.scene["robot"]
    base_z = robot.data.root_link_pos_w[:, 2]
    origin_z = (
        env.scene.env_origins[:, 2]
        if env.scene.env_origins is not None
        else torch.zeros(env.num_envs, device=env.device)
    )
    height_gain = (base_z - origin_z) - float(standing_height)

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
        step_height_tensor = torch.full_like(height_gain, float(step_height))

    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    steps = torch.clamp(height_gain / torch.clamp(step_height_tensor, min=1.0e-6), min=0.0)
    steps = torch.where(terrain_mask, steps, torch.zeros_like(steps))
    return steps * _upright_gate(env)


def stair_max_x_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Normalized radial progress toward the same target used by terrain mastery."""
    progress, _, _, terrain_mask = _radial_progress_components(
        env,
        move_up_distance_ratio=0.35,
        terrain_type_names=_STAIR_TERRAIN_TYPES,
    )
    return torch.where(terrain_mask, progress, torch.zeros_like(progress)) * _upright_gate(env)


def stair_height_gain(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """兼容旧目标任务的高度增益指标。"""
    del command_name
    robot = env.scene["robot"]
    base_z = robot.data.root_link_pos_w[:, 2]
    origin_z = (
        env.scene.env_origins[:, 2]
        if env.scene.env_origins is not None
        else torch.zeros(env.num_envs, device=env.device)
    )
    gain = (base_z - origin_z) - float(standing_height)
    gain = gain * _upright_gate(env)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return torch.where(terrain_mask, torch.clamp(gain, min=0.0), torch.zeros_like(gain))


def stair_climb_progress(
    env: ManagerBasedRlEnv,
    max_height_gain: float = 1.0,
    max_radial_progress: float = 4.0,
    radial_weight: float = 0.25,
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """奖励倒金字塔中的新增爬升和新增径向出坑进度。"""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    base_z = robot.data.root_link_pos_w[:, 2]
    if env.scene.env_origins is not None:
        origin_z = env.scene.env_origins[:, 2]
        radial_progress = torch.norm(
            robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2],
            dim=1,
        )
    else:
        origin_z = torch.zeros(env.num_envs, device=env.device)
        radial_progress = torch.zeros(env.num_envs, device=env.device)

    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    height_gain = (base_z - origin_z) - float(standing_height)
    height_gain = torch.where(terrain_mask, height_gain, torch.zeros_like(height_gain))
    radial_progress = torch.where(terrain_mask, radial_progress, torch.zeros_like(radial_progress))
    height_delta, radial_delta = state.climb_progress_delta(
        height_gain,
        radial_progress,
        max_height_gain=max_height_gain,
        max_radial_progress=max_radial_progress,
    )
    progress_delta = height_delta + float(radial_weight) * radial_delta
    return progress_delta / max(float(env.step_dt), 1.0e-6) * _upright_gate(env)


def stair_target_progress_delta(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    move_up_distance_ratio: float = 0.35,
    upright_threshold: float = -0.5,
    max_progress: float = 1.2,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """Reward only new progress along the same target direction used by mastery."""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    progress, _, _, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    progress = torch.where(terrain_mask, progress, torch.zeros_like(progress))
    progress_delta, _ = state.climb_progress_delta(
        progress,
        torch.zeros_like(progress),
        max_height_gain=float(max_progress),
        max_radial_progress=0.0,
    )
    upright = _curriculum_upright_mask(env, upright_threshold).float()
    return progress_delta / max(float(env.step_dt), 1.0e-6) * upright


def stair_no_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.2,
    move_up_distance_ratio: float = 0.35,
    progress_speed_threshold: float = 0.025,
    success_progress: float = 1.0,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """Penalty for commanded stair episodes that are not moving toward mastery success."""
    command = env.command_manager.get_command(command_name)
    progress, target_speed, _, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    move_up_distance = max(_terrain_size_x(env) * float(move_up_distance_ratio), 1.0e-6)
    progress_speed = target_speed / move_up_distance
    active = (
        terrain_mask
        & _curriculum_upright_mask(env, upright_threshold)
        & (command[:, 0] > float(command_threshold))
        & (progress < float(success_progress))
        & (progress_speed < float(progress_speed_threshold))
    )
    return active.float() * _upright_gate(env)


def stair_backtrack(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.2,
    move_up_distance_ratio: float = 0.35,
    backtrack_speed_threshold: float = 0.015,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """Penalty for moving back toward the pit center while commanded to climb out."""
    command = env.command_manager.get_command(command_name)
    _, target_speed, _, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    move_up_distance = max(_terrain_size_x(env) * float(move_up_distance_ratio), 1.0e-6)
    progress_speed = target_speed / move_up_distance
    active = (
        terrain_mask
        & _curriculum_upright_mask(env, upright_threshold)
        & (command[:, 0] > float(command_threshold))
        & (progress_speed < -float(backtrack_speed_threshold))
    )
    return active.float() * _upright_gate(env)


def stair_ctbc_side_height_gain(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_gain: float = 0.18,
    reward_window_s: float = 0.30,
    decay_power: float = 2.0,
    contact_sensor_name: str = "wheel_riser_sensor",
    contact_force_threshold: float = 1.0,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward early terrain-contacted side height gain without rewarding wheels in free air."""
    active, height_gain, _ = _ctbc_side_outcome_components(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
        asset_cfg=asset_cfg,
    )
    side_contact = _ctbc_side_terrain_contact(
        env,
        sensor_name=contact_sensor_name,
        force_threshold=contact_force_threshold,
    )
    time_weight = _ctbc_side_time_weight(
        env,
        reward_window_s=reward_window_s,
        decay_power=decay_power,
    )
    normalized = torch.clamp(height_gain / max(float(max_gain), 1.0e-6), 0.0, 1.0)
    gated = active & side_contact
    return torch.sum(normalized * time_weight * gated.float(), dim=1) * _upright_gate(env)


def stair_ctbc_side_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_progress: float = 0.35,
    reward_window_s: float = 0.30,
    decay_power: float = 2.0,
    contact_sensor_name: str = "wheel_riser_sensor",
    contact_force_threshold: float = 1.0,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward early terrain-contacted side progress along the target climb direction."""
    active, _, progress_gain = _ctbc_side_outcome_components(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
        asset_cfg=asset_cfg,
    )
    side_contact = _ctbc_side_terrain_contact(
        env,
        sensor_name=contact_sensor_name,
        force_threshold=contact_force_threshold,
    )
    time_weight = _ctbc_side_time_weight(
        env,
        reward_window_s=reward_window_s,
        decay_power=decay_power,
    )
    normalized = torch.clamp(progress_gain / max(float(max_progress), 1.0e-6), 0.0, 1.0)
    gated = active & side_contact
    return torch.sum(normalized * time_weight * gated.float(), dim=1) * _upright_gate(env)


def stair_ctbc_side_no_outcome(
    env: ManagerBasedRlEnv,
    command_name: str,
    grace_s: float = 0.06,
    min_height_gain: float = 0.025,
    min_progress: float = 0.035,
    deadline_s: float = 0.18,
    contact_sensor_name: str = "wheel_riser_sensor",
    contact_force_threshold: float = 1.0,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize a triggered side that fails to produce early contacted outcome."""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    active, height_gain, progress_gain = _ctbc_side_outcome_components(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
        asset_cfg=asset_cfg,
    )
    side_contact = _ctbc_side_terrain_contact(
        env,
        sensor_name=contact_sensor_name,
        force_threshold=contact_force_threshold,
    )
    elapsed = torch.clamp(state.ff_phase.float(), min=0.0) * float(state.control_dt)
    checked_window = (elapsed >= float(grace_s)) & (elapsed <= float(deadline_s))
    has_outcome = side_contact & (
        (height_gain >= float(min_height_gain)) | (progress_gain >= float(min_progress))
    )
    return torch.sum((active & checked_window & (~has_outcome)).float(), dim=1) * _upright_gate(env)


def stair_terrain_level(env: ManagerBasedRlEnv) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return torch.zeros(env.num_envs, device=env.device)
    for attr in ("terrain_levels", "env_terrain_level", "level"):
        value = getattr(terrain, attr, None)
        if isinstance(value, torch.Tensor):
            return value.to(device=env.device).float()
    return torch.zeros(env.num_envs, device=env.device)


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    move_up_distance_ratio: float = 0.35,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """台阶场景按目标方向做速度跟踪，不奖励斜着刷进度。"""
    del asset_cfg
    cmd = env.command_manager.get_command(command_name)
    _, target_speed, _, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    error = (target_speed - cmd[:, 0]) ** 2
    return torch.exp(-error / sigma) * _upright_gate(env) * terrain_mask.float()


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
    command = env.command_manager.get_command(command_name)
    commanded_forward = command[:, 0] > float(command_threshold)
    _, target_speed, _, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=0.35,
        terrain_type_names=terrain_type_names,
    )
    stalled = torch.abs(target_speed) < float(speed_threshold)
    riser_contact = state.riser_stall_active(min_duration_s)
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
    _, target_speed, _, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=0.35,
        terrain_type_names=terrain_type_names,
    )
    slow_forward = target_speed < float(forward_speed_threshold)
    slow_vertical = torch.abs(robot.data.root_link_lin_vel_w[:, 2]) < float(
        vertical_speed_threshold
    )
    return (
        commanded_forward & slow_forward & slow_vertical & terrain_mask
    ).float() * _upright_gate(env)


def stair_tangential_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.2,
    move_up_distance_ratio: float = 0.35,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """惩罚垂直于目标方向的横向速度，抑制斜着刷台阶进度。"""
    _, _, tangent_speed, _, terrain_mask = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    active = _stair_active_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        upright_threshold=upright_threshold,
        terrain_type_names=terrain_type_names,
    )
    return torch.abs(tangent_speed) * active.float() * terrain_mask.float() * _upright_gate(env)


def stair_heading_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.2,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """惩罚车体 yaw 和目标爬升方向不一致。"""
    robot = env.scene["robot"]
    current_yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    target_yaw = _target_yaw_w(env, command_name)
    error = torch.abs(_wrap_to_pi(target_yaw - current_yaw))
    active = _stair_active_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        upright_threshold=upright_threshold,
        terrain_type_names=terrain_type_names,
    )
    return (error / torch.pi) * active.float() * _upright_gate(env)


def stair_speed_deficit(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.2,
    min_speed_fraction: float = 0.65,
    move_up_distance_ratio: float = 0.35,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """惩罚有速度指令但沿目标方向速度明显不足。"""
    command = env.command_manager.get_command(command_name)
    _, target_speed, _, _, _ = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    deficit = torch.relu(float(min_speed_fraction) * command[:, 0] - target_speed)
    active = _stair_active_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        upright_threshold=upright_threshold,
        terrain_type_names=terrain_type_names,
    )
    return deficit * active.float() * _upright_gate(env)


def stair_early_velocity_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    window_s: float = 1.5,
    command_threshold: float = 0.2,
    move_up_distance_ratio: float = 0.35,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    """台阶阶段 episode 起步窗口内额外奖励快速跟上目标速度。"""
    gate = _stair_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    _, target_speed, _, _, _ = _target_progress_components(
        env,
        command_name=command_name,
        move_up_distance_ratio=move_up_distance_ratio,
        terrain_type_names=terrain_type_names,
    )
    early = env.episode_length_buf.to(device=env.device, dtype=torch.float32) * float(
        env.step_dt
    ) < float(window_s)
    active = _stair_active_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        upright_threshold=upright_threshold,
        terrain_type_names=terrain_type_names,
    ) & early
    reward = torch.exp(-((target_speed - command[:, 0]) ** 2) / float(sigma))
    return reward * active.float() * gate * _upright_gate(env)


def stair_early_speed_deficit(
    env: ManagerBasedRlEnv,
    command_name: str,
    window_s: float = 1.5,
    command_threshold: float = 0.2,
    min_speed_fraction: float = 0.65,
    move_up_distance_ratio: float = 0.35,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    walking_phase_iterations: int = 800,
    steps_per_policy_iter: int = 64,
) -> torch.Tensor:
    """台阶阶段起步窗口内更强惩罚慢热，抑制 GRU 固定热身策略。"""
    gate = _stair_phase_gate(env, walking_phase_iterations, steps_per_policy_iter)
    if not torch.any(gate):
        return torch.zeros(env.num_envs, device=env.device)
    early = env.episode_length_buf.to(device=env.device, dtype=torch.float32) * float(
        env.step_dt
    ) < float(window_s)
    return stair_speed_deficit(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        min_speed_fraction=min_speed_fraction,
        move_up_distance_ratio=move_up_distance_ratio,
        upright_threshold=upright_threshold,
        terrain_type_names=terrain_type_names,
    ) * early.float() * gate


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


def joint_mirror_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from se3_train.mdp.rewards import joint_mirror

    result = joint_mirror(env, asset_cfg=asset_cfg)
    if _get_stair_state(env) is None:
        return result
    return result * (1.0 - _ctbc_trigger_active(env))


def contact_forces_stair(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wheel terrain-contact force penalty used in stair training.

    Unlike earlier CTBC runs, this is not discounted during feedforward triggers.
    """
    from se3_train.mdp.rewards import contact_forces

    return contact_forces(
        env,
        threshold=threshold,
        sensor_name=sensor_name,
        asset_cfg=asset_cfg,
        use_recovery_gate=False,
    )


def stair_diagnostics(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """把台阶关键指标写入训练日志，不直接改变奖励。"""
    def _masked_mean(value: torch.Tensor) -> float:
        return _finite_mean_item(value, terrain_mask)

    steps = stair_steps_climbed(env, terrain_type_names=terrain_type_names)
    height_gain = stair_height_gain(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
    )
    terrain_level = stair_terrain_level(env)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    if command_name is not None:
        progress, target_speed, tangent_speed, lateral_distance, _ = _target_progress_components(
            env,
            command_name=command_name,
            move_up_distance_ratio=0.35,
            terrain_type_names=terrain_type_names,
        )
        robot = env.scene["robot"]
        heading_error = torch.abs(
            _wrap_to_pi(_target_yaw_w(env, command_name) - _yaw_from_quat_wxyz(robot.data.root_link_quat_w))
        )
    else:
        progress = torch.zeros(env.num_envs, device=env.device)
        target_speed = torch.zeros(env.num_envs, device=env.device)
        tangent_speed = torch.zeros(env.num_envs, device=env.device)
        lateral_distance = torch.zeros(env.num_envs, device=env.device)
        heading_error = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/obs_steps_climbed": _finite_mean_item(steps),
                "Stair/obs_height_gain": _finite_mean_item(height_gain),
                "Stair/obs_x_progress": _finite_mean_item(stair_max_x_progress(env)),
                "Stair/obs_terrain_level": _finite_mean_item(terrain_level),
                "Stair/diag_stair_env_rate": terrain_mask.float().mean().item(),
                "Stair/target_progress_mean": _masked_mean(progress),
                "Stair/target_speed_mean": _masked_mean(target_speed),
                "Stair/tangent_speed_abs_mean": _masked_mean(torch.abs(tangent_speed)),
                "Stair/lateral_distance_abs_mean": _masked_mean(torch.abs(lateral_distance)),
                "Stair/heading_error_deg_mean": _masked_mean(torch.rad2deg(heading_error)),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


__all__ = [
    *_FLAT_REWARD_ALL,
    "action_rate_no_ctbc",
    "contact_forces_stair",
    "flat_phase_leg_contact_penalty",
    "flat_phase_tracking_lin_vel",
    "flat_phase_wheel_contact_penalty",
    "joint_mirror_no_ctbc",
    "leg_power_no_ctbc",
    "leg_torques_no_ctbc",
    "stair_backtrack",
    "stair_climb_progress",
    "stair_commanded_stall",
    "stair_ctbc_side_forward_progress",
    "stair_ctbc_side_height_gain",
    "stair_ctbc_side_no_outcome",
    "stair_diagnostics",
    "stair_early_speed_deficit",
    "stair_early_velocity_tracking",
    "stair_forward_progress",
    "stair_heading_error",
    "stair_height_gain",
    "stair_max_x_progress",
    "stair_no_progress",
    "stair_phase_forward_progress",
    "stair_riser_stall",
    "stair_speed_deficit",
    "stair_steps_climbed",
    "stair_tangential_velocity",
    "stair_target_progress_delta",
    "stair_terrain_level",
    "stand_still_no_ctbc",
]
