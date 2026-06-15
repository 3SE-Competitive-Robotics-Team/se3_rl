"""CTBC 台阶任务奖励和诊断函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import rewards as base_rewards
from se3_train.mdp.joint_indices import wheel_joint_ids
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


def _ctbc_trigger_weight(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    return state.ctbc_trigger_weight()


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
    """机器人 z 高度相对 env 基准的抬升量，保留旧指标名。"""
    robot = env.scene["robot"]
    base_z = robot.data.root_link_pos_w[:, 2]
    origin_z = (
        env.scene.env_origins[:, 2]
        if env.scene.env_origins is not None
        else torch.zeros(env.num_envs, device=env.device)
    )
    return ((base_z - origin_z) - _DEFAULT_STANDING_HEIGHT) * _upright_gate(env)


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


def stair_terrain_level(env: ManagerBasedRlEnv) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return torch.zeros(env.num_envs, device=env.device)
    for attr in ("terrain_levels", "env_terrain_level", "level"):
        value = getattr(terrain, attr, None)
        if isinstance(value, torch.Tensor):
            return value.to(device=env.device).float()
    return torch.zeros(env.num_envs, device=env.device)


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
    return in_range.sum(dim=-1) * _ctbc_trigger_weight(env)


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
    air_time = torch.clamp(in_air * float(env.step_dt), max=0.5)
    return air_time.sum(dim=-1) * _ctbc_trigger_weight(env)


def stair_contact_number(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """实际触地/摆动状态与 CTBC 期望 stance 的匹配奖励。"""
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
    ff_active = state._ff_phase >= 0
    match = (in_contact & ~ff_active) | (~in_contact & ff_active)
    mismatch = ~match
    reward = match.float() - 1.3 * mismatch.float()
    return reward.sum(dim=-1) * _ctbc_trigger_weight(env)


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
    ff_active = (state._ff_phase >= 0).float()
    reward = torch.exp(-(ff_active * wheel_vel**2).sum(dim=-1))
    return reward * _ctbc_trigger_weight(env)


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """台阶场景前进速度跟踪，不惩罚合理的爬升 z 速度。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    error = (robot.data.root_link_lin_vel_b[:, 0] - cmd[:, 0]) ** 2
    return torch.exp(-error / sigma) * _upright_gate(env)


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


def stair_diagnostics(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """把台阶关键指标写入训练日志，不直接改变奖励。"""
    steps = stair_steps_climbed(env, terrain_type_names=terrain_type_names)
    height_gain = stair_height_gain(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
    )
    terrain_level = stair_terrain_level(env)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/obs_steps_climbed": steps.mean().item(),
                "Stair/obs_height_gain": height_gain.mean().item(),
                "Stair/obs_x_progress": stair_max_x_progress(env).mean().item(),
                "Stair/obs_terrain_level": terrain_level.mean().item(),
                "Stair/diag_stair_env_rate": terrain_mask.float().mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


__all__ = [
    *_FLAT_REWARD_ALL,
    "action_rate_no_ctbc",
    "contact_forces_no_ctbc",
    "flat_phase_leg_contact_penalty",
    "flat_phase_tracking_lin_vel",
    "flat_phase_wheel_contact_penalty",
    "leg_power_no_ctbc",
    "leg_torques_no_ctbc",
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
    "stair_riser_stall",
    "stair_steps_climbed",
    "stair_terrain_level",
    "stair_wheel_swing_zero_vel",
    "stand_still_no_ctbc",
]
