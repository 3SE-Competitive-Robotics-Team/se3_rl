"""CTBC 台阶任务奖励和诊断函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import recovery_state
from se3_train.mdp import rewards as mdp_rewards
from se3_train.mdp.joint_indices import wheel_joint_ids
from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_STAIR_TERRAIN_TYPES = ("forward_stairs",)
_FLAT_TERRAIN_TYPES = ("flat",)
_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height
_WHEEL_RADIUS_M = 0.060
_WHEEL_SUPPORT_CLEARANCE_TOL_M = 0.035
_WHEEL_SUPPORT_FORCE_THRESHOLD_N = 1.0
_TASK_MODE_STAIR = 0
_TASK_MODE_RECOVERY = 1
_TASK_MODE_FLAT = 2


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


def _flat_mode_gate(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...] = _FLAT_TERRAIN_TYPES,
) -> torch.Tensor:
    """只在 flat rehearsal mode 且真实平地 terrain 上打开奖励。"""
    return _task_mode_mask(env, (_TASK_MODE_FLAT,)) & _terrain_type_mask(env, terrain_type_names)


def _recovery_active_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """只在 recovery active 阶段打开 rehearsal 奖励。"""
    active = recovery_state.recovery_active_mask(env)
    if active.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return active


def flat_mode_tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float = 0.08,
    sigma_stand: float = 0.1,
    vz_weight: float = 2.0,
    use_upright_gate: bool = False,
    tracking_upright_full_cos: float = 0.7,
    terrain_type_names: tuple[str, ...] = _FLAT_TERRAIN_TYPES,
) -> torch.Tensor:
    """flat rehearsal 的原始平地线速度跟踪奖励。"""
    reward = mdp_rewards.tracking_lin_vel(
        env,
        command_name=command_name,
        sigma_move=sigma_move,
        sigma_stand=sigma_stand,
        vz_weight=vz_weight,
        use_upright_gate=use_upright_gate,
        tracking_upright_full_cos=tracking_upright_full_cos,
    )
    return reward * _flat_mode_gate(env, terrain_type_names).float()


def flat_mode_wheel_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    terrain_type_names: tuple[str, ...] = _FLAT_TERRAIN_TYPES,
) -> torch.Tensor:
    """flat rehearsal 中保持双轮接地。"""
    penalty = mdp_rewards.flat_wheel_contact_penalty(
        env,
        command_name=command_name,
        sensor_name=sensor_name,
        force_threshold=force_threshold,
    )
    return penalty * _flat_mode_gate(env, terrain_type_names).float()


def flat_mode_leg_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    terrain_type_names: tuple[str, ...] = _FLAT_TERRAIN_TYPES,
) -> torch.Tensor:
    """flat rehearsal 中禁止腿部触地。"""
    penalty = mdp_rewards.flat_leg_contact_penalty(
        env,
        command_name=command_name,
        sensor_name=sensor_name,
        force_threshold=force_threshold,
    )
    return penalty * _flat_mode_gate(env, terrain_type_names).float()


def recovery_active_tracking_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    height_sensor_name: str,
    kernel: str = "l2",
    use_upright_gate: bool = False,
    use_pose_end_gate: bool = False,
    use_inverted_free_upright_height_gate: bool = True,
    use_hard_inverted_height_gate: bool = True,
    hard_inverted_release_deg: float = 130.0,
    hard_inverted_full_deg: float = 170.0,
    hard_inverted_min_gate: float = 0.25,
    hard_inverted_wheel_sensor_name: str | None = "wheel_sensor",
    hard_inverted_force_threshold: float = 1.0,
    hard_inverted_wheel_contact_min_count: int = 2,
    hard_inverted_height_tolerance: float = 0.02,
) -> torch.Tensor:
    """recovery discovery 对齐的硬倒置高度 L2，只作用于 active recovery。"""
    reward = mdp_rewards.tracking_height(
        env,
        command_name=command_name,
        sigma=sigma,
        height_sensor_name=height_sensor_name,
        kernel=kernel,
        use_upright_gate=use_upright_gate,
        use_pose_end_gate=use_pose_end_gate,
        use_inverted_free_upright_height_gate=use_inverted_free_upright_height_gate,
        use_hard_inverted_height_gate=use_hard_inverted_height_gate,
        hard_inverted_release_deg=hard_inverted_release_deg,
        hard_inverted_full_deg=hard_inverted_full_deg,
        hard_inverted_min_gate=hard_inverted_min_gate,
        hard_inverted_wheel_sensor_name=hard_inverted_wheel_sensor_name,
        hard_inverted_force_threshold=hard_inverted_force_threshold,
        hard_inverted_wheel_contact_min_count=hard_inverted_wheel_contact_min_count,
        hard_inverted_height_tolerance=hard_inverted_height_tolerance,
    )
    return reward * _recovery_active_gate(env).float()


def recovery_active_upward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """recovery active 阶段的全姿态向上奖励。"""
    return mdp_rewards.upward(env) * _recovery_active_gate(env).float()


def recovery_active_lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    """recovery active 阶段的竖直速度惩罚。"""
    return mdp_rewards.lin_vel_z(env) * _recovery_active_gate(env).float()


def recovery_active_tracking_lin_vel(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的 x 速度跟踪奖励。"""
    return mdp_rewards.tracking_lin_vel(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_tracking_ang_vel(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的 yaw 速度跟踪奖励。"""
    return mdp_rewards.tracking_ang_vel(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    """recovery-discovery active 阶段的 roll/pitch 角速度惩罚。"""
    return mdp_rewards.ang_vel_xy(env) * _recovery_active_gate(env).float()


def recovery_active_upright_orientation_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
    gate_start_deg: float = 60.0,
    gate_full_deg: float = 20.0,
    roll_scale_rad: float = 0.14,
    pitch_scale_rad: float = 0.20,
    roll_weight: float = 1.5,
    pitch_weight: float = 1.0,
    max_penalty: float = 6.0,
) -> torch.Tensor:
    """recovery active 接近直立后压住 pitch/roll 误差。"""
    active = _recovery_active_gate(env)
    if not torch.any(active):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg = robot.data.projected_gravity_b
    tilt_deg = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    gate = torch.clamp((float(gate_start_deg) - tilt_deg) / gate_span, 0.0, 1.0)
    current_pitch = torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0))
    current_roll = torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0))
    pitch_error = current_pitch - cmd[:, 2]
    roll_error = current_roll - cmd[:, 3]
    pitch_term = (pitch_error / max(float(pitch_scale_rad), 1.0e-6)) ** 2
    roll_term = (roll_error / max(float(roll_scale_rad), 1.0e-6)) ** 2
    penalty = torch.clamp(
        float(roll_weight) * roll_term + float(pitch_weight) * pitch_term,
        max=float(max_penalty),
    )
    result = penalty * gate * active.float()
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["Recovery/diag_upright_orientation_penalty"] = _masked_mean(
            result, active
        )
    return result


def recovery_active_upright_zero_velocity_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    wheel_radius: float = _WHEEL_RADIUS_M,
    gate_start_deg: float = 45.0,
    gate_full_deg: float = 15.0,
    base_speed_scale: float = 0.15,
    wheel_speed_scale: float = 0.12,
    base_ang_vel_scale: float = 0.6,
    max_penalty: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """recovery active 接近站稳时抑制漂移和空转。"""
    active_recovery = _recovery_active_gate(env)
    if not torch.any(active_recovery):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    command_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    standing = command_norm <= float(command_threshold)
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)
    active = active_recovery & standing & (near_upright_gate > 0.0)

    base_vxy = robot.data.root_link_lin_vel_b[:, :2]
    base_speed_sq = torch.sum(base_vxy**2, dim=1)
    base_ang_vel_sq = torch.sum(robot.data.root_link_ang_vel_b**2, dim=1)
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    wheel_speed_sq = torch.mean(wheel_forward_speed**2, dim=1)
    penalty = (
        base_speed_sq / (float(base_speed_scale) ** 2)
        + wheel_speed_sq / (float(wheel_speed_scale) ** 2)
        + base_ang_vel_sq / (float(base_ang_vel_scale) ** 2)
    )
    result = torch.clamp(penalty, max=float(max_penalty)) * near_upright_gate * active.float()
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["Recovery/upright_zero_velocity_penalty"] = _masked_mean(result, active)
    return result


def recovery_active_wheel_air_velocity_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    velocity_scale: float = 1.0,
    max_penalty: float = 10000.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    log_prefix: str = "Recovery",
) -> torch.Tensor:
    """recovery active 阶段惩罚离地轮空转。"""
    return mdp_rewards.wheel_air_velocity_penalty(
        env,
        sensor_name=sensor_name,
        force_threshold=force_threshold,
        velocity_scale=velocity_scale,
        max_penalty=max_penalty,
        recovery_active_only=True,
        asset_cfg=asset_cfg,
        log_prefix=log_prefix,
    )


def recovery_active_stand_still(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的站立默认姿态惩罚。"""
    return mdp_rewards.stand_still(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_joint_pos_penalty(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的腿部关节位置惩罚。"""
    return mdp_rewards.joint_pos_penalty(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_leg_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """recovery-discovery active 阶段的腿部 action rate 惩罚。"""
    return mdp_rewards.leg_action_rate(env) * _recovery_active_gate(env).float()


def recovery_active_wheel_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """recovery-discovery active 阶段的轮部 action rate 惩罚。"""
    return mdp_rewards.wheel_action_rate(env) * _recovery_active_gate(env).float()


def recovery_active_action_smoothness(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的二阶 action 平滑惩罚。"""
    return mdp_rewards.action_smoothness(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_leg_contact_penalty(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的腿部触地惩罚。"""
    return mdp_rewards.leg_contact_penalty(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_wheel_contact_without_cmd(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段的零指令轮接触奖励。"""
    return mdp_rewards.feet_contact_without_cmd(env, **kwargs) * _recovery_active_gate(env).float()


def recovery_active_diagnostics(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """recovery-discovery active 阶段沿用 recovery diagnostics。"""
    return mdp_rewards.recovery_diagnostics(env, **kwargs) * _recovery_active_gate(env).float()


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


def _contact_force_max(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    try:
        sensor: ContactSensor = env.scene[sensor_name]
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = torch.linalg.vector_norm(_finite(data.force), dim=-1)
    while force_mag.ndim > 1:
        force_mag = force_mag.amax(dim=-1)
    return _finite(force_mag)


def _contact_mask(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold_n: float,
) -> torch.Tensor:
    return _contact_force_max(env, sensor_name) > float(force_threshold_n)


def _radial_distance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    origins = getattr(env.scene, "env_origins", None)
    if not isinstance(origins, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    root_pos = _finite(robot.data.root_link_pos_w[:, :2])
    radial = root_pos - origins[:, :2].to(device=env.device)
    return _finite(torch.linalg.vector_norm(radial, dim=1))


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


def _signed_x_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回世界系 +x 速度，避免把径向侧移误当成上台阶前进。"""
    robot = env.scene[asset_cfg.name]
    return _finite(robot.data.root_link_lin_vel_w[:, 0])


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


def stair_success_components(
    env: ManagerBasedRlEnv,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    min_success_steps: float = 1.0,
    success_height_tolerance_m: float = 0.015,
    forward_progress_m: float | None = None,
    step_depth_m: float = 0.50,
    forward_progress_step_fraction: float = 0.75,
    hold_duration_s: float = 0.20,
    upright_threshold: float = -0.90,
    max_vertical_speed_mps: float | None = 1.0,
    min_signed_x_velocity_mps: float | None = -0.10,
    max_ang_vel_radps: float | None = 2.5,
    max_lateral_offset_m: float | None = 0.55,
    max_support_drop_steps: float | None = 0.20,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    leg_contact_sensor_name: str = "leg_contact_sensor",
    base_contact_sensor_name: str = "collision_sensor",
    riser_contact_sensor_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    illegal_contact_force_threshold_n: float = 1.0,
    riser_contact_force_threshold_n: float = 1.0,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
    riser_normal_z_max: float = 0.5,
    riser_stall_duration_s: float = 0.15,
    record: bool = True,
    use_recorded_hold: bool = False,
) -> dict[str, torch.Tensor]:
    """返回台阶 strict success 及其子条件，避免瞬时高度和竖面卡住被误判。"""
    rise, wheel_heights = _wheel_terrain_measurements(env, height_sensor_name, asset_cfg)
    wheel_force = _wheel_contact_force(env, contact_sensor_name)
    wheel_contact = (wheel_force >= float(contact_force_threshold_n)).all(dim=1)
    near_support_height = (
        wheel_heights <= float(wheel_radius_m) + float(wheel_clearance_tol_m)
    ).all(dim=1)
    support_mask = (wheel_force >= float(contact_force_threshold_n)) & (
        wheel_heights <= float(wheel_radius_m) + float(wheel_clearance_tol_m)
    )
    riser_contact = _wheel_riser_contact_mask(
        env,
        riser_contact_sensor_name,
        riser_contact_force_threshold_n,
        riser_normal_z_max,
    )
    support_mask &= ~riser_contact
    supported_rise = torch.where(support_mask, torch.clamp(rise, min=0.0), torch.zeros_like(rise))
    current_both_rise = _finite(torch.min(supported_rise, dim=1).values)

    step_height = torch.clamp(_step_height_for_envs(env, step_height_range), min=1.0e-6)
    height_target = step_height * float(min_success_steps) - float(success_height_tolerance_m)
    height_ok = current_both_rise >= torch.clamp(height_target, min=0.0)

    if forward_progress_m is None:
        forward_target = (
            float(step_depth_m) * float(min_success_steps) * float(forward_progress_step_fraction)
        )
    else:
        forward_target = float(forward_progress_m)
    radial_distance = _radial_distance(env, asset_cfg)
    signed_x_progress = _signed_x_progress(env, asset_cfg)
    lateral_offset = torch.abs(_lateral_y_offset(env, asset_cfg))
    forward_ok = signed_x_progress >= max(0.0, forward_target)
    if max_lateral_offset_m is None:
        lateral_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        lateral_ok = lateral_offset <= float(max_lateral_offset_m)

    robot = env.scene[asset_cfg.name]
    projected_gravity_z = _finite(robot.data.projected_gravity_b[:, 2])
    upright_ok = projected_gravity_z <= float(upright_threshold)
    if max_vertical_speed_mps is None:
        vertical_speed_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        vertical_speed_ok = torch.abs(_finite(robot.data.root_link_lin_vel_w[:, 2])) <= float(
            max_vertical_speed_mps
        )
    if min_signed_x_velocity_mps is None:
        forward_velocity_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        forward_velocity_ok = _signed_x_velocity(env, asset_cfg) >= float(min_signed_x_velocity_mps)
    if max_ang_vel_radps is None:
        angular_velocity_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        angular_velocity_ok = torch.linalg.vector_norm(
            _finite(robot.data.root_link_ang_vel_b), dim=1
        ) <= float(max_ang_vel_radps)

    leg_contact = _contact_mask(env, leg_contact_sensor_name, illegal_contact_force_threshold_n)
    base_contact = _contact_mask(env, base_contact_sensor_name, illegal_contact_force_threshold_n)
    legal_contact_ok = ~(leg_contact | base_contact)

    state = _get_stair_state(env)
    if state is not None:
        riser_clear = ~state.riser_stall_active(riser_stall_duration_s)
        max_both_rise = _finite(state.max_wheel_supported_both_rise())
    else:
        riser_clear = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        max_both_rise = current_both_rise
    riser_clear &= ~riser_contact.any(dim=1)
    if max_support_drop_steps is None:
        support_drop_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        support_drop_ok = (max_both_rise - current_both_rise) <= (
            step_height * float(max_support_drop_steps)
        )

    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    stair_mode = _task_mode_mask(env, (_TASK_MODE_STAIR,))
    valid = terrain_mask & stair_mode
    candidate = (
        valid
        & wheel_contact
        & near_support_height
        & height_ok
        & forward_ok
        & lateral_ok
        & upright_ok
        & vertical_speed_ok
        & forward_velocity_ok
        & angular_velocity_ok
        & legal_contact_ok
        & riser_clear
        & support_drop_ok
    )

    if state is not None and record:
        duration = state.record_stair_success_candidate(
            candidate,
            step_index=int(getattr(env, "common_step_counter", 0)),
        )
    elif state is not None:
        duration = state.stair_success_duration()
    else:
        duration = torch.zeros(env.num_envs, device=env.device)

    hold_duration = max(0.0, float(hold_duration_s))
    if use_recorded_hold and state is not None:
        held = state.max_stair_success_duration() >= hold_duration
    elif hold_duration > 0.0:
        held = duration >= hold_duration
    else:
        held = candidate
    success = valid & held

    return {
        "success": success,
        "candidate": candidate,
        "height_ok": height_ok & valid,
        "forward_ok": forward_ok & valid,
        "upright_ok": upright_ok & valid,
        "vertical_speed_ok": vertical_speed_ok & valid,
        "forward_velocity_ok": forward_velocity_ok & valid,
        "angular_velocity_ok": angular_velocity_ok & valid,
        "lateral_ok": lateral_ok & valid,
        "support_drop_ok": support_drop_ok & valid,
        "legal_contact_ok": legal_contact_ok & valid,
        "riser_clear": riser_clear & valid,
        "wheel_contact": wheel_contact & valid,
        "near_support_height": near_support_height & valid,
        "current_both_rise": current_both_rise,
        "radial_distance": radial_distance,
        "signed_x_progress": signed_x_progress,
        "lateral_offset": lateral_offset,
        "support_drop": torch.clamp(max_both_rise - current_both_rise, min=0.0),
        "duration": duration,
        "step_height": step_height,
        "height_target": torch.clamp(height_target, min=0.0),
        "forward_target": torch.full((env.num_envs,), forward_target, device=env.device),
        "valid": valid,
    }


def stair_success_mask(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """返回 strict stair success 布尔 mask。"""
    return stair_success_components(env, **kwargs)["success"]


def stair_success(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """strict stair success 奖励，只在双轮上台面、前进并稳定保持后给正反馈。"""
    return stair_success_mask(env, **kwargs).float()


def _ctbc_trigger_weight(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    weight = state.ctbc_trigger_weight()
    recovery_active = recovery_state.recovery_active_mask(env)
    if recovery_active.shape[0] == env.num_envs:
        weight = torch.where(recovery_active, torch.zeros_like(weight), weight)
    return weight


def _ctbc_phase_weight(env: ManagerBasedRlEnv) -> torch.Tensor:
    """CTBC 相位奖励使用真实触发状态，不随物理前馈退火一起消失。"""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    weight = state.contact_triggered().float()
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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
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
        riser_sensor_name=riser_sensor_name,
        riser_contact_force_threshold_n=riser_contact_force_threshold_n,
        riser_normal_z_max=riser_normal_z_max,
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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
) -> torch.Tensor:
    """保留旧指标名；实际记录左右轮共同支撑地形的抬升量。"""
    del height_sensor_name, contact_sensor_name, terrain_type_names
    del contact_force_threshold_n, wheel_radius_m, wheel_clearance_tol_m
    del riser_sensor_name, riser_contact_force_threshold_n, riser_normal_z_max
    return _finite(torch.clamp(_signed_x_progress(env, asset_cfg), min=0.0) * _upright_gate(env))


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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
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
        riser_sensor_name=riser_sensor_name,
        riser_contact_force_threshold_n=riser_contact_force_threshold_n,
        riser_normal_z_max=riser_normal_z_max,
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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
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
        riser_sensor_name=riser_sensor_name,
        riser_contact_force_threshold_n=riser_contact_force_threshold_n,
        riser_normal_z_max=riser_normal_z_max,
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
    riser_sensor_name: str | None = None,
    riser_contact_force_threshold_n: float = 1.0,
    riser_normal_z_max: float = 0.5,
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
        riser_sensor_name=riser_sensor_name,
        riser_contact_force_threshold_n=riser_contact_force_threshold_n,
        riser_normal_z_max=riser_normal_z_max,
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
    return (in_range * active).sum(dim=-1) * _ctbc_phase_weight(env)


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
    return air_time.sum(dim=-1) * _ctbc_phase_weight(env)


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
    return reward * _ctbc_phase_weight(env)


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
    return reward * _ctbc_phase_weight(env)


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

    del radial_min_distance
    signed_x_vel = _signed_x_velocity(env, asset_cfg)
    signed_x_command = torch.clamp(command, min=0.0)
    signed_x_score = torch.exp(-((signed_x_vel - signed_x_command) ** 2) / sigma)

    blend = min(max(float(radial_velocity_blend), 0.0), 1.0)
    score = (1.0 - blend) * body_score + blend * signed_x_score
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
    del radial_min_distance
    radial_vel = _signed_x_velocity(env, asset_cfg)
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
    del radial_min_distance
    radial_vel = _signed_x_velocity(env, asset_cfg)
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
    step_height_range: tuple[float, float] = (0.05, 0.20),
    min_success_steps: float = 1.0,
    success_height_tolerance_m: float = 0.015,
    forward_progress_m: float | None = None,
    step_depth_m: float = 0.50,
    forward_progress_step_fraction: float = 0.75,
    hold_duration_s: float = 0.20,
    upright_threshold: float = -0.90,
    max_vertical_speed_mps: float | None = 1.0,
    min_signed_x_velocity_mps: float | None = -0.10,
    max_ang_vel_radps: float | None = 2.5,
    max_lateral_offset_m: float | None = 0.55,
    max_support_drop_steps: float | None = 0.20,
    height_sensor_name: str = "wheel_height_sensor",
    contact_sensor_name: str = "wheel_sensor",
    leg_contact_sensor_name: str = "leg_contact_sensor",
    base_contact_sensor_name: str = "collision_sensor",
    riser_contact_sensor_name: str | None = None,
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
    contact_force_threshold_n: float = _WHEEL_SUPPORT_FORCE_THRESHOLD_N,
    illegal_contact_force_threshold_n: float = 1.0,
    riser_contact_force_threshold_n: float = 1.0,
    wheel_radius_m: float = _WHEEL_RADIUS_M,
    wheel_clearance_tol_m: float = _WHEEL_SUPPORT_CLEARANCE_TOL_M,
    riser_normal_z_max: float = 0.5,
    riser_stall_duration_s: float = 0.15,
) -> torch.Tensor:
    """把台阶关键指标写入训练日志，不直接改变奖励。"""
    support_params = {
        "height_sensor_name": height_sensor_name,
        "contact_sensor_name": contact_sensor_name,
        "terrain_type_names": terrain_type_names,
        "contact_force_threshold_n": contact_force_threshold_n,
        "wheel_radius_m": wheel_radius_m,
        "wheel_clearance_tol_m": wheel_clearance_tol_m,
        "riser_sensor_name": riser_contact_sensor_name,
        "riser_contact_force_threshold_n": riser_contact_force_threshold_n,
        "riser_normal_z_max": riser_normal_z_max,
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
    flat_mode = _task_mode_mask(env, (_TASK_MODE_FLAT,))
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
        ctbc_phase_active = (
            state.contact_triggered().float()
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
        success_components = stair_success_components(
            env,
            step_height_range=step_height_range,
            min_success_steps=min_success_steps,
            success_height_tolerance_m=success_height_tolerance_m,
            forward_progress_m=forward_progress_m,
            step_depth_m=step_depth_m,
            forward_progress_step_fraction=forward_progress_step_fraction,
            hold_duration_s=hold_duration_s,
            upright_threshold=upright_threshold,
            max_vertical_speed_mps=max_vertical_speed_mps,
            min_signed_x_velocity_mps=min_signed_x_velocity_mps,
            max_ang_vel_radps=max_ang_vel_radps,
            max_lateral_offset_m=max_lateral_offset_m,
            max_support_drop_steps=max_support_drop_steps,
            height_sensor_name=height_sensor_name,
            contact_sensor_name=contact_sensor_name,
            leg_contact_sensor_name=leg_contact_sensor_name,
            base_contact_sensor_name=base_contact_sensor_name,
            riser_contact_sensor_name=riser_contact_sensor_name,
            terrain_type_names=terrain_type_names,
            contact_force_threshold_n=contact_force_threshold_n,
            illegal_contact_force_threshold_n=illegal_contact_force_threshold_n,
            riser_contact_force_threshold_n=riser_contact_force_threshold_n,
            wheel_radius_m=wheel_radius_m,
            wheel_clearance_tol_m=wheel_clearance_tol_m,
            riser_normal_z_max=riser_normal_z_max,
            riser_stall_duration_s=riser_stall_duration_s,
            record=True,
        )
        valid_success = success_components["valid"]
        action_term = env.action_manager.get_term("delayed_action")
        ctbc_delta = getattr(action_term, "ctbc_action_delta", None)
        if not isinstance(ctbc_delta, torch.Tensor):
            ctbc_delta = torch.zeros(env.num_envs, 6, device=env.device)
        ctbc_delta = _finite(ctbc_delta)
        flat_zero_mask = getattr(env, "_stair_flat_zero_command_mask", None)
        if isinstance(flat_zero_mask, torch.Tensor) and flat_zero_mask.shape[0] == env.num_envs:
            flat_zero_rate = _masked_mean(flat_zero_mask.float(), flat_mode)
        else:
            flat_zero_rate = 0.0
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
                "Stair/strict_success_rate": _masked_mean(
                    success_components["success"].float(),
                    valid_success,
                ),
                "Stair/strict_success_candidate_rate": _masked_mean(
                    success_components["candidate"].float(),
                    valid_success,
                ),
                "Stair/strict_success_height_cond_rate": _masked_mean(
                    success_components["height_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_forward_cond_rate": _masked_mean(
                    success_components["forward_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_lateral_cond_rate": _masked_mean(
                    success_components["lateral_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_upright_cond_rate": _masked_mean(
                    success_components["upright_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_forward_velocity_cond_rate": _masked_mean(
                    success_components["forward_velocity_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_ang_vel_cond_rate": _masked_mean(
                    success_components["angular_velocity_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_drop_cond_rate": _masked_mean(
                    success_components["support_drop_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_contact_cond_rate": _masked_mean(
                    success_components["legal_contact_ok"].float(),
                    valid_success,
                ),
                "Stair/strict_success_riser_clear_rate": _masked_mean(
                    success_components["riser_clear"].float(),
                    valid_success,
                ),
                "Stair/strict_success_duration_s": _masked_mean(
                    success_components["duration"],
                    valid_success,
                ),
                "Stair/strict_success_height_target_m": _masked_mean(
                    success_components["height_target"],
                    valid_success,
                ),
                "Stair/strict_success_forward_target_m": _masked_mean(
                    success_components["forward_target"],
                    valid_success,
                ),
                "Stair/strict_success_signed_x_progress_m": _masked_mean(
                    success_components["signed_x_progress"],
                    valid_success,
                ),
                "Stair/strict_success_lateral_offset_m": _masked_mean(
                    success_components["lateral_offset"],
                    valid_success,
                ),
                "Stair/obs_terrain_level": terrain_level.mean().item(),
                "Stair/diag_stair_env_rate": terrain_mask.float().mean().item(),
                "Stair/diag_task_mode_recovery_rate": recovery_mode.float().mean().item(),
                "Stair/diag_task_mode_flat_rate": flat_mode.float().mean().item(),
                "Stair/diag_task_mode_stair_rate": (
                    _task_mode_mask(env, (_TASK_MODE_STAIR,)).float().mean().item()
                ),
                "Stair/diag_flat_zero_command_rate": flat_zero_rate,
                "Stair/diag_body_vx": body_vx.mean().item(),
                "Stair/diag_radial_vx": radial_vx.mean().item(),
                "Stair/diag_radial_retreat_rate": (radial_vx < -0.05).float().mean().item(),
                "Stair/diag_ctbc_delta_abs_mean": torch.abs(ctbc_delta).mean().item(),
                "Stair/diag_ctbc_leg_delta_abs_mean": torch.abs(ctbc_delta[:, :4]).mean().item(),
                "Stair/diag_ctbc_wheel_delta_abs_mean": torch.abs(ctbc_delta[:, 4:6]).mean().item(),
                "Stair/diag_ctbc_phase_active_rate": ctbc_phase_active.mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


__all__ = [
    *_FLAT_REWARD_ALL,
    "action_rate_no_ctbc",
    "contact_forces_no_ctbc",
    "flat_mode_leg_contact_penalty",
    "flat_mode_tracking_lin_vel",
    "flat_mode_wheel_contact_penalty",
    "leg_power_no_ctbc",
    "leg_torques_no_ctbc",
    "recovery_active_lin_vel_z",
    "recovery_active_tracking_height",
    "recovery_active_upward",
    "recovery_active_upright_orientation_l2",
    "recovery_active_upright_zero_velocity_penalty",
    "recovery_active_wheel_air_velocity_penalty",
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
    "stair_success",
    "stair_success_components",
    "stair_success_mask",
    "stair_support_descent",
    "stair_support_height",
    "stair_terrain_level",
    "stair_wheel_support_rise",
    "stair_wheel_fore_aft_offset_penalty",
    "stair_wheel_swing_zero_vel",
    "stand_still_no_ctbc",
]
