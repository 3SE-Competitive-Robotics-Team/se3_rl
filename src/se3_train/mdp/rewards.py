"""SE3 轮腿机器人的奖励函数。

所有奖励函数接收 env 并返回 [num_envs] 的张量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from se3_shared import RobotConfig as SharedRobotConfig
from se3_shared import (
    output_to_policy_pos_torch,
    output_to_policy_vel_torch,
)
from se3_train.mdp import recovery_state
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.height_default_cache import get_policy_default_from_height_cache
from se3_train.mdp.joint_indices import (
    active_rod_angle_terms,
    is_closedchain_model,
    is_fourbar_surrogate_model,
    leg_actuator_ids,
    policy_leg_joint_ids,
    wheel_actuator_ids,
    wheel_joint_ids,
)
from se3_train.mdp.leg_alignment import wheel_alignment_ok, wheel_alignment_penalty

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SHARED_ROBOT = SharedRobotConfig()
_DEFAULT_REWARD_LOG_INTERVAL_STEPS = 64
_FOURBAR_WHEEL_RADIUS_M = 0.06
_ACTION_SMOOTH_PREV_ATTR = "_se3_action_smooth_prev_action"
_ACTION_SMOOTH_PREV_PREV_ATTR = "_se3_action_smooth_prev_prev_action"


def _recovery_reset_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回当前仍处于 recovery active 模式的 env。"""
    return recovery_state.recovery_active_mask(env)


def _recovery_episode_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回本 episode 是否由 recovery reset 开始。"""
    return recovery_state.recovery_episode_mask(env)


def _recovery_angle_buffer(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    """读取 recovery reset 时记录的初始姿态角。"""
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)
    return values.to(device=env.device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """计算 mask 内均值；空 mask 返回 0。"""
    if mask.any():
        return values[mask].float().mean().item()
    return 0.0


def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> float:
    """计算 mask 内最小值；空 mask 返回 0。"""
    if mask.any():
        return values[mask].float().min().item()
    return 0.0


def _should_log_step(
    env: ManagerBasedRlEnv, interval: int = _DEFAULT_REWARD_LOG_INTERVAL_STEPS
) -> bool:
    """按 policy step 降频写 host 标量日志。"""
    step = int(getattr(env, "common_step_counter", 0))
    interval = max(1, int(getattr(env, "_se3_reward_log_interval_steps", interval)))
    return interval <= 1 or (step - 1) % interval == 0


def _command_curriculum_metrics_enabled(env: ManagerBasedRlEnv) -> bool:
    """是否启用速度课程逐步累计指标。"""
    return bool(getattr(env, "_se3_enable_command_curriculum_metrics", False))


def _log_cached_reset_diagnostics(env: ManagerBasedRlEnv) -> None:
    """将 reset 事件缓存的诊断量转发到训练 logger。"""
    if not hasattr(env, "extras") or not isinstance(env.extras.get("log"), dict):
        return
    values = getattr(env, "_reset_robotlab_full_random_log_values", None)
    if isinstance(values, dict):
        env.extras["log"].update(values)


def _accumulate_command_curriculum_metric(
    env: ManagerBasedRlEnv,
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """累计速度课程使用的逐 episode 原始指标。"""
    if not _command_curriculum_metrics_enabled(env):
        return
    if values.shape[0] != env.num_envs or mask.shape[0] != env.num_envs:
        return

    sum_name = f"_command_curriculum_{name}_sum"
    count_name = f"_command_curriculum_{name}_count"
    sums = getattr(env, sum_name, None)
    counts = getattr(env, count_name, None)
    if not isinstance(sums, torch.Tensor) or sums.shape[0] != env.num_envs:
        sums = torch.zeros(env.num_envs, device=env.device)
        setattr(env, sum_name, sums)
    if not isinstance(counts, torch.Tensor) or counts.shape[0] != env.num_envs:
        counts = torch.zeros(env.num_envs, device=env.device)
        setattr(env, count_name, counts)

    valid = mask & torch.isfinite(values)
    if valid.any():
        sums[valid] += values[valid].detach().float()
        counts[valid] += 1.0


def _policy_leg_pos_and_default(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 policy 主动杆语义下的腿部当前位置和默认位置。"""
    leg_ids = policy_leg_joint_ids(robot)
    joint_pos = robot.data.joint_pos[:, leg_ids]
    default_pos = robot.data.default_joint_pos[:, leg_ids]
    if is_fourbar_surrogate_model(robot):
        joint_pos = output_to_policy_pos_torch(joint_pos)
        default_pos = output_to_policy_pos_torch(default_pos)
    return joint_pos, default_pos


def _policy_leg_pos_and_height_default(
    env: ManagerBasedRlEnv,
    robot,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前腿部位置和随高度指令变化的默认腿部姿态。"""
    joint_pos, default_pos = _policy_leg_pos_and_default(robot)
    if not (is_closedchain_model(robot) or is_fourbar_surrogate_model(robot)):
        return joint_pos, default_pos
    try:
        height_default = get_policy_default_from_height_cache(
            env,
            command_name,
            device=joint_pos.device,
            dtype=joint_pos.dtype,
        )
    except Exception:
        return joint_pos, default_pos
    return joint_pos, height_default


def _active_rod_angles(robot) -> torch.Tensor:
    """返回左右主动杆夹角。"""
    if is_fourbar_surrogate_model(robot):
        pos = output_to_policy_pos_torch(robot.data.joint_pos[:, policy_leg_joint_ids(robot)])
        angles = []
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = _SHARED_ROBOT.active_rod_angle_coeffs[side_idx]
            angles.append(front_coef * pos[:, front_idx] + back_coef * pos[:, back_idx])
        angle_tensor = torch.stack(angles, dim=1)
    else:
        angles = []
        for front_id, back_id, front_coef, back_coef in active_rod_angle_terms(robot):
            angles.append(
                front_coef * robot.data.joint_pos[:, front_id]
                + back_coef * robot.data.joint_pos[:, back_id]
            )
        angle_tensor = torch.stack(angles, dim=1)
    return angle_tensor


def _active_rod_angle_margins(robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回主动杆夹角到上下限的最小余量诊断。"""
    lower, upper = _SHARED_ROBOT.active_rod_angle_limits
    angle_tensor = _active_rod_angles(robot)

    lower_margin = angle_tensor - float(lower)
    upper_margin = float(upper) - angle_tensor
    min_lower = torch.min(lower_margin, dim=1).values
    min_upper = torch.min(upper_margin, dim=1).values
    min_margin = torch.minimum(min_lower, min_upper)
    return min_margin, min_lower, min_upper


def _wheel_clearance_stats(
    env: ManagerBasedRlEnv,
    robot,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回左右轮底离地最小高度和平均高度。"""
    attr_name = f"_recovery_diag_wheel_body_ids_{asset_cfg.name}"
    body_ids = getattr(env, attr_name, None)
    if not isinstance(body_ids, list) or len(body_ids) != 2:
        body_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
        setattr(env, attr_name, body_ids)
    wheel_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    wheel_bottom = wheel_pos_w[:, :, 2] - ground_z - _FOURBAR_WHEEL_RADIUS_M
    return wheel_bottom.min(dim=1).values, wheel_bottom.mean(dim=1)


def _contact_diagnostic_stats(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """读取接触传感器，返回接触率、单 env 最大力和平均力。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        zeros = torch.zeros(env.num_envs, device=env.device)
        return zeros, zeros, zeros

    force_mag = finite_contact_force_norm(data.force)
    if force_mag.ndim == 1:
        force_mag = force_mag.unsqueeze(1)
    contact = force_mag > float(force_threshold)
    contact_ratio = contact.float().mean(dim=1)
    max_force = force_mag.max(dim=1).values
    mean_force = force_mag.mean(dim=1)
    return contact_ratio, max_force, mean_force


def _policy_leg_vel(robot) -> torch.Tensor:
    """返回 policy 主动杆语义下的腿部速度。"""
    leg_ids = policy_leg_joint_ids(robot)
    joint_vel = robot.data.joint_vel[:, leg_ids]
    if is_fourbar_surrogate_model(robot):
        joint_vel = output_to_policy_vel_torch(robot.data.joint_pos[:, leg_ids], joint_vel)
    return joint_vel


def _policy_leg_torque_and_vel(
    env: ManagerBasedRlEnv,
    robot,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 policy 主动杆语义下的腿部电机力矩和速度。"""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            if getattr(term, "_entity", None) is not robot:
                continue
            torque = getattr(term, "policy_leg_torque", None)
            vel = getattr(term, "policy_leg_vel", None)
            if isinstance(torque, torch.Tensor) and isinstance(vel, torch.Tensor):
                return torque, vel

    if is_fourbar_surrogate_model(robot):
        raise RuntimeError("fourbar reward 缺少主动杆力矩缓存，请检查 action term")
    return robot.data.actuator_force[:, leg_actuator_ids(robot)], _policy_leg_vel(robot)


def _policy_leg_acc(
    env: ManagerBasedRlEnv,
    robot,
) -> torch.Tensor:
    """返回 policy 主动杆语义下的腿部加速度。"""
    if not is_fourbar_surrogate_model(robot):
        return robot.data.joint_acc[:, policy_leg_joint_ids(robot)]

    vel = _policy_leg_vel(robot)
    prev = getattr(env, "_reward_policy_leg_vel_prev", None)
    if not isinstance(prev, torch.Tensor) or prev.shape != vel.shape:
        env._reward_policy_leg_vel_prev = vel.detach().clone()
        return torch.zeros_like(vel)

    dt = max(float(env.step_dt), 1.0e-6)
    acc = (vel - prev) / dt
    first_step = env.episode_length_buf <= 1
    if first_step.any():
        acc[first_step] = 0.0
    prev[:] = vel.detach()
    return acc


def _policy_leg_mirror_diffs(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 policy 主动杆语义下的左右腿镜像误差。"""
    joint_pos, _ = _policy_leg_pos_and_default(robot)
    if is_closedchain_model(robot) or is_fourbar_surrogate_model(robot):
        return joint_pos[:, 0] + joint_pos[:, 2], joint_pos[:, 1] + joint_pos[:, 3]
    return joint_pos[:, 0] - joint_pos[:, 2], joint_pos[:, 1] - joint_pos[:, 3]


def _recovery_hard_tilt_mask(
    env: ManagerBasedRlEnv,
    min_initial_tilt_deg: float = 75.0,
) -> torch.Tensor:
    """识别初始大倾角 recovery 样本，不区分倾倒轴向。"""
    active = _recovery_reset_mask(env)
    return _recovery_hard_tilt_mask_for(env, active, min_initial_tilt_deg)


def _recovery_hard_tilt_episode_mask(
    env: ManagerBasedRlEnv,
    min_initial_tilt_deg: float = 75.0,
) -> torch.Tensor:
    """识别本 episode 中的初始大倾角 recovery 样本。"""
    episode = _recovery_episode_mask(env)
    return _recovery_hard_tilt_mask_for(env, episode, min_initial_tilt_deg)


def _recovery_hard_tilt_mask_for(
    env: ManagerBasedRlEnv,
    base_mask: torch.Tensor,
    min_initial_tilt_deg: float,
) -> torch.Tensor:
    """按传入 mask 识别初始大倾角样本。"""
    init_tilt = _recovery_angle_buffer(env, "_recovery_init_tilt")
    min_tilt = torch.deg2rad(torch.tensor(float(min_initial_tilt_deg), device=env.device))
    return base_mask & (init_tilt >= min_tilt)


def _recovery_success_components(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    upright_angle_deg: float,
    height_tolerance: float,
    ang_vel_threshold: float,
    force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算恢复成功及各个成功门控条件。"""
    active = recovery_state.recovery_active_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    upright_limit = torch.deg2rad(torch.tensor(float(upright_angle_deg), device=env.device))
    upright = tilt < upright_limit

    cmd = env.command_manager.get_command(command_name)
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height_ok = torch.abs(height_sensor.data.heights[:, 0] - cmd[:, 4]) < float(height_tolerance)

    ang_vel_norm = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)
    stable = ang_vel_norm < float(ang_vel_threshold)

    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    if data.force is None:
        wheel_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        force_mag = finite_contact_force_norm(data.force)
        wheel_contact = (force_mag > float(force_threshold)).any(dim=1)

    success = active & upright & height_ok & stable & wheel_contact
    return success, wheel_contact, active, upright, height_ok, stable


def _recovery_success_mask(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    upright_angle_deg: float,
    height_tolerance: float,
    ang_vel_threshold: float,
    force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算恢复成功、轮子接地和 recovery active mask。"""
    success, wheel_contact, active, _, _, _ = _recovery_success_components(
        env,
        sensor_name=sensor_name,
        height_sensor_name=height_sensor_name,
        command_name=command_name,
        upright_angle_deg=upright_angle_deg,
        height_tolerance=height_tolerance,
        ang_vel_threshold=ang_vel_threshold,
        force_threshold=force_threshold,
    )
    return success, wheel_contact, active


def _upright_factor(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """计算直立因子:clamp(-pg_z, 0, 0.7) / 0.7。"""
    return torch.clamp(-projected_gravity_z, 0.0, 0.7) / 0.7


def _tracking_upright_gate(
    projected_gravity_z: torch.Tensor,
    full_cos: float = 0.7,
) -> torch.Tensor:
    """计算速度跟踪专用直立门控，避免影响站姿和接触正则。"""
    full_cos = min(max(float(full_cos), 1.0e-6), 1.0)
    return torch.clamp(-projected_gravity_z, 0.0, full_cos) / full_cos


def _near_upright_gate(
    projected_gravity_z: torch.Tensor,
    gate_start_deg: float,
    gate_full_deg: float,
) -> torch.Tensor:
    """按机身倾角生成渐进直立门控，避免恢复早期被站立姿态惩罚束缚。"""
    tilt = torch.rad2deg(torch.acos(torch.clamp(-projected_gravity_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    return torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    """将 [0,1] 门控变成端点斜率为零的平滑曲线。"""
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _upward_score(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """全姿态直立分数：倒置为 0，侧躺为 1，直立为 4。"""
    return torch.square(1.0 - projected_gravity_z)


def _recovery_penalty_gate(
    env: ManagerBasedRlEnv, projected_gravity_z: torch.Tensor
) -> torch.Tensor:
    """倒地恢复早期不惩罚接触，接近直立后再恢复常规惩罚。"""
    grace_steps = int(getattr(env, "_recovery_grace_steps", 74))
    upright = _upright_factor(projected_gravity_z)
    recovery = recovery_state.recovery_active_mask(env)
    in_recovery_grace = recovery & (env.episode_length_buf < grace_steps)
    return torch.where(in_recovery_grace, torch.zeros_like(upright), upright)


def upward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """全局向上奖励，不区分 roll/pitch 轴向来源。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    reward = _upward_score(pg_z)

    if hasattr(env, "extras") and _should_log_step(env):
        tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        upright_15 = tilt < torch.deg2rad(torch.as_tensor(15.0, device=env.device))
        log = env.extras.setdefault("log", {})
        log.update(
            {
                "Locomotion/upward": reward.mean().item(),
                "SelfRight/tilt_deg": torch.rad2deg(tilt).mean().item(),
                "SelfRight/upright_15deg_rate": upright_15.float().mean().item(),
                "Locomotion/upright_gate": _upright_factor(pg_z).mean().item(),
            }
        )
        _log_cached_reset_diagnostics(env)

    return reward


def upward_progress(
    env: ManagerBasedRlEnv,
    delta_scale: float = 0.05,
    max_reward: float = 2.0,
) -> torch.Tensor:
    """全局向上进度奖励，不区分 roll/pitch 轴向来源。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    score = _upward_score(pg_z)

    prev_score = getattr(env, "_prev_upward_score", None)
    if not isinstance(prev_score, torch.Tensor) or prev_score.shape[0] != env.num_envs:
        prev_score = score.detach().clone()
        env._prev_upward_score = prev_score

    first_step = env.episode_length_buf <= 1
    delta = (score - prev_score) / max(float(delta_scale), 1.0e-6)
    reward = torch.clamp(delta, -float(max_reward), float(max_reward))
    reward = torch.where(first_step, torch.zeros_like(reward), reward)
    prev_score[:] = score.detach()

    if hasattr(env, "extras") and _should_log_step(env):
        log = env.extras.setdefault("log", {})
        log.update(
            {
                "SelfRight/upward_progress": reward.mean().item(),
                "SelfRight/upward_progress_pos_rate": (reward > 0.0).float().mean().item(),
                "SelfRight/upward_progress_neg_rate": (reward < 0.0).float().mean().item(),
            }
        )

    return reward


def tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float = 2.0,
    use_upright_gate: bool = True,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """x 方向速度跟踪,将 v_z 折入同一 exp 核消除目标冲突。

    reward = exp(-(error_x² + vz_weight·v_z²) / sigma)
    低速时 sigma 收紧(adaptive),直立门控。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    lin_vel = robot.data.root_link_lin_vel_b
    error_x = lin_vel[:, 0] - cmd[:, 0]
    vz = lin_vel[:, 2]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _tracking_upright_gate(pg_z, tracking_upright_full_cos)

    cmd_mag = torch.abs(cmd[:, 0])
    sigma = torch.where(cmd_mag < 0.2, sigma_stand, sigma_move)
    reward = torch.exp(-(error_x**2 + vz_weight * vz**2) / sigma)
    if use_upright_gate:
        reward = reward * gate
    jump_flag = cmd[:, 5] > 0.5 if cmd.shape[1] > 5 else torch.zeros_like(cmd_mag, dtype=torch.bool)
    moving = (cmd_mag >= 0.2) & ~jump_flag
    locomotion = ~jump_flag
    idle = (torch.abs(cmd[:, 0]) < 0.08) & (torch.abs(cmd[:, 1]) < 0.08) & locomotion

    _accumulate_command_curriculum_metric(env, "lin_score", reward, moving)
    _accumulate_command_curriculum_metric(env, "upright_score", gate, locomotion)

    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * 0.059,
            -wheel_vel[:, 1] * 0.059,
        ),
        dim=1,
    )
    wheel_speed_sq = torch.mean(wheel_forward_speed**2, dim=1)
    _accumulate_command_curriculum_metric(env, "idle_wheel_speed", torch.sqrt(wheel_speed_sq), idle)

    straight = moving & (torch.abs(cmd[:, 1]) < 0.20)
    straight_slip = torch.mean(
        (wheel_forward_speed - lin_vel[:, 0].unsqueeze(1)) ** 2,
        dim=1,
    ) / (0.45**2)
    _accumulate_command_curriculum_metric(env, "slip_penalty", straight_slip, straight)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        moving = cmd_mag >= 0.2
        env.extras["log"].update(
            {
                "Locomotion/cmd_vx_mean": _masked_mean(cmd[:, 0], moving),
                "Locomotion/base_vx_mean": _masked_mean(lin_vel[:, 0], moving),
                "Locomotion/base_vx_error_abs": _masked_mean(torch.abs(error_x), moving),
                "Locomotion/tracking_lin_vel_reward": _masked_mean(reward, moving),
                "Locomotion/tracking_upright_gate": _masked_mean(gate, locomotion),
            }
        )

    return reward


def tracking_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    sigma_cmd_scale: float = 0.0,
    ratio_blend: float = 0.0,
    use_upright_gate: bool = True,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """偏航角速度跟踪,高 yaw 命令下保留大误差学习梯度。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_z = robot.data.root_link_ang_vel_b[:, 2]
    error = ang_vel_z - cmd[:, 1]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _tracking_upright_gate(pg_z, tracking_upright_full_cos)
    cmd_mag = torch.abs(cmd[:, 1])
    effective_sigma = float(sigma) * (1.0 + float(sigma_cmd_scale) * cmd_mag)
    reward_gate = gate if use_upright_gate else torch.ones_like(gate)
    exp_reward = torch.exp(-(error**2) / effective_sigma) * reward_gate
    reward = exp_reward
    if ratio_blend > 0.0:
        blend = min(max(float(ratio_blend), 0.0), 1.0)
        # exp 核在大误差下梯度很快消失,比例项只负责把策略拉向正确方向。
        ratio_denom = torch.clamp(cmd_mag, min=1.0)
        ratio_reward = torch.clamp(1.0 - torch.abs(error) / ratio_denom, 0.0, 1.0) * reward_gate
        reward = (1.0 - blend) * exp_reward + blend * ratio_reward
    jump_flag = (
        cmd[:, 5] > 0.5
        if cmd.shape[1] > 5
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    moving = (torch.abs(cmd[:, 1]) >= 0.2) & ~jump_flag
    _accumulate_command_curriculum_metric(env, "yaw_score", reward, moving)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        env.extras["log"].update(
            {
                "Locomotion/cmd_yaw_rate_mean": _masked_mean(cmd[:, 1], moving),
                "Locomotion/base_yaw_rate_mean": _masked_mean(ang_vel_z, moving),
                "Locomotion/base_yaw_error_abs": _masked_mean(torch.abs(error), moving),
                "Locomotion/tracking_ang_vel_reward": _masked_mean(reward, moving),
                "Locomotion/tracking_ang_vel_exp_reward": _masked_mean(exp_reward, moving),
                "Locomotion/tracking_ang_vel_sigma": _masked_mean(effective_sigma, moving),
                "Locomotion/tracking_upright_gate": _masked_mean(gate, ~jump_flag),
            }
        )

    return reward


def tracking_lin_yaw_joint(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_lin_cmd: float = 0.2,
    min_yaw_cmd: float = 0.5,
    lin_error_scale: float = 0.75,
    yaw_error_scale: float = 2.0,
    use_upright_gate: bool = True,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """线速度和 yaw 同时非零时的联合跟踪奖励,防止二者互相让路。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    lin_vel_x = robot.data.root_link_lin_vel_b[:, 0]
    yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _tracking_upright_gate(pg_z, tracking_upright_full_cos)

    lin_cmd = cmd[:, 0]
    yaw_cmd = cmd[:, 1]
    jump_flag = (
        cmd[:, 5] > 0.5
        if cmd.shape[1] > 5
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    combined = (
        (torch.abs(lin_cmd) >= float(min_lin_cmd))
        & (torch.abs(yaw_cmd) >= float(min_yaw_cmd))
        & ~jump_flag
    )

    lin_denom = torch.clamp(torch.abs(lin_cmd), min=float(lin_error_scale))
    yaw_denom = torch.clamp(torch.abs(yaw_cmd), min=float(yaw_error_scale))
    lin_score = torch.clamp(1.0 - torch.abs(lin_vel_x - lin_cmd) / lin_denom, 0.0, 1.0)
    yaw_score = torch.clamp(1.0 - torch.abs(yaw_vel - yaw_cmd) / yaw_denom, 0.0, 1.0)
    reward_gate = gate if use_upright_gate else torch.ones_like(gate)
    reward = lin_score * yaw_score * reward_gate * combined.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        env.extras["log"].update(
            {
                "Locomotion/lin_yaw_joint_reward": _masked_mean(reward, combined),
                "Locomotion/lin_yaw_joint_lin_score": _masked_mean(lin_score, combined),
                "Locomotion/lin_yaw_joint_yaw_score": _masked_mean(yaw_score, combined),
                "Locomotion/lin_yaw_joint_cmd_ratio": combined.float().mean().item(),
                "Locomotion/tracking_upright_gate": _masked_mean(gate, ~jump_flag),
            }
        )

    return reward


def tracking_orientation_l2(
    env: ManagerBasedRlEnv, command_name: str, ignore_recovery: bool = False
) -> torch.Tensor:
    """pitch/roll 姿态 L2 惩罚，提供 Tron 风格的持续回正梯度。

    L2 惩罚会随误差平方增长，小倾斜也有明确负反馈。行走任务保留 pitch/roll
    指令语义，惩罚相对目标姿态的误差；跳跃任务中 pitch/roll 指令固定为 0，
    因此等价于 flat orientation L2。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg = robot.data.projected_gravity_b

    current_pitch = torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0))
    current_roll = torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0))

    pitch_error = current_pitch - cmd[:, 2]
    roll_error = current_roll - cmd[:, 3]
    penalty = pitch_error**2 + roll_error**2
    if ignore_recovery:
        penalty = penalty * (~_recovery_reset_mask(env)).float()
    return penalty


def recovery_upright_orientation_l2(
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
    """接近直立后惩罚 pitch/roll 分轴误差，抑制转弯时的横滚侧倾。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg = robot.data.projected_gravity_b

    gate = _near_upright_gate(
        pg[:, 2],
        gate_start_deg=float(gate_start_deg),
        gate_full_deg=float(gate_full_deg),
    )
    jump_flag = (
        cmd[:, 5] > 0.5
        if cmd.shape[1] > 5
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    active = (gate > 0.0) & ~jump_flag

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

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        yaw_abs = torch.abs(cmd[:, 1])
        turning = active & (yaw_abs >= 0.2)
        straight = active & (yaw_abs < 0.2)
        yaw_low = active & (yaw_abs < 1.0)
        yaw_mid = active & (yaw_abs >= 1.0) & (yaw_abs < 3.0)
        yaw_high = active & (yaw_abs >= 3.0)
        roll_abs_deg = torch.rad2deg(torch.abs(roll_error))
        pitch_abs_deg = torch.rad2deg(torch.abs(pitch_error))
        env.extras["log"].update(
            {
                "Recovery/diag_upright_orientation_penalty": _masked_mean(result, active),
                "Recovery/diag_upright_orientation_gate": _masked_mean(gate, active),
                "Recovery/diag_abs_roll_deg_turning": _masked_mean(roll_abs_deg, turning),
                "Recovery/diag_abs_roll_deg_straight": _masked_mean(roll_abs_deg, straight),
                "Recovery/diag_abs_roll_deg_by_yaw_cmd/low": _masked_mean(roll_abs_deg, yaw_low),
                "Recovery/diag_abs_roll_deg_by_yaw_cmd/mid": _masked_mean(roll_abs_deg, yaw_mid),
                "Recovery/diag_abs_roll_deg_by_yaw_cmd/high": _masked_mean(roll_abs_deg, yaw_high),
                "Recovery/diag_abs_pitch_deg_turning": _masked_mean(pitch_abs_deg, turning),
            }
        )

    return result


def tracking_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    height_sensor_name: str,
    ignore_recovery: bool = False,
    kernel: str = "exp",
    use_upright_gate: bool = False,
    min_upright_gate: float = 0.0,
    use_pose_end_gate: bool = False,
    upright_gate_angle_deg: float = 30.0,
    inverted_gate_angle_deg: float = 150.0,
) -> torch.Tensor:
    """高度跟踪项，支持 exp 奖励或 L2 误差惩罚。"""
    cmd = env.command_manager.get_command(command_name)

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    target_height = cmd[:, 4]
    error_sq = torch.square(height - target_height)
    if kernel == "exp":
        reward = torch.exp(-error_sq / sigma)
    elif kernel == "l2":
        reward = error_sq
    else:
        raise ValueError(f"未知高度跟踪核函数: {kernel}")

    robot = None
    if use_upright_gate or use_pose_end_gate:
        robot = env.scene["robot"]

    if use_upright_gate and robot is not None:
        gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
        gate = torch.clamp(gate, min=float(min_upright_gate))
        reward = reward * gate
        if (
            hasattr(env, "extras")
            and isinstance(env.extras.get("log"), dict)
            and _should_log_step(env)
        ):
            env.extras["log"]["Locomotion/height_gate"] = gate.mean().item()
    if use_pose_end_gate and robot is not None:
        pg_z = robot.data.projected_gravity_b[:, 2]
        upright_gate = torch.clamp(-pg_z, 0.0, 1.0)
        inverted_gate = torch.clamp(pg_z, 0.0, 1.0)
        gate = torch.maximum(upright_gate, inverted_gate)
        reward = reward * gate
        if (
            hasattr(env, "extras")
            and isinstance(env.extras.get("log"), dict)
            and _should_log_step(env)
        ):
            env.extras["log"].update(
                {
                    "Locomotion/height_pose_end_gate": gate.mean().item(),
                    "Locomotion/height_upright_end_gate": upright_gate.mean().item(),
                    "Locomotion/height_inverted_end_gate": inverted_gate.mean().item(),
                }
            )
    if ignore_recovery:
        reward = reward * (~_recovery_reset_mask(env)).float()
    return reward


def bad_tilt(
    env: ManagerBasedRlEnv,
    soft_limit_deg: float = 12.0,
    hard_limit_deg: float = 35.0,
    max_penalty: float = 4.0,
    ignore_recovery: bool = False,
) -> torch.Tensor:
    """坏姿态 barrier 惩罚。

    小倾斜由 tracking_orientation_l2 处理；超过 soft_limit 后快速加重惩罚，
    避免策略把明显歪斜当成可接受状态。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    soft = torch.deg2rad(torch.tensor(float(soft_limit_deg), device=env.device))
    hard = torch.deg2rad(torch.tensor(float(hard_limit_deg), device=env.device))
    span = torch.clamp(hard - soft, min=1.0e-6)
    excess = torch.clamp((tilt - soft) / span, min=0.0)
    penalty = torch.clamp(excess**2, max=float(max_penalty))
    if ignore_recovery:
        penalty = penalty * (~_recovery_reset_mask(env)).float()
    return penalty


def is_alive(env: ManagerBasedRlEnv, recovery_scale: float = 1.0) -> torch.Tensor:
    """存活奖励；倒地恢复样本可单独缩放，避免躺平刷 alive。"""
    reward = torch.ones(env.num_envs, device=env.device)
    if recovery_scale != 1.0:
        reward = torch.where(
            _recovery_reset_mask(env),
            reward * float(recovery_scale),
            reward,
        )
    return reward


def lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座 z 方向速度的平方,直立门控。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return robot.data.root_link_lin_vel_b[:, 2] ** 2 * gate


def ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    """横滚/俯仰角速度平方和,直立门控。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    ang_vel = robot.data.root_link_ang_vel_b
    return (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2) * gate


def angular_momentum(env: ManagerBasedRlEnv) -> torch.Tensor:
    """全身角动量范数平方,直立门控。

    使用 MuJoCo subtree_angmom[root_body] 获取整个机器人子树的角动量。
    对两轮倒立摆尤为重要——腿部激进动作产生的角动量脉冲会直接导致失稳。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    root_body_id = robot.data.indexing.root_body_id
    angmom = env.sim.data.subtree_angmom[:, root_body_id]
    return torch.sum(angmom**2, dim=-1) * gate


def leg_torques(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    recovery_scale: float | None = None,
) -> torch.Tensor:
    """腿部主动杆电机力矩平方和。"""
    robot = env.scene[asset_cfg.name]
    torques, _ = _policy_leg_torque_and_vel(env, robot)
    penalty = torch.sum(torques**2, dim=1)
    if recovery_scale is not None:
        penalty = torch.where(_recovery_reset_mask(env), penalty * float(recovery_scale), penalty)
    return penalty


def wheel_torques(
    env: ManagerBasedRlEnv,
    max_torque: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """轮子执行器力矩超出额定值的平方和。

    max_torque: 轮子电机额定最大力矩 (N·m)。
    """
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force[:, wheel_actuator_ids(robot)]
    excess = torch.clamp(torch.abs(torques) - max_torque, min=0.0)
    return torch.sum(excess**2, dim=1)


def leg_dof_acc(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部主动杆加速度平方和(排除轮子)。"""
    robot = env.scene[asset_cfg.name]
    acc = _policy_leg_acc(env, robot)
    return torch.sum(acc**2, dim=1)


def leg_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    recovery_scale: float | None = None,
) -> torch.Tensor:
    """腿部主动杆电机 |力矩 * 速度| 之和。"""
    robot = env.scene[asset_cfg.name]
    torques, vel = _policy_leg_torque_and_vel(env, robot)
    penalty = torch.sum(torch.abs(torques * vel), dim=1)
    if recovery_scale is not None:
        penalty = torch.where(_recovery_reset_mask(env), penalty * float(recovery_scale), penalty)
    return penalty


def action_rate(env: ManagerBasedRlEnv, recovery_scale: float | None = None) -> torch.Tensor:
    """当前动作与上一动作差值的平方和。"""
    action = env.action_manager.action
    penalty = torch.sum((action - env.action_manager.prev_action) ** 2, dim=1)
    if recovery_scale is not None:
        penalty = torch.where(_recovery_reset_mask(env), penalty * float(recovery_scale), penalty)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        action_abs = torch.abs(action)
        leg_action_abs = action_abs[:, :4]
        wheel_action_abs = action_abs[:, 4:6]
        max_abs_action = torch.max(action_abs, dim=1).values
        max_abs_leg_action = torch.max(leg_action_abs, dim=1).values
        max_abs_wheel_action = torch.max(wheel_action_abs, dim=1).values
        saturation_threshold = 0.95
        env.extras["log"].update(
            {
                "Locomotion/max_abs_action": max_abs_action.mean().item(),
                "Locomotion/max_abs_leg_action": max_abs_leg_action.mean().item(),
                "Locomotion/max_abs_wheel_action": max_abs_wheel_action.mean().item(),
                "Locomotion/raw_action_saturation_rate": (max_abs_action > saturation_threshold)
                .float()
                .mean()
                .item(),
                "Locomotion/leg_action_saturation_rate": (max_abs_leg_action > saturation_threshold)
                .float()
                .mean()
                .item(),
                "Locomotion/wheel_action_saturation_rate": (
                    max_abs_wheel_action > saturation_threshold
                )
                .float()
                .mean()
                .item(),
            }
        )
    return penalty


def _action_rate_slice(
    env: ManagerBasedRlEnv,
    start: int,
    stop: int,
    recovery_scale: float | None = None,
) -> torch.Tensor:
    """指定动作维度的一阶变化平方和。"""
    action = env.action_manager.action[:, start:stop]
    prev_action = env.action_manager.prev_action[:, start:stop]
    penalty = torch.sum((action - prev_action) ** 2, dim=1)
    if recovery_scale is not None:
        penalty = torch.where(_recovery_reset_mask(env), penalty * float(recovery_scale), penalty)
    return penalty


def leg_action_rate(env: ManagerBasedRlEnv, recovery_scale: float | None = None) -> torch.Tensor:
    """腿部动作一阶变化平方和。"""
    penalty = _action_rate_slice(env, 0, 4, recovery_scale=recovery_scale)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        env.extras["log"]["Recovery/diag_leg_action_rate"] = penalty.mean().item()
    return penalty


def wheel_action_rate(env: ManagerBasedRlEnv, recovery_scale: float | None = None) -> torch.Tensor:
    """轮子动作一阶变化平方和。"""
    penalty = _action_rate_slice(env, 4, 6, recovery_scale=recovery_scale)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        env.extras["log"]["Recovery/diag_wheel_action_rate"] = penalty.mean().item()
    return penalty


def action_smoothness(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    gate_start_deg: float = 60.0,
    gate_full_deg: float = 30.0,
    max_penalty: float = 80.0,
    leg_scale: float = 1.0,
    wheel_scale: float = 1.0,
) -> torch.Tensor:
    """惩罚动作二阶差分，压制接近直立后的高频来回抖动。"""
    action = env.action_manager.action
    prev = getattr(env, _ACTION_SMOOTH_PREV_ATTR, None)
    prev_prev = getattr(env, _ACTION_SMOOTH_PREV_PREV_ATTR, None)

    if not isinstance(prev, torch.Tensor) or prev.shape != action.shape:
        setattr(env, _ACTION_SMOOTH_PREV_ATTR, action.detach().clone())
        setattr(env, _ACTION_SMOOTH_PREV_PREV_ATTR, action.detach().clone())
        return torch.zeros(env.num_envs, device=env.device)
    if not isinstance(prev_prev, torch.Tensor) or prev_prev.shape != action.shape:
        setattr(env, _ACTION_SMOOTH_PREV_PREV_ATTR, prev.detach().clone())
        setattr(env, _ACTION_SMOOTH_PREV_ATTR, action.detach().clone())
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = action - 2.0 * prev + prev_prev
    setattr(env, _ACTION_SMOOTH_PREV_PREV_ATTR, prev.detach().clone())
    setattr(env, _ACTION_SMOOTH_PREV_ATTR, action.detach().clone())

    leg_penalty = torch.sum(action_acc[:, :4] ** 2, dim=1)
    wheel_penalty = torch.sum(action_acc[:, 4:6] ** 2, dim=1)
    penalty = float(leg_scale) * leg_penalty + float(wheel_scale) * wheel_penalty
    penalty = torch.clamp(penalty, max=float(max_penalty))

    robot = env.scene["robot"]
    gate = _near_upright_gate(
        robot.data.projected_gravity_b[:, 2],
        gate_start_deg=float(gate_start_deg),
        gate_full_deg=float(gate_full_deg),
    )
    active = gate > 0.0
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        if cmd.shape[1] > 5:
            active = active & ~(cmd[:, 5] > 0.5)

    startup = env.episode_length_buf < 3
    gated_penalty = torch.where(startup | ~active, torch.zeros_like(penalty), penalty * gate)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        log = env.extras["log"]
        log.update(
            {
                "Recovery/diag_action_smoothness_raw": _masked_mean(penalty, active),
                "Recovery/diag_action_smoothness": _masked_mean(gated_penalty, active),
                "Recovery/diag_leg_action_smoothness": _masked_mean(leg_penalty, active),
                "Recovery/diag_wheel_action_smoothness": _masked_mean(wheel_penalty, active),
                "Recovery/diag_action_smoothness_gate": _masked_mean(gate, active),
            }
        )

    return gated_penalty


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    ignore_recovery: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """站立时惩罚腿部偏离高度条件默认姿态。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    _ = default_height, height_tolerance
    joint_pos, default_pos = _policy_leg_pos_and_height_default(env, robot, command_name)
    diff = joint_pos - default_pos
    reward = torch.sum(diff**2, dim=1)

    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    vel_scale = (cmd_norm <= command_threshold).float()

    result = reward * vel_scale * gate
    if ignore_recovery:
        result = result * (~_recovery_reset_mask(env)).float()
    return result


def joint_pos_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """按 RobotLab 语义惩罚腿部关节偏离默认姿态，静止命令下惩罚更强。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    joint_pos, default_pos = _policy_leg_pos_and_height_default(env, robot, command_name)
    joint_error = torch.linalg.norm(joint_pos - default_pos, dim=1)
    command_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    body_vel = torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
    moving = (command_norm > float(command_threshold)) | (body_vel > float(velocity_threshold))
    scale = torch.where(
        moving,
        torch.ones_like(joint_error),
        torch.full_like(joint_error, float(stand_still_scale)),
    )
    penalty = joint_error * scale * gate

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "Locomotion/joint_pos_penalty": _masked_mean(penalty, active),
                "Locomotion/joint_pos_penalty_moving_rate": _masked_mean(moving.float(), active),
            }
        )

    return penalty


def recovery_diagnostics(
    env: ManagerBasedRlEnv,
    command_name: str,
    base_height_sensor_name: str,
    wheel_sensor_name: str,
    leg_contact_sensor_name: str,
    collision_sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    force_threshold: float = 1.0,
    contact_force_threshold: float = 35.0,
    action_saturation_threshold: float = 0.95,
    active_rod_margin_warning: float = 0.05,
    log_interval_steps: int = 1,
    core_log_interval_steps: int = 1,
) -> torch.Tensor:
    """记录 recovery one-policy 诊断量，返回 0 以避免改变奖励语义。"""
    zero = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "extras"):
        return zero

    step = int(getattr(env, "common_step_counter", 0))
    heavy_interval = max(1, int(log_interval_steps))
    core_interval = max(1, int(core_log_interval_steps))
    heavy_due = heavy_interval <= 1 or (step - 1) % heavy_interval == 0
    core_due = core_interval <= 1 or (step - 1) % core_interval == 0
    if not heavy_due and not core_due:
        return zero

    log = env.extras.setdefault("log", {})
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    pg = robot.data.projected_gravity_b
    pg_z = pg[:, 2]
    tilt_deg = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    roll_deg = torch.rad2deg(torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0)))
    pitch_deg = torch.rad2deg(torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0)))
    upright_gate = _upright_factor(pg_z)
    upright_15 = tilt_deg < 15.0
    upright_30 = tilt_deg < 30.0
    side_region = (tilt_deg >= 60.0) & (tilt_deg <= 120.0)
    inverted_130 = tilt_deg > 130.0
    inverted_150 = tilt_deg > 150.0

    base_lin_vel = robot.data.root_link_lin_vel_b
    base_ang_vel = robot.data.root_link_ang_vel_b
    base_vxy = torch.linalg.norm(base_lin_vel[:, :2], dim=1)
    base_vz_abs = torch.abs(base_lin_vel[:, 2])
    base_ang_xy = torch.linalg.norm(base_ang_vel[:, :2], dim=1)
    base_ang_norm = torch.linalg.norm(base_ang_vel, dim=1)

    height_sensor: TerrainHeightSensor = env.scene[base_height_sensor_name]
    base_height = torch.nan_to_num(
        height_sensor.data.heights[:, 0],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    target_height = (
        cmd[:, 4]
        if cmd.shape[1] > 4
        else torch.full_like(base_height, _SHARED_ROBOT.default_base_height)
    )
    height_error = base_height - target_height
    height_abs_error = torch.abs(height_error)
    height_low_1cm = base_height < (target_height - 0.01)
    height_low_2cm = base_height < (target_height - 0.02)
    height_low_4cm = base_height < (target_height - 0.04)
    height_ok_1cm = height_abs_error < 0.01
    height_ok_2cm = height_abs_error < 0.02
    height_ok_5cm = height_abs_error < 0.05

    cmd_vx = cmd[:, 0]
    cmd_yaw = cmd[:, 1]
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    moving = torch.abs(cmd_vx) >= 0.2
    turning = torch.abs(cmd_yaw) >= 0.2
    standing = cmd_speed < 0.1

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_manager = getattr(env, "action_manager", None)
    action_term = None
    if action_manager is not None:
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            if getattr(term, "_entity", None) is robot:
                action_term = term
                break
    unclipped_action = getattr(action_term, "unclipped_action", None)
    if not isinstance(unclipped_action, torch.Tensor) or unclipped_action.shape != action.shape:
        unclipped_action = action
    action_abs = torch.abs(action)
    unclipped_action_abs = torch.abs(unclipped_action)
    action_delta = action - prev_action
    max_abs_action = torch.max(action_abs, dim=1).values
    leg_action_abs = action_abs[:, :4]
    wheel_action_abs = action_abs[:, 4:6]
    unclipped_leg_action_abs = unclipped_action_abs[:, :4]
    unclipped_wheel_action_abs = unclipped_action_abs[:, 4:6]
    max_abs_leg_action = torch.max(leg_action_abs, dim=1).values
    max_abs_wheel_action = torch.max(wheel_action_abs, dim=1).values
    max_abs_unclipped_action = torch.max(unclipped_action_abs, dim=1).values
    max_abs_unclipped_leg_action = torch.max(unclipped_leg_action_abs, dim=1).values
    max_abs_unclipped_wheel_action = torch.max(unclipped_wheel_action_abs, dim=1).values
    action_saturated = max_abs_action > float(action_saturation_threshold)
    leg_action_saturated = max_abs_leg_action > float(action_saturation_threshold)
    wheel_action_saturated = max_abs_wheel_action > float(action_saturation_threshold)
    unclipped_action_saturated = max_abs_unclipped_action > float(action_saturation_threshold)
    unclipped_leg_action_saturated = max_abs_unclipped_leg_action > float(
        action_saturation_threshold
    )
    unclipped_wheel_action_saturated = max_abs_unclipped_wheel_action > float(
        action_saturation_threshold
    )

    active_rod_angle = _active_rod_angles(robot)
    active_target_clamp_rate = 0.0
    active_target_clamp_env = torch.zeros(env.num_envs, device=env.device)
    active_target_mean = 0.0
    active_target_error_env = torch.zeros(env.num_envs, device=env.device)
    active_target_error_mean = 0.0
    front_target_abs_mean = 0.0
    if action_term is not None:
        active_target = getattr(action_term, "active_rod_angle_target", None)
        active_clamped = getattr(action_term, "active_rod_angle_target_clamped", None)
        policy_target = getattr(action_term, "policy_leg_target", None)
        if isinstance(active_target, torch.Tensor):
            active_target_mean = active_target.float().mean().item()
            active_target_error_env = torch.abs(active_target - active_rod_angle).mean(dim=1)
            active_target_error_mean = active_target_error_env.mean().item()
        if isinstance(active_clamped, torch.Tensor):
            active_target_clamp_rate = active_clamped.float().mean().item()
            active_target_clamp_env = active_clamped.float().mean(dim=1).to(device=env.device)
        if isinstance(policy_target, torch.Tensor) and policy_target.shape[-1] >= 3:
            front_target_abs_mean = torch.abs(policy_target[:, (0, 2)]).mean().item()

    if core_due and not heavy_due:
        log.update(
            {
                "Recovery/diag_cmd_vx_mean": cmd_vx.mean().item(),
                "Recovery/diag_cmd_vx_abs_mean": torch.abs(cmd_vx).mean().item(),
                "Recovery/diag_cmd_yaw_rate_mean": cmd_yaw.mean().item(),
                "Recovery/diag_cmd_yaw_rate_abs_mean": torch.abs(cmd_yaw).mean().item(),
                "Recovery/diag_cmd_speed_mean": cmd_speed.mean().item(),
                "Recovery/diag_cmd_moving_ratio": moving.float().mean().item(),
                "Recovery/diag_cmd_turning_ratio": turning.float().mean().item(),
                "Recovery/diag_cmd_standing_ratio": standing.float().mean().item(),
                "Recovery/diag_cmd_height_m": target_height.mean().item(),
                "Recovery/diag_command_height_mean_m": target_height.mean().item(),
                "Recovery/diag_command_height_min_m": target_height.min().item(),
                "Recovery/diag_command_height_max_m": target_height.max().item(),
                "Recovery/diag_tilt_deg_mean": tilt_deg.mean().item(),
                "Recovery/diag_tilt_deg_max": tilt_deg.max().item(),
                "Recovery/diag_upright_15deg_rate": upright_15.float().mean().item(),
                "Recovery/diag_upright_30deg_rate": upright_30.float().mean().item(),
                "Recovery/diag_base_height_m": base_height.mean().item(),
                "Recovery/diag_height_error_signed_m": height_error.mean().item(),
                "Recovery/diag_height_error_abs_m": height_abs_error.mean().item(),
                "Recovery/diag_height_ok_2cm_rate": height_ok_2cm.float().mean().item(),
                "Recovery/diag_height_low_2cm_rate": height_low_2cm.float().mean().item(),
                "Recovery/diag_max_abs_action": max_abs_action.mean().item(),
                "Recovery/diag_max_abs_leg_action": max_abs_leg_action.mean().item(),
                "Recovery/diag_max_abs_wheel_action": max_abs_wheel_action.mean().item(),
                "Recovery/diag_raw_action_saturation_rate": action_saturated.float().mean().item(),
                "Recovery/diag_leg_action_saturation_rate": leg_action_saturated.float()
                .mean()
                .item(),
                "Recovery/diag_wheel_action_saturation_rate": wheel_action_saturated.float()
                .mean()
                .item(),
                "Recovery/diag_unclipped_raw_action_saturation_rate": unclipped_action_saturated.float()
                .mean()
                .item(),
                "Recovery/diag_unclipped_leg_action_saturation_rate": unclipped_leg_action_saturated.float()
                .mean()
                .item(),
                "Recovery/diag_unclipped_wheel_action_saturation_rate": unclipped_wheel_action_saturated.float()
                .mean()
                .item(),
                "Recovery/diag_action_delta_norm": torch.linalg.norm(action_delta, dim=1)
                .mean()
                .item(),
                "Recovery/diag_active_target_mean_rad": active_target_mean,
                "Recovery/diag_active_target_error_abs_rad": active_target_error_mean,
                "Recovery/diag_active_target_clamp_rate": active_target_clamp_rate,
            }
        )

    if not heavy_due:
        return zero

    joint_pos, fixed_default_pos = _policy_leg_pos_and_default(robot)
    _, default_pos = _policy_leg_pos_and_height_default(env, robot, command_name)
    fixed_joint_error_norm = torch.linalg.norm(joint_pos - fixed_default_pos, dim=1)
    joint_error = joint_pos - default_pos
    joint_error_norm = torch.linalg.norm(joint_error, dim=1)
    default_active_angles = []
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = _SHARED_ROBOT.active_rod_angle_coeffs[side_idx]
        default_active_angles.append(
            front_coef * default_pos[:, front_idx] + back_coef * default_pos[:, back_idx]
        )
    default_active_angle = torch.stack(default_active_angles, dim=1)
    joint_vel_norm = torch.linalg.norm(_policy_leg_vel(robot), dim=1)
    min_margin, lower_margin, upper_margin = _active_rod_angle_margins(robot)
    near_active_rod_limit = min_margin < float(active_rod_margin_warning)
    lower_limit_close = lower_margin < float(active_rod_margin_warning)
    upper_limit_close = upper_margin < float(active_rod_margin_warning)
    limit_penalty = dof_pos_limits(env, asset_cfg=asset_cfg)
    wheel_clearance_min, wheel_clearance_mean = _wheel_clearance_stats(env, robot, asset_cfg)

    wheel_contact_ratio, wheel_max_force, wheel_mean_force = _contact_diagnostic_stats(
        env, wheel_sensor_name, force_threshold
    )
    leg_contact_ratio, leg_max_force, _ = _contact_diagnostic_stats(
        env, leg_contact_sensor_name, force_threshold
    )
    collision_ratio, collision_max_force, _ = _contact_diagnostic_stats(
        env, collision_sensor_name, force_threshold
    )
    wheel_force_excess = torch.clamp(wheel_max_force - float(contact_force_threshold), min=0.0)

    log.update(
        {
            "Recovery/diag_cmd_vx_mean": cmd_vx.mean().item(),
            "Recovery/diag_cmd_vx_abs_mean": torch.abs(cmd_vx).mean().item(),
            "Recovery/diag_cmd_vx_std": cmd_vx.float().std(unbiased=False).item(),
            "Recovery/diag_cmd_yaw_rate_mean": cmd_yaw.mean().item(),
            "Recovery/diag_cmd_yaw_rate_abs_mean": torch.abs(cmd_yaw).mean().item(),
            "Recovery/diag_cmd_yaw_rate_std": cmd_yaw.float().std(unbiased=False).item(),
            "Recovery/diag_cmd_speed_mean": cmd_speed.mean().item(),
            "Recovery/diag_cmd_moving_ratio": moving.float().mean().item(),
            "Recovery/diag_cmd_turning_ratio": turning.float().mean().item(),
            "Recovery/diag_cmd_standing_ratio": standing.float().mean().item(),
            "Recovery/diag_cmd_height_m": target_height.mean().item(),
            "Recovery/diag_command_height_mean_m": target_height.mean().item(),
            "Recovery/diag_command_height_std_m": target_height.float().std(unbiased=False).item(),
            "Recovery/diag_command_height_min_m": target_height.min().item(),
            "Recovery/diag_command_height_max_m": target_height.max().item(),
            "Recovery/diag_tilt_deg_mean": tilt_deg.mean().item(),
            "Recovery/diag_tilt_deg_max": tilt_deg.max().item(),
            "Recovery/diag_abs_roll_deg": torch.abs(roll_deg).mean().item(),
            "Recovery/diag_abs_pitch_deg": torch.abs(pitch_deg).mean().item(),
            "Recovery/diag_roll_deg_mean": roll_deg.mean().item(),
            "Recovery/diag_pitch_deg_mean": pitch_deg.mean().item(),
            "Recovery/diag_upright_gate": upright_gate.mean().item(),
            "Recovery/diag_upright_15deg_rate": upright_15.float().mean().item(),
            "Recovery/diag_upright_30deg_rate": upright_30.float().mean().item(),
            "Recovery/diag_side_region_rate": side_region.float().mean().item(),
            "Recovery/diag_inverted_130deg_rate": inverted_130.float().mean().item(),
            "Recovery/diag_inverted_150deg_rate": inverted_150.float().mean().item(),
            "Recovery/diag_base_height_m": base_height.mean().item(),
            "Recovery/diag_base_height_min_m": base_height.min().item(),
            "Recovery/diag_base_height_max_m": base_height.max().item(),
            "Recovery/diag_height_error_signed_m": height_error.mean().item(),
            "Recovery/diag_height_error_abs_m": height_abs_error.mean().item(),
            "Recovery/diag_height_ok_1cm_rate": height_ok_1cm.float().mean().item(),
            "Recovery/diag_height_ok_2cm_rate": height_ok_2cm.float().mean().item(),
            "Recovery/diag_height_ok_5cm_rate": height_ok_5cm.float().mean().item(),
            "Recovery/diag_height_low_1cm_rate": height_low_1cm.float().mean().item(),
            "Recovery/diag_height_low_2cm_rate": height_low_2cm.float().mean().item(),
            "Recovery/diag_height_low_4cm_rate": height_low_4cm.float().mean().item(),
            "Recovery/diag_upright_low_height_rate": (upright_15 & height_low_2cm)
            .float()
            .mean()
            .item(),
            "Recovery/diag_near_upright_height_error_abs_m": _masked_mean(
                height_abs_error, upright_30
            ),
            "Recovery/diag_inverted_height_error_abs_m": _masked_mean(
                height_abs_error, inverted_130
            ),
            "Recovery/diag_base_vxy_speed": base_vxy.mean().item(),
            "Recovery/diag_base_vz_abs": base_vz_abs.mean().item(),
            "Recovery/diag_base_ang_vel_xy_norm": base_ang_xy.mean().item(),
            "Recovery/diag_base_ang_vel_norm": base_ang_norm.mean().item(),
            "Recovery/diag_near_upright_ang_vel_xy_norm": _masked_mean(base_ang_xy, upright_30),
            "Recovery/diag_near_upright_vxy_speed": _masked_mean(base_vxy, upright_30),
            "Recovery/diag_max_abs_action": max_abs_action.mean().item(),
            "Recovery/diag_max_abs_leg_action": max_abs_leg_action.mean().item(),
            "Recovery/diag_max_abs_wheel_action": max_abs_wheel_action.mean().item(),
            "Recovery/diag_unclipped_max_abs_action": max_abs_unclipped_action.mean().item(),
            "Recovery/diag_unclipped_max_abs_leg_action": max_abs_unclipped_leg_action.mean().item(),
            "Recovery/diag_unclipped_max_abs_wheel_action": max_abs_unclipped_wheel_action.mean().item(),
            "Recovery/diag_leg_action_abs_mean": leg_action_abs.mean().item(),
            "Recovery/diag_wheel_action_abs_mean": wheel_action_abs.mean().item(),
            "Recovery/diag_unclipped_leg_action_abs_mean": unclipped_leg_action_abs.mean().item(),
            "Recovery/diag_unclipped_wheel_action_abs_mean": unclipped_wheel_action_abs.mean().item(),
            "Recovery/diag_raw_action_saturation_rate": action_saturated.float().mean().item(),
            "Recovery/diag_leg_action_saturation_rate": leg_action_saturated.float().mean().item(),
            "Recovery/diag_wheel_action_saturation_rate": wheel_action_saturated.float()
            .mean()
            .item(),
            "Recovery/diag_unclipped_raw_action_saturation_rate": unclipped_action_saturated.float()
            .mean()
            .item(),
            "Recovery/diag_unclipped_leg_action_saturation_rate": unclipped_leg_action_saturated.float()
            .mean()
            .item(),
            "Recovery/diag_unclipped_wheel_action_saturation_rate": unclipped_wheel_action_saturated.float()
            .mean()
            .item(),
            "Recovery/diag_upright_action_saturation_rate": _masked_mean(
                action_saturated.float(), upright_15
            ),
            "Recovery/diag_action_delta_norm": torch.linalg.norm(action_delta, dim=1).mean().item(),
            "Recovery/diag_joint_error_norm_rad": joint_error_norm.mean().item(),
            "Recovery/diag_fixed_default_joint_error_norm_rad": fixed_joint_error_norm.mean().item(),
            "Recovery/diag_dynamic_default_joint_error_rad": joint_error_norm.mean().item(),
            "Recovery/diag_default_active_angle_mean_rad": default_active_angle.mean().item(),
            "Recovery/diag_default_lf0_mean_rad": default_pos[:, 0].mean().item(),
            "Recovery/diag_active_target_mean_rad": active_target_mean,
            "Recovery/diag_active_target_error_abs_rad": active_target_error_mean,
            "Recovery/diag_active_target_clamp_rate": active_target_clamp_rate,
            "Recovery/diag_front_target_abs_mean_rad": front_target_abs_mean,
            "Recovery/diag_near_upright_joint_error_norm_rad": _masked_mean(
                joint_error_norm, upright_30
            ),
            "Recovery/diag_joint_vel_norm": joint_vel_norm.mean().item(),
            "Recovery/diag_active_rod_angle_mean_rad": active_rod_angle.mean().item(),
            "Recovery/diag_active_rod_angle_std_rad": active_rod_angle.float()
            .std(unbiased=False)
            .item(),
            "Recovery/diag_active_rod_margin_rad": min_margin.mean().item(),
            "Recovery/diag_active_rod_margin_min_rad": min_margin.min().item(),
            "Recovery/diag_active_rod_lower_margin_rad": lower_margin.mean().item(),
            "Recovery/diag_active_rod_upper_margin_rad": upper_margin.mean().item(),
            "Recovery/diag_active_rod_near_limit_rate": near_active_rod_limit.float().mean().item(),
            "Recovery/diag_active_rod_near_lower_rate": lower_limit_close.float().mean().item(),
            "Recovery/diag_active_rod_near_upper_rate": upper_limit_close.float().mean().item(),
            "Recovery/diag_dof_pos_limits_raw": limit_penalty.mean().item(),
            "Recovery/diag_dof_pos_limits_active_rate": (limit_penalty > 0.0).float().mean().item(),
            "Recovery/diag_wheel_clearance_min_m": wheel_clearance_min.min().item(),
            "Recovery/diag_wheel_clearance_mean_m": wheel_clearance_mean.mean().item(),
            "Recovery/diag_wheel_penetration_rate": (wheel_clearance_min < -0.001)
            .float()
            .mean()
            .item(),
            "Recovery/diag_wheel_contact_ratio": wheel_contact_ratio.mean().item(),
            "Recovery/diag_wheel_full_contact_rate": (wheel_contact_ratio >= 1.0)
            .float()
            .mean()
            .item(),
            "Recovery/diag_wheel_any_contact_rate": (wheel_contact_ratio > 0.0)
            .float()
            .mean()
            .item(),
            "Recovery/diag_wheel_max_force_n": wheel_max_force.mean().item(),
            "Recovery/diag_wheel_mean_force_n": wheel_mean_force.mean().item(),
            "Recovery/diag_wheel_force_excess_n": wheel_force_excess.mean().item(),
            "Recovery/diag_leg_contact_rate": (leg_contact_ratio > 0.0).float().mean().item(),
            "Recovery/diag_leg_contact_ratio": leg_contact_ratio.mean().item(),
            "Recovery/diag_leg_max_force_n": leg_max_force.mean().item(),
            "Recovery/diag_collision_contact_rate": (collision_ratio > 0.0).float().mean().item(),
            "Recovery/diag_collision_contact_ratio": collision_ratio.mean().item(),
            "Recovery/diag_collision_max_force_n": collision_max_force.mean().item(),
            "Recovery/diag_onepolicy_ready_2cm_rate": (
                upright_15 & height_ok_2cm & (wheel_contact_ratio >= 1.0)
            )
            .float()
            .mean()
            .item(),
        }
    )

    for action_idx, action_name in enumerate(
        ("lf0", "left_active", "rf0", "right_active", "left_wheel", "right_wheel")
    ):
        if action_idx >= action_abs.shape[1]:
            continue
        dim_saturated = action_abs[:, action_idx] > float(action_saturation_threshold)
        log[f"Recovery/diag_action_abs_{action_name}"] = action_abs[:, action_idx].mean().item()
        log[f"Recovery/diag_action_saturation_rate_{action_name}"] = (
            dim_saturated.float().mean().item()
        )

    for joint_idx, joint_name in enumerate(("lf", "lb", "rf", "rb")):
        if joint_idx >= joint_error.shape[1]:
            continue
        log[f"Recovery/diag_joint_error_abs_{joint_name}_rad"] = (
            torch.abs(joint_error[:, joint_idx]).mean().item()
        )

    reset_bins = getattr(env, "_reset_init_tilt_bin", None)
    if isinstance(reset_bins, torch.Tensor) and reset_bins.shape[0] == env.num_envs:
        for bin_idx, bin_name in enumerate(("upright_noise", "near_fall", "hard_tilt", "inverted")):
            bin_mask = reset_bins == bin_idx
            log[f"Recovery/diag_tilt_deg_by_reset_bin/{bin_name}"] = _masked_mean(
                tilt_deg, bin_mask
            )
            log[f"Recovery/diag_upright_15deg_rate_by_reset_bin/{bin_name}"] = _masked_mean(
                upright_15.float(), bin_mask
            )
            log[f"Recovery/diag_height_error_abs_m_by_reset_bin/{bin_name}"] = _masked_mean(
                height_abs_error, bin_mask
            )
            log[f"Recovery/diag_action_saturation_rate_by_reset_bin/{bin_name}"] = _masked_mean(
                action_saturated.float(), bin_mask
            )
            log[f"Recovery/diag_active_rod_margin_min_rad_by_reset_bin/{bin_name}"] = _masked_min(
                min_margin, bin_mask
            )

    reset_pose_bins = getattr(env, "_reset_pose_bin", None)
    if isinstance(reset_pose_bins, torch.Tensor) and reset_pose_bins.shape[0] == env.num_envs:
        for bin_idx, bin_name in enumerate(("mixed_full", "pitch_inverted", "roll_side")):
            bin_mask = reset_pose_bins == bin_idx
            log[f"Recovery/diag_sample_rate_by_reset_pose/{bin_name}"] = (
                bin_mask.float().mean().item()
            )
            log[f"Recovery/diag_tilt_deg_by_reset_pose/{bin_name}"] = _masked_mean(
                tilt_deg, bin_mask
            )
            log[f"Recovery/diag_upright_15deg_rate_by_reset_pose/{bin_name}"] = _masked_mean(
                upright_15.float(), bin_mask
            )
            log[f"Recovery/diag_upright_30deg_rate_by_reset_pose/{bin_name}"] = _masked_mean(
                upright_30.float(), bin_mask
            )
            log[f"Recovery/diag_height_error_abs_m_by_reset_pose/{bin_name}"] = _masked_mean(
                height_abs_error, bin_mask
            )
            log[f"Recovery/diag_action_saturation_rate_by_reset_pose/{bin_name}"] = _masked_mean(
                action_saturated.float(), bin_mask
            )
            log[f"Recovery/diag_leg_action_saturation_rate_by_reset_pose/{bin_name}"] = (
                _masked_mean(leg_action_saturated.float(), bin_mask)
            )
            log[f"Recovery/diag_wheel_action_saturation_rate_by_reset_pose/{bin_name}"] = (
                _masked_mean(wheel_action_saturated.float(), bin_mask)
            )
            log[f"Recovery/diag_active_rod_margin_min_rad_by_reset_pose/{bin_name}"] = _masked_min(
                min_margin, bin_mask
            )

    for height_name, height_lo, height_hi in (
        ("cmd_20_22cm", 0.20, 0.22),
        ("cmd_22_24cm", 0.22, 0.24),
        ("cmd_24_28cm", 0.24, 0.28),
        ("cmd_28_32cm", 0.28, 0.32),
        ("cmd_32_36cm", 0.32, 0.36),
        ("cmd_36_38cm", 0.36, 0.38),
    ):
        height_mask = (target_height >= float(height_lo)) & (target_height < float(height_hi))
        log[f"Recovery/diag_sample_rate_by_cmd_height/{height_name}"] = (
            height_mask.float().mean().item()
        )
        log[f"Recovery/diag_upright_15deg_rate_by_cmd_height/{height_name}"] = _masked_mean(
            upright_15.float(), height_mask
        )
        log[f"Recovery/diag_upright_30deg_rate_by_cmd_height/{height_name}"] = _masked_mean(
            upright_30.float(), height_mask
        )
        log[f"Recovery/diag_joint_error_norm_by_cmd_height/{height_name}"] = _masked_mean(
            joint_error_norm, height_mask
        )
        log[f"Recovery/diag_fixed_joint_error_norm_by_cmd_height/{height_name}"] = _masked_mean(
            fixed_joint_error_norm, height_mask
        )
        log[f"Recovery/diag_active_target_error_abs_by_cmd_height/{height_name}"] = _masked_mean(
            active_target_error_env, height_mask
        )
        log[f"Recovery/diag_active_target_clamp_rate_by_cmd_height/{height_name}"] = _masked_mean(
            active_target_clamp_env, height_mask
        )
        log[f"Recovery/diag_height_error_abs_by_cmd_height/{height_name}"] = _masked_mean(
            height_abs_error, height_mask
        )
        log[f"Recovery/diag_height_error_signed_by_cmd_height/{height_name}"] = _masked_mean(
            height_error, height_mask
        )
        log[f"Recovery/diag_raw_action_saturation_rate_by_cmd_height/{height_name}"] = _masked_mean(
            action_saturated.float(), height_mask
        )
        log[f"Recovery/diag_leg_action_saturation_rate_by_cmd_height/{height_name}"] = _masked_mean(
            leg_action_saturated.float(), height_mask
        )
        log[f"Recovery/diag_wheel_action_saturation_rate_by_cmd_height/{height_name}"] = (
            _masked_mean(wheel_action_saturated.float(), height_mask)
        )
        log[f"Recovery/diag_active_rod_margin_min_by_cmd_height/{height_name}"] = _masked_min(
            min_margin, height_mask
        )
        log[f"Recovery/diag_wheel_contact_rate_by_cmd_height/{height_name}"] = _masked_mean(
            (wheel_contact_ratio >= 1.0).float(), height_mask
        )
        log[f"Recovery/diag_leg_contact_rate_by_cmd_height/{height_name}"] = _masked_mean(
            (leg_contact_ratio > 0.0).float(), height_mask
        )
        log[f"Recovery/diag_wheel_clearance_min_by_cmd_height/{height_name}"] = _masked_min(
            wheel_clearance_min, height_mask
        )

    return zero


def flat_leg_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """平地行走腿部触地惩罚，不使用直立门控。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = (
        cmd[:, 5] > 0.5
        if cmd.shape[1] > 5
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    active = ~jump_flag

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    has_contact = (force_mag > float(force_threshold)).any(dim=1)
    _accumulate_command_curriculum_metric(env, "leg_contact", has_contact.float(), active)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["Locomotion/flat_leg_contact_rate"] = _masked_mean(
            has_contact.float(), active
        )

    return has_contact.float() * active.float()


def flat_wheel_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """平地行走轮子离地惩罚，不使用直立门控。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = (
        cmd[:, 5] > 0.5
        if cmd.shape[1] > 5
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    active = ~jump_flag

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(force_threshold)
    contact_ratio = in_contact.float().mean(dim=1)
    penalty = 1.0 - contact_ratio

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Locomotion/flat_wheel_contact_ratio": _masked_mean(contact_ratio, active),
                "Locomotion/flat_wheel_full_contact_rate": _masked_mean(
                    (contact_ratio >= 1.0).float(), active
                ),
            }
        )

    return penalty * active.float()


def wheel_air_velocity_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    velocity_scale: float = 1.0,
    max_penalty: float = 10000.0,
    recovery_active_only: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    log_prefix: str = "Recovery",
) -> torch.Tensor:
    """Penalize wheel joint speed only while the corresponding wheel is airborne."""
    robot = env.scene[asset_cfg.name]
    wheel_vel = torch.nan_to_num(
        robot.data.joint_vel[:, wheel_joint_ids(robot)],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    if force_mag.ndim == 1:
        force_mag = force_mag.unsqueeze(1)
    elif force_mag.ndim > 2:
        force_mag = force_mag.flatten(start_dim=2).amax(dim=2)

    if force_mag.shape[1] == 1 and wheel_vel.shape[1] > 1:
        force_mag = force_mag.expand(-1, wheel_vel.shape[1])
    elif force_mag.shape[1] != wheel_vel.shape[1]:
        cols = min(force_mag.shape[1], wheel_vel.shape[1])
        force_mag = force_mag[:, :cols]
        wheel_vel = wheel_vel[:, :cols]

    in_contact = force_mag > float(force_threshold)
    air_mask = (~in_contact).float()
    scale = max(float(velocity_scale), 1.0e-6)
    penalty_per_wheel = air_mask * (wheel_vel / scale) ** 2
    penalty = torch.sum(penalty_per_wheel, dim=1)
    penalty = torch.clamp(penalty, max=float(max_penalty))

    if recovery_active_only:
        active = _recovery_reset_mask(env)
    else:
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    result = penalty * active.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        log_name = log_prefix.rstrip("/")
        air_ratio = air_mask.mean(dim=1)
        air_vel_abs = torch.mean(torch.abs(wheel_vel) * air_mask, dim=1)
        env.extras["log"].update(
            {
                f"{log_name}/wheel_air_velocity_penalty": _masked_mean(result, active),
                f"{log_name}/wheel_air_ratio": _masked_mean(air_ratio, active),
                f"{log_name}/wheel_air_joint_vel_abs": _masked_mean(air_vel_abs, active),
            }
        )

    return result


def upright_leg_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    min_upright_gate: float = 0.5,
) -> torch.Tensor:
    """接近直立后惩罚腿部触地，防止用小腿/连杆替代轮子支撑。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    robot = env.scene["robot"]
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    active = (~jump_flag) & (gate >= float(min_upright_gate))

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    has_contact = (force_mag > float(force_threshold)).any(dim=1)
    penalty = has_contact.float() * gate

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        env.extras["log"].update(
            {
                "Locomotion/upright_leg_contact_rate": _masked_mean(has_contact.float(), active),
                "Locomotion/upright_leg_contact_gate": _masked_mean(gate, active),
            }
        )

    return penalty * active.float()


def leg_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """腿部触地惩罚，不使用直立门控。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    has_contact = (force_mag > float(force_threshold)).any(dim=1)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        env.extras["log"]["Recovery/diag_leg_contact_penalty_rate"] = (
            has_contact.float().mean().item()
        )

    return has_contact.float()


def upright_wheel_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
    min_upright_gate: float = 0.5,
) -> torch.Tensor:
    """接近直立后惩罚轮子离地，要求平地支撑主要发生在两个轮子上。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    robot = env.scene["robot"]
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    active = (~jump_flag) & (gate >= float(min_upright_gate))

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(force_threshold)
    contact_ratio = in_contact.float().mean(dim=1)
    penalty = (1.0 - contact_ratio) * gate

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Locomotion/upright_wheel_contact_ratio": _masked_mean(contact_ratio, active),
                "Locomotion/upright_wheel_full_contact_rate": _masked_mean(
                    (contact_ratio >= 1.0).float(), active
                ),
            }
        )

    return penalty * active.float()


def upright_wheel_slip_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.059,
    idle_command_threshold: float = 0.08,
    straight_yaw_threshold: float = 0.20,
    min_upright_gate: float = 0.5,
    idle_wheel_speed_scale: float = 0.35,
    slip_speed_scale: float = 0.45,
    base_speed_scale: float = 0.20,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """接近直立后惩罚轮子空转和直行滑移，堵住轮子离地高速转的漏洞。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    vx_cmd = cmd[:, 0]
    yaw_cmd = cmd[:, 1]

    robot = env.scene["robot"]
    gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
    active = (~jump_flag) & (gate >= float(min_upright_gate))

    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    base_vel_b = robot.data.root_link_lin_vel_b
    base_vx = base_vel_b[:, 0]
    base_vxy_sq = base_vel_b[:, 0] ** 2 + base_vel_b[:, 1] ** 2

    idle = (torch.abs(vx_cmd) < float(idle_command_threshold)) & (
        torch.abs(yaw_cmd) < float(idle_command_threshold)
    )
    straight = (~idle) & (torch.abs(yaw_cmd) < float(straight_yaw_threshold))

    wheel_speed_sq = torch.mean(wheel_forward_speed**2, dim=1)
    idle_penalty = wheel_speed_sq / (float(idle_wheel_speed_scale) ** 2) + base_vxy_sq / (
        float(base_speed_scale) ** 2
    )
    straight_slip = torch.mean(
        (wheel_forward_speed - base_vx.unsqueeze(1)) ** 2,
        dim=1,
    ) / (float(slip_speed_scale) ** 2)
    penalty = torch.where(
        idle,
        idle_penalty,
        torch.where(straight, straight_slip, torch.zeros_like(straight_slip)),
    )
    penalty = torch.clamp(penalty, max=float(max_penalty))

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Locomotion/upright_idle_wheel_speed": _masked_mean(
                    torch.sqrt(wheel_speed_sq), active & idle
                ),
                "Locomotion/upright_wheel_slip_penalty": _masked_mean(penalty, active),
                "Locomotion/upright_straight_slip_penalty": _masked_mean(
                    straight_slip, active & straight
                ),
            }
        )

    return penalty * gate * active.float()


def recovery_upright(
    env: ManagerBasedRlEnv,
    sensor_name: str | None = None,
    height_sensor_name: str | None = None,
    command_name: str | None = None,
    upright_angle_deg: float = 15.0,
    height_tolerance: float = 0.05,
    ang_vel_threshold: float = 1.5,
    force_threshold: float = 1.0,
    power: float = 2.0,
) -> torch.Tensor:
    """倒地恢复期直立奖励。

    使用 projected_gravity 的 z 分量构造连续信号：侧躺约 0.5，直立为 1。
    这让策略在大倾角时也能得到非零恢复梯度。
    """
    robot = env.scene["robot"]
    episode = _recovery_episode_mask(env)
    active = _recovery_reset_mask(env)
    pg_z = robot.data.projected_gravity_b[:, 2]
    upright = torch.clamp((-pg_z + 1.0) * 0.5, 0.0, 1.0)

    if hasattr(env, "extras"):
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
        cache_reset = recovery_state.ensure_bool_buffer(env, "_recovery_cache_reset_mask")
        log = {
            "Recovery/reset_ratio": episode.float().mean().item(),
            "Recovery/active_ratio": active.float().mean().item(),
            "Recovery/cache_reset_ratio": _masked_mean(cache_reset.float(), episode),
            "Recovery/tilt_deg": tilt[episode].mean().item() if episode.any() else 0.0,
            "Recovery/upright_score": upright[active].mean().item() if active.any() else 0.0,
            "Recovery/stage_step": float(getattr(env, "_recovery_stage_step", 0)),
            "Recovery/stage_prob": float(getattr(env, "_recovery_stage_prob", 0.0)),
            "Recovery/stage_fallen_pose_prob": float(
                getattr(env, "_recovery_stage_fallen_pose_prob", 0.0)
            ),
            "Recovery/stage_cache_prob": float(getattr(env, "_recovery_stage_cache_prob", 0.0)),
        }
        if sensor_name is not None and height_sensor_name is not None and command_name is not None:
            (
                success,
                wheel_contact,
                success_active,
                upright_ok,
                height_ok,
                stable_ok,
            ) = _recovery_success_components(
                env,
                sensor_name=sensor_name,
                height_sensor_name=height_sensor_name,
                command_name=command_name,
                upright_angle_deg=upright_angle_deg,
                height_tolerance=height_tolerance,
                ang_vel_threshold=ang_vel_threshold,
                force_threshold=force_threshold,
            )
            init_yaw_abs = torch.abs(_recovery_angle_buffer(env, "_recovery_init_yaw"))
            init_tilt = _recovery_angle_buffer(env, "_recovery_init_tilt")
            hard_tilt = _recovery_hard_tilt_mask(env)
            hard_tilt_episode = _recovery_hard_tilt_episode_mask(env)
            time_to_success = recovery_state.ensure_long_buffer(
                env, "_recovery_time_to_success_steps"
            )
            ever_completed = time_to_success >= 0
            upright_height = upright_ok & height_ok
            upright_height_stable = upright_height & stable_ok
            log.update(
                {
                    "Recovery/success_rate": (episode & success).float().mean().item(),
                    "Recovery/success_active_rate": _masked_mean(success.float(), success_active),
                    "Recovery/upright_cond_rate": _masked_mean(upright_ok.float(), success_active),
                    "Recovery/height_cond_rate": _masked_mean(height_ok.float(), success_active),
                    "Recovery/stable_cond_rate": _masked_mean(stable_ok.float(), success_active),
                    "Recovery/wheel_contact_cond_rate": _masked_mean(
                        wheel_contact.float(), success_active
                    ),
                    "Recovery/upright_height_rate": _masked_mean(
                        upright_height.float(), success_active
                    ),
                    "Recovery/success_without_contact_rate": _masked_mean(
                        upright_height_stable.float(), success_active
                    ),
                    "Recovery/wheel_contact_rate": (success_active & wheel_contact)
                    .float()
                    .mean()
                    .item(),
                    "Recovery/init_yaw_abs_deg": _masked_mean(torch.rad2deg(init_yaw_abs), episode),
                    "Recovery/init_tilt_deg": _masked_mean(torch.rad2deg(init_tilt), episode),
                    "Recovery/hard_tilt_ratio": hard_tilt.float().mean().item(),
                    "Recovery/hard_tilt_success_rate": _masked_mean(success.float(), hard_tilt),
                    "Recovery/hard_tilt_upright_cond_rate": _masked_mean(
                        upright_ok.float(), hard_tilt
                    ),
                    "Recovery/hard_tilt_height_cond_rate": _masked_mean(
                        height_ok.float(), hard_tilt
                    ),
                    "Recovery/hard_tilt_stable_cond_rate": _masked_mean(
                        stable_ok.float(), hard_tilt
                    ),
                    "Recovery/hard_tilt_wheel_contact_cond_rate": _masked_mean(
                        wheel_contact.float(), hard_tilt
                    ),
                    "Recovery/hard_tilt_success_without_contact_rate": _masked_mean(
                        upright_height_stable.float(), hard_tilt
                    ),
                    "Recovery/hard_tilt_episode_ratio": hard_tilt_episode.float().mean().item(),
                    "Recovery/hard_tilt_ever_completed_rate": _masked_mean(
                        ever_completed.float(), hard_tilt_episode
                    ),
                }
            )
        env.extras.setdefault("log", {}).update(log)

    return upright.pow(float(power)) * active.float()


def recovery_progress(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    upright_delta_scale: float = 0.05,
    height_delta_scale: float = 0.03,
    max_reward: float = 4.0,
) -> torch.Tensor:
    """奖励恢复过程中直立程度和高度的单步正向进展。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    upright = torch.clamp((-pg_z + 1.0) * 0.5, 0.0, 1.0)

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]

    prev_upright = recovery_state.ensure_float_buffer(env, "_recovery_prev_upright")
    prev_height = recovery_state.ensure_float_buffer(env, "_recovery_prev_height")

    first_step = active & (env.episode_length_buf <= 1)
    prev_upright[first_step] = upright[first_step]
    prev_height[first_step] = height[first_step]

    upright_gain = torch.clamp(upright - prev_upright, min=0.0) / max(
        float(upright_delta_scale), 1.0e-6
    )
    height_gain = torch.clamp(height - prev_height, min=0.0) / max(
        float(height_delta_scale), 1.0e-6
    )
    reward = torch.clamp(upright_gain + height_gain, max=float(max_reward)) * active.float()

    prev_upright[active] = upright[active].detach()
    prev_height[active] = height[active].detach()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/progress_reward": reward[active].mean().item() if active.any() else 0.0,
                "Recovery/upright_gain": upright_gain[active].mean().item()
                if active.any()
                else 0.0,
                "Recovery/height_gain": height_gain[active].mean().item() if active.any() else 0.0,
            }
        )
    return reward


def recovery_hard_tilt_upright(
    env: ManagerBasedRlEnv,
    power: float = 1.0,
) -> torch.Tensor:
    """奖励大倾角样本进入可站立半球，不区分倾倒轴向。"""
    hard_tilt = _recovery_hard_tilt_mask(env)
    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    upright_half = torch.clamp(-pg_z, 0.0, 1.0)
    reward = upright_half.pow(float(power)) * hard_tilt.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/hard_tilt_upright_reward": _masked_mean(reward, hard_tilt),
                "Recovery/hard_tilt_upright_half": _masked_mean(upright_half, hard_tilt),
            }
        )
    return reward


def recovery_hard_tilt_supported_upright(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    near_upright_angle_deg: float = 30.0,
    height_tolerance: float = 0.05,
    ang_vel_threshold: float = 1.5,
    force_threshold: float = 1.0,
    near_upright_bonus: float = 0.5,
) -> torch.Tensor:
    """奖励大倾角样本在已有支撑条件下继续回正。"""
    _, wheel_contact, _, near_upright, height_ok, stable_ok = _recovery_success_components(
        env,
        sensor_name=sensor_name,
        height_sensor_name=height_sensor_name,
        command_name=command_name,
        upright_angle_deg=near_upright_angle_deg,
        height_tolerance=height_tolerance,
        ang_vel_threshold=ang_vel_threshold,
        force_threshold=force_threshold,
    )
    hard_tilt = _recovery_hard_tilt_mask(env)
    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    upright_half = torch.clamp(-pg_z, 0.0, 1.0)
    supported = wheel_contact & height_ok & stable_ok
    milestone = supported & near_upright
    reward = upright_half * supported.float() + milestone.float() * float(near_upright_bonus)
    reward = reward * hard_tilt.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/hard_tilt_supported_upright_reward": _masked_mean(reward, hard_tilt),
                "Recovery/hard_tilt_support_cond_rate": _masked_mean(supported.float(), hard_tilt),
                "Recovery/hard_tilt_near_upright_milestone_rate": _masked_mean(
                    milestone.float(), hard_tilt
                ),
            }
        )
    return reward


def recovery_stable_bonus(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    upright_angle_deg: float = 15.0,
    height_tolerance: float = 0.05,
    ang_vel_threshold: float = 1.5,
    force_threshold: float = 1.0,
    stable_steps_required: int = 32,
    per_step_bonus: float = 0.1,
    completion_bonus: float = 1.0,
) -> torch.Tensor:
    """连续站稳后退出 recovery active 模式，并给一次完成奖励。"""
    success, _, active = _recovery_success_mask(
        env,
        sensor_name=sensor_name,
        height_sensor_name=height_sensor_name,
        command_name=command_name,
        upright_angle_deg=upright_angle_deg,
        height_tolerance=height_tolerance,
        ang_vel_threshold=ang_vel_threshold,
        force_threshold=force_threshold,
    )
    completed = recovery_state.deactivate_recovered(env, success, stable_steps_required)
    stable_steps = recovery_state.ensure_long_buffer(env, "_recovery_success_steps")
    time_to_success = recovery_state.ensure_long_buffer(env, "_recovery_time_to_success_steps")
    episode = _recovery_episode_mask(env)

    reward = (
        success.float() * float(per_step_bonus) + completed.float() * float(completion_bonus)
    ) * active.float()

    if hasattr(env, "extras"):
        valid_time = episode & (time_to_success >= 0)
        ever_completed = episode & (time_to_success >= 0)
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/stable_steps": _masked_mean(stable_steps.float(), episode),
                "Recovery/stable_success_rate": _masked_mean(success.float(), active),
                "Recovery/completed_rate": _masked_mean(completed.float(), episode),
                "Recovery/completed_rate_step": _masked_mean(completed.float(), episode),
                "Recovery/ever_completed_rate": _masked_mean(ever_completed.float(), episode),
                "Recovery/time_to_success_steps": _masked_mean(time_to_success.float(), valid_time),
            }
        )
    return reward


def recovery_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    sigma: float = 0.04,
    gate_start_deg: float = 45.0,
    gate_full_deg: float = 15.0,
    min_gate: float = 0.0,
) -> torch.Tensor:
    """倒地恢复期 base 高度奖励，目标高度沿用当前站立高度指令。"""
    active = _recovery_reset_mask(env)
    cmd = env.command_manager.get_command(command_name)
    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    target_height = cmd[:, 4]
    reward = torch.exp(-torch.square(height - target_height) / float(sigma))
    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = _smoothstep01((float(gate_start_deg) - tilt) / gate_span)
    near_upright_gate = torch.clamp(near_upright_gate, min=float(min_gate))

    if hasattr(env, "extras"):
        hard_tilt = _recovery_hard_tilt_mask(env)
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/base_height_error_m": torch.abs(height - target_height)[active]
                .mean()
                .item()
                if active.any()
                else 0.0,
                "Recovery/height_gate": near_upright_gate[active].mean().item()
                if active.any()
                else 0.0,
                "Recovery/hard_tilt_height_gate": _masked_mean(near_upright_gate, hard_tilt),
            }
        )

    return reward * near_upright_gate * active.float()


def recovery_inverted_low_height_penalty(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    height_floor: float = _SHARED_ROBOT.default_base_height,
    height_scale: float = 0.05,
    max_normalized_low_height: float = 3.0,
    tilt_start_deg: float = 140.0,
    tilt_full_deg: float = 170.0,
) -> torch.Tensor:
    """只在接近完全倒置时惩罚低 base 高度，帮助策略脱离 upward 低梯度区。"""
    active = _recovery_reset_mask(env)
    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0, posinf=0.0, neginf=0.0)

    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(tilt_full_deg) - float(tilt_start_deg), 1.0e-6)
    inverted_gate = _smoothstep01((tilt - float(tilt_start_deg)) / gate_span)

    low_height = torch.clamp(
        (float(height_floor) - height) / max(float(height_scale), 1.0e-6),
        min=0.0,
        max=float(max_normalized_low_height),
    )
    penalty = torch.square(low_height) * inverted_gate * active.float()

    if hasattr(env, "extras"):
        inverted_region = active & (inverted_gate > 0.0)
        low_height_active = inverted_region & (low_height > 0.0)
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/inverted_region_rate": _masked_mean(inverted_region.float(), active),
                "RecoveryStand/inverted_height_gate": _masked_mean(inverted_gate, active),
                "RecoveryStand/inverted_low_height_penalty": _masked_mean(penalty, active),
                "RecoveryStand/inverted_low_height_rate": _masked_mean(
                    low_height_active.float(), active
                ),
            }
        )

    return penalty


def recovery_wheel_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    gate_start_deg: float = 120.0,
    gate_full_deg: float = 45.0,
) -> torch.Tensor:
    """倒地恢复期奖励轮子重新成为主要接地点。"""
    active = _recovery_reset_mask(env)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    if data.force is None:
        wheel_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        force_mag = finite_contact_force_norm(data.force)
        wheel_contact = (force_mag > float(force_threshold)).any(dim=1)

    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

    if hasattr(env, "extras"):
        hard_tilt = _recovery_hard_tilt_mask(env)
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/wheel_contact_cond_rate": _masked_mean(wheel_contact.float(), active),
                "Recovery/wheel_contact_gate": near_upright_gate[active].mean().item()
                if active.any()
                else 0.0,
                "Recovery/hard_tilt_wheel_contact_gate": _masked_mean(near_upright_gate, hard_tilt),
            }
        )

    return wheel_contact.float() * active.float()


def _contact_bool(env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float) -> torch.Tensor:
    """读取接触传感器，返回每个 env 是否有接触。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force_mag = finite_contact_force_norm(data.force)
    if force_mag.ndim == 1:
        return force_mag > float(force_threshold)
    return (force_mag > float(force_threshold)).any(dim=1)


def recovery_stand_wheel_contact(
    env: ManagerBasedRlEnv,
    left_wheel_sensor_name: str,
    right_wheel_sensor_name: str,
    force_threshold: float = 1.0,
    gate_start_deg: float = 120.0,
    gate_full_deg: float = 45.0,
) -> torch.Tensor:
    """接近直立后奖励左右轮都重新成为接地点。"""
    active = _recovery_reset_mask(env)
    left_contact = _contact_bool(env, left_wheel_sensor_name, force_threshold)
    right_contact = _contact_bool(env, right_wheel_sensor_name, force_threshold)
    dual_contact = left_contact & right_contact

    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/dual_wheel_contact_rate": _masked_mean(dual_contact.float(), active),
                "RecoveryStand/wheel_contact_gate": _masked_mean(near_upright_gate, active),
            }
        )

    return dual_contact.float() * near_upright_gate * active.float()


def recovery_stand_nonwheel_clearance(
    env: ManagerBasedRlEnv,
    nonwheel_sensor_name: str,
    force_threshold: float = 1.0,
    gate_start_deg: float = 75.0,
    gate_full_deg: float = 15.0,
) -> torch.Tensor:
    """接近直立后奖励清除机身和腿部等非轮接触，避免低趴静止解。"""
    active = _recovery_reset_mask(env)
    nonwheel_contact = _contact_bool(env, nonwheel_sensor_name, force_threshold)
    nonwheel_clear = ~nonwheel_contact

    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)
    reward = nonwheel_clear.float() * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/nonwheel_clearance_reward": _masked_mean(reward, active),
                "RecoveryStand/nonwheel_clearance_gate": _masked_mean(near_upright_gate, active),
                "RecoveryStand/nonwheel_clear_rate": _masked_mean(nonwheel_clear.float(), active),
            }
        )

    return reward


def recovery_stand_stillness(
    env: ManagerBasedRlEnv,
    wheel_radius: float = 0.059,
    gate_start_deg: float = 75.0,
    gate_full_deg: float = 15.0,
    base_speed_scale: float = 0.35,
    wheel_speed_scale: float = 0.45,
) -> torch.Tensor:
    """接近直立后奖励低机身线速度和低轮速，避免翻正后继续滚走。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

    base_speed = torch.linalg.norm(robot.data.root_link_lin_vel_b, dim=1)
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    wheel_speed = torch.linalg.norm(wheel_forward_speed, dim=1)
    base_score = 1.0 / (1.0 + (base_speed / float(base_speed_scale)) ** 2)
    wheel_score = 1.0 / (1.0 + (wheel_speed / float(wheel_speed_scale)) ** 2)
    reward = 0.5 * (base_score + wheel_score) * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/stillness_reward": _masked_mean(reward, active),
                "RecoveryStand/stillness_gate": _masked_mean(near_upright_gate, active),
                "RecoveryStand/base_lin_vel_norm": _masked_mean(base_speed, active),
                "RecoveryStand/wheel_forward_speed": _masked_mean(wheel_speed, active),
            }
        )

    return reward


def recovery_stand_orientation_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    gate_start_deg: float = 60.0,
    gate_full_deg: float = 15.0,
    roll_scale_rad: float = 0.08,
    pitch_scale_rad: float = 0.12,
    roll_weight: float = 1.5,
    pitch_weight: float = 1.0,
    max_penalty: float = 6.0,
) -> torch.Tensor:
    """接近直立后惩罚 pitch/roll 分轴误差，避免总 tilt 合格但 roll 歪着站。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg = robot.data.projected_gravity_b

    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

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
    result = penalty * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/final_orientation_penalty": _masked_mean(result, active),
                "RecoveryStand/abs_roll_deg": _masked_mean(
                    torch.rad2deg(torch.abs(roll_error)), active
                ),
                "RecoveryStand/abs_pitch_deg": _masked_mean(
                    torch.rad2deg(torch.abs(pitch_error)), active
                ),
                "RecoveryStand/orientation_gate": _masked_mean(near_upright_gate, active),
            }
        )

    return result


def recovery_stand_zero_velocity_penalty(
    env: ManagerBasedRlEnv,
    wheel_radius: float = 0.059,
    gate_start_deg: float = 45.0,
    gate_full_deg: float = 15.0,
    base_speed_scale: float = 0.05,
    wheel_speed_scale: float = 0.05,
    base_ang_vel_scale: float = 0.5,
    action_saturation_threshold: float = 0.95,
    max_penalty: float = 8.0,
) -> torch.Tensor:
    """接近直立后强惩罚机体平动、角速度和轮速，并记录限位诊断。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

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

    if hasattr(env, "extras"):
        min_margin, lower_margin, upper_margin = _active_rod_angle_margins(robot)
        action = env.action_manager.action
        action_abs = torch.abs(action)
        leg_action_abs = action_abs[:, :4]
        wheel_action_abs = action_abs[:, 4:6]
        max_abs_action = torch.max(action_abs, dim=1).values
        max_abs_leg_action = torch.max(leg_action_abs, dim=1).values
        max_abs_wheel_action = torch.max(wheel_action_abs, dim=1).values
        saturated = max_abs_action > float(action_saturation_threshold)
        leg_saturated = max_abs_leg_action > float(action_saturation_threshold)
        wheel_saturated = max_abs_wheel_action > float(action_saturation_threshold)
        log_values = {
            "RecoveryStand/zero_velocity_penalty": _masked_mean(result, active),
            "RecoveryStand/base_vxy_speed": _masked_mean(torch.sqrt(base_speed_sq), active),
            "RecoveryStand/wheel_idle_speed": _masked_mean(torch.sqrt(wheel_speed_sq), active),
            "RecoveryStand/base_ang_vel_norm": _masked_mean(torch.sqrt(base_ang_vel_sq), active),
            "RecoveryStand/zero_velocity_gate": _masked_mean(near_upright_gate, active),
            "RecoveryStand/active_rod_margin_rad": _masked_mean(min_margin, active),
            "RecoveryStand/active_rod_lower_margin_rad": _masked_mean(lower_margin, active),
            "RecoveryStand/active_rod_upper_margin_rad": _masked_mean(upper_margin, active),
            "RecoveryStand/max_abs_action": _masked_mean(max_abs_action, active),
            "RecoveryStand/max_abs_leg_action": _masked_mean(max_abs_leg_action, active),
            "RecoveryStand/max_abs_wheel_action": _masked_mean(max_abs_wheel_action, active),
            "RecoveryStand/raw_action_saturation_rate": _masked_mean(saturated.float(), active),
            "RecoveryStand/leg_action_saturation_rate": _masked_mean(leg_saturated.float(), active),
            "RecoveryStand/wheel_action_saturation_rate": _masked_mean(
                wheel_saturated.float(), active
            ),
        }
        for action_idx, action_name in enumerate(
            ("lf", "lb", "rf", "rb", "left_wheel", "right_wheel")
        ):
            dim_saturated = action_abs[:, action_idx] > float(action_saturation_threshold)
            log_values[f"RecoveryStand/action_abs_{action_name}"] = _masked_mean(
                action_abs[:, action_idx], active
            )
            log_values[f"RecoveryStand/action_saturation_rate_{action_name}"] = _masked_mean(
                dim_saturated.float(), active
            )
        env.extras.setdefault("log", {}).update(log_values)

    return result


def recovery_stand_leg_alignment(
    env: ManagerBasedRlEnv,
    gate_start_deg: float = 75.0,
    gate_full_deg: float = 15.0,
    min_lateral_distance: float = 0.40,
    max_lateral_distance: float = 0.46,
    max_fore_aft_offset: float = 0.03,
    lateral_scale: float = 0.04,
    fore_aft_scale: float = 0.03,
    fore_aft_weight: float = 1.5,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """接近直立后惩罚左右轮前后错位，避免站成前后劈叉。"""
    active = _recovery_reset_mask(env)
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

    penalty, lateral_distance, fore_aft_offset, alignment_ok = wheel_alignment_penalty(
        env,
        min_lateral_distance=min_lateral_distance,
        max_lateral_distance=max_lateral_distance,
        max_fore_aft_offset=max_fore_aft_offset,
        lateral_scale=lateral_scale,
        fore_aft_scale=fore_aft_scale,
        fore_aft_weight=fore_aft_weight,
        max_penalty=max_penalty,
        asset_cfg=asset_cfg,
    )
    result = penalty * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/leg_alignment_penalty": _masked_mean(result, active),
                "RecoveryStand/wheel_lateral_distance_m": _masked_mean(lateral_distance, active),
                "RecoveryStand/wheel_fore_aft_offset_m": _masked_mean(fore_aft_offset, active),
                "RecoveryStand/wheel_alignment_ok_rate": _masked_mean(alignment_ok.float(), active),
                "RecoveryStand/leg_alignment_gate": _masked_mean(near_upright_gate, active),
            }
        )

    return result


def recovery_stand_default_joint_pos(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    gate_start_deg: float = 60.0,
    gate_full_deg: float = 15.0,
    max_penalty: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """接近直立后惩罚腿部主动关节偏离默认位置，避免恢复完成后停在怪异腿型。"""
    active = _recovery_reset_mask(env)
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    near_upright_gate = _near_upright_gate(pg_z, gate_start_deg, gate_full_deg)

    joint_pos, default_pos = _policy_leg_pos_and_height_default(env, robot, command_name)
    joint_error = joint_pos - default_pos
    penalty = torch.mean(joint_error**2, dim=1)
    result = torch.clamp(penalty, max=float(max_penalty)) * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/default_joint_pos_penalty": _masked_mean(result, active),
                "RecoveryStand/default_joint_pos_error_rad": _masked_mean(
                    torch.sqrt(torch.mean(joint_error**2, dim=1)), active
                ),
                "RecoveryStand/default_joint_pos_gate": _masked_mean(near_upright_gate, active),
            }
        )

    return result


def recovery_stand_joint_mirror(
    env: ManagerBasedRlEnv,
    gate_start_deg: float = 60.0,
    gate_full_deg: float = 15.0,
    hip_weight: float = 1.0,
    knee_weight: float = 1.0,
    max_penalty: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """接近直立后惩罚左右腿镜像误差，避免用非对称腿型换取短期站稳。"""
    active = _recovery_reset_mask(env)
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    near_upright_gate = _near_upright_gate(pg_z, gate_start_deg, gate_full_deg)

    hip_diff, knee_diff = _policy_leg_mirror_diffs(robot)
    penalty = float(hip_weight) * hip_diff**2 + float(knee_weight) * knee_diff**2
    normalizer = max(float(hip_weight) + float(knee_weight), 1.0e-6)
    penalty = penalty / normalizer
    result = torch.clamp(penalty, max=float(max_penalty)) * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/joint_mirror_penalty": _masked_mean(result, active),
                "RecoveryStand/hip_mirror_error_rad": _masked_mean(torch.abs(hip_diff), active),
                "RecoveryStand/knee_mirror_error_rad": _masked_mean(torch.abs(knee_diff), active),
                "RecoveryStand/joint_mirror_gate": _masked_mean(near_upright_gate, active),
            }
        )

    return result


def recovery_success_bonus(
    env: ManagerBasedRlEnv,
    left_wheel_sensor_name: str,
    right_wheel_sensor_name: str,
    nonwheel_sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    upright_angle_deg: float = 15.0,
    max_abs_roll_deg: float = 3.0,
    max_abs_pitch_deg: float = 5.0,
    height_tolerance: float = 0.05,
    ang_vel_threshold: float = 0.5,
    lin_vel_threshold: float = 0.05,
    wheel_speed_threshold: float = 0.05,
    wheel_radius: float = 0.059,
    force_threshold: float = 1.0,
    stable_steps_required: int = 50,
    min_episode_steps: int = 50,
    min_wheel_lateral_distance: float = 0.40,
    max_wheel_lateral_distance: float = 0.46,
    max_wheel_fore_aft_offset: float = 0.03,
    completion_bonus: float = 10.0,
) -> torch.Tensor:
    """成功窗口完成时给一次性奖励。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    upright_limit = torch.deg2rad(torch.tensor(float(upright_angle_deg), device=env.device))
    upright_ok = tilt < upright_limit
    current_pitch = torch.asin(torch.clamp(robot.data.projected_gravity_b[:, 0], -1.0, 1.0))
    current_roll = torch.asin(torch.clamp(-robot.data.projected_gravity_b[:, 1], -1.0, 1.0))
    roll_limit = torch.deg2rad(torch.tensor(float(max_abs_roll_deg), device=env.device))
    pitch_limit = torch.deg2rad(torch.tensor(float(max_abs_pitch_deg), device=env.device))
    roll_ok = torch.abs(current_roll) < roll_limit
    pitch_ok = torch.abs(current_pitch) < pitch_limit

    cmd = env.command_manager.get_command(command_name)
    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    height_ok = torch.abs(height - cmd[:, 4]) < float(height_tolerance)

    ang_vel_ok = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1) < float(ang_vel_threshold)
    lin_vel_ok = torch.linalg.norm(robot.data.root_link_lin_vel_b, dim=1) < float(lin_vel_threshold)
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    wheel_speed_ok = torch.linalg.norm(wheel_forward_speed, dim=1) < float(wheel_speed_threshold)
    dual_contact = _contact_bool(env, left_wheel_sensor_name, force_threshold) & _contact_bool(
        env, right_wheel_sensor_name, force_threshold
    )
    nonwheel_clear = ~_contact_bool(env, nonwheel_sensor_name, force_threshold)
    wheel_alignment, _, _ = wheel_alignment_ok(
        env,
        min_lateral_distance=min_wheel_lateral_distance,
        max_lateral_distance=max_wheel_lateral_distance,
        max_fore_aft_offset=max_wheel_fore_aft_offset,
    )

    success = (
        active
        & upright_ok
        & roll_ok
        & pitch_ok
        & height_ok
        & ang_vel_ok
        & lin_vel_ok
        & wheel_speed_ok
        & dual_contact
        & nonwheel_clear
        & wheel_alignment
    )
    completed = recovery_state.update_success_window(
        env,
        success,
        stable_steps_required=stable_steps_required,
        min_episode_steps=min_episode_steps,
    )
    reward = completed.float() * float(completion_bonus)

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/success_bonus": reward.mean().item(),
                "RecoveryStand/success_bonus_rate": completed.float().mean().item(),
            }
        )

    return reward


def joint_mirror(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """左右关节位置差的平均平方,直立门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    hip_diff, knee_diff = _policy_leg_mirror_diffs(robot)
    diff = torch.stack((hip_diff, knee_diff), dim=1)
    num_pairs = 2
    return torch.sum(diff**2, dim=1) / num_pairs * gate


def collision(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    use_recovery_gate: bool = True,
) -> torch.Tensor:
    """受惩罚的身体接触计数，可关闭恢复姿态门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z) if use_recovery_gate else torch.ones_like(pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    contact_count = (force_mag > 0.1).float().sum(dim=1)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict) and _should_log_step(env):
        has_contact = contact_count > 0.0
        active = gate > 0.0
        env.extras["log"].update(
            {
                "Locomotion/upright_base_contact_rate": _masked_mean(has_contact.float(), active),
                "Locomotion/upright_base_contact_count": _masked_mean(contact_count, active),
            }
        )

    return contact_count * gate


def contact_forces(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    use_recovery_gate: bool = True,
) -> torch.Tensor:
    """轮子接触力超过阈值的部分,除以 100 归一化，可关闭恢复姿态门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z) if use_recovery_gate else torch.ones_like(pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    excess = torch.clamp(force_mag - threshold, min=0.0) / 100.0
    return torch.sum(excess, dim=1) * gate


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    force_threshold: float,
    cmd_threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """静止时轮子接触,直立门控。"""
    cmd = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    velocity_command = torch.linalg.norm(cmd[:, :2], dim=1)
    stationary = velocity_command < cmd_threshold

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    has_contact = (force_mag > force_threshold).float()
    return torch.sum(has_contact, dim=1) * gate * stationary.float()


def dof_pos_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部软限位余量惩罚：闭链使用同侧两主动杆夹角。"""
    robot = env.scene[asset_cfg.name]
    if is_closedchain_model(robot):
        lower, upper = _SHARED_ROBOT.active_rod_soft_angle_limits
        penalties = []
        for front_id, back_id, front_coef, back_coef in active_rod_angle_terms(robot):
            angle = (
                front_coef * robot.data.joint_pos[:, front_id]
                + back_coef * robot.data.joint_pos[:, back_id]
            )
            penalties.append(-(angle - float(lower)).clip(max=0.0))
            penalties.append((angle - float(upper)).clip(min=0.0))
        return torch.stack(penalties, dim=1).sum(dim=1)

    if is_fourbar_surrogate_model(robot):
        lower, upper = _SHARED_ROBOT.active_rod_soft_angle_limits
        pos = output_to_policy_pos_torch(robot.data.joint_pos[:, policy_leg_joint_ids(robot)])
        penalties = []
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = _SHARED_ROBOT.active_rod_angle_coeffs[side_idx]
            angle = front_coef * pos[:, front_idx] + back_coef * pos[:, back_idx]
            penalties.append(-(angle - float(lower)).clip(max=0.0))
            penalties.append((angle - float(upper)).clip(min=0.0))
        return torch.stack(penalties, dim=1).sum(dim=1)

    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
        return torch.zeros(env.num_envs, device=env.device)

    leg_ids = policy_leg_joint_ids(robot)
    pos = robot.data.joint_pos[:, leg_ids]
    limits = soft_limits[:, leg_ids]

    out_of_limits = -(pos - limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (pos - limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
