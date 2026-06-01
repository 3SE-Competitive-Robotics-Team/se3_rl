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
from se3_train.mdp import recovery_state
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.joint_indices import (
    active_rod_angle_terms,
    is_closedchain_model,
    leg_actuator_ids,
    output_leg_mirror_diffs,
    policy_leg_joint_ids,
    wheel_actuator_ids,
    wheel_joint_ids,
)
from se3_train.mdp.leg_alignment import wheel_alignment_ok, wheel_alignment_penalty

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SHARED_ROBOT = SharedRobotConfig()


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


def _upward_score(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """全姿态直立分数：倒置为 0，侧躺为 2，直立为 4。"""
    return torch.clamp(2.0 * (1.0 - projected_gravity_z), min=0.0, max=4.0)


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

    if hasattr(env, "extras"):
        tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        upright_15 = tilt < torch.deg2rad(torch.tensor(15.0, device=env.device))
        log = env.extras.setdefault("log", {})
        log.update(
            {
                "Locomotion/upward": reward.mean().item(),
                "SelfRight/tilt_deg": torch.rad2deg(tilt).mean().item(),
                "SelfRight/upright_15deg_rate": upright_15.float().mean().item(),
                "Locomotion/upright_gate": _upright_factor(pg_z).mean().item(),
            }
        )

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

    if hasattr(env, "extras"):
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
    gate = _upright_factor(pg_z)

    cmd_mag = torch.abs(cmd[:, 0])
    sigma = torch.where(cmd_mag < 0.2, sigma_stand, sigma_move)
    return torch.exp(-(error_x**2 + vz_weight * vz**2) / sigma) * gate


def tracking_ang_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float) -> torch.Tensor:
    """偏航角速度跟踪的 exp(-error^2/sigma),直立门控。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_z = robot.data.root_link_ang_vel_b[:, 2]
    error = ang_vel_z - cmd[:, 1]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return torch.exp(-(error**2) / sigma) * gate


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


def tracking_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    height_sensor_name: str,
    ignore_recovery: bool = False,
    use_upright_gate: bool = False,
    min_upright_gate: float = 0.0,
) -> torch.Tensor:
    """高度跟踪奖励，可保留倒地阶段的最小高度梯度。"""
    cmd = env.command_manager.get_command(command_name)

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    target_height = cmd[:, 4]
    error = torch.square(height - target_height)
    reward = torch.exp(-error / sigma)
    if use_upright_gate:
        robot = env.scene["robot"]
        gate = _upright_factor(robot.data.projected_gravity_b[:, 2])
        gate = torch.clamp(gate, min=float(min_upright_gate))
        reward = reward * gate
        if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
            env.extras["log"]["Locomotion/height_gate"] = gate.mean().item()
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
    """腿部电机力矩平方和，按 actuator 名称排除气弹簧。"""
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force[:, leg_actuator_ids(robot)]
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
    """腿部关节加速度平方和(排除轮子)。"""
    robot = env.scene[asset_cfg.name]
    acc = robot.data.joint_acc
    return torch.sum(acc[:, policy_leg_joint_ids(robot)] ** 2, dim=1)


def leg_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    recovery_scale: float | None = None,
) -> torch.Tensor:
    """腿部关节 |力矩 * 速度| 之和。"""
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force[:, leg_actuator_ids(robot)]
    vel = robot.data.joint_vel[:, policy_leg_joint_ids(robot)]
    penalty = torch.sum(torch.abs(torques * vel), dim=1)
    if recovery_scale is not None:
        penalty = torch.where(_recovery_reset_mask(env), penalty * float(recovery_scale), penalty)
    return penalty


def action_rate(env: ManagerBasedRlEnv, recovery_scale: float | None = None) -> torch.Tensor:
    """当前动作与上一动作差值的平方和。"""
    penalty = torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)
    if recovery_scale is not None:
        penalty = torch.where(_recovery_reset_mask(env), penalty * float(recovery_scale), penalty)
    return penalty


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    ignore_recovery: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """站立时关节偏差平方和,高度自适应 sigma。

    当 cmd_height 偏离 default_height 时,惩罚自动放松,
    避免高度指令与姿态惩罚的结构性矛盾。
    衰减因子: exp(-height_tolerance * (cmd_h - default_h)²)
    """
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    leg_ids = policy_leg_joint_ids(robot)
    diff = robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
    reward = torch.sum(diff**2, dim=1)

    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    vel_scale = (cmd_norm <= command_threshold).float()

    height_deviation = cmd[:, 4] - default_height
    height_scale = torch.exp(-height_tolerance * height_deviation**2)

    result = reward * vel_scale * height_scale * gate
    if ignore_recovery:
        result = result * (~_recovery_reset_mask(env)).float()
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

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Locomotion/upright_leg_contact_rate": _masked_mean(has_contact.float(), active),
                "Locomotion/upright_leg_contact_gate": _masked_mean(gate, active),
            }
        )

    return penalty * active.float()


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
    height = sensor.data.heights[:, 0]
    target_height = cmd[:, 4]
    reward = torch.exp(-torch.square(height - target_height) / float(sigma))
    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)
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
    max_penalty: float = 8.0,
) -> torch.Tensor:
    """接近直立后强惩罚机体平动和轮速，避免零命令自起后继续溜车。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    gate_span = max(float(gate_start_deg) - float(gate_full_deg), 1.0e-6)
    near_upright_gate = torch.clamp((float(gate_start_deg) - tilt) / gate_span, 0.0, 1.0)

    base_vxy = robot.data.root_link_lin_vel_b[:, :2]
    base_speed_sq = torch.sum(base_vxy**2, dim=1)
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    wheel_speed_sq = torch.mean(wheel_forward_speed**2, dim=1)
    penalty = base_speed_sq / (float(base_speed_scale) ** 2) + wheel_speed_sq / (
        float(wheel_speed_scale) ** 2
    )
    result = torch.clamp(penalty, max=float(max_penalty)) * near_upright_gate * active.float()

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "RecoveryStand/zero_velocity_penalty": _masked_mean(result, active),
                "RecoveryStand/base_vxy_speed": _masked_mean(torch.sqrt(base_speed_sq), active),
                "RecoveryStand/wheel_idle_speed": _masked_mean(torch.sqrt(wheel_speed_sq), active),
                "RecoveryStand/zero_velocity_gate": _masked_mean(near_upright_gate, active),
            }
        )

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

    hip_diff, knee_diff = output_leg_mirror_diffs(robot, robot.data.joint_pos)
    diff = torch.stack((hip_diff, knee_diff), dim=1)
    num_pairs = 2
    return torch.sum(diff**2, dim=1) / num_pairs * gate


def collision(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """受惩罚的身体接触计数,恢复惩罚门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    contact_count = (force_mag > 0.1).float().sum(dim=1)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
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
) -> torch.Tensor:
    """轮子接触力超过阈值的部分,除以 100 归一化,恢复门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z)

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

    stationary = torch.abs(cmd[:, 0]) < cmd_threshold

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
    """腿部限位违规惩罚：闭链使用同侧两主动杆夹角。"""
    robot = env.scene[asset_cfg.name]
    if is_closedchain_model(robot):
        lower, upper = _SHARED_ROBOT.active_rod_angle_limits
        penalties = []
        for front_id, back_id, front_coef, back_coef in active_rod_angle_terms(robot):
            angle = (
                front_coef * robot.data.joint_pos[:, front_id]
                + back_coef * robot.data.joint_pos[:, back_id]
            )
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
