"""跳跃专属奖励函数。

分两个阶段使用:
- PreTrain 阶段:地面主动起跳 + 目标高度条件化 + 轻姿态/限位约束
- Fine-tune 阶段:参考轨迹 tracking + 真实离地高度 + 空中/着陆质量约束
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared import JointGroup
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.jump_commands import JumpCommandTerm, ideal_takeoff_vel

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_PRETRAIN_MAX_BASE_HEIGHT_ATTR = "_jump_pretrain_max_base_height"
_PRETRAIN_MAX_WHEEL_HEIGHT_ATTR = "_jump_pretrain_max_wheel_height"
_PRETRAIN_PREV_JUMP_FLAG_ATTR = "_jump_pretrain_prev_jump_flag"
_ACTION_SMOOTH_PREV_ATTR = "_jump_action_smooth_prev_action"
_ACTION_SMOOTH_PREV_PREV_ATTR = "_jump_action_smooth_prev_prev_action"


def _get_jump_term(env: ManagerBasedRlEnv, command_name: str) -> JumpCommandTerm:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, JumpCommandTerm), (
        f"指令 '{command_name}' 必须是 JumpCommandTerm, 实际为 {type(term)}"
    )
    return term


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """计算掩码内均值;无样本时返回 0。"""
    if mask.any():
        return float(value[mask].mean().item())
    return 0.0


def _wheel_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """缓存左右轮 body 在 entity 内的局部索引。"""
    attr_name = f"_jump_wheel_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body,实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _wheel_bottom_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """返回左右轮底部相对地面的最小高度。"""
    wheel_bottom_h = _wheel_bottom_heights(env, asset_cfg, wheel_radius)
    return torch.min(wheel_bottom_h, dim=1).values


def _wheel_bottom_heights(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """返回左右轮底部相对地面的高度。"""
    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    wheel_bottom_h = wheel_pos_w[:, :, 2] - ground_z - float(wheel_radius)
    return wheel_bottom_h


def _wheel_alignment_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    center_lead_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算左右轮前后对齐误差的共享几何核。"""
    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_xy = robot.data.body_link_pos_w[:, body_ids, :2]

    com_x = robot.data.root_link_pos_w[:, 0]
    vx = robot.data.root_link_lin_vel_w[:, 0]
    ideal_x = com_x + float(center_lead_gain) * vx

    dist_l = torch.abs(wheel_xy[:, 0, 0] - ideal_x)
    dist_r = torch.abs(wheel_xy[:, 1, 0] - ideal_x)
    mean_dist = (dist_l + dist_r) / 2.0
    return ideal_x, dist_l, dist_r, mean_dist


def _update_pretrain_max_heights(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """维护 PreTrain 单次跳跃窗口内的轮组和 base 最大高度。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    robot = env.scene[asset_cfg.name]
    base_h = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    wheel_h = _wheel_bottom_height(env, asset_cfg, wheel_radius)

    max_base = getattr(env, _PRETRAIN_MAX_BASE_HEIGHT_ATTR, None)
    max_wheel = getattr(env, _PRETRAIN_MAX_WHEEL_HEIGHT_ATTR, None)
    if not isinstance(max_base, torch.Tensor) or max_base.shape != jump_flag.shape:
        max_base = base_h.clone()
    if not isinstance(max_wheel, torch.Tensor) or max_wheel.shape != jump_flag.shape:
        max_wheel = wheel_h.clone()

    prev_jump = getattr(env, _PRETRAIN_PREV_JUMP_FLAG_ATTR, None)
    if not isinstance(prev_jump, torch.Tensor) or prev_jump.shape != jump_flag.shape:
        prev_jump = torch.zeros_like(jump_flag)
    new_jump = jump_flag & ~prev_jump

    max_base = torch.where(new_jump, base_h, max_base)
    max_wheel = torch.where(new_jump, wheel_h, max_wheel)
    max_base = torch.where(jump_flag, torch.maximum(max_base, base_h), base_h)
    max_wheel = torch.where(jump_flag, torch.maximum(max_wheel, wheel_h), wheel_h)

    setattr(env, _PRETRAIN_MAX_BASE_HEIGHT_ATTR, max_base.detach())
    setattr(env, _PRETRAIN_MAX_WHEEL_HEIGHT_ATTR, max_wheel.detach())
    setattr(env, _PRETRAIN_PREV_JUMP_FLAG_ATTR, jump_flag.detach())
    return jump_flag, max_base, max_wheel


def _fixed_time_mask(env: ManagerBasedRlEnv, term: JumpCommandTerm, start_s: float) -> torch.Tensor:
    """返回从指定秒数之后开始激活的固定时间窗掩码。"""
    control_dt = max(float(getattr(env, "physics_dt", 0.002)) * 5.0, 1.0e-4)
    start_step = max(0, round(float(start_s) / control_dt))
    return term.traj_step >= start_step


# ---------------------------------------------------------------------------
# 阶段1(PreTrain)奖励
# ---------------------------------------------------------------------------


def jump_pretrain_height_success(
    env: ManagerBasedRlEnv,
    command_name: str,
    base_height_offset: float = 0.26,
    wheel_radius: float = 0.059,
    relative_tolerance: float = 0.45,
    falloff_ratio: float = 0.25,
    score_start_s: float = 1.1,
    base_weight: float = 0.5,
    wheel_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """PreTrain 稀疏高度奖励。

    PreTrain 不做精确高度匹配，只判断发力结果是否进入宽容忍带。

    高度相对误差在 relative_tolerance 内视为及格并给满分;超出后按误差
    超出量平滑衰减。这样 0.2m/0.3m 目标不会被固定 0.1m 弱跳误判为及格,
    也不会在已经接近目标时继续逼策略做精确高度控制。
    """
    term = _get_jump_term(env, command_name)
    jump_flag, max_base_h, max_wheel_h = _update_pretrain_max_heights(
        env,
        command_name,
        wheel_radius,
        asset_cfg,
    )
    cmd = env.command_manager.get_command(command_name)
    h_target = cmd[:, 6]
    base_target = h_target + float(base_height_offset)
    active = jump_flag & _fixed_time_mask(env, term, score_start_s)

    rel_tol = float(relative_tolerance)
    falloff = max(float(falloff_ratio), 1.0e-6)
    wheel_error_ratio = torch.abs(max_wheel_h - h_target) / h_target.clamp_min(1.0e-3)
    base_error_ratio = torch.abs(max_base_h - base_target) / base_target.clamp_min(1.0e-3)
    wheel_excess = torch.clamp(wheel_error_ratio - rel_tol, min=0.0)
    base_excess = torch.clamp(base_error_ratio - rel_tol, min=0.0)
    wheel_score = torch.exp(-((wheel_excess / falloff) ** 2))
    base_score = torch.exp(-((base_excess / falloff) ** 2))
    reward = float(wheel_weight) * wheel_score + float(base_weight) * base_score

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_pretrain_max_wheel_height": _mean_on_mask(max_wheel_h, active),
                "Jump/diag_pretrain_max_base_height": _mean_on_mask(max_base_h, active),
                "Jump/diag_pretrain_wheel_height_error_ratio": _mean_on_mask(
                    wheel_error_ratio, active
                ),
                "Jump/diag_pretrain_height_pass_rate": _mean_on_mask(
                    (wheel_error_ratio <= rel_tol).float(), active
                ),
                "Jump/diag_pretrain_height_reward": _mean_on_mask(reward, active),
            }
        )

    return reward * active.float()


def jump_pretrain_wheel_clearance_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.059,
    relative_tolerance: float = 0.45,
    falloff_ratio: float = 0.25,
    score_start_s: float = 0.75,
    progress_power: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """PreTrain 轮底离地结果进度奖励,使用 ±relative_tolerance 宽窗口。

    该项只看跳跃窗口内左右轮底的最大离地高度。低于及格下界时按进度给连续
    信用;进入 ±45% 窗口后满分;超过上界后平滑衰减。这样 PreTrain 的轮高
    仍是宽容忍窗口语义,同时比稀疏成功奖励更早给到轮底离地结果信用。
    """
    term = _get_jump_term(env, command_name)
    jump_flag, _, max_wheel_h = _update_pretrain_max_heights(
        env,
        command_name,
        wheel_radius,
        asset_cfg,
    )
    cmd = env.command_manager.get_command(command_name)
    h_target = cmd[:, 6]
    rel_tol = float(relative_tolerance)
    lower = h_target * max(0.0, 1.0 - rel_tol)
    upper = h_target * (1.0 + rel_tol)
    active = jump_flag & _fixed_time_mask(env, term, score_start_s)

    below_score = torch.clamp(max_wheel_h / lower.clamp_min(1.0e-3), min=0.0, max=1.0)
    if float(progress_power) != 1.0:
        below_score = below_score ** float(progress_power)
    above_excess = torch.clamp(max_wheel_h - upper, min=0.0)
    falloff_m = (h_target * max(float(falloff_ratio), 1.0e-6)).clamp_min(1.0e-3)
    above_score = torch.exp(-((above_excess / falloff_m) ** 2))
    in_band = (max_wheel_h >= lower) & (max_wheel_h <= upper)
    score = torch.where(max_wheel_h < lower, below_score, torch.ones_like(below_score))
    score = torch.where(in_band, torch.ones_like(score), score)
    score = torch.where(max_wheel_h > upper, above_score, score)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_pretrain_wheel_clearance_progress": _mean_on_mask(score, active),
                "Jump/diag_pretrain_wheel_pass_lower": _mean_on_mask(lower, active),
                "Jump/diag_pretrain_wheel_pass_upper": _mean_on_mask(upper, active),
                "Jump/diag_pretrain_wheel_clearance_lower_margin": _mean_on_mask(
                    max_wheel_h - lower, active
                ),
            }
        )

    return score * active.float()


def jump_pre_takeoff_wheel_lift_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.059,
    clearance_threshold: float = 0.015,
    scale: float = 0.04,
    max_penalty: float = 9.0,
    max_ref_preload_vz: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """起跳前轮子提前离地惩罚。

    正常跳跃应先接地蓄力,再在起跳段离地。若 preload 阶段轮子先小幅抬起,
    策略会绕开发力,把"提前收腿"当成起跳准备动作。该项只在参考蓄力阶段激活,
    对超过阈值的轮底离地高度扣分。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    preload = term.reference_preload_active(max_ref_preload_vz)
    takeoff = term.reference_takeoff_active(max_ref_preload_vz)
    active = jump_flag & preload & (~takeoff)

    wheel_h = _wheel_bottom_height(env, asset_cfg, wheel_radius)
    excess = torch.clamp(wheel_h - float(clearance_threshold), min=0.0)
    penalty = torch.clamp((excess / float(scale)) ** 2, max=float(max_penalty))

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_pre_takeoff_wheel_lift_m": _mean_on_mask(wheel_h, active),
                "Jump/diag_pre_takeoff_wheel_lift_raw": _mean_on_mask(penalty, active),
            }
        )

    return penalty * active.float()


def jump_vel_encourage(
    env: ManagerBasedRlEnv,
    command_name: str,
    weight_scale: float = 1.0,
) -> torch.Tensor:
    """跳跃奖励:参考空中阶段奖励 vz,防止地面期 vz 套利。

    RSI 配套设计:
    - reset_root_state_full 给 jump_flag=1 的 episode 注入 vz 初速度
    - reset_joints 同时设置空中收腿姿态
    - 策略从 reference stage=1(airborne)开始,立刻拿到此奖励
    - 地面期 vz 无奖励(策略通过探索和 RSI 体验学会自主起跳)
    """

    robot = env.scene["robot"]
    vz_w = robot.data.root_link_lin_vel_w[:, 2]
    jump_flag = env.command_manager.get_command(command_name)[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_air = term.jump_stage == 1

    # reference 空中阶段奖励正向 vz,下降段不额外惩罚。
    active = jump_flag & in_air
    return torch.clamp(vz_w, min=0.0) * active.float() * weight_scale


def jump_leg_contact_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """跳跃窗口腿部触地惩罚。

    PreTrain 中腿部接触不再硬终止,避免早期探索被秒杀;但触地必须进入
    reward,防止策略把"深蹲到腿碰地"当成可行跳跃前置动作。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    contact_count = (force_mag > force_threshold).float().sum(dim=1)
    return contact_count * jump_flag.float()


def jump_takeoff_drive(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_ref_takeoff_vz: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """起跳驱动奖励:在参考蹬地期奖励质心向上速度,解决地面期梯度断链。

    【为什么用质心 vz 而不是腿部关节速度】
    原实现奖励关节速度负分量(伸腿方向),可被"原地抖腿"hack:
    抖腿时关节速度正负交替,负分量均值非零,策略无需真正起跳即可套利。
    质心 vz > 0 只有一种来源:身体整体向上运动(真实起跳)。
    原地震荡时 vz 均值为 0,无法套利。

    【激活条件】
    - jump_flag=1(有跳跃指令)
    - reference takeoff 子相位(grounded 且参考 vz 已进入上升段)
    - vz > 0(质心已经向上运动,奖励蹬腿有效果的那一帧)

    jump_stage 由 reference motion 时间推进,蹬地辅助项只覆盖上升蹬地段。
    """
    from se3_train.mdp.jump_commands import JumpCommandTerm

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    # reference 子相位门控:只在 grounded 内的上升蹬地段激活。
    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        phase_takeoff = term.reference_takeoff_active(min_ref_takeoff_vz)
    else:
        phase_takeoff = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # 质心向上速度:按目标起跳速度归一化,避免 0.05m/s 与 1.4m/s 获得近似声量。
    robot = env.scene[asset_cfg.name]
    vz_w = robot.data.root_link_lin_vel_w[:, 2]
    h_target = cmd[:, 6]
    vz_ref = ideal_takeoff_vel(h_target).clamp_min(0.1)
    takeoff_signal = torch.clamp(vz_w / vz_ref, min=0.0, max=1.0)

    active = jump_flag & phase_takeoff
    return takeoff_signal * active.float()


def jump_takeoff_impulse(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_knee_extension_vel: float = 0.0,
    min_ref_takeoff_vz: float = 0.05,
    min_ref_knee_extension_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """地面蹬起冲量奖励：按参考伸膝时序奖励膝关节伸展速度。

    参考轨迹进入蹬地伸展段后才激活，让策略明确学习"什么时候伸腿"。
    """
    from se3_train.mdp.jump_commands import JumpCommandTerm

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        phase_takeoff = term.reference_takeoff_active(min_ref_takeoff_vz)
        ref_q_vel = term.reference_joint_velocity()
        ref_knee_vel = ref_q_vel[:, [1, 4]]
        ref_extension_vel = torch.clamp(-torch.mean(ref_knee_vel, dim=1), min=0.0)
    else:
        phase_takeoff = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        ref_extension_vel = torch.ones(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]

    knee_indices = [JointGroup.LEGS[1], JointGroup.LEGS[3]]
    knee_vel = robot.data.joint_vel[:, knee_indices]
    extension_vel = torch.clamp(-torch.mean(knee_vel, dim=1) - min_knee_extension_vel, min=0.0)

    target_extension_vel = torch.clamp(ref_extension_vel, min=min_ref_knee_extension_vel)
    extension_tracking = torch.clamp(extension_vel / target_extension_vel, min=0.0, max=1.0)
    active = jump_flag & phase_takeoff
    return extension_tracking * active.float()


def jump_takeoff_vz_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    tolerance: float = 0.35,
    min_vz_ratio: float = 0.3,
    min_ref_takeoff_vz: float = 0.05,
    progress_power: float = 1.0,
    tracking_mix: float = 0.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """地面起跳速度目标奖励:低速给进度引导,高速做目标速度 tracking。

    旧版在 `vz > min_vz_ratio * vz_ref` 后才激活。当前策略只能产生
    0.02~0.11m/s,低于 0.1m 目标高度的激活阈值,因此速度目标奖励长期为 0。
    这里改成两段式:
    - 低于阈值:奖励 `(vz / vz_ref)^p`,从 0 开始提供密集探索梯度
    - 高于阈值:混合进 exp tracking,继续推动接近目标起跳速度

    2026-05-21 PostTrain 经验:
    旧版 L1 tracking 在 tolerance 较宽时会让 1.0m/s 弱跳也拿到较高奖励,
    PPO 会逐步把 1.4m/s 起跳收敛成 1.0m/s 稳定弱跳。这里提高阈值并加入
    progress_power,让弱跳仍有梯度,但分数明显低于接近目标起跳速度的样本。
    """
    from se3_train.mdp.jump_commands import JumpCommandTerm

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        phase_takeoff = term.reference_takeoff_active(min_ref_takeoff_vz)
    else:
        phase_takeoff = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene[asset_cfg.name]
    vz_w = robot.data.root_link_lin_vel_w[:, 2]
    vz_ref = ideal_takeoff_vel(h_target).clamp_min(0.1)
    threshold = min_vz_ratio * vz_ref

    progress = torch.clamp(vz_w / vz_ref, min=0.0, max=1.0)
    progress_reward = progress ** float(progress_power)
    tracking_reward = torch.exp(-((vz_w - vz_ref) ** 2) / (float(tolerance) ** 2))
    mixed_tracking = (1.0 - float(tracking_mix)) * progress_reward + float(
        tracking_mix
    ) * tracking_reward

    reward = torch.where(vz_w < threshold, progress_reward, mixed_tracking)
    active = jump_flag & phase_takeoff
    return reward * active.float()


def jump_dof_pos_limits_strict(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """起跳/飞行阶段对膝关节(lf1/rf1)下限方向额外惩罚,防止过伸。

    jump_flag=1 时激活。与 dof_pos_limits 不同,此惩罚只看下限方向,
    权重在 env_cfg 里独立控制。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    robot = env.scene[asset_cfg.name]
    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 只看膝关节:JointGroup.LEGS 中索引 1/3(lf1/rf1)
    knee_indices = [JointGroup.LEGS[1], JointGroup.LEGS[3]]
    pos = robot.data.joint_pos[:, knee_indices]
    limits = soft_limits[:, knee_indices]

    # 超出下限的量
    below_lower = -(pos - limits[:, :, 0]).clamp(max=0.0)
    penalty = torch.sum(below_lower, dim=1)

    return penalty * jump_flag.float()


def jump_orientation(
    env: ManagerBasedRlEnv,
    command_name: str,
    pitch_weight: float = 1.0,
    roll_weight: float = 1.0,
    takeoff_scale: float = 0.35,
    air_scale: float = 1.0,
    landing_scale: float = 0.8,
) -> torch.Tensor:
    """跳跃阶段姿态惩罚:同时惩罚 pitch(pg_x)和 roll(pg_y)的平方和。

    旧版只惩罚 roll(pg_y),导致起跳时前后倾斜(pitch)无约束,
    引起视角抖动和落地不稳。轮腿机器人跳跃时 pitch 和 roll 都需要控制:
    - pitch(pg_x):前后倾,起跳蹬力方向偏斜时产生
    - roll(pg_y):左右倾,两腿不对称时产生
    2026-05-21 sim2sim sweep 显示 roll 最大只有约 2.6°,tilt 主要来自
    pitch 约 20°。因此 Fine-tune 可用 pitch_weight 单独加重前后点头。

    使用平方惩罚而不是 abs 惩罚:轻微起跳扰动更温和,大倾角会被
    jump_tilt_barrier 额外放大。只要 jump_flag=1,就使用跳跃自己的阶段约束;
    严格平地 pitch 约束只属于 jump_flag=0 的行走/静站段。takeoff_scale 保持中等值,
    避免真正蹬地窗口被压死;真正不能过早启用的是 hard barrier,而不是连续 L2 回正梯度。
    """
    robot = env.scene["robot"]
    pg = robot.data.projected_gravity_b

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_takeoff = term.reference_takeoff_active()
    in_air = term.jump_stage == 1
    in_landing = term.jump_stage == 2

    stage_scale = torch.where(
        term.jump_stage == 0,
        torch.full((env.num_envs,), float(takeoff_scale), device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )
    stage_scale = torch.where(
        in_takeoff,
        torch.full_like(stage_scale, float(takeoff_scale)),
        stage_scale,
    )
    stage_scale = torch.where(
        in_air,
        torch.full_like(stage_scale, float(air_scale)),
        stage_scale,
    )
    stage_scale = torch.where(
        in_landing,
        torch.full_like(stage_scale, float(landing_scale)),
        stage_scale,
    )

    # projected_gravity_b: x 对应 pitch,y 对应 roll。
    penalty = float(pitch_weight) * pg[:, 0] ** 2 + float(roll_weight) * pg[:, 1] ** 2
    return penalty * stage_scale * jump_flag.float()


def jump_tilt_barrier(
    env: ManagerBasedRlEnv,
    command_name: str,
    soft_limit_deg: float = 25.0,
    hard_limit_deg: float = 45.0,
    landing_scale: float = 0.5,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """跳跃姿态分段强惩罚:轻微 tilt 不管,大 tilt 快速加重。

    jump_orientation 提供连续的小姿态梯度;本项负责 25° 以上的明显空中倾斜。
    它是软硬之间的 barrier:不会像终止一样直接截断 episode,但 35°~45° 的
    样本会明显扣分,避免策略长期接受大 tilt 来换高度 tracking。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_air = term.jump_stage == 1
    in_landing = term.jump_stage == 2
    active = jump_flag & (in_air | in_landing)

    soft = torch.deg2rad(torch.tensor(float(soft_limit_deg), device=env.device))
    hard = torch.deg2rad(torch.tensor(float(hard_limit_deg), device=env.device))
    span = torch.clamp(hard - soft, min=1.0e-3)
    excess = torch.clamp((tilt - soft) / span, min=0.0)
    penalty = torch.clamp(excess**2, max=float(max_penalty))
    stage_scale = torch.where(
        in_landing,
        torch.full_like(penalty, float(landing_scale)),
        torch.ones_like(penalty),
    )

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})["Jump/diag_tilt_barrier_raw"] = (
            penalty[active].mean() if active.any() else torch.zeros((), device=env.device)
        )

    return penalty * stage_scale * active.float()


def jump_ang_vel_xy(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """飞行阶段横滚/俯仰角速度平方和,抑制空中翻滚。

    仅在 reference flight 阶段激活。
    """
    robot = env.scene["robot"]
    ang_vel = robot.data.root_link_ang_vel_b

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_air = term.jump_stage == 1
    active = jump_flag & in_air

    return (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2) * active.float()


def jump_ang_vel_z(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """跳跃窗口 yaw 角速度跟踪惩罚,抑制起跳和空中拧腰。

    yaw 漂移主要来自起跳左右动作不对称,等到空中再惩罚已经偏晚。
    因此该项在完整 jump_flag 窗口激活,并跟踪 command[1],让 sim2sim 的
    yaw PID 信号在跳跃期间也能进入训练目标。
    """
    robot = env.scene["robot"]
    ang_vel = robot.data.root_link_ang_vel_b

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    yaw_error = ang_vel[:, 2] - cmd[:, 1]
    return (yaw_error**2) * jump_flag.float()


def jump_wheel_counterspin(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """跳跃窗口左右轮差速惩罚,减少空中绕 yaw 轴反扭矩。

    MJCF 中左右轮 joint axis 相反:左轮是 +Y,右轮是 -Y。因此在 joint
    广义速度坐标里,左右轮异号更接近平移,同号会制造明显 yaw 扭转。
    该项惩罚左右广义轮速之和,保留策略用异号轮速做落地恢复的自由度。
    """
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    yaw_drive = wheel_vel[:, 0] + wheel_vel[:, 1]
    return (yaw_drive**2) * jump_flag.float()


def jump_wheel_ground_slip(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    wheel_radius: float = 0.059,
    contact_force_threshold: float = 1.0,
    longitudinal_scale: float = 0.35,
    lateral_scale: float = 0.20,
    max_penalty: float = 9.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """接地轮滑移惩罚。

    轮子接地时,轮缘滚动速度应接近机身前向速度,机身侧向速度也应接近 0。
    只要出现明显切向滑移,就按坏行为扣分;该项不奖励更大的轮地反力。

    MJCF 中左右轮 joint axis 相反。广义速度里左轮 `+w`、右轮 `-w` 才表示
    同向前滚,因此右轮滚动速度取负号。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(contact_force_threshold)

    robot = env.scene[asset_cfg.name]
    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )

    base_vel_b = robot.data.root_link_lin_vel_b
    forward_slip = wheel_forward_speed - base_vel_b[:, 0].unsqueeze(1)
    lateral_slip = base_vel_b[:, 1].unsqueeze(1).expand_as(forward_slip)

    penalty_per_wheel = (forward_slip / float(longitudinal_scale)) ** 2 + (
        lateral_slip / float(lateral_scale)
    ) ** 2
    penalty = torch.sum(penalty_per_wheel * in_contact.float(), dim=1)
    penalty = torch.clamp(penalty, max=float(max_penalty))
    active = jump_flag & in_contact.any(dim=1)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_wheel_ground_slip_raw": _mean_on_mask(penalty, active),
                "Jump/diag_wheel_ground_slip_contact_ratio": float(
                    in_contact.any(dim=1).float().mean().item()
                ),
                "Jump/diag_wheel_forward_slip_abs": _mean_on_mask(
                    torch.sum(torch.abs(forward_slip) * in_contact.float(), dim=1)
                    / in_contact.float().sum(dim=1).clamp_min(1.0),
                    active,
                ),
                "Jump/diag_wheel_lateral_slip_abs": _mean_on_mask(
                    torch.abs(base_vel_b[:, 1]),
                    active,
                ),
            }
        )

    return penalty * active.float()


def jump_landing_horizontal_motion_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    wheel_radius: float = 0.059,
    contact_force_threshold: float = 1.0,
    base_speed_scale: float = 0.25,
    wheel_speed_scale: float = 0.35,
    max_penalty: float = 9.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """落地阶段水平运动惩罚。

    落地后继续向前滚,即使轮缘速度与机身速度匹配,也属于原地跳任务的坏行为。
    因此该项直接惩罚 landing 阶段的机身水平速度和接地轮滚动速度,而不是奖励
    滚动匹配效率。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_landing = term.jump_stage == 2

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(contact_force_threshold)
    has_contact = in_contact.any(dim=1)

    robot = env.scene[asset_cfg.name]
    base_vel_b = robot.data.root_link_lin_vel_b
    base_vxy_sq = base_vel_b[:, 0] ** 2 + base_vel_b[:, 1] ** 2

    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    contact_count = in_contact.float().sum(dim=1).clamp_min(1.0)
    wheel_speed_sq = torch.sum((wheel_forward_speed**2) * in_contact.float(), dim=1)
    wheel_speed_sq = wheel_speed_sq / contact_count

    penalty = base_vxy_sq / (float(base_speed_scale) ** 2) + wheel_speed_sq / (
        float(wheel_speed_scale) ** 2
    )
    penalty = torch.clamp(penalty, max=float(max_penalty))
    active = jump_flag & in_landing & has_contact

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_landing_horizontal_motion_raw": _mean_on_mask(penalty, active),
                "Jump/diag_landing_base_vxy": _mean_on_mask(torch.sqrt(base_vxy_sq), active),
                "Jump/diag_landing_wheel_speed_abs": _mean_on_mask(
                    torch.sqrt(wheel_speed_sq), active
                ),
            }
        )

    return penalty * active.float()


def jump_action_mirror(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """跳跃窗口动作镜像惩罚,直接压制左右腿目标分叉和轮子同号拧腰。"""
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    action = env.action_manager.action

    leg_mirror = (action[:, 0] - action[:, 2]) ** 2 + (action[:, 1] - action[:, 3]) ** 2
    # 左右轮 joint axis 相反,同号广义轮速更容易制造 yaw 扭转。
    wheel_yaw = (action[:, 4] + action[:, 5]) ** 2
    return (leg_mirror + 0.5 * wheel_yaw) * jump_flag.float()


def landing_symmetry(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """着陆阶段左右轮力对称性惩罚,减少偏载着陆。

    penalty = |F_left - F_right| / (F_left + F_right + 1)

    仅在 landing stage(2)时激活。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_landing = term.jump_stage == 2
    active = jump_flag & in_landing

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)  # [num_envs, 2]
    f_left = force_mag[:, 0]
    f_right = force_mag[:, 1]

    asymmetry = torch.abs(f_left - f_right) / (f_left + f_right + 1.0)

    return asymmetry * active.float()


def tracking_lin_vel_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float = 2.0,
) -> torch.Tensor:
    """jump_flag=1 时关闭速度跟踪奖励,避免 vz2 惩罚压制起跳。"""
    from se3_train.mdp.rewards import tracking_lin_vel

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = tracking_lin_vel(
        env,
        command_name=command_name,
        sigma_move=sigma_move,
        sigma_stand=sigma_stand,
        vz_weight=vz_weight,
    )
    return result * (~jump_flag).float()


def leg_torques_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """jump_flag=1 时关闭腿部力矩惩罚,允许起跳时输出峰值力矩。

    力矩上限已从 rated_torque(20 N·m) 放开到 stall_torque(40 N·m),
    若保留 leg_torques 惩罚,策略在起跳瞬间会被高额惩罚压制峰值力矩输出。
    jump_flag=1 时清零,jump_flag=0 时正常惩罚(保持行走效率约束)。
    """
    from se3_train.mdp.rewards import leg_torques

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = leg_torques(env, asset_cfg=asset_cfg)
    return result * (~jump_flag).float()


def leg_power_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """jump_flag=1 时关闭腿部功率惩罚,允许起跳时高功率输出。

    与 leg_torques_no_jump 配套:起跳瞬间高力矩 × 高速度 = 高功率,
    若不豁免 leg_power,峰值功率惩罚会同样压制起跳动作。
    """
    from se3_train.mdp.rewards import leg_power

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = leg_power(env, asset_cfg=asset_cfg)
    return result * (~jump_flag).float()


# ---------------------------------------------------------------------------
# 新增:轮子离地高度、左右对称性、空中轮速约束
# ---------------------------------------------------------------------------


def jump_wheel_clr_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    base_height_offset: float = 0.26,
    wheel_radius: float = 0.059,
    relative_tolerance: float = 0.15,
    falloff_ratio: float = 0.10,
    apex_ref_vz_window: float = 0.35,
) -> torch.Tensor:
    """参考 apex 附近的轮子离地高度跟踪,相对误差超出容忍带后扣分。

    轮子离地高度 = 轮子 body 到地面的距离(wheel_height_sensor 测量值)。
    目标 = jump_target_height(指令中的目标跳跃高度)。

    Fine-tune 负责精确高度控制:相对误差在 relative_tolerance 内不扣分,
    超出后按相对超出量平滑扣分。默认 15%,对应用户定义的精调及格带。

    只在参考垂直速度接近 0 的 apex 窗口激活。直接在整个上升段跟踪最终高度
    会惩罚合法的刚离地阶段,和逐帧轨迹 tracking 产生时序冲突。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]  # 目标跳跃高度(轮子离地)

    term = _get_jump_term(env, command_name)
    in_air = term.jump_stage == 1
    ref_vz = term.reference_root_velocity()[:, 2]
    near_apex = torch.abs(ref_vz) <= float(apex_ref_vz_window)
    active = jump_flag & in_air & near_apex

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    wheel_h = sensor.data.heights[:, 0] - wheel_radius  # 轮底离地高度

    rel_error = torch.abs(wheel_h - h_target) / h_target.clamp_min(1.0e-3)
    rel_excess = torch.clamp(rel_error - float(relative_tolerance), min=0.0)
    penalty = (rel_excess / max(float(falloff_ratio), 1.0e-6)) ** 2

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_wheel_clr_error_ratio": _mean_on_mask(rel_error, active),
                "Jump/diag_wheel_clr_pass_rate": _mean_on_mask(
                    (rel_error <= float(relative_tolerance)).float(), active
                ),
            }
        )

    return penalty * active.float()


def jump_joint_mirror(
    env: ManagerBasedRlEnv,
    command_name: str,
    hip_weight: float = 4.0,
    knee_weight: float = 1.5,
    grounded_scale: float = 1.0,
    takeoff_scale: float = 4.0,
    air_scale: float = 3.0,
    landing_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """跳跃窗口左右关节对称性惩罚:起跳 + 空中 + 着陆阶段激活。

    惩罚 lf0-rf0 和 lf1-rf1 的加权差值平方和,抑制左右腿不对称动作。
    行走时已有 joint_mirror(weight=-0.179),但跳跃时需要更强的对称约束。

    激活条件:jump_flag=1。蹬地发生在 stage 0 末尾,必须从起跳前就约束;
    起跳和空中阶段权重更高,专门压制 sim2sim 中的双腿分叉。
    """
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    q = robot.data.joint_pos
    diff_hip = q[:, JointGroup.LEGS[0]] - q[:, JointGroup.LEGS[2]]  # lf0 - rf0
    diff_knee = q[:, JointGroup.LEGS[1]] - q[:, JointGroup.LEGS[3]]  # lf1 - rf1

    mirror_penalty = float(hip_weight) * diff_hip**2 + float(knee_weight) * diff_knee**2

    term = env.command_manager.get_term(command_name)
    stage_scale = torch.full_like(mirror_penalty, float(grounded_scale))
    if isinstance(term, JumpCommandTerm):
        stage_scale = torch.where(
            term.reference_takeoff_active(),
            torch.full_like(stage_scale, float(takeoff_scale)),
            stage_scale,
        )
        stage_scale = torch.where(
            term.jump_stage == 1,
            torch.full_like(stage_scale, float(air_scale)),
            stage_scale,
        )
        stage_scale = torch.where(
            term.jump_stage == 2,
            torch.full_like(stage_scale, float(landing_scale)),
            stage_scale,
        )

    return mirror_penalty * stage_scale * jump_flag.float()


def wheel_distance_regularization(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_lateral_distance: float = 0.40,
    max_lateral_distance: float = 0.46,
    max_fore_aft_offset: float = 0.03,
    lateral_scale: float = 0.04,
    fore_aft_scale: float = 0.03,
    fore_aft_weight: float = 1.5,
    standing_scale: float = 1.0,
    grounded_scale: float = 0.4,
    takeoff_scale: float = 1.2,
    air_scale: float = 1.0,
    landing_scale: float = 1.4,
    low_speed_threshold: float = 0.10,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """显式轮距约束:约束左右轮横向间距,并惩罚前后错位。

    Tron1 的 pen_feet_distance 是直接约束脚间距;SerialLeg 之前只靠
    joint_mirror 间接约束,无法直接发现"两个轮一前一后"的站立姿态。
    这里在 base 坐标系下计算左右轮中心差:
    - y 方向:横向轮距,默认站立约 0.433m,因此约束在 0.40~0.46m
    - x 方向:前后错位,超过 0.03m 后惩罚

    非跳跃低速站立和跳跃全阶段都会激活;高速行走期不激活,避免限制步态调整。
    """
    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    delta_w = wheel_pos_w[:, 0, :] - wheel_pos_w[:, 1, :]
    delta_b = quat_apply_inverse(robot.data.root_link_quat_w, delta_w)

    lateral_distance = torch.abs(delta_b[:, 1])
    fore_aft_offset = torch.abs(delta_b[:, 0])

    lateral_low = torch.clamp(float(min_lateral_distance) - lateral_distance, min=0.0)
    lateral_high = torch.clamp(lateral_distance - float(max_lateral_distance), min=0.0)
    lateral_error = lateral_low + lateral_high
    fore_aft_error = torch.clamp(fore_aft_offset - float(max_fore_aft_offset), min=0.0)

    penalty = (lateral_error / float(lateral_scale)) ** 2 + float(fore_aft_weight) * (
        fore_aft_error / float(fore_aft_scale)
    ) ** 2
    penalty = torch.clamp(penalty, max=float(max_penalty))

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    low_speed = (torch.abs(cmd[:, 0]) < float(low_speed_threshold)) & (
        torch.abs(cmd[:, 1]) < float(low_speed_threshold)
    )

    stage_scale = torch.zeros_like(penalty)
    stage_scale = torch.where(
        (~jump_flag) & low_speed,
        torch.full_like(stage_scale, float(standing_scale)),
        stage_scale,
    )

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        jump_scale = torch.full_like(stage_scale, float(grounded_scale))
        jump_scale = torch.where(
            term.reference_takeoff_active(),
            torch.full_like(jump_scale, float(takeoff_scale)),
            jump_scale,
        )
        jump_scale = torch.where(
            term.jump_stage == 1,
            torch.full_like(jump_scale, float(air_scale)),
            jump_scale,
        )
        jump_scale = torch.where(
            term.jump_stage == 2,
            torch.full_like(jump_scale, float(landing_scale)),
            jump_scale,
        )
        stage_scale = torch.where(jump_flag, jump_scale, stage_scale)

    if hasattr(env, "extras"):
        active = stage_scale > 0.0
        env.extras.setdefault("log", {})["Jump/diag_wheel_lateral_distance_m"] = (
            lateral_distance[active].mean() if active.any() else torch.zeros((), device=env.device)
        )
        env.extras.setdefault("log", {})["Jump/diag_wheel_fore_aft_offset_m"] = (
            fore_aft_offset[active].mean() if active.any() else torch.zeros((), device=env.device)
        )
        env.extras.setdefault("log", {})["Jump/diag_wheel_distance_raw"] = (
            penalty[active].mean() if active.any() else torch.zeros((), device=env.device)
        )

    return penalty * stage_scale


def flat_wheel_center_alignment_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    contact_sensor_name: str,
    contact_force_threshold: float = 1.0,
    low_speed_threshold: float = 0.10,
    center_lead_gain: float = 0.03,
    center_tolerance: float = 0.05,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """平地段轮前后对齐惩罚。

    这里和落地稳定性使用同一套几何语法：左右轮分别对齐到
    ideal_x = COM_x + k * vx。平地只是在低速静站时激活，落地则在
    landing 相位激活。两者共享同一个几何核，避免一处改动另一处漏掉。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    low_speed = (torch.abs(cmd[:, 0]) < float(low_speed_threshold)) & (
        torch.abs(cmd[:, 1]) < float(low_speed_threshold)
    )

    sensor: ContactSensor = env.scene[contact_sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(contact_force_threshold)
    has_contact = in_contact.any(dim=1)

    _ideal_x, dist_l, dist_r, mean_dist = _wheel_alignment_error(
        env,
        asset_cfg,
        center_lead_gain,
    )
    penalty = torch.clamp(
        (mean_dist / float(center_tolerance)) ** 2,
        max=float(max_penalty),
    )
    active = (~jump_flag) & low_speed & has_contact

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_wheel_center_error_m": _mean_on_mask(mean_dist, active),
                "Jump/diag_flat_wheel_center_penalty": _mean_on_mask(penalty, active),
                "Jump/diag_flat_wheel_center_contact_ratio": float(
                    has_contact.float().mean().item()
                ),
                "Jump/diag_flat_wheel_center_left_error_m": _mean_on_mask(dist_l, active),
                "Jump/diag_flat_wheel_center_right_error_m": _mean_on_mask(dist_r, active),
            }
        )

    return penalty * active.float()


def jump_wheel_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """空中阶段轮速惩罚:抑制策略用轮速补偿姿态或做无效旋转。

    sim2sim 数据显示轮速在空中持续加速(l_wheel=+34, r_wheel=-79 rad/s),
    策略通过异号大轮速产生陀螺效应来维持姿态,而不是靠腿部正确起跳。
    惩罚空中期轮速平方和,迫使策略用更纯粹的腿部动作控制姿态。

    激活条件:jump_flag=1 且 stage==1(airborne)。
    """
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = _get_jump_term(env, command_name)
    in_air = term.jump_stage == 1
    active = jump_flag & in_air

    from se3_shared import JointGroup

    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]  # [num_envs, 2]
    return torch.sum(wheel_vel**2, dim=1) * active.float()


def feet_contact_without_cmd_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    force_threshold: float,
    cmd_threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """jump_flag=1 时关闭 feet_contact_without_cmd 奖励。

    原奖励在「无速度指令且轮子接地」时给正奖励。
    当 jump_flag=1 时,机器人坐在地面蹲健就能拥有这个奖励,会鼓励「蹲着不动」而非起跳。
    """
    from se3_train.mdp.rewards import feet_contact_without_cmd

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = feet_contact_without_cmd(
        env,
        command_name=command_name,
        force_threshold=force_threshold,
        cmd_threshold=cmd_threshold,
        sensor_name=sensor_name,
        asset_cfg=asset_cfg,
    )
    return result * (~jump_flag).float()


def stand_still_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """jump_flag=1 时关闭站立静止惩罚,允许起跳时腿部大幅运动。"""
    from se3_train.mdp.rewards import stand_still

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = stand_still(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        default_height=default_height,
        height_tolerance=height_tolerance,
        asset_cfg=asset_cfg,
    )
    return result * (~jump_flag).float()


def standing_joint_mirror_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    hip_weight: float = 4.0,
    knee_weight: float = 1.5,
    low_speed_sigma: float = 0.35,
    low_speed_floor: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """静站期左右腿镜像惩罚,压制左右轮前后错位。

    sim2sim 中观察到静站时左右轮沿机身 x 方向错开,根因表现为左右髋关节
    角度不对称。该项只在非跳跃时激活,低速和静站给强约束,避免硬阈值导致
    静站样本太稀疏;jump_flag=1 时完全关闭,避免限制起跳蹲展动作。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    stationary = cmd_speed < command_threshold

    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = torch.clamp(-pg_z, 0.0, 0.7) / 0.7
    low_speed_gate = torch.exp(-(cmd_speed**2) / (2.0 * float(low_speed_sigma) ** 2))
    low_speed_gate = torch.clamp(low_speed_gate, min=float(low_speed_floor), max=1.0)
    low_speed_gate = torch.where(stationary, torch.ones_like(low_speed_gate), low_speed_gate)

    hip_diff = (
        robot.data.joint_pos[:, JointGroup.LEGS[0]] - robot.data.joint_pos[:, JointGroup.LEGS[2]]
    )
    knee_diff = (
        robot.data.joint_pos[:, JointGroup.LEGS[1]] - robot.data.joint_pos[:, JointGroup.LEGS[3]]
    )
    penalty = float(hip_weight) * hip_diff**2 + float(knee_weight) * knee_diff**2
    return penalty * low_speed_gate * (~jump_flag).float() * gate


# ---------------------------------------------------------------------------
# 行走奖励 jump 门控补充:tracking_height、tracking_ang_vel、action_rate
# ---------------------------------------------------------------------------


def tracking_height_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    height_sensor_name: str,
) -> torch.Tensor:
    """jump_flag=1 时关闭行走高度跟踪奖励,避免与跳跃轨迹 pose tracking 正面冲突。

    tracking_height 在 jump_flag=1 时惩罚 base_z 偏离行走默认高度(0.26m),
    而跳跃轨迹在起跳段要求 base_z 明显升高,两者梯度完全相反。
    """
    from se3_train.mdp.rewards import tracking_height

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = tracking_height(
        env,
        command_name=command_name,
        sigma=sigma,
        height_sensor_name=height_sensor_name,
    )
    return result * (~jump_flag).float()


def tracking_ang_vel_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
) -> torch.Tensor:
    """jump_flag=1 时关闭偏航速度跟踪奖励。

    起跳/飞行期偏航变化是正常的,不应持续惩罚。
    """
    from se3_train.mdp.rewards import tracking_ang_vel

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = tracking_ang_vel(env, command_name=command_name, sigma=sigma)
    return result * (~jump_flag).float()


def tracking_orientation_l2_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """jump_flag=1 时关闭行走期姿态 L2 惩罚。

    跳跃阶段由 jump_orientation / jump_tilt_barrier 负责姿态质量;行走和静站阶段
    使用 L2 惩罚提供持续回正梯度。
    """
    from se3_train.mdp.rewards import tracking_orientation_l2

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = tracking_orientation_l2(env, command_name=command_name)
    return result * (~jump_flag).float()


def flat_orientation_l2_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """平地段 Unitree 风格直立姿态 L2 惩罚。

    直接惩罚 projected_gravity 在机身 x/y 方向的平方和。机身完全直立时
    x/y 为 0;只要出现 pitch 或 roll 偏离,就会产生连续负反馈。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    pg = robot.data.projected_gravity_b
    flat = ~jump_flag
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    idle = flat & (cmd_norm < 0.08)
    penalty = torch.sum(torch.square(pg[:, :2]), dim=1)
    signed_pitch = torch.atan2(pg[:, 0], -pg[:, 2])

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_orientation_l2_raw": _mean_on_mask(penalty, flat),
                "Jump/diag_idle_orientation_l2_raw": _mean_on_mask(penalty, idle),
                "Jump/diag_flat_pitch_signed_deg": _mean_on_mask(
                    torch.rad2deg(signed_pitch),
                    flat,
                ),
                "Jump/diag_idle_pitch_signed_deg": _mean_on_mask(
                    torch.rad2deg(signed_pitch),
                    idle,
                ),
            }
        )

    return penalty * flat.float()


def action_rate_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    idle_command_threshold: float = 0.08,
    idle_scale: float = 1.0,
    moving_scale: float = 1.0,
    max_penalty: float = 80.0,
) -> torch.Tensor:
    """jump_flag=1 时关闭 action_rate 惩罚,并在 flat idle 期加强平滑。

    起跳的核心是 action 突变(蹲→展腿),action_rate 惩罚会直接压制这个动作。
    flat idle 期没有动作突变需求,额外压低动作变化,避免站立/低速段抖动。
    """
    from se3_train.mdp.rewards import action_rate

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    flat = ~jump_flag
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    idle = flat & (cmd_norm < float(idle_command_threshold))

    result = action_rate(env)
    clipped = torch.clamp(result, max=float(max_penalty))
    scale = torch.full_like(clipped, float(moving_scale))
    scale = torch.where(idle, torch.full_like(scale, float(idle_scale)), scale)

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        log["Jump/diag_flat_action_rate_raw"] = (
            result[flat].mean() if flat.any() else torch.zeros((), device=env.device)
        )
        log["Jump/diag_idle_action_rate_raw"] = (
            result[idle].mean() if idle.any() else torch.zeros((), device=env.device)
        )
    return clipped * scale * flat.float()


def idle_wheel_motion_penalty_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    wheel_radius: float = 0.059,
    idle_command_threshold: float = 0.08,
    contact_force_threshold: float = 1.0,
    base_speed_scale: float = 0.18,
    wheel_speed_scale: float = 0.22,
    max_penalty: float = 9.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """静止期轮组运动惩罚。

    flat idle 期的目标是原地不动。只惩罚 action_rate 会让策略学到慢速来回滚轮;
    该项直接惩罚接地轮滚动速度和机身水平速度,避免站立时轮组前后抖动。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    idle = (~jump_flag) & (cmd_norm < float(idle_command_threshold))

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(contact_force_threshold)
    has_contact = in_contact.any(dim=1)

    robot = env.scene[asset_cfg.name]
    base_vel_b = robot.data.root_link_lin_vel_b
    base_vxy_sq = base_vel_b[:, 0] ** 2 + base_vel_b[:, 1] ** 2

    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    contact_count = in_contact.float().sum(dim=1).clamp_min(1.0)
    wheel_speed_sq = torch.sum((wheel_forward_speed**2) * in_contact.float(), dim=1)
    wheel_speed_sq = wheel_speed_sq / contact_count

    penalty = base_vxy_sq / (float(base_speed_scale) ** 2) + wheel_speed_sq / (
        float(wheel_speed_scale) ** 2
    )
    penalty = torch.clamp(penalty, max=float(max_penalty))
    active = idle & has_contact

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_idle_wheel_motion_raw": _mean_on_mask(penalty, active),
                "Jump/diag_idle_base_vxy": _mean_on_mask(torch.sqrt(base_vxy_sq), active),
                "Jump/diag_idle_wheel_speed_abs": _mean_on_mask(torch.sqrt(wheel_speed_sq), active),
            }
        )

    return penalty * active.float()


def flat_wheel_ground_slip_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    wheel_radius: float = 0.059,
    contact_force_threshold: float = 1.0,
    longitudinal_scale: float = 0.28,
    lateral_scale: float = 0.20,
    max_penalty: float = 9.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """平地段轮地滑移惩罚。

    直接约束接地轮的滚动速度和机身速度一致，防止策略在平地阶段用
    "轮子空转 + 机身慢滑" 的方式骗过速度跟踪。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    flat = ~jump_flag

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(contact_force_threshold)

    robot = env.scene[asset_cfg.name]
    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )

    base_vel_b = robot.data.root_link_lin_vel_b
    forward_slip = wheel_forward_speed - base_vel_b[:, 0].unsqueeze(1)
    lateral_slip = base_vel_b[:, 1].unsqueeze(1).expand_as(forward_slip)

    penalty_per_wheel = (forward_slip / float(longitudinal_scale)) ** 2 + (
        lateral_slip / float(lateral_scale)
    ) ** 2
    penalty = torch.sum(penalty_per_wheel * in_contact.float(), dim=1)
    penalty = torch.clamp(penalty, max=float(max_penalty))
    active = flat & in_contact.any(dim=1)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_wheel_ground_slip_raw": _mean_on_mask(penalty, active),
                "Jump/diag_flat_wheel_ground_slip_contact_ratio": float(
                    in_contact.any(dim=1).float().mean().item()
                ),
            }
        )

    return penalty * active.float()


def flat_base_lin_vel_z_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    low_speed_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """平地段基座竖直速度惩罚。

    直接压制 base_link 在 z 方向的小幅上下振荡，避免策略学出“高度大体对
    但一直抖、偶尔小跳”的平地姿态。只在非跳跃且低速时激活，跳跃阶段完全
    豁免，避免干扰起跳和落地的竖直动量。
    """
    from se3_train.mdp.rewards import lin_vel_z

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    low_speed = (torch.abs(cmd[:, 0]) < float(low_speed_threshold)) & (
        torch.abs(cmd[:, 1]) < float(low_speed_threshold)
    )

    robot = env.scene[asset_cfg.name]
    vz = robot.data.root_link_lin_vel_b[:, 2]
    penalty = lin_vel_z(env)

    active = (~jump_flag) & low_speed
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_base_vz_raw": _mean_on_mask(torch.abs(vz), active),
                "Jump/diag_flat_base_vz_penalty": _mean_on_mask(penalty, active),
            }
        )

    return penalty * active.float()


def flat_wheel_air_penalty_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.059,
    idle_command_threshold: float = 0.08,
    clearance_tolerance: float = 0.003,
    clearance_scale: float = 0.015,
    max_penalty: float = 25.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """平地静止期轮子离地惩罚。

    只在 jump_flag=0 且低速 idle 时激活。任一轮底明显离地都会扣分,
    直接压制 sim2sim 里看到的原地小跳;跳跃阶段完全豁免,避免和起跳目标冲突。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    idle = (~jump_flag) & (cmd_norm < float(idle_command_threshold))

    wheel_bottom_h = _wheel_bottom_heights(env, asset_cfg, wheel_radius)
    lift = torch.clamp(wheel_bottom_h - float(clearance_tolerance), min=0.0)
    penalty = torch.sum((lift / max(float(clearance_scale), 1.0e-6)) ** 2, dim=1)
    penalty = torch.clamp(penalty, max=float(max_penalty))

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        max_lift = torch.clamp(torch.max(wheel_bottom_h, dim=1).values, min=0.0)
        env.extras["log"].update(
            {
                "Jump/diag_flat_wheel_air_penalty": _mean_on_mask(penalty, idle),
                "Jump/diag_flat_wheel_max_lift_m": _mean_on_mask(max_lift, idle),
                "Jump/diag_flat_wheel_grounded_rate": _mean_on_mask(
                    (max_lift <= float(clearance_tolerance)).float(),
                    idle,
                ),
            }
        )

    return penalty * idle.float()


def flat_base_height_penalty_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    sigma: float = 0.05,
) -> torch.Tensor:
    """平地段 base 高度惩罚。

    直接惩罚 base height 偏离当前 command height 的平方误差。
    这比只靠 tracking_height 的正奖励更明确，方便把平地站高收敛到
    我们指定的随机区间内。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    flat = ~jump_flag

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    target_height = cmd[:, 4]
    penalty = torch.square(height - target_height) / (float(sigma) ** 2)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_base_height_error_m": _mean_on_mask(
                    torch.abs(height - target_height), flat
                ),
                "Jump/diag_flat_base_height_penalty": _mean_on_mask(penalty, flat),
            }
        )

    return penalty * flat.float()


def action_smoothness_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    idle_command_threshold: float = 0.08,
    idle_scale: float = 1.5,
    moving_scale: float = 0.5,
    max_penalty: float = 80.0,
) -> torch.Tensor:
    """平地段二阶动作平滑惩罚。

    参考 Tron1 的 ActionSmoothnessPenalty。action_rate 只惩罚一阶变化,周期性
    前后抖动仍可能以较小一阶变化存在;二阶项直接惩罚动作曲率,压制来回抽动。
    jump_flag=1 时关闭,保留起跳动作自由度。
    """
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

    penalty = torch.sum((action - 2.0 * prev + prev_prev) ** 2, dim=1)
    setattr(env, _ACTION_SMOOTH_PREV_PREV_ATTR, prev.detach().clone())
    setattr(env, _ACTION_SMOOTH_PREV_ATTR, action.detach().clone())

    startup = env.episode_length_buf < 3
    penalty = torch.where(startup, torch.zeros_like(penalty), penalty)

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    flat = ~jump_flag
    cmd_norm = torch.linalg.norm(cmd[:, :3], dim=1)
    idle = flat & (cmd_norm < float(idle_command_threshold))

    clipped = torch.clamp(penalty, max=float(max_penalty))
    scale = torch.full_like(clipped, float(moving_scale))
    scale = torch.where(idle, torch.full_like(scale, float(idle_scale)), scale)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_action_smoothness_raw": (
                    penalty[flat].mean() if flat.any() else torch.zeros((), device=env.device)
                ),
                "Jump/diag_idle_action_smoothness_raw": (
                    penalty[idle].mean() if idle.any() else torch.zeros((), device=env.device)
                ),
            }
        )

    return clipped * scale * flat.float()


def action_rate_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    grounded_scale: float = 0.4,
    takeoff_scale: float = 0.25,
    air_scale: float = 1.0,
    landing_scale: float = 1.2,
    max_penalty: float = 80.0,
) -> torch.Tensor:
    """jump_flag=1 时保留阶段化动作变化惩罚,压制高频抖动。

    起跳需要从蹲到展的动作变化,不能用行走期同等 action_rate 惩罚。
    2026-05-21 sim2sim sweep 显示动作尖峰主要集中在跳跃窗口内,最大
    action_delta_sq_sum 超过 200。这里按参考阶段缩放:
    - preload/grounded:弱惩罚,只压无意义抖动
    - takeoff:最弱惩罚,保留蹬地动作自由度
    - air/landing:强惩罚,压制空中抽动和落地瞬间动作跳变

    max_penalty 用于截断极端样本,避免少量尖峰主导 PPO 更新。
    """
    from se3_train.mdp.rewards import action_rate

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    result = action_rate(env)

    term = env.command_manager.get_term(command_name)
    stage_scale = torch.full_like(result, float(grounded_scale))
    if isinstance(term, JumpCommandTerm):
        stage_scale = torch.where(
            term.reference_takeoff_active(),
            torch.full_like(stage_scale, float(takeoff_scale)),
            stage_scale,
        )
        stage_scale = torch.where(
            term.jump_stage == 1,
            torch.full_like(stage_scale, float(air_scale)),
            stage_scale,
        )
        stage_scale = torch.where(
            term.jump_stage == 2,
            torch.full_like(stage_scale, float(landing_scale)),
            stage_scale,
        )

    clipped = torch.clamp(result, max=float(max_penalty))
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})["Jump/diag_jump_action_rate_raw"] = (
            result[jump_flag].mean() if jump_flag.any() else torch.zeros((), device=env.device)
        )
        env.extras.setdefault("log", {})["Jump/diag_jump_action_rate_clipped"] = (
            clipped[jump_flag].mean() if jump_flag.any() else torch.zeros((), device=env.device)
        )
    return clipped * stage_scale * jump_flag.float()


# ---------------------------------------------------------------------------
# 「原地垂直跳」专项约束
# ---------------------------------------------------------------------------


def jump_pre_takeoff_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_vxy: float = 0.3,
    stillness_window_steps: int = 50,
    max_ref_preload_vz: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """起跳前水平静止奖励:在下降蓄力子相位鼓励水平速度归零。

    设计目标:让机器人先原地静止再起跳,而不是边走边跳。
    奖励形式:exp(-vxy2 / sigma_vxy2),静止时为 1,vxy=sigma_vxy 时约 0.37。

    激活限制:
    - jump_flag=1 + reference preload(grounded 且参考 vz 尚未上升)
    - 另加 stillness_window_steps 时间窗口限制:jump 窗口开始后的前 N 步才有奖励,超过则决策必须起跳。
    """
    from se3_train.mdp.jump_commands import JumpCommandTerm

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        phase_preload = term.reference_preload_active(max_ref_preload_vz)
        pre_takeoff_only = term.traj_step <= stillness_window_steps
    else:
        phase_preload = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        pre_takeoff_only = env.episode_length_buf <= stillness_window_steps

    robot = env.scene[asset_cfg.name]
    vel = robot.data.root_link_lin_vel_w  # [num_envs, 3]
    vxy_sq = vel[:, 0] ** 2 + vel[:, 1] ** 2

    stillness = torch.exp(-vxy_sq / (sigma_vxy**2))
    active = jump_flag & phase_preload & pre_takeoff_only
    return stillness * active.float()


def _landing_recovery_reward(
    stillness: torch.Tensor, height_recovery: torch.Tensor
) -> torch.Tensor:
    """落地恢复奖励的组合方式:加权和,让两项各自独立传梯度。

    乘积形式在高度未恢复时 height_recovery≈0,会阻断 stillness 梯度传播;
    加权和确保静止和高度恢复各自都有梯度,不会互相屏蔽。
    """
    return 0.6 * stillness + 0.4 * height_recovery


def jump_landing_recovery(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_vxy: float = 0.4,
    sigma_h: float = 0.03,
    target_height: float = 0.26,
    height_sensor_name: str = "base_height_sensor",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """落地恢复奖励:landing 阶段同时奖励水平速度归零和高度回到站立位。

    设计目标:落地后机器人停在原地,而不是继续滑行或倒地。
    奖励 = stillness × height_recovery:两项同时为好才拿满分。

    激活阶段:reference landing。真实腿部接触只用于日志留档,不参与奖励阶段判断。
    """

    from se3_train.mdp.jump_commands import JumpCommandTerm

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        in_landing = term.jump_stage == 2
    else:
        in_landing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    active = jump_flag & in_landing
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    vel = robot.data.root_link_lin_vel_w
    vxy_sq = vel[:, 0] ** 2 + vel[:, 1] ** 2
    stillness = torch.exp(-vxy_sq / (sigma_vxy**2))

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    h = sensor.data.heights[:, 0]
    h_err_sq = (h - target_height) ** 2
    height_recovery = torch.exp(-h_err_sq / (sigma_h**2))

    # 加权和替代乘积:两项独立传梯度,防止高度未恢复时阻断水平静止梯度传播
    return _landing_recovery_reward(stillness, height_recovery) * active.float()


def jump_landing_base_height_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str = "base_height_sensor",
    target_height: float = 0.26,
    tolerance: float = 0.035,
    scale: float = 0.06,
) -> torch.Tensor:
    """落地阶段 base 高度显式惩罚。

    Tron1 使用高权重 pen_base_height 直接惩罚低身位。SerialLeg 跳跃过程不能
    全程固定 base 高度,否则会和起跳高度 tracking 冲突;因此只在 landing 阶段
    要求回到站立高度,防止落地后蹲太深或趴低。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        in_landing = term.jump_stage == 2
    else:
        in_landing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    height_error = torch.clamp(torch.abs(height - float(target_height)) - float(tolerance), min=0.0)
    penalty = (height_error / float(scale)) ** 2
    return penalty * (jump_flag & in_landing).float()


def jump_takeoff_horizontal_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_vxy: float = 0.25,
    min_vz_ratio: float = 0.2,
    min_ref_takeoff_vz: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """起跳时水平速度惩罚:在上升蹬地子相位惩罚水平速度,强制垂直起跳。

    设计目标:策略必须先停下来再蹬腿,而不是带着水平动量起跳。
    激活条件与 jump_takeoff_drive 相同(reference takeoff + jump_flag=1),
    但进一步要求 vz > 0(已经开始蹬起)时才惩罚水平速度。
    这样不会惩罚下降蓄力阶段,只惩罚「边跑边跳」的瞬间。

    返回正惩罚值,env_cfg 中使用负权重。
    惩罚形式:1 - exp(-vxy2 / sigma2),水平速度为 0 时惩罚 0,越快惩罚越大。
    """
    from se3_train.mdp.jump_commands import JumpCommandTerm

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]

    term = env.command_manager.get_term(command_name)
    if isinstance(term, JumpCommandTerm):
        phase_takeoff = term.reference_takeoff_active(min_ref_takeoff_vz)
    else:
        phase_takeoff = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene[asset_cfg.name]
    vel = robot.data.root_link_lin_vel_w
    vz = vel[:, 2]
    vz_ref = ideal_takeoff_vel(h_target)
    vxy_sq = vel[:, 0] ** 2 + vel[:, 1] ** 2

    # 只在已经蹬出一部分向上速度时才惩罚水平速度,不干扰早期蓄力/蹬地探索
    taking_off = vz > (min_vz_ratio * vz_ref)
    penalty = 1.0 - torch.exp(-vxy_sq / (sigma_vxy**2))

    active = jump_flag & phase_takeoff & taking_off
    return penalty * active.float()


def jump_landing_stability_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str = "wheel_sensor",
    wheel_radius: float = 0.059,
    contact_force_threshold: float = 1.0,
    k_gain: float = 0.03,
    tolerance: float = 0.10,
    max_penalty: float = 9.0,
    min_landing_vz: float = -0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """落地稳定性惩罚：左右轮分别计算理想落点。

    物理直觉：落地瞬间若轮子触地点偏离理想位置，地面反力会产生绕 COM 的力矩，
    迫使轮子滑移来抵消。

    每个轮子独立计算理想落点：
    - ideal_x = COM_x + k_gain * vx（两个轮子相同的前后方向目标）
    - ideal_y = 轮子当前侧向位置（几何固定，不可优化）
    - 轮子只有一个沿矢状面的旋转自由度，侧向位置由机身决定

    激活条件：轮子着地 且 COM 下降段 vz < min_landing_vz 且 jump_flag=1
    惩罚：|actual_x - ideal_x| 左右轮各自计算后取平均
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    in_contact = (force_mag > float(contact_force_threshold)).any(dim=1)

    robot = env.scene[asset_cfg.name]
    vz = robot.data.root_link_lin_vel_w[:, 2]

    # 只在真正从高处落下时才激活（vz < min_landing_vz）
    landing_active = jump_flag & in_contact & (vz < float(min_landing_vz))

    _, dist_l, dist_r, mean_dist = _wheel_alignment_error(env, asset_cfg, k_gain)

    tol = float(tolerance)
    excess = torch.clamp(mean_dist - tol, min=0.0)
    penalty = torch.clamp(excess / tol, max=float(max_penalty))

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_landing_ideal_dist_m": _mean_on_mask(mean_dist, landing_active),
                "Jump/diag_landing_stability_raw": _mean_on_mask(penalty, landing_active),
                "Jump/diag_landing_active_ratio": _mean_on_mask(
                    landing_active.float(), torch.ones_like(landing_active)
                ),
                "Jump/diag_landing_left_ideal_dist_m": _mean_on_mask(dist_l, landing_active),
                "Jump/diag_landing_right_ideal_dist_m": _mean_on_mask(dist_r, landing_active),
            }
        )

    return penalty * landing_active.float()
