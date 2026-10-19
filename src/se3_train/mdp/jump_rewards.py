"""跳跃相关的共享奖励函数。

旧跳跃线（Jump-PreTrain / FineTune）2026-10-19 删除后，这里只保留 Flat、Rough、Stair
仍在引用的项；旧线专属的奖励项见 git 历史（846c4fcd）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.joint_indices import (
    wheel_joint_ids,
)
from se3_train.mdp.jump_commands import JumpCommandTerm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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


def _recovery_reset_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 recovery reset 标记；普通任务没有该标记时全 False。"""
    mask = getattr(env, "_recovery_reset_mask", None)
    if not isinstance(mask, torch.Tensor) or mask.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return mask.to(device=env.device, dtype=torch.bool)


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


# ---------------------------------------------------------------------------
# 阶段1(PreTrain)奖励
# ---------------------------------------------------------------------------


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
    recovery = _recovery_reset_mask(env)
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
    active = (~jump_flag) & (~recovery) & low_speed & has_contact

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


# ---------------------------------------------------------------------------
# 行走奖励 jump 门控补充:tracking_height、tracking_ang_vel、action_rate
# ---------------------------------------------------------------------------


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

    wheel_vel = robot.data.joint_vel[:, list(wheel_joint_ids(robot))]
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
    wheel_vel = robot.data.joint_vel[:, list(wheel_joint_ids(robot))]
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
    recovery = _recovery_reset_mask(env)
    low_speed = (torch.abs(cmd[:, 0]) < float(low_speed_threshold)) & (
        torch.abs(cmd[:, 1]) < float(low_speed_threshold)
    )

    robot = env.scene[asset_cfg.name]
    vz = robot.data.root_link_lin_vel_b[:, 2]
    penalty = lin_vel_z(env)

    active = (~jump_flag) & (~recovery) & low_speed
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_base_vz_raw": _mean_on_mask(torch.abs(vz), active),
                "Jump/diag_flat_base_vz_penalty": _mean_on_mask(penalty, active),
            }
        )

    return penalty * active.float()


def flat_wheel_contact_penalty_no_jump(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    idle_command_threshold: float = 0.08,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """平地静止期轮子缺失接触惩罚。

    只在 jump_flag=0 且低速 idle 时激活。两个轮子都接地时惩罚为 0,
    单轮离地惩罚 0.5,双轮离地惩罚 1.0。该项只使用接触传感器,
    避免用轮高假设绑定到 flat 地形几何。
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
    contact_ratio = in_contact.float().mean(dim=1)
    penalty = 1.0 - contact_ratio

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Jump/diag_flat_wheel_contact_penalty": _mean_on_mask(penalty, idle),
                "Jump/diag_flat_wheel_contact_ratio": _mean_on_mask(contact_ratio, idle),
                "Jump/diag_flat_wheel_full_contact_rate": _mean_on_mask(
                    (contact_ratio >= 1.0).float(), idle
                ),
            }
        )

    return penalty * idle.float()


# ---------------------------------------------------------------------------
# 「原地垂直跳」专项约束
# ---------------------------------------------------------------------------
