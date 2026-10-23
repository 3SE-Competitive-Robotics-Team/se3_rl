"""本任务的奖励包装：数学全部复用 Flat 基线，只按地形列改生效范围或核参数。

1. （已删除）`command_velocity_error_on_terrain`：A6–M35 用的速度违令二次罚。M37 起删除：lin_vel_scale 3.0 下
   1.0 m/s 误差只罚 0.20、斜率不到 1，是常数税不是远端梯度（m37_reward_prune_20260927）。
2. `base_height_penalty_support_on_terrain`（M21）：机身高度罚，在指定列（上台阶列）把地面参考从 base_link
   正下方射线换成两轮下方射线（轮子支撑面），其余列与 Flat 原函数逐位相同。旧口径下机身沿一越过台阶边，
   参考地面瞬间抬一阶，误差夹满 0.15 就按封顶 −9/s 罚到机身升完这一阶，把"机身先过沿、轮子随后收腿提上来"
   的过渡期罚成了起跳/走梯（见 env_cfg 的 ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS 注释）。
   `base_height_penalty_off_terrain` 是 M1/M2 用过的按列置零包装（M3 起全列生效、不再挂在配置里，留作对照工具）。
2b. `base_height_penalty_window_on_terrain`（M27 对照 M21，M34 起默认）：上台阶列的地面参考改成机身周围 77 点窗口均值
   （复旦口径），罚改成有界的 1 − exp(−e²/σ²)，M35 起 ±5 cm 死区；由 env_cfg 的 ROUGH_STAIR_HEIGHT_REFERENCE 选择口径。
3. `tracking_ang_vel_off_terrain`：yaw 跟踪在台阶列置零（A10）。台阶列 yaw 指令恒 0、σ=0.25，
   完全静止就能拿满 73% 的正奖励，站着不动是正收益均衡；指令侧归零 + 奖励侧归零缺一不可。
4. `tracking_lin_vel_terrain_vz`：非平地列关掉核里的 vz 项（A7，爬升必须有垂直速度），台阶列单独
   放宽运动核分母（A11，误差约 1 m/s 时仍有半额奖励）；顺带记按列拆开的速度诊断 `Rough/*_terrain`。
5. `off_column`：把任意 Flat 奖励项在指定列上置零的通用包装，M2 用它在台阶列关掉 is_alive、
   flat_wheel_contact、collision（见 env_cfg.py 的 ROUGH_STAIRS_ZEROED_REWARDS 注释）。
5b. `column_scaled`：同一件事的连续版，指定列乘一个系数而不是置零。M22 用它给窄核速度跟踪分列定权
   （台阶列 w=1、其余四列 w=3，见 env_cfg.py 的 ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT 注释）。
6. （已删除）`wheel_fore_aft_offset`（M18/M19）与 `wheel_height_diff`（M23）：左右轮心前后错位 Δx² 与高度差 Δz² 的几何罚。
   M37 起删除：用户决定对称只保留 joint_mirror 一条关节空间定价。推导见 m18_wheel_offset_20260921、
   m23_wheel_dz_20260921；复现用 commit 5af0d00（M35，最后一个带这两项的入口）。

掩码为 None（非课程地形、平面地形、列名对不上）时全部退化成 Flat 基线的行为。
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.distributed

from se3_shared import JointGroup
from se3_shared.action_history import periodic_policy_action_second_difference_torch
from se3_shared.torch_constants import device_constant
from se3_train.mdp.action_period import front_action_periods_from_env
from se3_train.mdp.rewards import (
    _DEFAULT_ASSET_CFG,
    _policy_leg_torque_and_vel,
    _recovery_reset_mask,
    _tracking_upright_gate,
    _wheel_pos_body_frame,
)
from se3_train.mdp.terrain_height import frame_height_above_terrain, ground_height_estimate
from se3_train.tasks.flat.rewards import (
    flat_base_height_penalty_no_jump,
    tracking_lin_vel,
)

from .columns import column_mask, non_flat_column_mask

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def tracking_lin_vel_narrow(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """全地形、全速度指令的窄核奖励；sigma 为指数分母，不重复更新宽核课程指标。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = robot.data.root_link_lin_vel_b[:, 0] - command[:, 0]
    gate = _tracking_upright_gate(robot.data.projected_gravity_b[:, 2], tracking_upright_full_cos)
    return torch.exp(-error.square() / sigma) * gate


def base_height_penalty_off_terrain(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    sigma: float = 0.05,
    max_error: float | None = 0.15,
) -> torch.Tensor:
    """机身高度 L2 罚，在指定子地形列上置零，其余列保持原强度。"""
    penalty = flat_base_height_penalty_no_jump(
        env,
        command_name=command_name,
        height_sensor_name=height_sensor_name,
        sigma=sigma,
        max_error=max_error,
    )
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return penalty
    return penalty * (~mask).float()


def base_height_penalty_support_on_terrain(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    support_sensor_name: str = "stair_reward_height",
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    sigma: float = 0.05,
    max_error: float | None = 0.15,
) -> torch.Tensor:
    """机身高度 L2 罚：指定列的地面参考改为轮子支撑面，其余列与 Flat 原函数逐位相同。

    支撑面 = `support_sensor_name`（两轮下方射线）有效命中的均值（`ground_height_estimate`），
    高度 = 机身射线 frame_z − 支撑面；误差夹 ±max_error、σ、跳跃/扶正掩码与 Flat 原函数一致。
    M21（2026-09-21 用户定）：旧口径（机身正下方射线）下机身沿一越过台阶边，参考地面瞬间抬一阶，
    误差夹满 0.15 → 峰值 −9/s，直到机身升完这一阶（M15 每道 20 cm 立面累计 −2.7…−3.1，整段四级 −10），
    把目标动作"机身先过沿、轮子随后收腿提上来"的过渡期按最高费率罚，奖励天然偏向缩短过渡的
    起跳（M18）/走梯（M17）。支撑面口径下轮子还在下一阶时参考不跳，机身前倾压紧时误差只是几厘米，
    轮子过沿时机身已随之升起。平地上两种口径逐位相同（整圈射线都命中同一平面）。
    没有分列信息（plane）时退化为 Flat 原函数。顺带记台阶列两种口径的 |误差| 均值。
    """
    penalty = flat_base_height_penalty_no_jump(
        env,
        command_name=command_name,
        height_sensor_name=height_sensor_name,
        sigma=sigma,
        max_error=max_error,
    )
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return penalty
    cmd = env.command_manager.get_command(command_name)
    active = (~(cmd[:, 5] > 0.5)) & (~_recovery_reset_mask(env))
    frame_z = env.scene[height_sensor_name].data.frame_pos_w[:, 0, 2]
    height = frame_z - ground_height_estimate(env, support_sensor_name)
    error = height - cmd[:, 4]
    if max_error is not None:
        error = torch.clamp(error, -float(max_error), float(max_error))
    support = error.square() / (float(sigma) ** 2) * active.float()
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        keep = (mask & active).float()
        n = keep.sum().clamp(min=1.0)
        ray_error = frame_height_above_terrain(env, height_sensor_name) - cmd[:, 4]
        log["Rough/base_height_err_support_stairs"] = ((height - cmd[:, 4]).abs() * keep).sum() / n
        log["Rough/base_height_err_ray_stairs"] = (ray_error.abs() * keep).sum() / n
    return torch.where(mask, support, penalty)


def base_height_penalty_window_on_terrain(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    window_sensor_name: str,
    support_sensor_name: str = "stair_reward_height",
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    sigma: float = 0.05,
    max_error: float | None = 0.15,
    dead_zone_m: float = 0.0,
) -> torch.Tensor:
    """机身高度罚：指定列改用机身周围窗口的地面均值作参考、罚改成有界形状，其余列与 Flat 原函数逐位相同。

    对照 M21 的支撑面口径（2026-09-25 用户定）。参考来自 yly-true/fudan_rl_wheel_leg 上台阶 v3：
    高度 = 机身 z − 机身周围 11×7 点（x ±0.5、y ±0.3 m，yaw 对齐）的地面均值，这里直接复用 critic 的
    同尺寸高度扫描（`window_sensor_name`）。机身接近台阶时窗口前沿先扫到上一阶，参考提前抬高、过沿时
    平滑过渡，等于奖励"机身先过沿"；M21 的支撑面在机身过沿时参考不动，这份激励也没了。
    罚取 1 − exp(−e²/σ²)：小误差时与原二次罚 e²/σ² 曲率相同，大误差封顶 1（乘权重 −4 即每秒最多 −4），
    原口径夹 ±0.15 m 时峰值是 (0.15/σ)² = 2.25。`max_error` 只作用于其余列的 Flat 原函数。
    顺带记台阶列窗口与支撑面两种口径的 |误差| 均值。

    `dead_zone_m`（M35，2026-09-26 用户定）：只对指定列的窗口罚生效，|误差| 先减去死区再进核，死区内免费；
    日志仍记原始误差。M34-7800 第 9 级确定性回放的账本（.scratch/m34_eval/window_height_err.py）：窗口口径误差
    68–86% 的时间偏低（p10 −8…−10 cm），罚款 80% 落在过沿过渡段，单边只罚偏低几乎不省钱（0.62 → 0.58/s），
    ±5 cm 死区把每秒罚从 0.62/0.85/0.98（爬升 0.73/0.81/0.89 m/s）压到 0.09/0.15/0.22，即把这项随速度上涨的
    部分从每快 0.08 m/s 多付 0.22/s 压到 0.06/s。0 = 原样（M27/M34）。
    """
    penalty = flat_base_height_penalty_no_jump(
        env,
        command_name=command_name,
        height_sensor_name=height_sensor_name,
        sigma=sigma,
        max_error=max_error,
    )
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return penalty
    cmd = env.command_manager.get_command(command_name)
    active = (~(cmd[:, 5] > 0.5)) & (~_recovery_reset_mask(env))
    frame_z = env.scene[height_sensor_name].data.frame_pos_w[:, 0, 2]
    error = frame_z - ground_height_estimate(env, window_sensor_name) - cmd[:, 4]
    shaped = error
    if float(dead_zone_m) > 0.0:
        shaped = torch.sign(error) * (error.abs() - float(dead_zone_m)).clamp(min=0.0)
    window = (1.0 - torch.exp(-shaped.square() / (float(sigma) ** 2))) * active.float()
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        keep = (mask & active).float()
        n = keep.sum().clamp(min=1.0)
        support_error = frame_z - ground_height_estimate(env, support_sensor_name) - cmd[:, 4]
        log["Rough/base_height_err_window_stairs"] = (error.abs() * keep).sum() / n
        log["Rough/base_height_err_support_stairs"] = (support_error.abs() * keep).sum() / n
    return torch.where(mask, window, penalty)


def wheel_fore_aft_offset(
    env: ManagerBasedRlEnv,
    dead_zone_m: float = 0.0,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """左右轮心在机身系里的前后错位 Δx，超出死区的部分取平方（m²），全列生效，直立门控（M49，2026-09-29 用户定）。

    只罚水平分量、不罚竖直分量：左右腿长差（机身系 Δz）留给 roll 指令用，平地上腿长不等导致的机身侧倾
    已由 tracking_orientation_l2 计价。机身系而非世界系，俯仰不产生假误差。与 M18–M36 的同名项（M37 删除）
    同式，但全列生效、默认无死区——上台阶目标是"腿一收双轮同抬"，不要一先一后的走梯。
    顺带记全列与台阶列的 |Δx|，以及机身系左右腿长差 |Δz|。
    """
    robot = env.scene[_DEFAULT_ASSET_CFG.name]
    wheel_b = _wheel_pos_body_frame(env, _DEFAULT_ASSET_CFG)  # [N, 2, 3]，顺序 (左, 右)
    dx = wheel_b[:, 0, 0] - wheel_b[:, 1, 0]
    excess = torch.clamp(dx.abs() - float(dead_zone_m), min=0.0)
    gate = _tracking_upright_gate(robot.data.projected_gravity_b[:, 2], tracking_upright_full_cos)
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        dz = (wheel_b[:, 0, 2] - wheel_b[:, 1, 2]).abs()
        log["Rough/wheel_dx_abs"] = dx.abs().mean()
        log["Rough/wheel_dz_body_abs"] = dz.mean()
        mask = column_mask(env, ("stairs_up", "stairs_two_step_up"))
        if mask is not None:
            keep = mask.float()
            log["Rough/wheel_dx_abs_stairs"] = (dx.abs() * keep).sum() / keep.sum().clamp(min=1.0)
    return excess.square() * gate


def off_column(
    env: ManagerBasedRlEnv,
    inner,
    params: dict,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
) -> torch.Tensor:
    """任意奖励项在指定子地形列上置零：`inner` 是原函数，`params` 是它自己的参数；其余列逐位不变。"""
    value = inner(env, **params)
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return value
    return value * (~mask).float()


def column_scaled(
    env: ManagerBasedRlEnv,
    inner,
    params: dict,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    scale: float = 1.0,
) -> torch.Tensor:
    """任意奖励项在指定子地形列上乘 `scale`，其余列逐位不变；scale=0 与 `off_column` 等价。

    M22 用来给窄核速度跟踪分列定权：RewardTermCfg 只有一个 weight，分列权重只能靠这层包装。
    """
    value = inner(env, **params)
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return value
    return torch.where(mask, value * float(scale), value)


def tracking_lin_vel_terrain_vz(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float = 2.0,
    terrain_vz_weight: float = 0.0,
    flat_type_names: tuple[str, ...] = ("flat",),
    use_upright_gate: bool = True,
    tracking_upright_full_cos: float = 0.7,
    stair_sigma_move: float | None = None,
    stair_type_names: tuple[str, ...] = ("stairs_up",),
    zero_vz_when_jumping: bool = False,
) -> torch.Tensor:
    """x 速度跟踪：非平地列把核里的 vz 项换成 `terrain_vz_weight`，台阶列把运动核换成 `stair_sigma_move`。

    zero_vz_when_jumping（RJ1）：跳跃参考播放中的 env 去掉 vz 项（腾空 vz≈2 m/s 时整项归零，跳跃中前进速度没有塑形，同 J3）。

    逐 env 的权重/核张量直接喂给 `tracking_lin_vel`，按元素广播；观测、静站判定、课程累加与
    `Locomotion/*` 记账均复用 Flat 基线。
    """
    mask = non_flat_column_mask(env, flat_type_names)
    weight: float | torch.Tensor = float(vz_weight)
    if mask is not None:
        weight = torch.where(
            mask,
            device_constant(float(terrain_vz_weight), device=env.device),
            device_constant(float(vz_weight), device=env.device),
        )
    if zero_vz_when_jumping:
        jumping = env.command_manager.get_term(command_name).active
        weight = torch.where(jumping, torch.zeros_like(jumping, dtype=torch.float), weight)
    move_sigma: float | torch.Tensor = sigma_move
    stair_mask = column_mask(env, stair_type_names) if stair_sigma_move is not None else None
    if stair_mask is not None:
        move_sigma = torch.where(stair_mask, float(stair_sigma_move), float(sigma_move))
    reward = tracking_lin_vel(
        env,
        command_name=command_name,
        sigma_move=move_sigma,
        sigma_stand=sigma_stand,
        vz_weight=weight,
        use_upright_gate=use_upright_gate,
        tracking_upright_full_cos=tracking_upright_full_cos,
    )
    if mask is None:
        return reward

    # 按列拆开的诊断：Locomotion/* 是全体均值，地形列的误差被平地列稀释了看不出来。
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        terrain = mask.float()
        flat = (~mask).float()
        n_t = terrain.sum().clamp(min=1.0)
        n_f = flat.sum().clamp(min=1.0)
        cmd_vx = env.command_manager.get_command(command_name)[:, 0]
        base_vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
        log.update(
            {
                "Rough/tracking_lin_vel_terrain": (reward * terrain).sum() / n_t,
                "Rough/tracking_lin_vel_flat": (reward * flat).sum() / n_f,
                "Rough/cmd_vx_terrain": (cmd_vx * terrain).sum() / n_t,
                "Rough/base_vx_terrain": (base_vx * terrain).sum() / n_t,
                "Rough/base_vx_error_terrain": ((cmd_vx - base_vx).abs() * terrain).sum() / n_t,
            }
        )
        # `*_terrain` 是全部非平地列，M9 起混进了按平地方式发 ±2.4 对称指令的下台阶与坡道列，
        # 看不出上台阶列本身跟不跟得上；单独记一份只看上台阶列的。
        stairs_mask = column_mask(env, stair_type_names)
        if stairs_mask is not None:
            stairs = stairs_mask.float()
            n_s = stairs.sum().clamp(min=1.0)
            log.update(
                {
                    "Rough/cmd_vx_stairs": (cmd_vx * stairs).sum() / n_s,
                    "Rough/base_vx_stairs": (base_vx * stairs).sum() / n_s,
                    "Rough/base_vx_error_stairs": ((cmd_vx - base_vx).abs() * stairs).sum() / n_s,
                }
            )
    return reward


_HEADING_TARGET_ATTR = "_rough_stair_heading_target"


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def stair_heading_hold(
    env: ManagerBasedRlEnv,
    command_name: str,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
) -> torch.Tensor:
    """台阶列航向保持：目标航向 = 回合起点航向 + ∫yaw 指令 dt，返回航向误差平方（rad²），其余列为 0（M54，2026-09-30 用户定）。

    M53 斜向撞立面时先触面的轮子被挡住、另一侧继续走，机身被动绕被挡轮转正（`.scratch/m53_yaw_ledger/`：出生偏 30° 转 −24°，
    强制左右轮同速反而转 −32°）；原定价只有 tracking_ang_vel 按瞬时角速度收费（转动 0.2 s 约 0.6），转完后航向变化不计价。
    本项让航向偏差持续计价，策略要么抵抗被动扭转，要么转完再转回来。目标航向在回合第一步取当前航向（reset 事件已按
    stair_facing_yaw 摆好出生朝向），之后每步按指令 yaw 角速度积分；台阶列 yaw 指令恒为 0 时目标即出生朝向。
    """
    robot = env.scene[_DEFAULT_ASSET_CFG.name]
    yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    cmd_yaw_rate = env.command_manager.get_command(command_name)[:, 1]
    target = getattr(env, _HEADING_TARGET_ATTR, None)
    if not isinstance(target, torch.Tensor) or target.shape != yaw.shape:
        target = yaw.clone()
    new_episode = env.episode_length_buf <= 1
    target = torch.where(new_episode, yaw, target + cmd_yaw_rate * float(env.step_dt))
    setattr(env, _HEADING_TARGET_ATTR, target)
    err = torch.atan2(torch.sin(yaw - target), torch.cos(yaw - target))
    mask = column_mask(env, terrain_type_names)
    keep = torch.ones_like(err) if mask is None else mask.float()
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict) and mask is not None:
        n = keep.sum().clamp(min=1.0)
        log["Rough/stair_heading_error_deg"] = torch.rad2deg((err.abs() * keep).sum() / n)
    return err.square() * keep


def is_terminated_except(env: ManagerBasedRlEnv, exclude_terms: tuple[str, ...]) -> torch.Tensor:
    """非超时终止，但不算 exclude_terms 里的终止项（RJ1 默认排除偏离参考终止）。"""
    manager = env.termination_manager
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in manager.active_terms:
        if name in exclude_terms or manager.get_term_cfg(name).time_out:
            continue
        done |= manager.get_term(name)
    return done.float()


def termination_term(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
    """单个终止项本步是否触发（0 / 1）；权重取 −罚分 / step_dt 即每次触发恰好扣一次罚分（同 fall_penalty 按事件计）。"""
    return env.termination_manager.get_term(term_name).float()


def jump_env_only(
    env: ManagerBasedRlEnv, inner, params: dict, command_name: str = "velocity_height"
) -> torch.Tensor:
    """任意奖励项只在跳跃样本上计（RJ1：模仿奖励的站姿帧是 0.22 m，非跳跃样本高度指令 0.20–0.38 不能被它拉向站姿）。"""
    return inner(env, **params) * env.command_manager.get_term(command_name).jump_env.float()


# ---- 倒地约束：拉格朗日乘子（2026-10-22 用户定，见 env_cfg.ROUGH_FALL_LAGRANGE）----
# 罚分单价 λ 不再人定：按倒地起身时的约束代价是否达标逐轮调整，涨到"越界省下的时间不再划算"为止。
# v2（同日）：只在"倒地出生、尚未起身"的 env（M）里计罚与统计 J，M 内奖励按 1/(1+Σλ) 归一化，功率只计正功。
FALL_LAGRANGE_CONSTRAINTS = ("leg_speed", "body_ang_vel", "wheel_speed", "leg_power")
_FALL_LAGRANGE_STATE_ATTR = "_se3_fall_lagrange_state"
_FALL_NORMALIZE_CHECKED_ATTR = "_se3_fall_lagrange_normalize_checked"


def fall_constraint_active(env: ManagerBasedRlEnv) -> torch.Tensor:
    """M：倒地出生、尚未起身的 env（起身判据见 events.FALL_RECOVERED_TILT_DEG / FALL_RECOVERED_MIN_HEIGHT_M，
    第一次满足后本 episode 不再算 M）。没有倒地出生时全 False。"""
    from .events import _FALL_RECOVERED_ATTR, FALL_SPAWN_MASK_ATTR

    fall = getattr(env, FALL_SPAWN_MASK_ATTR, None)
    recovered = getattr(env, _FALL_RECOVERED_ATTR, None)
    if fall is None or recovered is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return fall & ~recovered


class _FallLagrangeState:
    """四项倒地约束共用的状态：本步代价、3 s 起身窗口的累计代价、按轮的 J 窗口与武装信号窗口、各项 λ。"""

    def __init__(self, env: ManagerBasedRlEnv, params: dict) -> None:
        self.params = params
        self.limits = tuple(float(params["limits"][name]) for name in FALL_LAGRANGE_CONSTRAINTS)
        n = len(FALL_LAGRANGE_CONSTRAINTS)
        self.lambdas = [0.0] * n
        self.lambda_t = torch.zeros(n, device=env.device)
        self.scale = 1.0  # M 内奖励缩放 1/(1+Σλ)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.acc_active = torch.zeros(2, device=env.device)  # [M 内 env·步数, 全部 env·步数]
        self.cost = torch.zeros(env.num_envs, n, device=env.device)
        self.finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.window_cost = torch.zeros_like(self.cost)
        self.window_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.acc_cost = torch.zeros(n, device=env.device)
        self.acc_windows = torch.zeros(1, device=env.device)
        self.acc_nonfinite = torch.zeros(1, device=env.device)  # 本轮代价非有限的 env·步数
        n_cls = len(params["window_class_names"])
        self.acc_arm = torch.zeros(2 * n_cls, device=env.device)  # [命中, 样本] × 类别
        self.cost_window: deque[tuple[list[float], float]] = deque(
            maxlen=int(params["cost_window_iters"])
        )
        self.arm_window: deque[float] = deque(maxlen=int(params["arm_window_iters"]))
        self.armed = False
        self.last_step = -1
        self.class_ids: torch.Tensor | None = None
        ids, _ = env.scene["robot"].find_joints(JointGroup.WHEEL_NAMES, preserve_order=True)
        self.wheel_ids = torch.tensor(ids, device=env.device, dtype=torch.long)

    def step(self, env: ManagerBasedRlEnv) -> None:
        """每个 env.step 只算一次（四个奖励项共用）：本步代价、窗口累计，轮末跨卡汇总并更新 λ。"""
        step = int(env.common_step_counter)
        if step == self.last_step:
            return
        self.last_step = step
        self.active = fall_constraint_active(env)
        self.acc_active[0] += self.active.sum()
        self.acc_active[1] += float(env.num_envs)
        self._compute_cost(env)
        self._accumulate(env)
        if step % int(self.params["steps_per_policy_iter"]) == 0:
            self._end_iteration(env)

    def _compute_cost(self, env: ManagerBasedRlEnv) -> None:
        """本步代价，只在 M 内非零（M 外的倒地 / 走路 / 跳跃一概不计）。"""
        robot = env.scene["robot"]
        torque, leg_vel = _policy_leg_torque_and_vel(env, robot)
        leg_lim, ang_lim, wheel_lim, power_lim = self.limits
        wheel = robot.data.joint_vel[:, self.wheel_ids]
        omega = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)
        # 只计正功（电机输出的机械功）：倒地时腿被地面反推产生的制动功率不受策略控制，R1 参考里占 |P| 的 45%。
        power = torch.clamp(torque * leg_vel, min=0.0).sum(dim=1)
        cost = torch.stack(
            (
                (torch.clamp(leg_vel.abs() - leg_lim, min=0.0) / leg_lim).sum(dim=1),
                torch.clamp(omega - ang_lim, min=0.0) / ang_lim,
                (torch.clamp(wheel.abs() - wheel_lim, min=0.0) / wheel_lim).sum(dim=1),
                torch.clamp(power - power_lim, min=0.0) / power_lim,
            ),
            dim=1,
        )
        # 物理偶发发散时（catastrophic_state 会终止该 env）代价是 nan / inf：置 0，窗口累计里作废该 env 当前窗口。
        # p0cdt5d9 第 537 轮起 J 全是 nan：单个 env 发散，"乘 0/1 掩码"的 nan × 0 = nan 把整轮求和带坏，50 轮窗口里一直是 nan。
        self.finite = torch.isfinite(cost).all(dim=1)
        keep = (self.finite & self.active).unsqueeze(1)
        self.cost = torch.where(keep, cost, torch.zeros_like(cost))
        self.acc_nonfinite += (~self.finite).sum()

    def _accumulate(self, env: ManagerBasedRlEnv) -> None:
        """侧躺 / 倒置出生 env 出生后 horizon_s 内、起身之前的代价积分（J 的样本；代价在 M 外为 0，
        积分区间即 [出生, min(起身, horizon_s)]），以及武装信号（horizon_s 内起身率）。"""
        from .events import (
            _FALL_CACHE_ATTR,
            _FALL_CLASS_ATTR,
            FALL_SPAWN_MASK_ATTR,
            fall_recovery_within,
        )

        fall_mask = getattr(env, FALL_SPAWN_MASK_ATTR, None)
        cache = getattr(env, _FALL_CACHE_ATTR, None)
        if fall_mask is None or cache is None:
            return
        if self.class_ids is None:
            names = tuple(self.params["window_class_names"])
            self.class_ids = torch.tensor(
                [cache.class_names.index(name) for name in names], device=env.device
            )
        horizon = float(self.params["horizon_s"])
        dt = float(env.step_dt)
        window_steps = round(horizon / dt)
        steps = env.episode_length_buf
        tracked = fall_mask & torch.isin(getattr(env, _FALL_CLASS_ATTR), self.class_ids)
        new_episode = steps == 1
        self.window_cost[new_episode] = 0.0
        self.window_valid[new_episode] = True
        self.window_valid &= self.finite
        # 掩码一律用 torch.where 选择，不用乘 0/1（nan × 0 仍是 nan）。
        in_window = (tracked & (steps <= window_steps)).unsqueeze(1)
        self.window_cost = torch.where(
            in_window, self.window_cost + self.cost * dt, self.window_cost
        )
        done = tracked & (steps == window_steps) & self.window_valid
        self.acc_cost += torch.where(
            done.unsqueeze(1), self.window_cost, torch.zeros_like(self.window_cost)
        ).sum(dim=0)
        self.acc_windows += done.sum()

        within = fall_recovery_within(env, horizon)
        if within is not None:
            rate, count, _ = within
            hits = rate[self.class_ids] * count[self.class_ids]
            self.acc_arm += torch.stack((hits, count[self.class_ids]), dim=1).reshape(-1)

    def _end_iteration(self, env: ManagerBasedRlEnv) -> None:
        """轮末：多卡 all_reduce 汇总本轮样本，推进 J / 武装窗口，武装后按 J 更新 λ（所有卡同一份 λ）。

        由奖励项在 env.step 里触发：每张卡每步都算奖励，轮末这一步在所有卡上是同一个 common_step_counter，
        集合通信的调用次数一致。课程项只在有 env reset 时才被调用，某张卡某步没有 reset 就会错位死锁，不能放那里。
        """
        packed = torch.cat(
            (self.acc_cost, self.acc_windows, self.acc_nonfinite, self.acc_active, self.acc_arm)
        )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        values = packed.tolist()
        self.acc_cost.zero_()
        self.acc_windows.zero_()
        self.acc_nonfinite.zero_()
        self.acc_active.zero_()
        self.acc_arm.zero_()
        n = len(FALL_LAGRANGE_CONSTRAINTS)
        cost_sums, windows, nonfinite = values[:n], values[n], values[n + 1]
        active_steps, env_steps, arm = values[n + 2], values[n + 3], values[n + 4 :]
        self.cost_window.append((cost_sums, windows))
        n_cls = len(arm) // 2
        rates = [arm[2 * i] / arm[2 * i + 1] for i in range(n_cls) if arm[2 * i + 1] > 0]
        if n_cls > 0 and len(rates) == n_cls:
            self.arm_window.append(min(rates))
        arm_signal = sum(self.arm_window) / len(self.arm_window) if self.arm_window else 0.0
        if (
            not self.armed
            and len(self.arm_window) == self.arm_window.maxlen
            and arm_signal >= float(self.params["arm_threshold"])
        ):
            self.armed = True

        log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else {}
        total_windows = sum(count for _, count in self.cost_window)
        if total_windows > 0:
            target = float(self.params["target_cost_s"])
            err_lo, err_hi = (float(v) for v in self.params["error_clip"])
            for i, name in enumerate(FALL_LAGRANGE_CONSTRAINTS):
                j = sum(sums[i] for sums, _ in self.cost_window) / total_windows
                # 兜底：非有限代价已在累计时排除；万一 J 仍非有限就跳过本轮更新（Python 的 min/max 会把 nan 传进 λ）。
                if self.armed and math.isfinite(j):
                    error = min(max((j - target) / target, err_lo), err_hi)
                    self.lambdas[i] = min(
                        max(self.lambdas[i] + float(self.params["lambda_lr"]) * error, 0.0),
                        float(self.params["lambda_max"]),
                    )
                log[f"FallLagrange/J_{name}_s"] = torch.tensor(j, device=env.device)
            self.lambda_t = torch.tensor(self.lambdas, device=env.device)
            self.scale = 1.0 / (1.0 + sum(self.lambdas))
        for i, name in enumerate(FALL_LAGRANGE_CONSTRAINTS):
            log[f"FallLagrange/lambda_{name}"] = torch.tensor(self.lambdas[i], device=env.device)
        log["FallLagrange/armed"] = torch.tensor(float(self.armed), device=env.device)
        log["FallLagrange/arm_signal"] = torch.tensor(arm_signal, device=env.device)
        log["FallLagrange/windows_per_iter"] = torch.tensor(windows, device=env.device)
        log["FallLagrange/nonfinite_envs"] = torch.tensor(nonfinite, device=env.device)
        log["FallLagrange/scale"] = torch.tensor(self.scale, device=env.device)
        log["FallLagrange/active_frac"] = torch.tensor(
            active_steps / max(env_steps, 1.0), device=env.device
        )


def _fall_lagrange_state(env: ManagerBasedRlEnv, params: dict) -> _FallLagrangeState:
    state = getattr(env, _FALL_LAGRANGE_STATE_ATTR, None)
    if state is None:
        state = _FallLagrangeState(env, params)
        setattr(env, _FALL_LAGRANGE_STATE_ATTR, state)
    return state


def fall_lagrange_penalty(
    env: ManagerBasedRlEnv,
    constraint: str,
    steps_per_policy_iter: int,
    limits: dict[str, float],
    price_k: float = 300.0,
    window_class_names: tuple[str, ...] = ("side", "inverted"),
    horizon_s: float = 3.0,
    target_cost_s: float = 0.005,
    lambda_lr: float = 0.02,
    error_clip: tuple[float, float] = (-1.0, 5.0),
    lambda_max: float = 10.0,
    cost_window_iters: int = 50,
    arm_window_iters: int = 100,
    arm_threshold: float = 0.6,
) -> torch.Tensor:
    """倒地约束的拉格朗日罚 K·λ·c（权重取 −1）；四项共用一份状态，参数须一致（env_cfg 用同一个字典展开）。

    只在 M（倒地出生、尚未起身的 env，见 fall_constraint_active）里计；M 内的总奖励再由最后一项
    fall_lagrange_normalize 除以 (1 + Σλ)，即 r′ = (r − K·Σλᵢcᵢ)/(1 + Σλᵢ)，M 外 r′ = r。
    v1（dfde5296 / 704d362a）按世界系倾角 > 30° 计罚、λ 无上限：武装后 λ 涨到几千，起跳 / 陡坡 / 上台阶也被重罚，
    16jwmav8 与 brosdfzt 武装后 200 轮内地形等级 5.9 → 1.5 / 0.08、起身率归零。

    代价 c = 归一化超限量，线性 hinge（精确罚：越界一点也不划算；平方罚在限值附近边际代价趋于 0）：
    - leg_speed：Σ_4 电机 max(0, |q̇| − 限值) / 限值；body_ang_vel：max(0, ‖ω_b‖ − 限值) / 限值；
    - wheel_speed：Σ_2 轮 max(0, |ω| − 限值) / 限值；leg_power：max(0, Σ max(0, τ·q̇) − 限值) / 限值（只计正功）。
    K（price_k）是单位超限量的最高价格：λ 大时 M 内任务奖励（含倒地每秒约 −110 的时间代价）被缩到 0，提前起身的好处
    只剩起身后不缩放的走路奖励（每秒约 +14），K 太小约束就压不住；K=300 时 λ ≳ 3 即可压住（见 env_cfg 注释）。

    λ 的对偶更新（每轮一次，多卡 all_reduce 后所有卡同一份）：J = 侧躺 / 倒置出生 env 在 [出生, min(起身, horizon_s)]
    内的 ∫c dt（单位 s，取最近 cost_window_iters 轮完成的窗口求均值），e = clip((J − d) / d, error_clip)，
    λ ← clip(λ + lambda_lr · e, 0, lambda_max)。λ 已在归一化尺度上（λ=10 时约束权重 10/11），再大意义不大。
    武装：λ 起初恒 0；`window_class_names` 里较差一类的 horizon_s 内起身率按轮进 arm_window_iters 轮窗口，窗口满且
    均值 ≥ arm_threshold 后开始更新，此后不再关闭（翻身被发现之前就涨 λ，只会把翻身压没）。
    不设崩塌保护：策略不起身时没有越界、J < d，λ 自己往下走。checkpoint 不保存 λ，续训从 0 重新武装。
    物理偶发发散的 env（代价 nan / inf）当步代价与罚分置 0、当前窗口作废不计入 J；J 万一非有限则跳过更新。
    日志：`FallLagrange/lambda_*`、`J_*_s`、`armed`、`arm_signal`、`windows_per_iter`、`nonfinite_envs`、
    `scale`（1/(1+Σλ)）、`active_frac`（M 内 env·步占比）。
    """
    params = {
        "steps_per_policy_iter": steps_per_policy_iter,
        "limits": limits,
        "price_k": price_k,
        "window_class_names": window_class_names,
        "horizon_s": horizon_s,
        "target_cost_s": target_cost_s,
        "lambda_lr": lambda_lr,
        "error_clip": error_clip,
        "lambda_max": lambda_max,
        "cost_window_iters": cost_window_iters,
        "arm_window_iters": arm_window_iters,
        "arm_threshold": arm_threshold,
    }
    state = _fall_lagrange_state(env, params)
    if state.params != params:
        raise ValueError(f"倒地约束 {constraint} 的参数与其他项不一致，四项须用同一份配置")
    state.step(env)
    i = FALL_LAGRANGE_CONSTRAINTS.index(constraint)
    return float(price_k) * state.lambda_t[i] * state.cost[:, i]


def fall_lagrange_normalize(env: ManagerBasedRlEnv) -> torch.Tensor:
    """M 内总奖励归一化：返回补差的奖励率 (1/(1+Σλ) − 1) · 本步已累加总奖励率，权重须为 1、必须是最后一个奖励项。

    RewardManager 按字典序逐项累加，本项读累加到自己之前的 `_reward_buf`（含 fall_lagrange_* 罚分与 fall_penalty），
    加上补差后 M 内 env 的本步总奖励 = 原总奖励 / (1 + Σλ)，M 外不变。首次调用时核对本项排在最后，不对就报错。
    只在 M 内缩放（不对全部 env 缩放）：全部缩放会把走路奖励缩小 1+Σλ 倍，在优势归一化里被约束项淹没。
    """
    manager = env.reward_manager
    if not getattr(env, _FALL_NORMALIZE_CHECKED_ATTR, False):
        names = list(manager.active_terms)
        if names[-1] != "fall_lagrange_normalize":
            raise ValueError(
                f"fall_lagrange_normalize 必须是最后一个奖励项，实际排在它之后的有 "
                f"{names[names.index('fall_lagrange_normalize') + 1 :]}"
            )
        setattr(env, _FALL_NORMALIZE_CHECKED_ATTR, True)
    state = getattr(env, _FALL_LAGRANGE_STATE_ATTR, None)
    if state is None or state.scale == 1.0:
        return torch.zeros(env.num_envs, device=env.device)
    rate = manager._reward_buf / (float(env.step_dt) if manager._scale_by_dt else 1.0)
    return torch.where(state.active, rate * (state.scale - 1.0), torch.zeros_like(rate))


def fall_lagrange_scale(env: ManagerBasedRlEnv) -> float:
    """当前 M 内奖励缩放 1/(1+Σλ)；还没有拉格朗日状态时为 1。"""
    state = getattr(env, _FALL_LAGRANGE_STATE_ATTR, None)
    return 1.0 if state is None else float(state.scale)


def fall_constraints_armed(env: ManagerBasedRlEnv) -> float:
    """倒地约束是否已武装（拉格朗日乘子开始更新）：未武装 0、已武装 1。"""
    state = getattr(env, _FALL_LAGRANGE_STATE_ATTR, None)
    return 1.0 if state is not None and state.armed else 0.0


_FALL_SMOOTH_PREV_ATTR = "_se3_fall_smooth_prev_action"
_FALL_SMOOTH_PREV_PREV_ATTR = "_se3_fall_smooth_prev_prev_action"


def fall_action_smoothness(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    max_penalty: float = 320.0,
    leg_scale: float = 1.0,
    wheel_scale: float = 2.0,
) -> torch.Tensor:
    """倒地起身时的动作二阶差分罚：算式同 mdp.rewards.action_smoothness，只在 M（倒地出生、尚未起身）里、
    倒地约束武装后才计（2026-10-22 起；之前按倾角门控，会落到起跳 / 陡坡的走路 env 上）。

    没有物理限值，不适合拉格朗日乘子，保留 b3odjuk6 收到 s=1 时的固定权重；武装前为 0，不挡翻身的发现。
    用自己的上一步 / 上上步动作缓存——与 action_smoothness 共用缓存会被同一步两次调用互相覆盖。
    """
    action = env.action_manager.action
    prev = getattr(env, _FALL_SMOOTH_PREV_ATTR, None)
    prev_prev = getattr(env, _FALL_SMOOTH_PREV_PREV_ATTR, None)
    if not isinstance(prev, torch.Tensor) or not isinstance(prev_prev, torch.Tensor):
        setattr(env, _FALL_SMOOTH_PREV_ATTR, action.detach().clone())
        setattr(env, _FALL_SMOOTH_PREV_PREV_ATTR, action.detach().clone())
        return torch.zeros(env.num_envs, device=env.device)
    action_acc = periodic_policy_action_second_difference_torch(
        action, prev, prev_prev, front_action_period=front_action_periods_from_env(env)
    )
    prev_prev.copy_(prev)
    prev.copy_(action.detach())
    penalty = float(leg_scale) * action_acc[:, :4].square().sum(dim=1) + float(
        wheel_scale
    ) * action_acc[:, 4:6].square().sum(dim=1)
    penalty = torch.clamp(penalty, max=float(max_penalty))
    gate = fall_constraint_active(env).float() * fall_constraints_armed(env)
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        if cmd.shape[1] > 5:
            gate = gate * (cmd[:, 5] <= 0.5).float()
    startup = env.episode_length_buf < 3
    return torch.where(startup, torch.zeros_like(penalty), penalty * gate)


__all__ = [
    "FALL_LAGRANGE_CONSTRAINTS",
    "base_height_penalty_off_terrain",
    "base_height_penalty_support_on_terrain",
    "base_height_penalty_window_on_terrain",
    "column_scaled",
    "fall_action_smoothness",
    "fall_constraint_active",
    "fall_constraints_armed",
    "fall_lagrange_normalize",
    "fall_lagrange_penalty",
    "fall_lagrange_scale",
    "is_terminated_except",
    "jump_env_only",
    "off_column",
    "stair_heading_hold",
    "termination_term",
    "tracking_lin_vel_narrow",
    "tracking_lin_vel_terrain_vz",
    "wheel_fore_aft_offset",
]
