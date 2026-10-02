"""本任务的奖励包装：数学全部复用 Flat 基线，只按地形列改生效范围或核参数。

1. （已删除）`command_velocity_error_on_terrain`：A6–M35 用的速度违令二次罚。M37 起删除：lin_vel_scale 3.0 下
   1.0 m/s 误差只罚 0.20、斜率不到 1，是常数税不是远端梯度（docs/plan/m37_reward_prune_20260927.md）。
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
   M37 起删除：用户决定对称只保留 joint_mirror 一条关节空间定价。推导见 docs/plan/m18_wheel_offset_20260921.md、
   m23_wheel_dz_20260921.md；复现用 commit 5af0d00（M35，最后一个带这两项的入口）。

掩码为 None（非课程地形、平面地形、列名对不上）时全部退化成 Flat 基线的行为。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.rewards import (
    _DEFAULT_ASSET_CFG,
    _recovery_reset_mask,
    _tracking_upright_gate,
    _wheel_pos_body_frame,
)
from se3_train.mdp.terrain_height import frame_height_above_terrain, ground_height_estimate
from se3_train.tasks.flat.rewards import (
    flat_base_height_penalty_no_jump,
    tracking_ang_vel,
    tracking_lin_vel,
)

from .columns import column_mask, non_flat_column_mask

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def tracking_lin_vel_enhance(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 2.5,
) -> torch.Tensor:
    """复旦 `_reward_tracking_lin_vel_enhance`：exp(−e²/σ) − 1，e 为机身系 vx 与指令之差。

    值域 (−1, 0]，是"宽核减一"的有界罚：小误差处接近 0，大误差处给 −1 的常数梯度尾巴。
    与 mdp.rewards.tracking_lin_vel（σ 0.25、复旦尖核）成对使用（M43，2026-09-28）；σ 是指数分母。
    全列生效、无门控、无 vz 项，与复旦一致。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    error = robot.data.root_link_lin_vel_b[:, 0] - cmd[:, 0]
    return torch.exp(-error.square() / float(sigma)) - 1.0


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


def tracking_ang_vel_off_terrain(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    sigma_cmd_scale: float = 0.0,
    ratio_blend: float = 0.0,
    use_upright_gate: bool = True,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """yaw 角速度跟踪，在指定子地形列上置零，其余列与 Flat 基线逐位相同。"""
    reward = tracking_ang_vel(
        env,
        command_name=command_name,
        sigma=sigma,
        sigma_cmd_scale=sigma_cmd_scale,
        ratio_blend=ratio_blend,
        use_upright_gate=use_upright_gate,
        tracking_upright_full_cos=tracking_upright_full_cos,
    )
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return reward
    return reward * (~mask).float()


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


def _fudan_height_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    window_sensor_name: str,
    stair_type_names: tuple[str, ...],
    stair_support_sensor_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """复旦式高度对的误差 e = 机身 z − 地面参考 − 高度指令，返回 (e, 窗口口径 e, 活跃掩码, 台阶列掩码)。

    地面参考默认是机身周围 77 点窗口均值（M45）。`stair_support_sensor_name` 非空时台阶类列改用两轮轮心正下方
    的支撑面均值（M46，2026-09-29 用户定）：M45-1000 回放里窗口口径在立面前被上一阶抬高约半阶（9 级 −10.5 cm），
    而支撑面口径只有 −1…−3 cm；两轮不在同一阶时取两阶平均，对分腿收温和的钱。活跃 = 非跳跃、非恢复 reset。
    """
    cmd = env.command_manager.get_command(command_name)
    active = (~(cmd[:, 5] > 0.5)) & (~_recovery_reset_mask(env))
    frame_z = env.scene[height_sensor_name].data.frame_pos_w[:, 0, 2]
    window_error = frame_z - ground_height_estimate(env, window_sensor_name) - cmd[:, 4]
    mask = column_mask(env, stair_type_names)
    error = window_error
    if stair_support_sensor_name is not None and mask is not None:
        support_error = frame_z - ground_height_estimate(env, stair_support_sensor_name) - cmd[:, 4]
        error = torch.where(mask, support_error, window_error)
    return error, window_error, active, mask


def base_height_fudan(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    window_sensor_name: str,
    stair_type_names: tuple[str, ...] = ("stairs_up",),
    stair_support_sensor_name: str | None = None,
) -> torch.Tensor:
    """复旦上台阶3 `_reward_base_height`（正权重分支）：1.5·exp(−1000·e²)，全列统一、无死区（M45，2026-09-29）。

    按原式、不做复旦的逐项每秒 ±1 裁剪（用户定），峰值 1.5；核宽约 3.2 cm，1–5 cm 区间损失涨得最快。
    地面参考见 `_fudan_height_error`（M45 全列窗口口径；M46 台阶类列改支撑面）。
    顺带记全列与台阶列的带符号误差（计酬口径），以及台阶列窗口口径误差（与 M45 可比）。
    """
    error, window_error, active, mask = _fudan_height_error(
        env,
        command_name,
        height_sensor_name,
        window_sensor_name,
        stair_type_names,
        stair_support_sensor_name,
    )
    reward = 1.5 * torch.exp(-1000.0 * error.square())
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        act = active.float()
        n = act.sum().clamp(min=1.0)
        log["Rough/base_height_err_window_all"] = (error * act).sum() / n
        log["Rough/base_height_abs_err_window_all"] = (error.abs() * act).sum() / n
        if mask is not None:
            keep = (mask & active).float()
            m = keep.sum().clamp(min=1.0)
            log["Rough/base_height_err_window_stairs_signed"] = (window_error * keep).sum() / m
            log["Rough/base_height_err_window_stairs"] = (window_error.abs() * keep).sum() / m
            log["Rough/base_height_err_reward_stairs_signed"] = (error * keep).sum() / m
    return reward * active.float()


def base_height_fudan_enhance(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    window_sensor_name: str,
    stair_type_names: tuple[str, ...] = ("stairs_up",),
    stair_support_sensor_name: str | None = None,
) -> torch.Tensor:
    """复旦上台阶3 `_reward_base_height_enhance`：exp(−e²/0.01) − 1，值域 (−1, 0]，全列统一（M45）；参考同 base_height_fudan。"""
    error, _, active, _ = _fudan_height_error(
        env,
        command_name,
        height_sensor_name,
        window_sensor_name,
        stair_type_names,
        stair_support_sensor_name,
    )
    return (torch.exp(-error.square() / 0.01) - 1.0) * active.float()


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
            torch.tensor(float(terrain_vz_weight), device=env.device),
            torch.tensor(float(vz_weight), device=env.device),
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
    """非超时终止，但不算 exclude_terms 里的终止项（RJ1：偏离参考提前终止不吃 −500 摔倒罚）。"""
    manager = env.termination_manager
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in manager.active_terms:
        if name in exclude_terms or manager.get_term_cfg(name).time_out:
            continue
        done |= manager.get_term(name)
    return done.float()


def jump_env_only(
    env: ManagerBasedRlEnv, inner, params: dict, command_name: str = "velocity_height"
) -> torch.Tensor:
    """任意奖励项只在跳跃样本上计（RJ1：模仿奖励的站姿帧是 0.22 m，非跳跃样本高度指令 0.20–0.38 不能被它拉向站姿）。"""
    return inner(env, **params) * env.command_manager.get_term(command_name).jump_env.float()


__all__ = [
    "base_height_fudan",
    "base_height_fudan_enhance",
    "base_height_penalty_off_terrain",
    "base_height_penalty_support_on_terrain",
    "base_height_penalty_window_on_terrain",
    "column_scaled",
    "is_terminated_except",
    "jump_env_only",
    "off_column",
    "stair_heading_hold",
    "tracking_ang_vel_off_terrain",
    "tracking_lin_vel_enhance",
    "tracking_lin_vel_narrow",
    "tracking_lin_vel_terrain_vz",
    "wheel_fore_aft_offset",
]
