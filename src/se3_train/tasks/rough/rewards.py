"""本任务的奖励包装：数学全部复用 Flat 基线，只按地形列改生效范围或核参数。

1. `command_velocity_error_on_terrain`：速度违令二次罚，只在指定列生效。Flat 基线已删掉它
   （平地上误差小且短暂，等于奖励指令阶跃后猛冲），但台阶列换列后跟踪误差长期大于 0.4 m/s，
   tracking_lin_vel 的高斯核在那里是平的零、没有梯度（A5 跟踪 2.2 → 0.5 后再没回来），A6 起
   在台阶列加回来；A15 扩到全部六列（生效列由 env_cfg 配）。
2. `base_height_penalty_support_on_terrain`（M21）：机身高度罚，在指定列（上台阶列）把地面参考从 base_link
   正下方射线换成两轮下方射线（轮子支撑面），其余列与 Flat 原函数逐位相同。旧口径下机身沿一越过台阶边，
   参考地面瞬间抬一阶，误差夹满 0.15 就按封顶 −9/s 罚到机身升完这一阶，把"机身先过沿、轮子随后收腿提上来"
   的过渡期罚成了起跳/走梯（见 env_cfg 的 ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS 注释）。
   `base_height_penalty_off_terrain` 是 M1/M2 用过的按列置零包装（M3 起全列生效、不再挂在配置里，留作对照工具）。
3. `tracking_ang_vel_off_terrain`：yaw 跟踪在台阶列置零（A10）。台阶列 yaw 指令恒 0、σ=0.25，
   完全静止就能拿满 73% 的正奖励，站着不动是正收益均衡；指令侧归零 + 奖励侧归零缺一不可。
4. `tracking_lin_vel_terrain_vz`：非平地列关掉核里的 vz 项（A7，爬升必须有垂直速度），台阶列单独
   放宽运动核分母（A11，误差约 1 m/s 时仍有半额奖励）；顺带记按列拆开的速度诊断 `Rough/*_terrain`。
5. `off_column`：把任意 Flat 奖励项在指定列上置零的通用包装，M2 用它在台阶列关掉 is_alive、
   flat_wheel_contact、collision（见 env_cfg.py 的 ROUGH_STAIRS_ZEROED_REWARDS 注释）。
5b. `column_scaled`：同一件事的连续版，指定列乘一个系数而不是置零。M22 用它给窄核速度跟踪分列定权
   （台阶列 w=1、其余四列 w=3，见 env_cfg.py 的 ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT 注释）。
6b. `wheel_height_diff`（M23 引入）：左右轮心的世界系高度差 |Δz| 超出 8 cm 死区的部分取平方，只在上台阶列生效。
   量的是"一只轮已经上了一级、另一只还没上"这件事本身。爬升段 |Δz| 的 p95 把两种流形切得很干净：
   目标流形（M15 2.7/4.2、M18 0.2–2.7、M21-2600 1.2/4.1）≤ 4.2 cm，走梯（M17 14.1/20.4、M22 14.0/18.4）≥ 14 cm；
   而前后错位 Δx 在两组里完全重叠（M15 均值 −10.5/−9.2 比 M22 的 −7.4/−7.8 还大），罚 Δx 会先罚掉目标流形。
6. `wheel_fore_aft_offset`（M18 引入，M19 改形）：左右轮心在机身系里的前后错位 Δx 超出死区 10 cm 的部分取平方，
   几何量、不经关节空间；只在平地列生效（M18 的"全列除台阶"把爬梯的一先一后也压平了）。M17 评测发现策略靠"跨立"
   （左右轮前后错开 20–29 cm）把俯仰平衡变成静定问题来做精确跟踪，joint_mirror −0.179 在 Δx=28 cm
   只花 0.18/s，形同免费（docs/plan/m17_narrow_w3_20260920.md 附录）。

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
    command_velocity_error,
    flat_base_height_penalty_no_jump,
    tracking_ang_vel,
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


def wheel_fore_aft_offset(
    env: ManagerBasedRlEnv,
    apply_type_names: tuple[str, ...] = ("flat",),
    dead_zone_m: float = 0.10,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """左右轮心在机身系里的前后错位 Δx：超出死区的部分取平方（m²），只在指定列生效，直立门控；顺带记 |Δx| 均值。

    M18 用的是"全列（除台阶列）+ 无死区"，结果共享策略把两轮齐平学成全局习惯，台阶列也丢掉了
    M17 那种一先一后的走梯方式，只剩深前倾双轮同抬（docs/plan/m18_wheel_offset_20260921.md 中途核查）。
    M19 起改成只罚平地列、死区 10 cm：静站/平地行驶时 20–29 cm 的错位仍被罚，10 cm 以内不管，
    坡列/下台阶列不罚，爬梯时的一先一后不受影响。没有分列信息（plane）时全体生效。
    """
    robot = env.scene[_DEFAULT_ASSET_CFG.name]
    wheel_b = _wheel_pos_body_frame(env, _DEFAULT_ASSET_CFG)  # [N, 2, 3]，顺序 (左, 右)
    dx = wheel_b[:, 0, 0] - wheel_b[:, 1, 0]
    excess = torch.clamp(dx.abs() - float(dead_zone_m), min=0.0)
    gate = _tracking_upright_gate(robot.data.projected_gravity_b[:, 2], tracking_upright_full_cos)
    penalty = excess.square() * gate
    mask = column_mask(env, apply_type_names)
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        log["Rough/wheel_dx_abs"] = dx.abs().mean()
        if mask is not None:
            keep = mask.float()
            log["Rough/wheel_dx_abs_flat"] = (dx.abs() * keep).sum() / keep.sum().clamp(min=1.0)
            log["Rough/wheel_dx_abs_off_flat"] = (dx.abs() * (1.0 - keep)).sum() / (1.0 - keep).sum().clamp(
                min=1.0
            )
    if mask is None:
        return penalty
    return penalty * mask.float()


def wheel_height_diff(
    env: ManagerBasedRlEnv,
    apply_type_names: tuple[str, ...] = ("stairs_up",),
    dead_zone_m: float = 0.08,
    tracking_upright_full_cos: float = 0.7,
) -> torch.Tensor:
    """左右轮心的世界系高度差 |Δz| 超出死区的部分取平方（m²），只在指定列生效，直立门控。

    M23（2026-09-21 用户定）：堵死"一只轮先上一级、另一只在下一级推地"的走梯。死区 8 cm 留在目标流形之上
    （M15-7999 爬升段 |Δz| p95 只有 2.7–4.2 cm、峰值 5.8），所以爬升本身的摆动免费；
    走梯的 14–20 cm 会被罚 0.14–0.40/s，与台阶列窄核奖励（0.125/s）同量级。
    没有分列信息（plane）时恒 0：这是台阶专项，平面上不该凭空多一项罚。
    """
    robot = env.scene[_DEFAULT_ASSET_CFG.name]
    wheel_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    wheel_z = robot.data.body_link_pos_w[:, wheel_ids, 2]
    dz = wheel_z[:, 0] - wheel_z[:, 1]
    excess = torch.clamp(dz.abs() - float(dead_zone_m), min=0.0)
    gate = _tracking_upright_gate(robot.data.projected_gravity_b[:, 2], tracking_upright_full_cos)
    penalty = excess.square() * gate
    mask = column_mask(env, apply_type_names)
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        log["Rough/wheel_dz_abs"] = dz.abs().mean()
        if mask is not None:
            keep = mask.float()
            log["Rough/wheel_dz_abs_stairs"] = (dz.abs() * keep).sum() / keep.sum().clamp(min=1.0)
    if mask is None:
        return torch.zeros_like(penalty)
    return penalty * mask.float()


def command_velocity_error_on_terrain(
    env: ManagerBasedRlEnv,
    command_name: str,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    lin_vel_scale: float = 0.5,
    yaw_vel_scale: float = 1.0,
    lin_deadband: float = 0.05,
    yaw_deadband: float = 0.10,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """速度违令二次罚，只在指定的子地形列上生效（其余列恒 0）。"""
    penalty = command_velocity_error(
        env,
        command_name=command_name,
        lin_vel_scale=lin_vel_scale,
        yaw_vel_scale=yaw_vel_scale,
        lin_deadband=lin_deadband,
        yaw_deadband=yaw_deadband,
        max_penalty=max_penalty,
    )
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        # 没有分列信息时不生效：Flat 基线里这一项已被删除，默默全局加回来会改掉基线。
        return torch.zeros_like(penalty)
    penalty = penalty * mask.float()
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        log["Rough/command_velocity_error_terrain"] = penalty.sum() / mask.sum().clamp(min=1)
    return penalty


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
) -> torch.Tensor:
    """x 速度跟踪：非平地列把核里的 vz 项换成 `terrain_vz_weight`，台阶列把运动核换成 `stair_sigma_move`。

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
    return reward


__all__ = [
    "base_height_penalty_off_terrain",
    "base_height_penalty_support_on_terrain",
    "wheel_fore_aft_offset",
    "wheel_height_diff",
    "column_scaled",
    "command_velocity_error_on_terrain",
    "off_column",
    "tracking_ang_vel_off_terrain",
    "tracking_lin_vel_narrow",
    "tracking_lin_vel_terrain_vz",
]
