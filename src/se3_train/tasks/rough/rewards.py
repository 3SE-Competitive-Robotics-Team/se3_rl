"""本任务的奖励包装：数学全部复用 Flat 基线，只按地形列改生效范围或核参数。

1. `command_velocity_error_on_terrain`：速度违令二次罚，只在指定列生效。Flat 基线已删掉它
   （平地上误差小且短暂，等于奖励指令阶跃后猛冲），但台阶列换列后跟踪误差长期大于 0.4 m/s，
   tracking_lin_vel 的高斯核在那里是平的零、没有梯度（A5 跟踪 2.2 → 0.5 后再没回来），A6 起
   在台阶列加回来；A15 扩到全部六列（生效列由 env_cfg 配）。
2. `base_height_penalty_off_terrain`：机身高度罚按列置零的包装。M1/M2 在台阶列置零（爬台阶时机身相对
   脚下地面的高度本来就会大幅偏离指令，这项二次罚等于按爬升幅度罚钱，姿态交给地形感知高度下限管）；
   M3 起 env_cfg 传空列名（ROUGH_BASE_HEIGHT_OFF_COLUMNS），包装退化为全列生效的 Flat 原函数。
3. `tracking_ang_vel_off_terrain`：yaw 跟踪在台阶列置零（A10）。台阶列 yaw 指令恒 0、σ=0.25，
   完全静止就能拿满 73% 的正奖励，站着不动是正收益均衡；指令侧归零 + 奖励侧归零缺一不可。
4. `tracking_lin_vel_terrain_vz`：非平地列关掉核里的 vz 项（A7，爬升必须有垂直速度），台阶列单独
   放宽运动核分母（A11，误差约 1 m/s 时仍有半额奖励）；顺带记按列拆开的速度诊断 `Rough/*_terrain`。
5. `off_column`：把任意 Flat 奖励项在指定列上置零的通用包装，M2 用它在台阶列关掉 is_alive、
   flat_wheel_contact、collision（见 env_cfg.py 的 ROUGH_STAIRS_ZEROED_REWARDS 注释）。

掩码为 None（非课程地形、平面地形、列名对不上）时全部退化成 Flat 基线的行为。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.rewards import _tracking_upright_gate
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
    "command_velocity_error_on_terrain",
    "off_column",
    "tracking_ang_vel_off_terrain",
    "tracking_lin_vel_narrow",
    "tracking_lin_vel_terrain_vz",
]
