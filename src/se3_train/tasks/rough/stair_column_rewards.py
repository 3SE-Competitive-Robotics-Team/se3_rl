"""把参考实现的那 10 项台阶奖励原样搬到 stairs_up 列，并提供按列开关任意奖励项的包装器。

来源：se3_rl_competiition 仓库 `cloud-changes` 分支的 `src/se3_train/tasks/stair`。

动机
----
2026-09-11 对照该分支发现：他们台阶任务只有 10 项奖励、权重全在 0.01–1.5 之间；我们 A15 有
25 项、跨 5 个数量级（0.0002–25），而且最重的三项（flat_leg_contact -25、collision -16、
tracking_orientation_l2 -12）他们一项都没有。与其逐项猜哪条有害，不如把 stairs_up 整列
换成他们那 10 项做一次干净对照。

10 项的函数体照抄，不复用我们同名的实现——我们的 `tracking_lin_vel` 带
sigma_move/sigma_stand/vz_weight 三套参数与静站分支，`tracking_ang_vel` 带
sigma_cmd_scale/ratio_blend，复用就不是「一模一样」了。

唯一没照抄的是一个几何常数，见 `CHASSIS_BOTTOM_OFFSET`。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from .rewards import terrain_column_mask

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

CHASSIS_BOTTOM_OFFSET = 0.12
"""base_link 原点到底盘最低平面的竖直距离(m)，取自我们自己的 MJCF。

参考实现用的是它那份模型的实测值 0.1042，**不能照搬**：这是几何量，不是设计选择，照抄会把
离地高度整体算错 1.6 cm。我们的依据是 `ROUGH_BODY_COLLISION_BOTTOM_OFFSET = -0.12`
（COACD 碰撞网格 z 范围 [-0.1376, 0.1118]），与地形感知高度下限用同一个数，保证
「够不够得着台阶」与「底盘离地多少」两处口径一致。
"""

STAIR_PREV_TERRAIN_HEIGHT_ATTR = "_se3_stair_prev_terrain_height"
"""`ref_step_height_progress` 的跨帧缓存；reset 时置 NaN，下次调用重新初始化。"""


# --------------------------------------------------------------------------
# 按列开关：把任意奖励项限制在某些列上生效，或在某些列上置零。
# --------------------------------------------------------------------------
def _gate(
    env: ManagerBasedRlEnv,
    inner: Callable[..., torch.Tensor],
    params: dict[str, Any],
    terrain_type_names: tuple[str, ...],
    active_on_column: bool,
) -> torch.Tensor:
    value = inner(env, **params)
    mask = terrain_column_mask(env, terrain_type_names)
    if mask is None:
        # 没有分列信息（非课程地形，如 play/评测场景）时按列开关无从谈起：
        # 只在该列生效的项返回 0，在该列置零的项原样返回，两边都退化成「没有这一列」。
        return torch.zeros_like(value) if active_on_column else value
    gate = mask if active_on_column else ~mask
    return value * gate.to(value.dtype)


def on_column(
    env: ManagerBasedRlEnv,
    inner: Callable[..., torch.Tensor],
    params: dict[str, Any] | None = None,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
) -> torch.Tensor:
    """只在指定列上生效，其余列恒 0。"""
    return _gate(env, inner, params or {}, terrain_type_names, True)


def off_column(
    env: ManagerBasedRlEnv,
    inner: Callable[..., torch.Tensor],
    params: dict[str, Any] | None = None,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
) -> torch.Tensor:
    """在指定列上置零，其余列逐位不变。"""
    return _gate(env, inner, params or {}, terrain_type_names, False)


# --------------------------------------------------------------------------
# 参考实现的 10 项，函数体照抄。
# --------------------------------------------------------------------------
def _positive_sigma(sigma: float) -> float:
    return max(float(sigma), 1.0e-6)


def ref_tracking_lin_vel(env, command_name: str, sigma: float) -> torch.Tensor:
    """奖励机身 x 方向速度跟踪。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = robot.data.root_link_lin_vel_b[:, 0] - command[:, 0]
    return torch.exp(-(error**2) / _positive_sigma(sigma))


def ref_tracking_ang_vel(env, command_name: str, sigma: float) -> torch.Tensor:
    """奖励机身 yaw 角速度跟踪。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = robot.data.root_link_ang_vel_b[:, 2] - command[:, 1]
    return torch.exp(-(error**2) / _positive_sigma(sigma))


def ref_tracking_orientation_l2(env, command_name: str) -> torch.Tensor:
    """惩罚机身 pitch 与 roll 相对指令的平方误差。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    projected_gravity = robot.data.projected_gravity_b
    pitch = torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0))
    roll = torch.asin(torch.clamp(-projected_gravity[:, 1], -1.0, 1.0))
    return (pitch - command[:, 2]) ** 2 + (roll - command[:, 3]) ** 2


def ref_action_rate(env) -> torch.Tensor:
    """惩罚相邻控制周期的动作变化。"""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(delta**2, dim=1)


def ref_is_alive(env) -> torch.Tensor:
    """存活奖励：每步恒 1。"""
    return torch.ones(env.num_envs, device=env.device)


def ref_climb_progress(
    env, command_name: str, scale: float = 1.0, min_vx: float = 0.1
) -> torch.Tensor:
    """爬升进度奖励：世界系净上升速度，仅在有前进指令时生效（无状态）。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    vz = robot.data.root_link_lin_vel_w[:, 2]
    fwd = (command[:, 0].abs() > min_vx).float()
    return scale * torch.clamp(vz, 0.0, 0.1) * fwd


def terrain_height(env) -> torch.Tensor:
    """脚下台阶面的世界系 z 高度(m)：base_link 世界 z 减去 critic 高度传感器读数。"""
    base_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    clearance = env.scene["critic_height_sensor"].data.heights.squeeze(-1)
    return base_z - clearance


def chassis_clearance(env) -> torch.Tensor:
    """车底盘平面离地高度(m)。几何常数用我们自己的 `CHASSIS_BOTTOM_OFFSET`。"""
    base_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return base_z - CHASSIS_BOTTOM_OFFSET - terrain_height(env)


def ref_tracking_height(env, command_name: str, sigma: float) -> torch.Tensor:
    """楼梯专用高度跟踪：用脚下 clearance 与指令比对，台阶绝对高度自动抵消。

    注意这是**指数正奖励**（上限 1.0），不是我们 `flat_base_height` 那种无界二次罚。
    2026-09-11 的奖励账本显示，二次罚形式下走路必然产生的机身起伏一项就吃掉 4.15/秒，
    而速度跟踪满分才 4.0/秒——正奖励形式结构上不会出现这种"走路永远不划算"。
    """
    command = env.command_manager.get_command(command_name)
    clearance = env.scene["critic_height_sensor"].data.heights.squeeze(-1)
    error = clearance - command[:, 4]
    return torch.exp(-error * error / sigma / sigma)


def ref_step_height_progress(
    env,
    command_name: str,
    scale: float = 1.0,
    min_vx: float = 0.1,
    max_step: float = 0.05,
) -> torch.Tensor:
    """净爬升进度（带跨帧缓存）：真踩上更高一级台阶才发糖，比纯 vz 稳且防抖骗奖。"""
    command = env.command_manager.get_command(command_name)
    now = terrain_height(env)

    prev = getattr(env, STAIR_PREV_TERRAIN_HEIGHT_ATTR, None)
    if not isinstance(prev, torch.Tensor) or prev.shape[0] != env.num_envs:
        before = now.clone()
    else:
        before = prev.clone()
        nan_mask = torch.isnan(before)
        if torch.any(nan_mask):
            before[nan_mask] = now[nan_mask]

    delta = now - before
    fwd = (command[:, 0].abs() > min_vx).float()
    reward = scale * torch.clamp(delta, 0.0, max_step) * fwd
    setattr(env, STAIR_PREV_TERRAIN_HEIGHT_ATTR, now.clone())
    return reward


def ref_chassis_clearance_penalty(env, weight: float = 1.0, sigma: float = 0.08) -> torch.Tensor:
    """底盘不刮地惩罚：`-weight * exp(-clearance / sigma)`，离地越小罚越重。"""
    return -weight * torch.exp(-chassis_clearance(env) / float(sigma))


def ref_leg_alignment_penalty(
    env,
    min_lateral_distance: float = 0.40,
    max_lateral_distance: float = 0.46,
    max_fore_aft_offset: float = 0.06,
    lateral_scale: float = 0.04,
    fore_aft_scale: float = 0.03,
    fore_aft_weight: float = 1.5,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """逼左右腿前后对称（治「一前一后斜着画弧」）。

    罚的是左右轮在**车身系的前后错位距离**，不是关节角之和。2026-09-11 实测：卡死构型的
    `joint_mirror`（rod1+rod2）只有 0.28，比正常爬台阶时的 0.34 还小——那个量抓不住这个
    模态，而前后错位能抓住（卡死时两杆共模差 1.71 弧度）。
    """
    from se3_train.mdp.leg_alignment import wheel_alignment_penalty

    penalty, _, _, _ = wheel_alignment_penalty(
        env,
        min_lateral_distance=min_lateral_distance,
        max_lateral_distance=max_lateral_distance,
        max_fore_aft_offset=max_fore_aft_offset,
        lateral_scale=lateral_scale,
        fore_aft_scale=fore_aft_scale,
        fore_aft_weight=fore_aft_weight,
        max_penalty=max_penalty,
    )
    return penalty


def reset_step_height_cache(env, env_ids) -> None:
    """reset 时把缓存对应行置 NaN，避免首帧拿上一局的脏值白发一次奖励。"""
    prev = getattr(env, STAIR_PREV_TERRAIN_HEIGHT_ATTR, None)
    if isinstance(prev, torch.Tensor) and prev.shape[0] == env.num_envs:
        prev[env_ids] = float("nan")


__all__ = [
    "CHASSIS_BOTTOM_OFFSET",
    "STAIR_PREV_TERRAIN_HEIGHT_ATTR",
    "chassis_clearance",
    "off_column",
    "on_column",
    "ref_action_rate",
    "ref_chassis_clearance_penalty",
    "ref_climb_progress",
    "ref_is_alive",
    "ref_leg_alignment_penalty",
    "ref_step_height_progress",
    "ref_tracking_ang_vel",
    "ref_tracking_height",
    "ref_tracking_lin_vel",
    "ref_tracking_orientation_l2",
    "reset_step_height_cache",
    "terrain_height",
]
