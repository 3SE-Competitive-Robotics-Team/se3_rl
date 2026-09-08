"""本任务使用的奖励函数：平地那套 + 两个按地形列开关的包装。

2026-09-08 用户定（A6）：台阶列的定价与平地列分开，只动两项，都不改数值只改生效范围。

1. **速度违令二次罚只在台阶列加回来。** `command_velocity_error` 是 52ba696 为了补
   `tracking_lin_vel` 高斯核（σ_move=0.08）在误差 >0.4 m/s 处的零梯度而加的，2026-09-06
   （D7 对 D4）在平地上删掉成为 Flat 默认——删它的依据全部来自平地：那里误差小且短暂，
   99% 的代价来自指令阶跃后 1 s 内，等于在奖励"指令一跳就猛冲"。台阶列是另一个区间：
   换列后跟踪误差长期大于 0.4 m/s，高斯核在那儿是平的零，策略拿不到"往指令方向靠"的梯度
   （A5：跟踪 2.2 → 0.5 后再没回来）。所以按列分开定价，平地列仍然不带这一项。

2. **机身高度罚在台阶列置零。** `flat_base_height` 是 `(clamp(err, ±0.15) / 0.05)²` 的无界
   二次罚，误差 0.15 m 就已经 36/s。爬台阶时机身相对脚下地面的高度本来就会大幅偏离指令，
   这项罚等于按爬升幅度罚钱。置零后台阶列的姿态改由 AMP 风格奖励和地形感知高度下限
   （见 commands.py）来管。

两个包装都只做掩码乘法，不改被包装函数的任何参数；掩码为 None（非课程地形、平面地形、
列名对不上）时退化成 Flat 基线的行为：速度罚不生效、高度罚照常。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_ALL
from se3_train.tasks.flat.rewards import command_velocity_error, flat_base_height_penalty_no_jump

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def terrain_column_mask(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...],
) -> torch.Tensor | None:
    """返回“在这些子地形列上”的 env 掩码；非课程地形或列名对不上时返回 None。

    只在课程模式（每种子地形独占一列）下有定义：`terrain_types` 即列号，列号对应
    `sub_terrains` 的键序。与 events.set_curriculum_env_mask、ctbc._terrain_inactive_mask
    同一套口径。
    """
    if not terrain_type_names:
        return None
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if generator is None or terrain_types is None or not generator.curriculum:
        return None
    names = list(generator.sub_terrains.keys())
    cols = [names.index(n) for n in terrain_type_names if n in names]
    if not cols:
        return None
    types = terrain_types.to(device=env.device, dtype=torch.long)
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for col in cols:
        mask |= types == col
    return mask


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
    mask = terrain_column_mask(env, terrain_type_names)
    if mask is None:
        # 没有分列信息时不生效：Flat 基线里这一项已被删除，默默全局加回来会改掉基线。
        return torch.zeros_like(penalty)
    penalty = penalty * mask.float()
    # 基类记的 Locomotion/command_velocity_error_* 是全体均值，被没生效的列稀释了，
    # 这里单独记一份只看生效列的。
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        active = mask.float()
        log["Rough/command_velocity_error_terrain"] = penalty.sum() / active.sum().clamp(min=1.0)
    return penalty


def base_height_penalty_off_terrain(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    sigma: float = 0.05,
    max_error: float | None = 0.15,
) -> torch.Tensor:
    """机身高度 L2 罚，在指定的子地形列上置零，其余列与 Flat 基线逐位相同。"""
    penalty = flat_base_height_penalty_no_jump(
        env,
        command_name=command_name,
        height_sensor_name=height_sensor_name,
        sigma=sigma,
        max_error=max_error,
    )
    mask = terrain_column_mask(env, terrain_type_names)
    if mask is None:
        return penalty
    return penalty * (~mask).float()


__all__ = [
    *_FLAT_ALL,
    "base_height_penalty_off_terrain",
    "command_velocity_error_on_terrain",
    "terrain_column_mask",
]
