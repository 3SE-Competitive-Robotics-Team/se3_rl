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

上面两个包装都只做掩码乘法，不改被包装函数的任何参数；掩码为 None（非课程地形、平面地形、
列名对不上）时退化成 Flat 基线的行为：速度罚不生效、高度罚照常。

3. **非平地列把 `tracking_lin_vel` 的 vz 项关掉（2026-09-08 用户定，A7）。** 核是
   `exp(-(err_x² + vz_weight·vz²)/σ)`，vz 是机身垂直速度。爬台阶和上坡**必须**有垂直速度，
   而这一项按 vz² 扣分：vz 0.2 m/s 就把核乘掉 0.37，斜坡上 1 m/s 走 16° 坡的 vz 是 0.28。
   这是 7280006（"vz 折入 tracking 核、删掉独立 lin_vel_z 项"）给平地设计的——平地上 vz 本该是 0。
   平地列保持 2.0 不变，非平地列取 0。逐 env 的权重张量喂给同一个函数，不复制任何观测或奖励数学。

   顺带在这里记按列拆开的诊断（`Locomotion/*` 是全体均值，看不出地形列到底差多少）：
   `Rough/tracking_lin_vel_{flat,terrain}`、`Rough/{cmd_vx,base_vx,base_vx_error}_terrain`。
   A6 只能从 `Rough/command_velocity_error_terrain` 反解出地形列误差 ≈1.5 m/s，太绕。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_ALL
from se3_train.tasks.flat.rewards import (
    command_velocity_error,
    flat_base_height_penalty_no_jump,
    tracking_ang_vel,
    tracking_lin_vel,
)

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
    """yaw 角速度跟踪，在指定子地形列上置零，其余列与 Flat 基线逐位相同。

    2026-09-09 用户定（A10）。A9 的逐项拆分：台阶列 `tracking_ang_vel` = +2.739/s，
    占该列全部正奖励 3.753 的 73%，而它是**静止就能拿满**的——yaw 指令 ±0.2、σ=0.25，
    不动时误差约 0.1、核值 0.96。配上 is_alive 的 +1.0，站着不动净收益 +0.053/s 为正，
    而爬台阶要拿摔倒的风险去换 command_velocity_error 那点梯度，理性选择就是不动。
    连同 `stair_ang_vel_yaw_range=(0,0)`（指令侧）一起，把这份「不动的工资」彻底取消。
    """
    reward = tracking_ang_vel(
        env,
        command_name=command_name,
        sigma=sigma,
        sigma_cmd_scale=sigma_cmd_scale,
        ratio_blend=ratio_blend,
        use_upright_gate=use_upright_gate,
        tracking_upright_full_cos=tracking_upright_full_cos,
    )
    mask = terrain_column_mask(env, terrain_type_names)
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


def non_flat_column_mask(
    env: ManagerBasedRlEnv,
    flat_type_names: tuple[str, ...] = ("flat",),
) -> torch.Tensor | None:
    """返回“不在平地列”的 env 掩码；非课程地形时返回 None。

    与 commands.RoughCommandTerm._build_terrain_override_mask 同一套口径：分列定价按
    `terrain_type_names` 点名生效列，而 vz 项是“凡是要爬升的列都关”，用取反更稳
    （新增子地形时不用记得来加名字）。
    """
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if generator is None or terrain_types is None or not generator.curriculum:
        return None
    names = list(generator.sub_terrains.keys())
    flat_cols = [names.index(n) for n in flat_type_names if n in names]
    if not flat_cols:
        return None
    types = terrain_types.to(device=env.device, dtype=torch.long)
    is_flat = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for col in flat_cols:
        is_flat |= types == col
    return ~is_flat


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
    """x 速度跟踪，非平地列把核里的 vz 项换成 `terrain_vz_weight`（默认 0）。

    逐 env 的权重张量直接喂给 `tracking_lin_vel`，核里 `vz_weight * vz**2` 按元素广播，
    观测、静站判定、课程累加与 `Locomotion/*` 记账均复用 Flat 基线。
    台阶列还可通过 stair_sigma_move 单独设置运动核分母，其余列和静站核保持原值。
    掩码为 None 时退化成 Flat 行为。
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
    stair_mask = (
        terrain_column_mask(env, stair_type_names) if stair_sigma_move is not None else None
    )
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
    *_FLAT_ALL,
    "base_height_penalty_off_terrain",
    "tracking_ang_vel_off_terrain",
    "command_velocity_error_on_terrain",
    "non_flat_column_mask",
    "terrain_column_mask",
    "tracking_lin_vel_terrain_vz",
]
