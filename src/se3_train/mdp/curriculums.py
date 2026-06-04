"""SE3 轮腿机器人的课程学习函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from se3_train.mdp.commands import VelocityHeightCommandCfg


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """按训练步数阶梯式扩大速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    step = env.common_step_counter
    for stage in velocity_stages:
        if step >= stage["step"]:
            if "lin_vel_x_range" in stage:
                cfg.lin_vel_x_range = stage["lin_vel_x_range"]
            if "ang_vel_yaw_range" in stage:
                cfg.ang_vel_yaw_range = stage["ang_vel_yaw_range"]
    return {
        "step_counter": torch.tensor(float(step)),
        "lin_vel_x_max": torch.tensor(cfg.lin_vel_x_range[1]),
        "ang_vel_yaw_max": torch.tensor(cfg.ang_vel_yaw_range[1]),
    }


def commands_vel_linear(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start_step: int,
    end_step: int,
    start_lin_vel_x_range: tuple[float, float],
    end_lin_vel_x_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0),
) -> dict[str, torch.Tensor]:
    """按训练步数线性扩大前向速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    step = env.common_step_counter
    span = max(end_step - start_step, 1)
    progress = min(max((step - start_step) / span, 0.0), 1.0)
    cfg.lin_vel_x_range = (
        start_lin_vel_x_range[0] + progress * (end_lin_vel_x_range[0] - start_lin_vel_x_range[0]),
        start_lin_vel_x_range[1] + progress * (end_lin_vel_x_range[1] - start_lin_vel_x_range[1]),
    )
    cfg.ang_vel_yaw_range = ang_vel_yaw_range
    return {
        "step_counter": torch.tensor(float(step)),
        "lin_vel_x_min": torch.tensor(cfg.lin_vel_x_range[0]),
        "lin_vel_x_max": torch.tensor(cfg.lin_vel_x_range[1]),
        "ang_vel_yaw_max": torch.tensor(cfg.ang_vel_yaw_range[1]),
        "progress": torch.tensor(progress),
    }


def terrain_distribution_linear(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    start_proportions: tuple[float, ...],
    end_proportions: tuple[float, ...],
    start_max_level: int = 0,
    end_max_level: int | None = None,
) -> dict[str, torch.Tensor]:
    """按训练步数线性调整地形类型占比和最大难度等级。"""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter)),
            "progress": torch.tensor(0.0),
            "non_flat_ratio": torch.tensor(0.0),
            "max_level": torch.tensor(0.0),
        }

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter)),
            "progress": torch.tensor(0.0),
            "non_flat_ratio": torch.tensor(0.0),
            "max_level": torch.tensor(0.0),
        }

    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    if len(start_proportions) != num_cols or len(end_proportions) != num_cols:
        raise ValueError(
            "地形占比数量必须和 terrain columns 一致: "
            f"start={len(start_proportions)}, end={len(end_proportions)}, cols={num_cols}"
        )

    step = env.common_step_counter
    span = max(end_step - start_step, 1)
    progress = min(max((step - start_step) / span, 0.0), 1.0)

    start_probs = torch.tensor(start_proportions, device=env.device, dtype=torch.float)
    end_probs = torch.tensor(end_proportions, device=env.device, dtype=torch.float)
    probs = start_probs + progress * (end_probs - start_probs)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / torch.clamp(probs.sum(), min=1.0e-6)

    target_end_level = num_rows - 1 if end_max_level is None else min(end_max_level, num_rows - 1)
    max_level = round(start_max_level + progress * (target_end_level - start_max_level))
    max_level = max(0, min(max_level, num_rows - 1))

    terrain.terrain_types[env_ids] = torch.multinomial(probs, len(env_ids), replacement=True)
    terrain.terrain_levels[env_ids] = torch.randint(
        0,
        max_level + 1,
        (len(env_ids),),
        device=env.device,
    )
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    terrain_types = terrain.terrain_types[env_ids]
    non_flat = terrain_types != 0
    obstacle = terrain_types >= 2
    extras = {
        "step_counter": torch.tensor(float(step)),
        "progress": torch.tensor(progress),
        "flat_ratio": torch.mean((terrain_types == 0).float()),
        "rough_ratio": torch.mean((terrain_types == 1).float()),
        "obstacle_ratio": torch.mean(obstacle.float()),
        "non_flat_ratio": torch.mean(non_flat.float()),
        "max_level": torch.tensor(float(max_level)),
    }
    if num_cols >= 3:
        extras["box_ratio"] = torch.mean((terrain_types == 2).float())
    if num_cols >= 4:
        extras["stairs_up_ratio"] = torch.mean((terrain_types == 3).float())
        extras["stairs_ratio"] = torch.mean((terrain_types >= 3).float())
    if num_cols >= 5:
        extras["stairs_down_ratio"] = torch.mean((terrain_types == 4).float())
    return extras


def gait_terrain_distribution_linear(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    start_proportions: tuple[float, ...],
    end_proportions: tuple[float, ...],
    start_max_level: int = 0,
    end_max_level: int | None = None,
) -> dict[str, torch.Tensor]:
    """兼容旧 GAIT 课程名，实际使用通用地形占比课程。"""
    return terrain_distribution_linear(
        env=env,
        env_ids=env_ids,
        start_step=start_step,
        end_step=end_step,
        start_proportions=start_proportions,
        end_proportions=end_proportions,
        start_max_level=start_max_level,
        end_max_level=end_max_level,
    )


def push_disturbance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    push_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """按训练步数逐步增大推扰动强度。

    修改 env 上存储的 push velocity_range 配置。
    push_stages 格式: [{"step": 0, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, ...]
    """
    del env_ids
    step = env.common_step_counter
    current_velocity_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

    for stage in push_stages:
        if step >= stage["step"]:
            current_velocity_range = stage["velocity_range"]

    # interval push 事件读取这个动态配置，避免课程项只更新监控值而不影响实际扰动。
    env._push_velocity_range = current_velocity_range
    current_max = max(
        max(abs(axis_range[0]), abs(axis_range[1]))
        for axis_range in current_velocity_range.values()
    )

    return {
        "step_counter": torch.tensor(float(step)),
        "push_vel_max": torch.tensor(current_max),
    }
