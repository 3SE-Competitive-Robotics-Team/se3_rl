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
