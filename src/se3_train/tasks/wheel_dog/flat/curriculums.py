"""WheelDog 平地任务课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    """按全局 step 阶段推进 XY 速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    step = int(env.common_step_counter)
    active = velocity_stages[0]
    for stage in velocity_stages:
        if step >= int(stage["step"]):
            active = stage
    term.cfg.lin_vel_x_range = tuple(active["lin_vel_x_range"])
    term.cfg.lin_vel_y_range = tuple(active["lin_vel_y_range"])
    term.cfg.ang_vel_yaw_range = tuple(active.get("ang_vel_yaw_range", (0.0, 0.0)))
    return {
        "cmd_x_limit": torch.tensor(abs(term.cfg.lin_vel_x_range[1]), device=env.device),
        "cmd_y_limit": torch.tensor(abs(term.cfg.lin_vel_y_range[1]), device=env.device),
        "cmd_stage_step": torch.tensor(float(active["step"]), device=env.device),
    }


def push_disturbance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    push_stages: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
    """按全局 step 阶段推进外部速度扰动范围。"""
    del env_ids
    step = int(env.common_step_counter)
    active = push_stages[0]
    for stage in push_stages:
        if step >= int(stage["step"]):
            active = stage
    env._wheel_dog_push_velocity_range = active["velocity_range"]
    vx = env._wheel_dog_push_velocity_range.get("x", (0.0, 0.0))
    vy = env._wheel_dog_push_velocity_range.get("y", (0.0, 0.0))
    return {
        "push_x_limit": torch.tensor(abs(vx[1]), device=env.device),
        "push_y_limit": torch.tensor(abs(vy[1]), device=env.device),
        "push_stage_step": torch.tensor(float(active["step"]), device=env.device),
    }


__all__ = ["commands_vel", "push_disturbance"]
