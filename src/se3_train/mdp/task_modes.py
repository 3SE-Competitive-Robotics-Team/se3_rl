"""Task Mode 门控工具。

奖励函数只通过这些 helper 判断当前模式，避免在各处散写 command 列索引。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import TaskMode

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


TASK_MODE_ID_INDEX = 8
MODE_BLEND_INDEX = 9
PREV_TASK_MODE_ID_INDEX = 10


def task_mode_ids(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """读取当前 Task Mode id。旧 8 维指令默认视为 jump/wheel 二元模式。"""
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] > TASK_MODE_ID_INDEX:
        return cmd[:, TASK_MODE_ID_INDEX].round().long()
    jump_flag = (
        cmd[:, 5] > 0.5 if cmd.shape[1] > 5 else torch.zeros(env.num_envs, device=cmd.device)
    )
    return torch.where(
        jump_flag,
        torch.full_like(jump_flag, int(TaskMode.JUMP), dtype=torch.long),
        torch.full_like(jump_flag, int(TaskMode.WHEEL), dtype=torch.long),
    )


def mode_mask(env: ManagerBasedRlEnv, command_name: str, *modes: TaskMode | int) -> torch.Tensor:
    """返回指定 Task Mode 的布尔掩码。"""
    ids = task_mode_ids(env, command_name)
    result = torch.zeros_like(ids, dtype=torch.bool)
    for mode in modes:
        result |= ids == int(mode)
    return result


def mode_weight(env: ManagerBasedRlEnv, command_name: str, *modes: TaskMode | int) -> torch.Tensor:
    """返回指定 Task Mode 的 float gate。

    11 维 TaskMode 指令使用 mode_blend 在旧 mode 与新 mode 间线性平滑。
    旧 8 维跳跃指令保持硬 gate。
    """
    cmd = env.command_manager.get_command(command_name)
    if cmd.shape[1] <= PREV_TASK_MODE_ID_INDEX:
        return mode_mask(env, command_name, *modes).float()

    current = cmd[:, TASK_MODE_ID_INDEX].round().long()
    previous = cmd[:, PREV_TASK_MODE_ID_INDEX].round().long()
    blend = cmd[:, MODE_BLEND_INDEX].clamp(0.0, 1.0)
    result = torch.zeros(env.num_envs, device=cmd.device)
    for mode in modes:
        mode_id = int(mode)
        result += (previous == mode_id).float() * (1.0 - blend)
        result += (current == mode_id).float() * blend
    return torch.clamp(result, 0.0, 1.0)


def jump_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """返回 jump mode 掩码。"""
    return mode_mask(env, command_name, TaskMode.JUMP)


def non_jump_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """返回非 jump mode 掩码。"""
    return ~jump_mask(env, command_name)


__all__ = [
    "MODE_BLEND_INDEX",
    "PREV_TASK_MODE_ID_INDEX",
    "TASK_MODE_ID_INDEX",
    "jump_mask",
    "mode_mask",
    "mode_weight",
    "non_jump_mask",
    "task_mode_ids",
]
