"""本任务使用的终止条件：平地那套 + 墙阻挡转 time_out。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.tasks.flat.terminations import *  # noqa: F403
from se3_train.tasks.flat.terminations import __all__ as _FLAT_ALL

from .commands import WALL_BLOCKED_ATTR

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def wall_blocked(env: ManagerBasedRlEnv) -> torch.Tensor:
    """前方障碍高到跨不过去时结束该 episode。

    对应参考仓库 `StepUpStateMachine.apply_done_masks` 的
    `time_out |= wall_reset`：地形不可通过不是策略失败，注册时必须带
    `time_out=True`，让 PPO 按截断 bootstrap 而不是按失败记 0 值。
    标志由 `StepUpCommandTerm` 每步写入 env 属性。
    """
    flag = getattr(env, WALL_BLOCKED_ATTR, None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return flag


__all__ = [*_FLAT_ALL, "wall_blocked"]
