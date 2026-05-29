"""本任务使用的奖励函数。"""

from __future__ import annotations

from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL
from se3_train.mdp.rewards import upward

__all__ = [*_FLAT_REWARD_ALL, "upward"]
