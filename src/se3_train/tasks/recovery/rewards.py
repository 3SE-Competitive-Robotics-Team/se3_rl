"""本任务使用的奖励函数。"""

from __future__ import annotations

from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL
from se3_train.mdp.rewards import (
    is_alive,
    recovery_height,
    recovery_hard_roll_deep_upright,
    recovery_hard_roll_upright,
    recovery_progress,
    recovery_stable_bonus,
    recovery_upright,
    recovery_wheel_contact,
)

__all__ = [
    *_FLAT_REWARD_ALL,
    "is_alive",
    "recovery_height",
    "recovery_hard_roll_deep_upright",
    "recovery_hard_roll_upright",
    "recovery_progress",
    "recovery_stable_bonus",
    "recovery_upright",
    "recovery_wheel_contact",
]
