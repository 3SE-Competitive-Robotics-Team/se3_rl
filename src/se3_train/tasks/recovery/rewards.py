"""本任务使用的奖励函数。"""

from __future__ import annotations

from se3_train.mdp.rewards import (
    recovery_hard_roll_success,
    recovery_hard_roll_upright,
    recovery_height,
    recovery_stability,
    recovery_success,
    recovery_upright,
    recovery_wheel_contact,
)
from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

__all__ = [
    *_FLAT_REWARD_ALL,
    "recovery_hard_roll_success",
    "recovery_hard_roll_upright",
    "recovery_height",
    "recovery_stability",
    "recovery_success",
    "recovery_upright",
    "recovery_wheel_contact",
]
