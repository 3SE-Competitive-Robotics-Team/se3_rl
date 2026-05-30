"""本任务使用的奖励函数。"""

from __future__ import annotations

from se3_train.mdp.rewards import (
    upward,
    upward_progress,
    upright_leg_contact_penalty,
    upright_wheel_contact_penalty,
    upright_wheel_slip_penalty,
)
from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

__all__ = [
    *_FLAT_REWARD_ALL,
    "upward",
    "upward_progress",
    "upright_leg_contact_penalty",
    "upright_wheel_contact_penalty",
    "upright_wheel_slip_penalty",
]
