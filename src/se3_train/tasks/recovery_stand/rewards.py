"""本任务使用的奖励函数。"""

from __future__ import annotations

from se3_train.tasks.recovery.rewards import *  # noqa: F403
from se3_train.tasks.recovery.rewards import __all__ as _RECOVERY_REWARD_ALL
from se3_train.mdp.rewards import (
    recovery_height,
    recovery_stand_default_joint_pos,
    recovery_stand_joint_mirror,
    recovery_stand_orientation_penalty,
    recovery_stand_leg_alignment,
    recovery_stand_nonwheel_clearance,
    recovery_stand_stillness,
    recovery_stand_wheel_contact,
    recovery_stand_zero_velocity_penalty,
    recovery_success_bonus,
)

__all__ = [
    *_RECOVERY_REWARD_ALL,
    "recovery_height",
    "recovery_stand_default_joint_pos",
    "recovery_stand_joint_mirror",
    "recovery_stand_orientation_penalty",
    "recovery_stand_leg_alignment",
    "recovery_stand_nonwheel_clearance",
    "recovery_stand_stillness",
    "recovery_stand_wheel_contact",
    "recovery_stand_zero_velocity_penalty",
    "recovery_success_bonus",
]
