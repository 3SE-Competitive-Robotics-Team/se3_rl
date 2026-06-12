"""本任务使用的奖励函数。"""

from __future__ import annotations

from se3_train.mdp.rewards import (
    action_smoothness,
    leg_action_rate,
    leg_contact_penalty,
    recovery_diagnostics,
    recovery_upright_orientation_l2,
    recovery_upright_zero_velocity_penalty,
    upright_leg_contact_penalty,
    upright_wheel_contact_penalty,
    upright_wheel_slip_penalty,
    upward,
    upward_progress,
    wheel_action_rate,
    wheel_air_velocity_penalty,
)
from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

__all__ = [
    *_FLAT_REWARD_ALL,
    "action_smoothness",
    "leg_action_rate",
    "leg_contact_penalty",
    "recovery_diagnostics",
    "recovery_upright_orientation_l2",
    "recovery_upright_zero_velocity_penalty",
    "upward",
    "upward_progress",
    "upright_leg_contact_penalty",
    "upright_wheel_contact_penalty",
    "upright_wheel_slip_penalty",
    "wheel_action_rate",
    "wheel_air_velocity_penalty",
]
