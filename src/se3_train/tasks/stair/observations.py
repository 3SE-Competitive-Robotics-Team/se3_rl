"""台阶任务观测函数。"""

from __future__ import annotations

from se3_train.mdp.observations import _finite_clamp
from se3_train.tasks.flat.observations import *  # noqa: F403
from se3_train.tasks.flat.observations import __all__ as _FLAT_OBS_ALL


def last_actions_obs(env):
    """上一步实际下发的 6 维动作，包含 CTBC 前馈注入。"""
    return _finite_clamp(env.action_manager.get_term("delayed_action").raw_action)


__all__ = [*_FLAT_OBS_ALL, "last_actions_obs"]
