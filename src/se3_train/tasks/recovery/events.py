"""本任务使用的事件函数。"""

from __future__ import annotations

from se3_train.mdp.events import reset_root_state_full_angle_random
from se3_train.tasks.flat.events import *  # noqa: F403
from se3_train.tasks.flat.events import __all__ as _FLAT_EVENT_ALL

__all__ = [*_FLAT_EVENT_ALL, "reset_root_state_full_angle_random"]
