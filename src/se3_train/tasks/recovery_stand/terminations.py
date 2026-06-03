"""本任务使用的终止条件。"""

from __future__ import annotations

from se3_train.mdp.terminations import recovery_success
from se3_train.tasks.recovery.terminations import *  # noqa: F403
from se3_train.tasks.recovery.terminations import __all__ as _RECOVERY_TERMINATION_ALL

__all__ = [*_RECOVERY_TERMINATION_ALL, "recovery_success"]
