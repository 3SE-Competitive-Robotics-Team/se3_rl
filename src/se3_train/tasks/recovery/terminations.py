"""本任务使用的终止条件。"""

from __future__ import annotations

from se3_train.tasks.flat.terminations import *  # noqa: F403
from se3_train.tasks.flat.terminations import __all__ as _FLAT_TERMINATION_ALL
from se3_train.mdp.terminations import recovery_stagnation

__all__ = [*_FLAT_TERMINATION_ALL, "recovery_stagnation"]
