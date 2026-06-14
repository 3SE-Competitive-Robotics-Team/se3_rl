"""本任务使用的终止条件。"""

from __future__ import annotations

from se3_train.mdp.terminations import catastrophic_state, recovery_stagnation
from se3_train.tasks.flat.terminations import *  # noqa: F403
from se3_train.tasks.flat.terminations import __all__ as _FLAT_TERMINATION_ALL

__all__ = [*_FLAT_TERMINATION_ALL, "catastrophic_state", "recovery_stagnation"]
