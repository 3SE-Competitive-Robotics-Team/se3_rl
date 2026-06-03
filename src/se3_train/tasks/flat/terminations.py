"""本任务使用的终止条件。"""

from __future__ import annotations

from se3_train.mdp.terminations import bad_orientation_delayed, leg_contact, time_out

__all__ = ["bad_orientation_delayed", "leg_contact", "time_out"]
