"""本任务使用的跳跃终止条件。"""

from __future__ import annotations

from se3_train.mdp.jump_terminations import knee_hyperextension
from se3_train.mdp.terminations import bad_orientation_delayed, leg_contact, time_out

__all__ = ["bad_orientation_delayed", "knee_hyperextension", "leg_contact", "time_out"]
