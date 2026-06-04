"""FlowMatch task 组使用的终止条件。"""

from __future__ import annotations

from se3_train.mdp.terminations import (
    BodyContactPenalty,
    bad_orientation_delayed,
    base_link_contact_delayed,
    gait_low_base_height_delayed,
    leg_contact,
    time_out,
)

__all__ = [
    "BodyContactPenalty",
    "bad_orientation_delayed",
    "base_link_contact_delayed",
    "gait_low_base_height_delayed",
    "leg_contact",
    "time_out",
]
