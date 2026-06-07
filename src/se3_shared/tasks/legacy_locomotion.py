"""旧行走/跳跃任务的 32D actor 观测契约。"""

from __future__ import annotations

from .base import COMMON_LOCOMOTION_TERMS, ObservationLayout, ObservationTermSpec, TaskContract

LEGACY_LOCOMOTION_CONTRACT = TaskContract(
    name="legacy_locomotion",
    observation=ObservationLayout(
        terms=(*COMMON_LOCOMOTION_TERMS, ObservationTermSpec("jump_commands", 3))
    ),
)

LEGACY_LOCOMOTION_NUM_OBS = LEGACY_LOCOMOTION_CONTRACT.num_obs


__all__ = ["LEGACY_LOCOMOTION_CONTRACT", "LEGACY_LOCOMOTION_NUM_OBS"]
