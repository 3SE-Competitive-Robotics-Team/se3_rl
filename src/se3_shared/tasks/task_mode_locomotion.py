"""TaskMode 统一任务的 42D actor 观测契约。"""

from __future__ import annotations

from .base import COMMON_LOCOMOTION_TERMS, ObservationLayout, ObservationTermSpec, TaskContract

TASK_MODE_TERM_NAME = "task_mode"
TASK_MODE_TAIL_DIM = 13

TASK_MODE_LOCOMOTION_CONTRACT = TaskContract(
    name="task_mode_locomotion",
    observation=ObservationLayout(
        terms=(
            *COMMON_LOCOMOTION_TERMS,
            ObservationTermSpec(TASK_MODE_TERM_NAME, TASK_MODE_TAIL_DIM),
        )
    ),
    is_task_mode=True,
)

TASK_MODE_LOCOMOTION_NUM_OBS = TASK_MODE_LOCOMOTION_CONTRACT.num_obs
TASK_MODE_LOCOMOTION_TASK_MODE_SLICE = TASK_MODE_LOCOMOTION_CONTRACT.observation.require_slice(
    TASK_MODE_TERM_NAME
)


__all__ = [
    "TASK_MODE_LOCOMOTION_CONTRACT",
    "TASK_MODE_LOCOMOTION_NUM_OBS",
    "TASK_MODE_LOCOMOTION_TASK_MODE_SLICE",
    "TASK_MODE_TAIL_DIM",
    "TASK_MODE_TERM_NAME",
]
