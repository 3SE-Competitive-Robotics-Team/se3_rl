"""任务级共享契约。"""

from .base import COMMON_LOCOMOTION_TERMS, ObservationLayout, ObservationTermSpec, TaskContract
from .legacy_locomotion import LEGACY_LOCOMOTION_CONTRACT, LEGACY_LOCOMOTION_NUM_OBS
from .registry import TASK_CONTRACTS, task_contract, task_contract_for_num_obs
from .task_mode_locomotion import (
    TASK_MODE_LOCOMOTION_CONTRACT,
    TASK_MODE_LOCOMOTION_NUM_OBS,
    TASK_MODE_LOCOMOTION_TASK_MODE_SLICE,
    TASK_MODE_TAIL_DIM,
    TASK_MODE_TERM_NAME,
)

__all__ = [
    "COMMON_LOCOMOTION_TERMS",
    "LEGACY_LOCOMOTION_CONTRACT",
    "LEGACY_LOCOMOTION_NUM_OBS",
    "TASK_CONTRACTS",
    "TASK_MODE_LOCOMOTION_CONTRACT",
    "TASK_MODE_LOCOMOTION_NUM_OBS",
    "TASK_MODE_LOCOMOTION_TASK_MODE_SLICE",
    "TASK_MODE_TAIL_DIM",
    "TASK_MODE_TERM_NAME",
    "ObservationLayout",
    "ObservationTermSpec",
    "TaskContract",
    "task_contract",
    "task_contract_for_num_obs",
]
