"""SE3 轮腿机器人共享配置。

训练（se3_train）和验证（se3_sim2sim）的单一参数来源。
"""

from .action_delay import ActionDelayConfig, delay_seconds_to_steps
from .fourbar import (
    FOURBAR_SURROGATE_MARKER,
    is_fourbar_surrogate_name_set,
    output_to_policy_pos_np,
    output_to_policy_pos_torch,
    output_to_policy_vel_np,
    output_to_policy_vel_torch,
    policy_to_output_pos_np,
    policy_to_output_pos_torch,
    policy_to_output_torque_np,
    policy_to_output_torque_torch,
    policy_to_output_vel_torch,
)
from .height_default import policy_default_from_height_np, policy_default_from_height_torch
from .motor import DM8009P, M3508_C620_14, M3508_HEXROLL, MotorSpec
from .observation import ObservationConfig
from .robot import FourbarRobotConfig, Joint, JointGroup, RobotConfig, Termination
from .task_mode import (
    TASK_MODE_COUNT,
    TASK_MODE_NAMES,
    TASK_MODE_SEMANTIC_DIM,
    TASK_MODE_SEMANTICS,
    TaskMode,
    TaskModeSemantic,
)
from .tasks import (
    LEGACY_LOCOMOTION_CONTRACT,
    LEGACY_LOCOMOTION_NUM_OBS,
    TASK_CONTRACTS,
    TASK_MODE_LOCOMOTION_CONTRACT,
    TASK_MODE_LOCOMOTION_NUM_OBS,
    TASK_MODE_LOCOMOTION_TASK_MODE_SLICE,
    TASK_MODE_TAIL_DIM,
    TASK_MODE_TERM_NAME,
    ObservationLayout,
    ObservationTermSpec,
    TaskContract,
    task_contract,
    task_contract_for_num_obs,
)

__all__ = [
    "DM8009P",
    "FOURBAR_SURROGATE_MARKER",
    "LEGACY_LOCOMOTION_CONTRACT",
    "LEGACY_LOCOMOTION_NUM_OBS",
    "M3508_C620_14",
    "M3508_HEXROLL",
    "TASK_CONTRACTS",
    "TASK_MODE_COUNT",
    "TASK_MODE_LOCOMOTION_CONTRACT",
    "TASK_MODE_LOCOMOTION_NUM_OBS",
    "TASK_MODE_LOCOMOTION_TASK_MODE_SLICE",
    "TASK_MODE_NAMES",
    "TASK_MODE_SEMANTICS",
    "TASK_MODE_SEMANTIC_DIM",
    "TASK_MODE_TAIL_DIM",
    "TASK_MODE_TERM_NAME",
    "ActionDelayConfig",
    "FourbarRobotConfig",
    "Joint",
    "JointGroup",
    "MotorSpec",
    "ObservationConfig",
    "ObservationLayout",
    "ObservationTermSpec",
    "RobotConfig",
    "TaskContract",
    "TaskMode",
    "TaskModeSemantic",
    "Termination",
    "delay_seconds_to_steps",
    "is_fourbar_surrogate_name_set",
    "output_to_policy_pos_np",
    "output_to_policy_pos_torch",
    "output_to_policy_vel_np",
    "output_to_policy_vel_torch",
    "policy_default_from_height_np",
    "policy_default_from_height_torch",
    "policy_to_output_pos_np",
    "policy_to_output_pos_torch",
    "policy_to_output_torque_np",
    "policy_to_output_torque_torch",
    "policy_to_output_vel_torch",
    "task_contract",
    "task_contract_for_num_obs",
]
