"""SE3 轮腿机器人共享配置。

训练（se3_train）和验证（se3_sim2sim）的单一参数来源。
"""

from .action_delay import ActionDelayConfig, delay_seconds_to_steps
from .ctbc_feedforward import (
    REFERENCE_CTBC_CONTACT_WINDOW,
    REFERENCE_CTBC_FF_AMPLITUDE,
    REFERENCE_CTBC_FF_PERIOD_S,
    REFERENCE_CTBC_FORCE_THRESHOLD_N,
    REFERENCE_CTBC_HIP_RATIO,
    REFERENCE_CTBC_KNEE_RATIO,
    REFERENCE_CTBC_LEG_SCALE,
    current_leg_action_scales_np,
    reference_ctbc_bias_to_current_action_np,
    reference_ctbc_bias_to_current_action_torch,
)
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
from .robot import Joint, JointGroup, RobotConfig, Termination

__all__ = [
    "DM8009P",
    "FOURBAR_SURROGATE_MARKER",
    "M3508_C620_14",
    "M3508_HEXROLL",
    "REFERENCE_CTBC_CONTACT_WINDOW",
    "REFERENCE_CTBC_FF_AMPLITUDE",
    "REFERENCE_CTBC_FF_PERIOD_S",
    "REFERENCE_CTBC_FORCE_THRESHOLD_N",
    "REFERENCE_CTBC_HIP_RATIO",
    "REFERENCE_CTBC_KNEE_RATIO",
    "REFERENCE_CTBC_LEG_SCALE",
    "ActionDelayConfig",
    "Joint",
    "JointGroup",
    "MotorSpec",
    "ObservationConfig",
    "RobotConfig",
    "Termination",
    "current_leg_action_scales_np",
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
    "reference_ctbc_bias_to_current_action_np",
    "reference_ctbc_bias_to_current_action_torch",
]
