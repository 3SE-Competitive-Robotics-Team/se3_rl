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
from .leg_policy import (
    policy_leg_phase_active_obs_np,
    policy_leg_phase_active_obs_torch,
    policy_leg_position_error_np,
    policy_leg_position_error_torch,
)
from .motor import DM8009P, M3508_C620_14, M3508_HEXROLL, MotorSpec
from .observation import ObservationConfig
from .policy_io import (
    DecodedPolicyAction,
    PolicyActionDecoder,
    PolicyObservationResult,
    build_policy_observation_np,
)
from .robot import Joint, JointGroup, RobotConfig, Termination

__all__ = [
    "DM8009P",
    "FOURBAR_SURROGATE_MARKER",
    "M3508_C620_14",
    "M3508_HEXROLL",
    "ActionDelayConfig",
    "DecodedPolicyAction",
    "Joint",
    "JointGroup",
    "MotorSpec",
    "ObservationConfig",
    "PolicyActionDecoder",
    "PolicyObservationResult",
    "RobotConfig",
    "Termination",
    "build_policy_observation_np",
    "delay_seconds_to_steps",
    "is_fourbar_surrogate_name_set",
    "output_to_policy_pos_np",
    "output_to_policy_pos_torch",
    "output_to_policy_vel_np",
    "output_to_policy_vel_torch",
    "policy_default_from_height_np",
    "policy_default_from_height_torch",
    "policy_leg_phase_active_obs_np",
    "policy_leg_phase_active_obs_torch",
    "policy_leg_position_error_np",
    "policy_leg_position_error_torch",
    "policy_to_output_pos_np",
    "policy_to_output_pos_torch",
    "policy_to_output_torque_np",
    "policy_to_output_torque_torch",
    "policy_to_output_vel_torch",
]
