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
)
from .motor import DM8009P, M3508_HEXROLL, MotorSpec
from .observation import ObservationConfig
from .robot import Joint, JointGroup, RobotConfig, Termination

__all__ = [
    "DM8009P",
    "FOURBAR_SURROGATE_MARKER",
    "M3508_HEXROLL",
    "ActionDelayConfig",
    "Joint",
    "JointGroup",
    "MotorSpec",
    "ObservationConfig",
    "RobotConfig",
    "Termination",
    "delay_seconds_to_steps",
    "is_fourbar_surrogate_name_set",
    "output_to_policy_pos_np",
    "output_to_policy_pos_torch",
    "output_to_policy_vel_np",
    "output_to_policy_vel_torch",
    "policy_to_output_pos_np",
    "policy_to_output_pos_torch",
    "policy_to_output_torque_np",
    "policy_to_output_torque_torch",
]
