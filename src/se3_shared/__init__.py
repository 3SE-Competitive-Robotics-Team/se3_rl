"""SE3 轮腿机器人共享配置。

训练（se3_train）和验证（se3_sim2sim）的单一参数来源。
"""

from .action_delay import ActionDelayConfig, delay_seconds_to_steps
from .motor import DM8009P, M3508_HEXROLL, MotorSpec
from .observation import ObservationConfig
from .robot import Joint, JointGroup, RobotConfig, Termination

__all__ = [
    "DM8009P",
    "M3508_HEXROLL",
    "ActionDelayConfig",
    "Joint",
    "JointGroup",
    "MotorSpec",
    "ObservationConfig",
    "RobotConfig",
    "Termination",
    "delay_seconds_to_steps",
]
