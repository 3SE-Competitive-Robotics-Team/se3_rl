"""SE3 轮腿机器人共享配置。

训练（se3_train）和验证（se3_sim2sim）的单一参数来源。
"""

from .action_delay import ActionDelayConfig, delay_seconds_to_steps
from .observation import ObservationConfig
from .robot import Joint, JointGroup, RobotConfig, Termination

__all__ = [
    "ActionDelayConfig",
    "Joint",
    "JointGroup",
    "ObservationConfig",
    "RobotConfig",
    "Termination",
    "delay_seconds_to_steps",
]
