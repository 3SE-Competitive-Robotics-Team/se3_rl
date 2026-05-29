"""机器人关节布局与控制参数 — 训练和验证共享的单一来源。"""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

from pydantic import BaseModel, Field

from .action_delay import ActionDelayConfig
from .motor import DM8009P, M3508_HEXROLL


class Joint(IntEnum):
    """SerialLeg 受控关节语义枚举。

    值为 MJLab joint_pos / joint_vel 张量中的列索引。

    MJLab joint_pos 列布局（6 维）：
        0: lf0_Joint       -> LF0
        1: lf1_Joint       -> LF1
        2: l_wheel_Joint   -> L_WHEEL
        3: rf0_Joint       -> RF0
        4: rf1_Joint       -> RF1
        5: r_wheel_Joint   -> R_WHEEL
    """

    LF0 = 0
    LF1 = 1
    L_WHEEL = 2
    RF0 = 3
    RF1 = 4
    R_WHEEL = 5

    @property
    def mjcf_name(self) -> str:
        return _MJCF_NAMES[self]


_MJCF_NAMES: dict[Joint, str] = {
    Joint.LF0: "lf0_Joint",
    Joint.LF1: "lf1_Joint",
    Joint.L_WHEEL: "l_wheel_Joint",
    Joint.RF0: "rf0_Joint",
    Joint.RF1: "rf1_Joint",
    Joint.R_WHEEL: "r_wheel_Joint",
}


class JointGroup:
    """预定义的关节索引分组，替代散布在各处的魔法索引列表。

    LEGS / WHEELS / ALL 中的值是 MJLab joint_pos (6 维) 的列索引。
    CTRL_LEGS / CTRL_WHEELS 是受控关节 6 维数组（sim2sim dof_pos、actuator 输出）中的位置索引。
    LEG_ACTUATORS / WHEEL_ACTUATORS 是 actuator_force (6 维) 的列索引，与关节索引无关。
    """

    LEGS: ClassVar[list[int]] = [Joint.LF0, Joint.LF1, Joint.RF0, Joint.RF1]
    WHEELS: ClassVar[list[int]] = [Joint.L_WHEEL, Joint.R_WHEEL]
    CTRL_LEGS: ClassVar[list[int]] = [0, 1, 3, 4]
    CTRL_WHEELS: ClassVar[list[int]] = [2, 5]
    LEG_ACTUATORS: ClassVar[list[int]] = [0, 1, 2, 3]
    WHEEL_ACTUATORS: ClassVar[list[int]] = [4, 5]
    ALL: ClassVar[list[int]] = [
        Joint.LF0,
        Joint.LF1,
        Joint.L_WHEEL,
        Joint.RF0,
        Joint.RF1,
        Joint.R_WHEEL,
    ]

    @staticmethod
    def joint_names() -> tuple[str, ...]:
        return tuple(j.mjcf_name for j in Joint)


class Termination(BaseModel):
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12


class RobotConfig(BaseModel):
    """机器人物理参数 — 训练和验证共享的单一来源。"""

    leg_kp: float = 40.0
    leg_kd: float = 2.0
    wheel_kd: float = 0.5
    torque_limits: tuple[float, ...] = (
        DM8009P.stall_torque,  # 40 N·m 峰值，允许起跳时短时大力矩（连续额定 20 N·m）
        DM8009P.stall_torque,
        M3508_HEXROLL.rated_torque,
        DM8009P.stall_torque,
        DM8009P.stall_torque,
        M3508_HEXROLL.rated_torque,
    )
    default_dof_pos: tuple[float, ...] = (0.4610, 0.4742, 0.0, 0.4610, 0.4742, 0.0)
    default_base_height: float = 0.22
    action_scale: tuple[float, ...] = (0.35, 0.25, 0.35, 0.25, 20.0, 20.0)
    sim_dt: float = 0.002
    control_decimation: int = 5
    action_delay: ActionDelayConfig = Field(default_factory=ActionDelayConfig)
    termination: Termination = Termination()

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.control_decimation
