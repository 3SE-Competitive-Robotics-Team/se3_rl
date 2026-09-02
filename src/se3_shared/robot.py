"""机器人关节布局与控制参数 — 训练和验证共享的单一来源。"""

from __future__ import annotations

from enum import IntEnum
from math import radians
from typing import ClassVar

from pydantic import BaseModel, Field

from .action_delay import ActionDelayConfig
from .motor import DM8009P, M3508_C620_14

_ACTIVE_ROD_ANGLE_RANGE_DEG: float = 129.95 - 43.46
_ACTIVE_ROD_ANGLE_LIMITS: tuple[float, float] = (
    0.0,
    radians(_ACTIVE_ROD_ANGLE_RANGE_DEG),
)


class Joint(IntEnum):
    """SerialLeg policy 关节语义枚举。

    值为 policy-order 数组中的列索引，不再等同于 MJCF qpos 顺序。

    policy 动作/actor 观测布局（6 维）：
        0: lf0_Joint          -> LF0 / 左前主动杆
        1: l_drive_bar_Joint  -> LB / 左后主动杆
        2: rf0_Joint          -> RF0 / 右前主动杆
        3: r_drive_bar_Joint  -> RB / 右后主动杆
        4: l_wheel_Joint      -> L_WHEEL
        5: r_wheel_Joint      -> R_WHEEL
    """

    LF0 = 0
    LB = 1
    RF0 = 2
    RB = 3
    L_WHEEL = 4
    R_WHEEL = 5

    @property
    def mjcf_name(self) -> str:
        return _MJCF_NAMES[self]


_MJCF_NAMES: dict[Joint, str] = {
    Joint.LF0: "lf0_Joint",
    Joint.LB: "l_drive_bar_Joint",
    Joint.RF0: "rf0_Joint",
    Joint.RB: "r_drive_bar_Joint",
    Joint.L_WHEEL: "l_wheel_Joint",
    Joint.R_WHEEL: "r_wheel_Joint",
}


class JointGroup:
    """预定义的 policy-order 分组和 MJCF 名称。

    LEGS / WHEELS / ALL 只适用于 policy-order 的 6 维数组。
    训练端读取 MJLab joint_pos / actuator_force 时必须按名称解析，
    不能把这些索引当作 MJCF qpos 或 actuator_force 的自然顺序。
    """

    LEGS: ClassVar[list[int]] = [Joint.LF0, Joint.LB, Joint.RF0, Joint.RB]
    WHEELS: ClassVar[list[int]] = [Joint.L_WHEEL, Joint.R_WHEEL]
    CTRL_LEGS: ClassVar[list[int]] = LEGS
    CTRL_WHEELS: ClassVar[list[int]] = WHEELS
    LEG_ACTUATORS: ClassVar[list[int]] = [0, 1, 2, 3]
    WHEEL_ACTUATORS: ClassVar[list[int]] = [4, 5]
    ALL: ClassVar[list[int]] = [
        Joint.LF0,
        Joint.LB,
        Joint.RF0,
        Joint.RB,
        Joint.L_WHEEL,
        Joint.R_WHEEL,
    ]
    POLICY_JOINT_NAMES: ClassVar[tuple[str, ...]] = tuple(j.mjcf_name for j in Joint)
    POLICY_LEG_NAMES: ClassVar[tuple[str, ...]] = (
        "lf0_Joint",
        "l_drive_bar_Joint",
        "rf0_Joint",
        "r_drive_bar_Joint",
    )
    WHEEL_NAMES: ClassVar[tuple[str, ...]] = ("l_wheel_Joint", "r_wheel_Joint")
    OUTPUT_LEG_NAMES: ClassVar[tuple[str, ...]] = (
        "lf0_Joint",
        "lf1_Joint",
        "rf0_Joint",
        "rf1_Joint",
    )
    OUTPUT_KNEE_NAMES: ClassVar[tuple[str, ...]] = ("lf1_Joint", "rf1_Joint")
    CLOSEDCHAIN_PASSIVE_JOINT_NAMES: ClassVar[tuple[str, ...]] = (
        "lf1_Joint",
        "l_coupler_Joint",
        "rf1_Joint",
        "r_coupler_Joint",
    )
    POLICY_MOTOR_ACTUATOR_NAMES: ClassVar[tuple[str, ...]] = tuple(
        f"{name}_motor" for name in POLICY_JOINT_NAMES
    )
    KNEE_SPRING_ACTUATOR_NAMES: ClassVar[tuple[str, ...]] = (
        "l_knee_gas_spring",
        "r_knee_gas_spring",
    )

    @staticmethod
    def joint_names() -> tuple[str, ...]:
        return JointGroup.POLICY_JOINT_NAMES


class Termination(BaseModel):
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12


class RobotConfig(BaseModel):
    """机器人物理参数 — 训练和验证共享的单一来源。"""

    leg_kp: float = 60.0
    leg_kd: float = 3.0
    # 轮子速度环增益（N·m·s/rad，输出轴级）：力矩 = wheel_kd × (目标轮速 − 实际轮速)。
    # 2026-09-02 0.08 → 0.2：按 M3508 Kt 0.3 N·m/A（19.2:1）折算 14:1 输出轴 0.219 N·m/A，
    # C620 ±16384 ↔ ±20 A，kd_sim ≈ 0.0357 × 固件速度环 Kp[count/rpm]，0.2 ↔ Kp≈5.6。
    # 0.2 使满幅动作（scale 15 rad/s）≈ 堵转力矩 3.3 N·m，库仑死区 0.15 N·m 对应 0.75 rad/s。
    # 再高需同时降轮 scale，否则探索噪声力矩顶到额定限幅。
    wheel_kd: float = 0.2
    knee_gas_spring_force: float = 300.0
    # 关闭前馈，让策略直接面对带弹簧的 plant 并学会利用这份抗重力力矩；
    # 开启则电机自己抵消弹簧，等于放弃硬件收益还多占 10-18 N·m 包络。
    knee_gas_spring_compensation_enabled: bool = False
    torque_limits: tuple[float, ...] = (
        DM8009P.stall_torque,  # 40 N·m 峰值，允许起跳时短时大力矩（连续额定 20 N·m）
        DM8009P.stall_torque,
        DM8009P.stall_torque,
        DM8009P.stall_torque,
        M3508_C620_14.rated_torque,
        M3508_C620_14.rated_torque,
    )
    default_dof_pos: tuple[float, ...] = (
        -0.275422946189,
        -1.592100148957,
        0.275422946189,
        1.592100148957,
        0.0,
        0.0,
    )
    default_output_knee_pos: tuple[float, float] = (-1.242259649307, 1.242259649307)
    default_coupler_pos: tuple[float, float] = (1.401266340000, -1.401269410000)
    active_rod_angle_limits: tuple[float, float] = _ACTIVE_ROD_ANGLE_LIMITS
    active_rod_lower_target_overdrive: float = 0.20
    active_rod_soft_limit_factor: float = 1.0
    active_rod_angle_coeffs: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, -1.0),
        (-1.0, 1.0),
    )
    default_base_height: float = 0.22
    # 腿部 action 使用 [lf0, active_angle, rf0, active_angle] 语义；active 的零点是机械夹角中点。
    # 注意：recovery 任务族（grouped/ungrouped/expert 主线）的轮 scale 被
    # se3_shared.recovery.RECOVERY_WHEEL_ACTION_SCALE=15 显式覆写（2026-09-01 σ-gate 修复），
    # 此处 45 仅对 flat/jump 遗留线生效；那两条线的动作空间罚项未做 ×9 补偿，
    # 重启训练前必须先补偿再改 scale。
    action_scale: tuple[float, ...] = (
        0.25,
        0.25,
        0.25,
        0.25,
        45.0,
        45.0,
    )
    action_clip: float | None = 100.0
    sim_dt: float = 0.005
    control_decimation: int = 4
    action_delay: ActionDelayConfig = Field(default_factory=ActionDelayConfig)
    termination: Termination = Termination()

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.control_decimation

    @property
    def default_active_rod_angles(self) -> tuple[float, float]:
        """返回左右腿当前装配分支下的两主动杆夹角。"""
        left_front, left_back = self.active_rod_angle_coeffs[0]
        right_front, right_back = self.active_rod_angle_coeffs[1]
        return (
            left_front * self.default_dof_pos[Joint.LF0]
            + left_back * self.default_dof_pos[Joint.LB],
            right_front * self.default_dof_pos[Joint.RF0]
            + right_back * self.default_dof_pos[Joint.RB],
        )

    @property
    def active_rod_soft_angle_limits(self) -> tuple[float, float]:
        """返回主动杆夹角软限位，按硬限位中心收缩。"""
        lower, upper = self.active_rod_angle_limits
        factor = min(max(float(self.active_rod_soft_limit_factor), 0.0), 1.0)
        center = 0.5 * (float(lower) + float(upper))
        half_range = 0.5 * (float(upper) - float(lower)) * factor
        return center - half_range, center + half_range

    @property
    def default_model_joint_pos(self) -> dict[str, float]:
        """返回正式 OBB 闭链 MJCF 的默认关节角映射。"""
        policy = dict(zip(JointGroup.POLICY_JOINT_NAMES, self.default_dof_pos, strict=True))
        policy.update(
            {
                "lf1_Joint": self.default_output_knee_pos[0],
                "rf1_Joint": self.default_output_knee_pos[1],
                "l_coupler_Joint": self.default_coupler_pos[0],
                "r_coupler_Joint": self.default_coupler_pos[1],
            }
        )
        return policy
