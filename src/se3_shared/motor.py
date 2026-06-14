"""电机物理参数：训练和 sim2sim 共享的 T-N 包络来源。

所有转速和力矩都按输出轴级别记录，已经包含减速比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotorSpec:
    """DC 电机 T-N 曲线参数。"""

    name: str
    """电机型号名称。"""

    rated_voltage: float
    """额定电压 (V)。"""

    gear_ratio: float
    """总减速比，>1 表示减速。"""

    stall_torque: float
    """T-N 包络零速截距 (N·m)。"""

    no_load_speed: float
    """空载转速，输出轴 (rad/s)。"""

    rated_torque: float
    """连续可用输出轴力矩 (N·m)。"""

    rated_current: float
    """额定或最大持续电流 (A)。"""

    stall_current: float
    """用于粗略估算的限流电流 (A)。"""

    phase_resistance: float
    """相电阻 (Ω)。"""

    torque_speed_curve: tuple[tuple[float, float], ...] = ()
    """输出轴 T-N 包络点 ``(速度 rad/s, 最大扭矩 N·m)``，速度必须严格递增。"""

    @property
    def no_load_speed_rpm(self) -> float:
        """空载转速 (rpm)。"""
        return self.no_load_speed * 60.0 / (2.0 * math.pi)

    @property
    def rotor_kt(self) -> float:
        """转子级转矩常数 Kt (N·m/A)。"""
        return self.stall_torque / (self.stall_current * self.gear_ratio)

    @property
    def rotor_ke(self) -> float:
        """转子级反电动势常数 Ke (V·s/rad)。"""
        return self.rated_voltage / (self.no_load_speed * self.gear_ratio)

    def torque_limit_np(self, velocity: np.ndarray | float) -> np.ndarray:
        """按输出轴速度返回允许的最大扭矩绝对值。"""

        speed = np.abs(np.asarray(velocity, dtype=np.float64))
        if self.torque_speed_curve:
            curve = np.asarray(self.torque_speed_curve, dtype=np.float64)
            return np.interp(
                speed,
                curve[:, 0],
                curve[:, 1],
                left=float(curve[0, 1]),
                right=float(curve[-1, 1]),
            )

        vel_at_effort_limit = self.no_load_speed * (1.0 + self.rated_torque / self.stall_torque)
        speed_clipped = np.clip(speed, 0.0, vel_at_effort_limit)
        return np.minimum(
            self.stall_torque * (1.0 - speed_clipped / self.no_load_speed),
            self.rated_torque,
        ).clip(min=0.0)

    def clip_effort_np(
        self,
        effort: np.ndarray | float,
        velocity: np.ndarray | float,
    ) -> np.ndarray:
        """按 T-N 包络限制输出扭矩。"""

        effort_arr = np.asarray(effort, dtype=np.float64)
        if self.torque_speed_curve:
            limit = self.torque_limit_np(velocity)
            return np.clip(effort_arr, -limit, limit)

        velocity_arr = np.asarray(velocity, dtype=np.float64)
        vel_at_effort_limit = self.no_load_speed * (1.0 + self.rated_torque / self.stall_torque)
        vel_clipped = np.clip(
            velocity_arr,
            -vel_at_effort_limit,
            vel_at_effort_limit,
        )
        top = self.stall_torque * (1.0 - vel_clipped / self.no_load_speed)
        bottom = self.stall_torque * (-1.0 - vel_clipped / self.no_load_speed)
        return np.clip(
            effort_arr,
            np.maximum(bottom, -self.rated_torque),
            np.minimum(top, self.rated_torque),
        )


# ─── 3508 + C620，轮子改 14:1 减速比 ─────────────────────────────────────

_M3508_WHEEL_GEAR_RATIO = 14.0
_M3508_14_NO_LOAD_SPEED = 71.81
_M3508_14_MAX_TORQUE = 3.71

# 由 C620 电流闭环负载特性图数字化，再按 19:1 -> 14:1 映射得到。
# 原图只覆盖到约 32.9 rad/s；更低速度沿用最后测得的 3.71 N·m 上限。
M3508_C620_14_TORQUE_SPEED_CURVE: tuple[tuple[float, float], ...] = (
    (0.00, 3.71),
    (32.93, 3.71),
    (49.07, 3.61),
    (57.82, 3.54),
    (63.65, 3.46),
    (65.29, 3.39),
    (65.70, 3.32),
    (67.43, 2.95),
    (69.28, 2.21),
    (70.90, 1.47),
    (71.37, 0.74),
    (71.81, 0.00),
)

M3508_C620_14 = MotorSpec(
    name="M3508-C620-14to1",
    rated_voltage=24.0,
    gear_ratio=_M3508_WHEEL_GEAR_RATIO,
    stall_torque=_M3508_14_MAX_TORQUE,
    no_load_speed=_M3508_14_NO_LOAD_SPEED,
    rated_torque=_M3508_14_MAX_TORQUE,
    rated_current=20.0,
    stall_current=20.0,
    phase_resistance=0.194,
    torque_speed_curve=M3508_C620_14_TORQUE_SPEED_CURVE,
)

# ─── 3508 + hexroll 减速箱（main 既有轮子语义）───────────────────────────

M3508_HEXROLL = MotorSpec(
    name="M3508-Hexroll-C620",
    rated_voltage=24.0,
    gear_ratio=268.0 / 17.0,
    stall_torque=3.69,
    no_load_speed=587.0 * 2.0 * math.pi / 60.0,
    rated_torque=2.46,
    rated_current=10.0,
    stall_current=2.5,
    phase_resistance=0.194,
)


# ─── DM-8009P（腿部关节）──────────────────────────────────────────────────

DM8009P = MotorSpec(
    name="DM-8009P-2EC",
    rated_voltage=24.0,
    gear_ratio=9.0,
    stall_torque=40.0,
    no_load_speed=160.0 * 2.0 * math.pi / 60.0,
    rated_torque=20.0,
    rated_current=20.0,
    stall_current=50.0,
    phase_resistance=0.145,
)
