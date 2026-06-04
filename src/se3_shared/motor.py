"""电机物理参数：训练和 sim2sim 共享的 T-N 包络来源。

所有转速和力矩都按输出轴级别记录，已经包含减速比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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


# ─── 3508 + C620，轮子改 14:1 减速比 ─────────────────────────────────────

_M3508_P19_NOMINAL_RATIO = 19.0
_M3508_P19_NO_LOAD_RPM = 482.0
_M3508_P19_RATED_TORQUE = 3.0
_M3508_P19_RATED_TORQUE_SPEED_RPM = 469.0
_M3508_WHEEL_GEAR_RATIO = 14.0
_M3508_TORQUE_SCALE = _M3508_WHEEL_GEAR_RATIO / _M3508_P19_NOMINAL_RATIO

M3508_C620_14 = MotorSpec(
    name="M3508-C620-14to1",
    rated_voltage=24.0,
    gear_ratio=_M3508_WHEEL_GEAR_RATIO,
    # C620 官方数据为 3 N·m 下仍可到 469 rpm。这里的 stall_torque
    # 是 T-N 包络外推截距，只决定高速掉矩斜率；实际力矩由 rated_torque 限幅。
    stall_torque=(
        _M3508_P19_RATED_TORQUE
        * _M3508_TORQUE_SCALE
        / (1.0 - _M3508_P19_RATED_TORQUE_SPEED_RPM / _M3508_P19_NO_LOAD_RPM)
    ),
    no_load_speed=(
        _M3508_P19_NO_LOAD_RPM
        * _M3508_P19_NOMINAL_RATIO
        / _M3508_WHEEL_GEAR_RATIO
        * 2.0
        * math.pi
        / 60.0
    ),
    rated_torque=_M3508_P19_RATED_TORQUE * _M3508_TORQUE_SCALE,
    rated_current=20.0,
    stall_current=20.0,
    phase_resistance=0.194,
)

# 旧名字保留为兼容别名，新增代码优先使用 M3508_C620_14。
M3508_HEXROLL = M3508_C620_14


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
