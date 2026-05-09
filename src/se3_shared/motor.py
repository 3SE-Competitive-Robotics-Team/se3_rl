"""电机物理参数 — 训练和 sim2sim 共享的 T-N 曲线约束来源。

每个电机 dataclass 包含从数据手册提取的关键参数，用于：
- 训练端 DcMotorActuatorCfg 的 saturation_effort / velocity_limit
- sim2sim 端 MuJoCo dcmotor 的 nominal / saturation 配置
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MotorSpec:
    """DC 电机 T-N 曲线参数（输出轴级别，已含减速比）。"""

    name: str
    """电机型号名称。"""

    rated_voltage: float
    """额定电压 (V)。"""

    gear_ratio: float
    """总减速比(>1 表示减速)。"""

    stall_torque: float
    """堵转转矩 — 输出轴 (N·m)。对应 T-N 曲线零速截距。"""

    no_load_speed: float
    """空载转速 — 输出轴 (rad/s)。对应 T-N 曲线零矩截距。"""

    rated_torque: float
    """额定连续转矩 — 输出轴 (N·m)。长时间运行安全上限。"""

    rated_current: float
    """额定连续电流 (A)。"""

    stall_current: float
    """堵转电流 (A)。注意 C620 等电调可能主动限流。"""

    phase_resistance: float
    """相电阻 (Ω)。对 dcmotor 物理模型使用。"""

    @property
    def no_load_speed_rpm(self) -> float:
        """空载转速 (rpm)。"""
        return self.no_load_speed * 60.0 / (2.0 * math.pi)

    @property
    def rotor_kt(self) -> float:
        """转子级转矩常数 Kt (N·m/A) — 从堵转点推导。

        Kt_rotor = stall_torque / (stall_current × gear_ratio × efficiency)
        这里用简化公式 = stall_torque / (stall_current × gear_ratio)，
        因为 C620 限流堵转值已经包含了传动损耗。
        """
        return self.stall_torque / (self.stall_current * self.gear_ratio)

    @property
    def rotor_ke(self) -> float:
        """转子级反电动势常数 Ke (V·s/rad) — 从空载转速推导。

        Ke = V / (ω_no_load × gear_ratio)（忽略空载电流的 IR 压降）
        """
        return self.rated_voltage / (self.no_load_speed * self.gear_ratio)


# ─── 3508 + hexroll 减速箱（轮子）──────────────────────────────────────────────

M3508_HEXROLL = MotorSpec(
    name="M3508-Hexroll-C620",
    rated_voltage=24.0,
    gear_ratio=268.0 / 17.0,  # ≈ 15.76:1
    stall_torque=3.69,  # N·m（输出轴，C620 限流下）
    no_load_speed=587.0 * 2.0 * math.pi / 60.0,  # 587 rpm → rad/s ≈ 61.5
    rated_torque=2.46,  # N·m（输出轴连续）
    rated_current=10.0,  # A
    stall_current=2.5,  # A（C620 限流）
    phase_resistance=0.194,  # Ω（相-中性）
)

# ─── DM-8009P（腿部关节）──────────────────────────────────────────────────────

DM8009P = MotorSpec(
    name="DM-8009P-2EC",
    rated_voltage=24.0,
    gear_ratio=9.0,
    stall_torque=40.0,  # N·m（输出轴峰值）
    no_load_speed=160.0 * 2.0 * math.pi / 60.0,  # 160 rpm → rad/s ≈ 16.76
    rated_torque=20.0,  # N·m（输出轴连续）
    rated_current=20.0,  # A
    stall_current=50.0,  # A（峰值）
    phase_resistance=0.145,  # Ω（8009P 变体）
)
