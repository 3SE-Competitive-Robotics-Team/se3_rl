"""yaw 轴闭环 PID 控制器。"""

from __future__ import annotations

import numpy as np

from .config import MAX_YAW_RATE_RAD_S, YawPidConfig
from .math_utils import wrap_angle


class YawPidController:
    """根据当前 yaw 生成受限的 yaw_rate 指令。"""

    __slots__ = (
        "cfg",
        "command",
        "current_yaw",
        "enabled",
        "error",
        "integral",
        "kd",
        "ki",
        "kp",
        "max_rate",
        "prev_error",
        "target_yaw",
    )

    def __init__(self, cfg: YawPidConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.target_yaw = float(cfg.target_yaw_rad)
        self.kp = float(cfg.kp)
        self.ki = float(cfg.ki)
        self.kd = float(cfg.kd)
        self.max_rate = float(cfg.max_rate)
        if self.max_rate <= 0.0:
            raise ValueError(f"max_rate must be positive, got {self.max_rate}")
        if self.max_rate > MAX_YAW_RATE_RAD_S:
            raise ValueError(f"max_rate must be <= {MAX_YAW_RATE_RAD_S}, got {self.max_rate}")
        self.integral = 0.0
        self.prev_error = 0.0
        self.current_yaw = 0.0
        self.error = 0.0
        self.command = 0.0

    def reset(self, current_yaw: float) -> float:
        """重置积分和微分状态，并返回初始 yaw_rate 指令。"""

        self.current_yaw = float(current_yaw)
        self.error = float(wrap_angle(self.target_yaw - self.current_yaw))
        self.prev_error = self.error
        self.integral = 0.0
        self.command = self._clip(self.kp * self.error)
        return self.command

    def set_target_yaw(self, target_yaw: float) -> None:
        """更新目标 yaw，并重置积分/微分状态，避免旧误差残留。"""

        self.target_yaw = float(wrap_angle(float(target_yaw)))
        self.cfg.target_yaw_rad = self.target_yaw
        self.error = float(wrap_angle(self.target_yaw - self.current_yaw))
        self.prev_error = self.error
        self.integral = 0.0

    def update(self, current_yaw: float, dt: float) -> float:
        """根据当前 yaw 和控制周期更新 yaw_rate 指令。"""

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.current_yaw = float(current_yaw)
        self.error = float(wrap_angle(self.target_yaw - self.current_yaw))
        derivative = (self.error - self.prev_error) / float(dt)
        if self.ki != 0.0:
            self.integral += self.error * float(dt)
            limit = abs(self.max_rate / self.ki)
            self.integral = float(np.clip(self.integral, -limit, limit))
        else:
            self.integral = 0.0
        raw_command = self.kp * self.error + self.ki * self.integral + self.kd * derivative
        self.command = self._clip(raw_command)
        self.prev_error = self.error
        return self.command

    def telemetry(self) -> dict[str, float]:
        """返回便于调试的控制状态。"""

        return {
            "enabled": float(self.enabled),
            "target_yaw": float(self.target_yaw),
            "current_yaw": float(self.current_yaw),
            "error": float(self.error),
            "command": float(self.command),
            "integral": float(self.integral),
            "prev_error": float(self.prev_error),
            "kp": float(self.kp),
            "ki": float(self.ki),
            "kd": float(self.kd),
            "max_rate": float(self.max_rate),
        }

    def _clip(self, value: float) -> float:
        return float(np.clip(float(value), -self.max_rate, self.max_rate))
