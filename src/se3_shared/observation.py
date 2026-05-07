"""观测空间配置 — 训练和验证共享的缩放系数。"""

from __future__ import annotations

from pydantic import BaseModel


class ObservationConfig(BaseModel):
    """29D 策略输入的缩放系数，修改一处即两端同步生效。"""

    ang_vel_scale: float = 0.25
    command_scale: tuple[float, ...] = (2.0, 0.25, 5.0, 5.0, 5.0)
    leg_vel_scale: float = 0.25
    wheel_vel_scale: float = 0.05
    clip_value: float = 100.0
    num_obs: int = 29
    num_actions: int = 6
