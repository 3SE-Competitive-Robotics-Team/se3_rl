"""训练和 sim2sim 共享的动作延迟配置。"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def delay_seconds_to_steps(delay_s: float, sim_dt: float) -> int:
    """把秒级延迟量化为物理步数。"""
    if sim_dt <= 0.0:
        raise ValueError(f"sim_dt must be positive, got {sim_dt}")
    if delay_s <= 0.0:
        return 0
    return max(0, math.floor(delay_s / sim_dt + 0.5))


class ActionDelayConfig(BaseModel):
    """策略动作到 actuator target 之间的传输延迟。"""

    enabled: bool = True
    delay_s: float = Field(default=0.005, ge=0.0)
    randomize: bool = True
    min_delay_s: float = Field(default=0.004, ge=0.0)
    max_delay_s: float = Field(default=0.006, ge=0.0)
    resample: Literal["reset"] = "reset"

    @model_validator(mode="after")
    def _validate_range(self) -> ActionDelayConfig:
        if self.min_delay_s > self.max_delay_s:
            raise ValueError(
                f"min_delay_s must be <= max_delay_s, got {self.min_delay_s} > {self.max_delay_s}"
            )
        if not self.enabled:
            return self
        if self.randomize and not (self.min_delay_s <= self.delay_s <= self.max_delay_s):
            raise ValueError(
                "delay_s must be inside [min_delay_s, max_delay_s] when randomize is enabled, "
                f"got {self.delay_s} not in [{self.min_delay_s}, {self.max_delay_s}]"
            )
        return self

    def nominal_steps(self, sim_dt: float) -> int:
        """返回固定延迟模式下的步数。"""
        if not self.enabled:
            return 0
        return delay_seconds_to_steps(self.delay_s, sim_dt)

    def step_bounds(self, sim_dt: float) -> tuple[int, int]:
        """返回随机延迟模式的闭区间步数。"""
        if not self.enabled:
            return (0, 0)
        if not self.randomize:
            steps = self.nominal_steps(sim_dt)
            return (steps, steps)
        min_steps = delay_seconds_to_steps(self.min_delay_s, sim_dt)
        max_steps = delay_seconds_to_steps(self.max_delay_s, sim_dt)
        if min_steps > max_steps:
            min_steps, max_steps = max_steps, min_steps
        return (min_steps, max_steps)

    def actual_delay_s(self, steps: int, sim_dt: float) -> float:
        """把实际使用的延迟步数转回秒。"""
        del self
        return max(0, int(steps)) * float(sim_dt)
