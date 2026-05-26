"""sim2sim 指令历程（Command Course）：预定义的指令扫描序列。

历程是一个接近工程测试/演示意图的概念，把多次 sim2sim 运行需要手动切换的参数
（前进速度、跳跃高度等）编排成自动化序列，确保所有 checkpoints 在相同条件下评估。

当前支持两个历程：
- walk-sweep：行走速度扫描，vx 从 0.1 到 0.6 m/s 每档 5s，扫完后回到 0
- jump-sweep：跳跃高度扫描，依次触发 0.1～0.6m 跳跃，每次落地后间隔 5s 跳下一档
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CourseType(StrEnum):
    """历程类型。"""

    WALK_SWEEP = "walk-sweep"
    """行走速度扫描：前进速度 vx 从 0.1 匀速递增至 0.6 m/s，每档持续 5s。"""

    JUMP_SWEEP = "jump-sweep"
    """跳跃高度扫描：依次触发 0.1～0.6m 跳跃，每次落地后间隔 5s 跳下一档。"""

    NONE = "none"
    """禁用历程，使用固定指令（旧行为，与 --command 一致）。"""


@dataclass
class CourseConfig:
    """历程配置。"""

    mode: CourseType = CourseType.NONE

    # 行走速度扫描参数
    walk_sweep_velocities: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    """前进速度序列 (m/s)，每个速度保持 walk_sweep_segment_duration_s 秒。"""

    walk_sweep_segment_duration_s: float = 5.0
    """每档前进速度的保持时间（秒）。"""

    # 跳跃高度扫描参数
    jump_sweep_heights: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    """跳跃高度序列 (m)，每次落地后间隔 jump_sweep_interval_s 秒触发下一跳。"""

    jump_sweep_interval_s: float = 5.0
    """跳跃落地到下一次触发的时间间隔（秒）。"""


class WalkSpeedSweep:
    """行走速度扫描历程：按时间推进 command[0] (vx)。"""

    def __init__(self, config: CourseConfig, control_dt: float) -> None:
        self._velocities = tuple(config.walk_sweep_velocities)
        segment_steps = max(1, round(config.walk_sweep_segment_duration_s / control_dt))
        self._segment_steps = int(segment_steps)
        self._segment_idx = 0
        self._step_in_segment = 0
        self._sweep_done = False

    @property
    def current_vx(self) -> float:
        """返回当前控制步应使用的 vx 值。扫描完成后返回 0。"""
        if self._sweep_done or self._segment_idx >= len(self._velocities):
            return 0.0
        return float(self._velocities[self._segment_idx])

    def step(self) -> None:
        """每个控制步调用一次，推进内部计时器。"""
        if self._sweep_done:
            return
        self._step_in_segment += 1
        if self._step_in_segment >= self._segment_steps:
            self._step_in_segment = 0
            self._segment_idx += 1
            if self._segment_idx >= len(self._velocities):
                self._sweep_done = True

    @property
    def done(self) -> bool:
        return self._sweep_done

    def report(self) -> dict[str, object]:
        """返回当前状态的描述字典，用于打印日志。"""
        if self._sweep_done:
            return {"stage": "done", "vx": 0.0}
        return {
            "stage": "sweeping",
            "vx": float(self._velocities[self._segment_idx]),
            "segment": self._segment_idx + 1,
            "total_segments": len(self._velocities),
        }


class JumpHeightSweep:
    """跳跃高度扫描历程：每次跳跃完成后自动切换到下一个高度。"""

    def __init__(self, config: CourseConfig) -> None:
        self._heights = tuple(float(h) for h in config.jump_sweep_heights)
        self._index = 0
        self._completed = len(self._heights) == 0

    @property
    def current_height(self) -> float:
        """返回当前应使用的跳跃目标高度。"""
        return float(self._heights[self._index % len(self._heights)])

    def advance(self) -> float:
        """切换到下一个跳跃高度。返回新的当前高度。"""
        if self._index >= len(self._heights) - 1:
            self._completed = True
            return self.current_height
        self._index += 1
        return self.current_height

    @property
    def done(self) -> bool:
        return self._completed

    def report(self) -> dict[str, object]:
        return {
            "height": self.current_height,
            "index": self._index + 1,
            "total": len(self._heights),
        }


def create_course(
    config: CourseConfig, control_dt: float
) -> WalkSpeedSweep | JumpHeightSweep | None:
    """根据 CourseConfig 创建对应的历程对象。

    返回 None 表示 CourseType.NONE（不启用历程）。
    """
    if config.mode == CourseType.WALK_SWEEP:
        return WalkSpeedSweep(config, control_dt)
    if config.mode == CourseType.JUMP_SWEEP:
        return JumpHeightSweep(config)
    return None
