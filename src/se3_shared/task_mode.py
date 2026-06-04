"""Task Mode 定义。

Task Mode 是训练端和部署端共享的高层行为指令。策略仍然输出同一套
腿部位置目标和轮子速度目标，mode 只决定当前应该激活哪组奖励/约束。
"""

from __future__ import annotations

from enum import IntEnum

TaskModeSemantic = tuple[float, float, float, float]


class TaskMode(IntEnum):
    """轮腿统一策略支持的任务模式。"""

    WHEEL = 0
    GAIT = 1
    WHEEL_LEG = 2
    GAIT_WHEEL = 3
    JUMP = 4


TASK_MODE_COUNT = len(TaskMode)
TASK_MODE_NAMES: tuple[str, ...] = tuple(mode.name.lower() for mode in TaskMode)
TASK_MODE_SEMANTICS: tuple[TaskModeSemantic, ...] = (
    (1.0, 0.0, 0.0, 0.0),  # wheel:轮子主驱动，腿稳定机身
    (0.0, 1.0, 0.0, 0.0),  # gait:腿主导步态，轮子少参与
    (1.0, 0.5, 1.0, 0.0),  # wheel_leg:轮子主驱，腿负责抬升、越障、解卡
    (0.5, 1.0, 0.0, 0.0),  # gait_wheel:腿主导行走，轮子辅助推进和转向
    (0.0, 0.0, 0.0, 1.0),  # jump:起跳、腾空、落地
)
TASK_MODE_SEMANTIC_DIM = len(TASK_MODE_SEMANTICS[0])


__all__ = [
    "TASK_MODE_COUNT",
    "TASK_MODE_NAMES",
    "TASK_MODE_SEMANTICS",
    "TASK_MODE_SEMANTIC_DIM",
    "TaskMode",
]
