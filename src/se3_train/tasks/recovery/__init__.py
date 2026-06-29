"""Recovery discovery 共享 MDP 实现。

这个包保留 reward/event/curriculum/base cfg 等共享代码，但不再注册独立
``SE3-WheelLegged-Recovery-GRU`` 任务。正式 recovery 训练入口只保留
``SE3-WheelLegged-Recovery-Discovery-GRU``。
"""

from __future__ import annotations

__all__: list[str] = []
