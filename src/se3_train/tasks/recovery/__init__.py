"""Recovery-Discovery 的内部共享实现。

该包仅保留环境基配置、奖励、事件和课程等共享代码，不注册独立任务。
正式训练入口统一由 ``se3_train.tasks.recovery_discovery`` 提供。
"""

from __future__ import annotations

__all__: list[str] = []
