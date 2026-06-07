"""本任务使用的奖励函数。"""

from __future__ import annotations

from typing import Any

from se3_train.tasks.flow_match.common import rewards as _rewards

__all__ = list(_rewards.__all__)


def __getattr__(name: str) -> Any:
    """转发共享奖励函数。"""
    if name in __all__:
        return getattr(_rewards, name)
    raise AttributeError(name)
