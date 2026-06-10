"""本任务使用的事件函数。"""

from __future__ import annotations

from typing import Any

from se3_train.tasks.flow_match.common import events as _events

__all__ = list(_events.__all__)


def __getattr__(name: str) -> Any:
    """转发共享事件函数。"""
    if name in __all__:
        return getattr(_events, name)
    raise AttributeError(name)
