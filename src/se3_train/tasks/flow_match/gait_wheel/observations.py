"""本任务使用的 actor/critic 观测函数。"""

from __future__ import annotations

from typing import Any

from se3_train.tasks.flow_match.common import observations as _observations

__all__ = list(_observations.__all__)


def __getattr__(name: str) -> Any:
    """转发共享观测函数。"""
    if name in __all__:
        return getattr(_observations, name)
    raise AttributeError(name)
