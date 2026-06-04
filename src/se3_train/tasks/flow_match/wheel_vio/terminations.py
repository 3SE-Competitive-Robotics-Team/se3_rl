"""本任务使用的终止条件。"""

from __future__ import annotations

from typing import Any

from se3_train.tasks.flow_match.wheel import terminations as _terminations

__all__ = list(_terminations.__all__)


def __getattr__(name: str) -> Any:
    """转发 WHEEL 任务终止条件。"""
    if name in __all__:
        return getattr(_terminations, name)
    raise AttributeError(name)
