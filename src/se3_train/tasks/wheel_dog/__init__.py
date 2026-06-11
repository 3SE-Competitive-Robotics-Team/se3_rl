"""WheelDog 机器人任务组。"""

from __future__ import annotations

from . import flat


def register() -> None:
    """注册 WheelDog 任务组内全部正式任务。"""
    flat.register()


__all__ = ["register"]
