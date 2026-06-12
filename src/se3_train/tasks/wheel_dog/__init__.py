"""WheelDog 机器人任务组。"""

from __future__ import annotations

from . import blind_climb, flat


def register() -> None:
    """注册 WheelDog 任务组内全部正式任务。"""
    flat.register()
    blind_climb.register()


__all__ = ["register"]
