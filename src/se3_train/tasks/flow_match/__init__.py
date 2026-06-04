"""FlowMatch 蒸馏前置 task 组。"""

from __future__ import annotations

from . import (
    gait_finetune,
    gait_pretrain,
    gait_wheel,
    jump,
    wheel,
    wheel_leg,
    wheel_vio,
)


def register() -> None:
    """注册 FlowMatch task 组内全部正式任务。"""
    wheel.register()
    wheel_vio.register()
    gait_pretrain.register()
    gait_finetune.register()
    wheel_leg.register()
    gait_wheel.register()
    jump.register()


__all__ = ["register"]
