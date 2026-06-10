"""FlowMatch 蒸馏前置 task 组。"""

from __future__ import annotations

from . import (
    gait_stage1,
    gait_stage2,
    gait_stage3,
    gait_wheel,
    jump,
    loco_script,
    wheel,
    wheel_leg,
)


def register() -> None:
    """注册 FlowMatch task 组内全部正式任务。"""
    wheel.register()
    gait_stage1.register()
    gait_stage2.register()
    gait_stage3.register()
    wheel_leg.register()
    gait_wheel.register()
    jump.register()
    loco_script.register()


__all__ = ["register"]
