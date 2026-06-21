"""SE3 训练任务包。

每个子目录表示一个可独立注册的 MJLab task。新实验必须在对应 task 目录内
收拢观测、奖励、指令、课程、终止条件和注册代码。
"""

from __future__ import annotations

from . import (
    flat,
    jump_finetune,
    jump_pretrain,
    recovery,
    recovery_discovery,
    recovery_finetune,
    rough,
    stair,
    wheel_dog,
)


def register_all_tasks() -> None:
    """注册当前包内全部训练任务。"""
    rough.register()
    flat.register()
    recovery.register()
    recovery_discovery.register()
    recovery_finetune.register()
    stair.register()
    jump_pretrain.register()
    jump_finetune.register()
    wheel_dog.register()


__all__ = ["register_all_tasks"]
