"""SE3 训练任务包。

每个子目录表示一个可独立注册的 MJLab task。新实验必须在对应 task 目录内
收拢观测、奖励、指令、课程、终止条件和注册代码。
"""

from __future__ import annotations

from . import flat, jump, jump_obs42, jump_pretrain, rough


def register_all_tasks() -> None:
    """注册当前包内全部训练任务。"""
    rough.register()
    flat.register()
    jump_pretrain.register()
    jump.register()
    jump_obs42.register()


__all__ = ["register_all_tasks"]
