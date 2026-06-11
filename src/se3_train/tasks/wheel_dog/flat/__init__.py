"""WheelDog 平地速度跟随 GRU 任务。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.tasks.common import Se3OnPolicyRunner

from .env_cfg import env_cfg
from .rl_cfg import rl_cfg

TASK_ID = "SE3-WheelDog-Flat-GRU"


def register() -> None:
    """注册 WheelDog 平地速度跟随任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=rl_cfg(),
        runner_cls=Se3OnPolicyRunner,
    )


__all__ = ["TASK_ID", "env_cfg", "register", "rl_cfg"]
