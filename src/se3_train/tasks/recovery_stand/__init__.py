"""纯倒地自起站立 GRU 任务。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import env_cfg
from .rl_cfg import rl_cfg

TASK_ID = "SE3-WheelLegged-Recovery-Stand-GRU"


def register() -> None:
    """注册纯倒地自起站立任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=rl_cfg(),
    )


__all__ = ["TASK_ID", "env_cfg", "register", "rl_cfg"]
