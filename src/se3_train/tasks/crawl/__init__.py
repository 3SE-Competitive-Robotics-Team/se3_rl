"""趴地移动任务：机身压在底盘导轮上、两主轮在后方推动的平地移动技能。

34 维 actor 观测与 Flat 相同（高度指令固定为趴地高度），单帧 MLP，从零训练。
用带导轮的变体 MJCF，见 env_cfg.py 与 docs/plan/crawl_guide_wheels_20261017.md。
"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg
from .rl_cfg import mlp_rl_cfg

TASK_ID = "SE3-WheelLegged-Crawl-MLP"


def register() -> None:
    """注册趴地移动 MLP 任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["TASK_ID", "register"]
