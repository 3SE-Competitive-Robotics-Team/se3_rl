"""纯倒地自启 Recovery 专家任务（专家数据管线 E1）。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import expert_env_cfg
from .rl_cfg import mlp_rl_cfg

EXPERT_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Expert-MLP"


def register() -> None:
    """注册纯 recover 专家任务（五帧 History-MLP）。"""
    register_mjlab_task(
        task_id=EXPERT_MLP_TASK_ID,
        env_cfg=expert_env_cfg(),
        play_env_cfg=expert_env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXPERT_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXPERT_MLP_TASK_ID",
    "expert_env_cfg",
    "mlp_rl_cfg",
    "register",
]
