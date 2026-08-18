"""统一的倒地自启 Recovery-Discovery GRU 与 MLP 任务。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg, history_env_cfg, ungrouped_env_cfg
from .rl_cfg import mlp_rl_cfg, rl_cfg, ungrouped_mlp_rl_cfg

TASK_ID = "SE3-WheelLegged-Recovery-Discovery-GRU"
MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-MLP"
HISTORY_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-History-MLP"
GROUPED_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-Grouped-MLP"
UNGROUPED_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-Ungrouped-MLP"


def register() -> None:
    """注册环境与 PPO 一致、仅网络结构不同的 GRU 和 MLP 任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=rl_cfg(),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=MLP_TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=mlp_rl_cfg(),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=HISTORY_MLP_TASK_ID,
        env_cfg=history_env_cfg(),
        play_env_cfg=history_env_cfg(play=True),
        rl_cfg=mlp_rl_cfg(),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=GROUPED_MLP_TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=mlp_rl_cfg(),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=UNGROUPED_MLP_TASK_ID,
        env_cfg=ungrouped_env_cfg(),
        play_env_cfg=ungrouped_env_cfg(play=True),
        rl_cfg=ungrouped_mlp_rl_cfg(),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "GROUPED_MLP_TASK_ID",
    "HISTORY_MLP_TASK_ID",
    "MLP_TASK_ID",
    "TASK_ID",
    "UNGROUPED_MLP_TASK_ID",
    "env_cfg",
    "history_env_cfg",
    "mlp_rl_cfg",
    "register",
    "rl_cfg",
    "ungrouped_env_cfg",
    "ungrouped_mlp_rl_cfg",
]
