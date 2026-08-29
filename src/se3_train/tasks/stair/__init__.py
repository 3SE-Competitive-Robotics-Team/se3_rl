"""倒金字塔台阶 GRU 任务。"""

from __future__ import annotations

import os

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3StairWarmStartRunner

from .env_cfg import env_cfg, history_mlp_env_cfg
from .rl_cfg import mlp_rl_cfg, rl_cfg

TASK_ID = "SE3-WheelLegged-Stair-GRU"
MLP_TASK_ID = "SE3-WheelLegged-Stair-MLP"
TRAIN_VIEW_TASK_ID = f"{TASK_ID}-TrainView"
WATCH_USE_TRAIN_ENV_VAR = "SE3_WATCH_USE_TRAIN_ENV"


def _use_training_play_env() -> bool:
    raw = os.environ.get(WATCH_USE_TRAIN_ENV_VAR, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def register() -> None:
    """注册倒金字塔台阶 GRU、MLP 和训练视图任务。"""
    task_play_env_cfg = env_cfg() if _use_training_play_env() else env_cfg(play=True)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=task_play_env_cfg,
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
        runner_cls=Se3StairWarmStartRunner,
    )
    register_mjlab_task(
        task_id=MLP_TASK_ID,
        env_cfg=history_mlp_env_cfg(),
        play_env_cfg=(
            history_mlp_env_cfg() if _use_training_play_env() else history_mlp_env_cfg(play=True)
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), MLP_TASK_ID),
        runner_cls=Se3StairWarmStartRunner,
    )
    register_mjlab_task(
        task_id=TRAIN_VIEW_TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(),
        rl_cfg=bind_task_name(rl_cfg(), TRAIN_VIEW_TASK_ID),
        runner_cls=Se3StairWarmStartRunner,
    )


__all__ = [
    "MLP_TASK_ID",
    "TASK_ID",
    "TRAIN_VIEW_TASK_ID",
    "env_cfg",
    "history_mlp_env_cfg",
    "mlp_rl_cfg",
    "register",
    "rl_cfg",
]
