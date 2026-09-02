"""平地行走任务：GRU、单帧 MLP 与五帧 History-MLP。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
    env_cfg,
    history_env_cfg,
)
from .rl_cfg import mlp_rl_cfg, rl_cfg

TASK_ID = "SE3-WheelLegged-Flat-GRU"
MLP_TASK_ID = "SE3-WheelLegged-Flat-MLP"
HISTORY_MLP_TASK_ID = "SE3-WheelLegged-Flat-History-MLP"


def register() -> None:
    """注册平地 GRU、单帧 MLP 与五帧 History-MLP 行走任务。"""
    # 2026-09-02 轮 scale 45→15 + 动作空间罚项 (15/45)² 补偿，以及 action_smoothness 弹簧时代重定价
    # （-0.12/cap 320/轮 2×）：仅 Flat 三个任务生效，env_cfg() 默认仍为 45 + LEGACY 供
    # rough/stair/jump/flow_match 继承线保持旧契约。
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=MLP_TASK_ID,
        env_cfg=env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # 五帧展平历史 MLP：网络与 PPO 配置同 Flat-MLP，仅 actor 观测为 34×5=170 维。
    register_mjlab_task(
        task_id=HISTORY_MLP_TASK_ID,
        env_cfg=history_env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=history_env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), HISTORY_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "FLAT_ACTION_SMOOTHNESS_SPRING",
    "FLAT_WHEEL_ACTION_SCALE",
    "HISTORY_MLP_TASK_ID",
    "MLP_TASK_ID",
    "TASK_ID",
    "env_cfg",
    "history_env_cfg",
    "mlp_rl_cfg",
    "register",
    "rl_cfg",
]
