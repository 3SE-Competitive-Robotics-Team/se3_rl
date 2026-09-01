"""统一的倒地自启 Recovery-Discovery GRU 与 MLP 任务。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import (
    DiscoveryRewardProfile,
    env_cfg,
    history_env_cfg,
    ungrouped_env_cfg,
)
from .rl_cfg import mlp_rl_cfg, rl_cfg, ungrouped_mlp_rl_cfg

TASK_ID = "SE3-WheelLegged-Recovery-Discovery-GRU"
MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-MLP"
HISTORY_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-History-MLP"
GROUPED_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Loco-Grouped-MLP"
GROUPED_GENTLE_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-MLP"
UNGROUPED_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-Ungrouped-MLP"
UNGROUPED_REFORM_A_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-Ungrouped-ReformA-MLP"
UNGROUPED_ARRIVAL_MLP_TASK_ID = "SE3-WheelLegged-Recovery-Discovery-Ungrouped-Arrival-MLP"


def register() -> None:
    """注册 GRU、单帧 MLP、五帧 History-MLP 与分组对照任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=MLP_TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=HISTORY_MLP_TASK_ID,
        env_cfg=history_env_cfg(),
        play_env_cfg=history_env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), HISTORY_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=GROUPED_MLP_TASK_ID,
        env_cfg=history_env_cfg(),
        play_env_cfg=history_env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), GROUPED_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=GROUPED_GENTLE_MLP_TASK_ID,
        env_cfg=history_env_cfg(reward_profile=DiscoveryRewardProfile.GENTLE),
        play_env_cfg=history_env_cfg(
            play=True,
            reward_profile=DiscoveryRewardProfile.GENTLE,
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), GROUPED_GENTLE_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=UNGROUPED_MLP_TASK_ID,
        env_cfg=ungrouped_env_cfg(),
        play_env_cfg=ungrouped_env_cfg(play=True),
        rl_cfg=bind_task_name(ungrouped_mlp_rl_cfg(), UNGROUPED_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=UNGROUPED_REFORM_A_MLP_TASK_ID,
        env_cfg=ungrouped_env_cfg(reward_profile=DiscoveryRewardProfile.REFORM_A),
        play_env_cfg=ungrouped_env_cfg(
            play=True,
            reward_profile=DiscoveryRewardProfile.REFORM_A,
        ),
        rl_cfg=bind_task_name(ungrouped_mlp_rl_cfg(), UNGROUPED_REFORM_A_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=UNGROUPED_ARRIVAL_MLP_TASK_ID,
        env_cfg=ungrouped_env_cfg(reward_profile=DiscoveryRewardProfile.REFORM_AB),
        play_env_cfg=ungrouped_env_cfg(
            play=True,
            reward_profile=DiscoveryRewardProfile.REFORM_AB,
        ),
        rl_cfg=bind_task_name(ungrouped_mlp_rl_cfg(), UNGROUPED_ARRIVAL_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "GROUPED_GENTLE_MLP_TASK_ID",
    "GROUPED_MLP_TASK_ID",
    "HISTORY_MLP_TASK_ID",
    "MLP_TASK_ID",
    "TASK_ID",
    "UNGROUPED_ARRIVAL_MLP_TASK_ID",
    "UNGROUPED_MLP_TASK_ID",
    "UNGROUPED_REFORM_A_MLP_TASK_ID",
    "DiscoveryRewardProfile",
    "env_cfg",
    "history_env_cfg",
    "mlp_rl_cfg",
    "register",
    "rl_cfg",
    "ungrouped_env_cfg",
    "ungrouped_mlp_rl_cfg",
]
