"""崎岖地形 MLP 行走任务（含地形课程与 step_up 状态机）。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg
from .rl_cfg import amp_rl_cfg, rl_cfg
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# 只含上/下台阶与平地的定向评测入口：地形课程照常，用 terrain level 定位策略能上到多高的台阶。
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# 只换地形、不开状态机的单变量对照：用来量“台阶前瞻辅助”本身值多少。
NO_STEP_UP_TASK_ID = "SE3-WheelLegged-Rough-NoStepUp"
# AMP：加 amp 观测组 + 判别器风格奖励（se3_train.amp），专家数据集见 docs/amp_dataset.md。
AMP_TASK_ID = "SE3-WheelLegged-Rough-AMP"


def register() -> None:
    """注册崎岖地形行走任务与两个对照入口。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=STAIR_EVAL_TASK_ID,
        env_cfg=env_cfg(terrain_generator=stair_only_terrains_cfg()),
        play_env_cfg=env_cfg(play=True, terrain_generator=stair_only_terrains_cfg()),
        rl_cfg=bind_task_name(rl_cfg(), STAIR_EVAL_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=NO_STEP_UP_TASK_ID,
        env_cfg=env_cfg(step_up_enabled=False),
        play_env_cfg=env_cfg(play=True, step_up_enabled=False),
        rl_cfg=bind_task_name(rl_cfg(), NO_STEP_UP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=AMP_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(amp_rl_cfg(), AMP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "AMP_TASK_ID",
    "NO_STEP_UP_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "env_cfg",
    "register",
    "rl_cfg",
]
