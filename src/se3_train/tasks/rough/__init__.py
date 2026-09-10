"""崎岖地形 MLP 行走任务（含地形课程与地形感知高度下限）。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg
from .reward_ablation import REWARD_ABLATIONS, reward_ablation_env_cfg
from .rl_cfg import amp_rl_cfg, rl_cfg
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# 只含上/下台阶与平地的定向评测入口：地形课程照常，用 terrain level 定位策略能上到多高的台阶。
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# AMP：加 amp 观测组 + 判别器风格奖励（se3_train.amp），专家数据集见 docs/amp_dataset.md。
AMP_TASK_ID = "SE3-WheelLegged-Rough-AMP"
# 消融入口（2026-09-09）：A12 实测 AMP 在台阶列贡献 +9.1/秒，占该列奖励信号约 90%，
# 而判别器只分到 ±0.24、style_reward 长期贴 0.6——疑似发的是常数存活奖金而非姿态信息。
# C1 只改风格奖励权重 15→3（A2 的值）；C2 只换成时序打乱的专家数据（逐帧分布不变，
# 跨帧结构破坏，见 scripts/make_shuffled_amp_dataset.py）。其余与 AMP_TASK_ID 逐项相同。
AMP_ABL_W3_TASK_ID = "SE3-WheelLegged-Rough-AMP-AblW3"
AMP_ABL_SHUFFLED_TASK_ID = "SE3-WheelLegged-Rough-AMP-AblShuffled"
AMP_ABL_SHUFFLED_DATASET = "assets/amp/fudan_stairs20_shuffled/amp_training.pkl"


def register() -> None:
    """注册崎岖地形行走任务与定向评测、AMP 两个入口。"""
    for task_id, (progress_weight, support_weight) in REWARD_ABLATIONS.items():
        register_mjlab_task(
            task_id=task_id,
            env_cfg=reward_ablation_env_cfg(
                progress_weight=progress_weight, support_weight=support_weight
            ),
            play_env_cfg=reward_ablation_env_cfg(
                progress_weight=progress_weight, support_weight=support_weight, play=True
            ),
            rl_cfg=bind_task_name(rl_cfg(), task_id),
            runner_cls=Se3ProfiledOnPolicyRunner,
        )
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
        task_id=AMP_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(amp_rl_cfg(), AMP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=AMP_ABL_W3_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(amp_rl_cfg(reward_weight=3.0), AMP_ABL_W3_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=AMP_ABL_SHUFFLED_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(
            amp_rl_cfg(dataset_root=AMP_ABL_SHUFFLED_DATASET), AMP_ABL_SHUFFLED_TASK_ID
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "AMP_ABL_SHUFFLED_DATASET",
    "AMP_ABL_SHUFFLED_TASK_ID",
    "AMP_ABL_W3_TASK_ID",
    "AMP_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "env_cfg",
    "register",
    "rl_cfg",
]
