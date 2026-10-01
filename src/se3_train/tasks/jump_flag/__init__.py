"""跳跃 flag 任务（J5）：单独的跳跃策略，jump_flag 窗口触发、按状态给跳跃奖励、无参考帧无相位，见 env_cfg.py。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

from .env_cfg import env_cfg

TASK_ID = "SE3-WheelLegged-Jump-Flag-MLP"
EXP_J6_TASK_ID = "SE3-WheelLegged-Jump-Flag-Exp-J6"
"""J6 = J5 + 起跳段解析参考跟踪奖励（只进奖励，观测仍 34 维）。"""


def register() -> None:
    """注册跳跃 flag 任务，PPO 与 Flat-MLP 基线相同。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J6_TASK_ID,
        env_cfg=env_cfg(takeoff_mimic=True),
        play_env_cfg=env_cfg(play=True, takeoff_mimic=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J6_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["EXP_J6_TASK_ID", "TASK_ID", "register"]
