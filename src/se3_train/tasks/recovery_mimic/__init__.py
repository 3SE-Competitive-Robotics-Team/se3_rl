"""起身 mimic 任务：跟踪手调起身参考的教师策略（MLP、Rough 30 维本体观测 + 9 维参考特权观测、从头训），配置见 env_cfg.py。

教师之后在 Rough 训练里倒地时监督主策略（动作蒸馏）；动作空间与执行链和 Rough 完全一致。
"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

from .env_cfg import env_cfg

TASK_ID = "SE3-WheelLegged-Recovery-Mimic-MLP"


def register() -> None:
    """注册起身 mimic 任务，PPO 与 Flat-MLP 基线相同。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["TASK_ID", "register"]
