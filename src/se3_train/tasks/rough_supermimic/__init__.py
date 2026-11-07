"""Rough-SuperMimic（2026-11-06 用户定，第 1 阶段）：SuperMimic 世界模型 + BPTT 训练 Rough 速度跟踪，纯平地。

两个入口共用同一份 env（见 env_cfg.py），单变量 = 训练算法：
- `SE3-WheelLegged-Rough-SuperMimic`：SuperMimicRunner（se3_train.supermimic）
- `SE3-WheelLegged-Rough-SuperMimic-PPO`：同 env 的 Rough MLP PPO 对照
设计与验收见 docs/plan/rough_supermimic_20261106.md。临时入口，结论后并入或删除。
"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.runner import Se3ProfiledOnPolicyRunner
from se3_train.supermimic.runner import SuperMimicRunner
from se3_train.tasks.rough.rl_cfg import rl_cfg as rough_rl_cfg

from .env_cfg import env_cfg
from .rl_cfg import supermimic_rl_cfg

TASK_ID = "SE3-WheelLegged-Rough-SuperMimic"
PPO_TASK_ID = "SE3-WheelLegged-Rough-SuperMimic-PPO"


def register() -> None:
    """注册 SuperMimic 入口与同 env 的 PPO 对照入口。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(supermimic_rl_cfg(), TASK_ID),
        runner_cls=SuperMimicRunner,
    )
    register_mjlab_task(
        task_id=PPO_TASK_ID,
        env_cfg=env_cfg(replay=False),
        play_env_cfg=env_cfg(play=True, replay=False),
        rl_cfg=bind_task_name(rough_rl_cfg(), PPO_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["PPO_TASK_ID", "TASK_ID", "register"]
