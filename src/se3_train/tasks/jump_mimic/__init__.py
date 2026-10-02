"""跳跃 mimic 任务：单独的跳跃策略（MLP、34 维观测、一维相位、从头训），配置见 env_cfg.py。

J1–J10 的对照（2026-10-01 至 10-02）定下的配置即默认：无下蹲参考、前进跳、无 RSI、jump_phase 一维相位；
各对照入口已删除，复现用对应 commit（J7 09493b2、J9 4c7ce37、J10 e3c3a0a）。rough 默认配置（RJ1）按同一套合入跳跃。
"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

from .env_cfg import env_cfg

TASK_ID = "SE3-WheelLegged-Jump-Mimic-MLP"


def register() -> None:
    """注册跳跃 mimic 任务，PPO 与 Flat-MLP 基线相同。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["TASK_ID", "register"]
