"""跳跃 mimic 任务：单独的跳跃策略（MLP、单帧 + 参考帧、从头训），见 env_cfg.py。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

from .env_cfg import JUMP_MIMIC_J2_MAX_HEIGHT_ERROR, env_cfg

TASK_ID = "SE3-WheelLegged-Jump-Mimic-MLP"
EXP_J2_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J2"
"""J2 = J1 + 机身高度偏离终止阈值 0.25 → 0.12 m（不起跳即被终止）。"""


def register() -> None:
    """注册跳跃 mimic 任务，PPO 与 Flat-MLP 基线相同。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(mlp_rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J2_TASK_ID,
        env_cfg=env_cfg(max_height_error=JUMP_MIMIC_J2_MAX_HEIGHT_ERROR),
        play_env_cfg=env_cfg(play=True, max_height_error=JUMP_MIMIC_J2_MAX_HEIGHT_ERROR),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J2_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["EXP_J2_TASK_ID", "TASK_ID", "register"]
