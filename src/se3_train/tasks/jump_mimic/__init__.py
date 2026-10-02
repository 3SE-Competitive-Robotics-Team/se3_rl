"""跳跃 mimic 任务：单独的跳跃策略（MLP、单帧 + 参考帧、从头训），见 env_cfg.py。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

from .env_cfg import (
    JUMP_MIMIC_J2_MAX_HEIGHT_ERROR,
    JUMP_MIMIC_J3_MAX_LIN_VEL_X,
    JUMP_MIMIC_J4_REFERENCE_HEIGHTS,
    JUMP_MIMIC_J10_PHASE_TIME_SCALE_S,
    env_cfg,
)
from .reference import NOCROUCH_REFERENCE_DIR

TASK_ID = "SE3-WheelLegged-Jump-Mimic-MLP"
EXP_J2_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J2"
"""J2 = J1 + 机身高度偏离终止阈值 0.25 → 0.12 m（不起跳即被终止）。"""
EXP_J3_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J3"
"""J3 = J2 + 放开 vx 指令（±1.5 m/s）做前进跳。"""
_J3 = {
    "max_height_error": JUMP_MIMIC_J2_MAX_HEIGHT_ERROR,
    "max_lin_vel_x": JUMP_MIMIC_J3_MAX_LIN_VEL_X,
}
EXP_J4_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J4"
"""J4 = J3 + 0.50 m 参考（四条参考 0.20/0.30/0.40/0.50）。"""
_J4 = {**_J3, "reference_heights": JUMP_MIMIC_J4_REFERENCE_HEIGHTS}
EXP_J7_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J7"
"""J7 = J4 + 参考去掉起始停顿与下蹲、站姿 0.22 m（flag 一到就蹬，jump_ref_v2_nocrouch_h022）。"""
_J7 = {**_J4, "reference_dir": NOCROUCH_REFERENCE_DIR}
EXP_J8_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J8"
"""J8 = J7 + actor 与 critic 都不看参考帧 / 参考时钟（actor 34 维，POMDP），模仿奖励、偏离终止、RSI 不变。"""
_J8 = {**_J7, "reference_obs": False}
EXP_J9_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J9"
"""J9 = J7 + 去掉 RSI（rsi_prob 0.5 → 0），验证不靠参考中途初始化能否学会跳。"""
_J9 = {**_J7, "rsi_prob": 0.0}
EXP_J10_TASK_ID = "SE3-WheelLegged-Jump-Mimic-Exp-J10"
"""J10 = J9 + 去掉 actor 与 critic 的参考帧 / 参考时钟，改给一维相位 jump_phase = 参考时刻 / 1.5 s（actor 34 维）。"""
_J10 = {
    **_J9,
    "reference_obs": False,
    "phase_time_scale_s": JUMP_MIMIC_J10_PHASE_TIME_SCALE_S,
}


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
    register_mjlab_task(
        task_id=EXP_J3_TASK_ID,
        env_cfg=env_cfg(**_J3),
        play_env_cfg=env_cfg(play=True, **_J3),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J3_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J4_TASK_ID,
        env_cfg=env_cfg(**_J4),
        play_env_cfg=env_cfg(play=True, **_J4),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J4_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J7_TASK_ID,
        env_cfg=env_cfg(**_J7),
        play_env_cfg=env_cfg(play=True, **_J7),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J7_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J8_TASK_ID,
        env_cfg=env_cfg(**_J8),
        play_env_cfg=env_cfg(play=True, **_J8),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J8_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J9_TASK_ID,
        env_cfg=env_cfg(**_J9),
        play_env_cfg=env_cfg(play=True, **_J9),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J9_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_J10_TASK_ID,
        env_cfg=env_cfg(**_J10),
        play_env_cfg=env_cfg(play=True, **_J10),
        rl_cfg=bind_task_name(mlp_rl_cfg(), EXP_J10_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_J2_TASK_ID",
    "EXP_J3_TASK_ID",
    "EXP_J4_TASK_ID",
    "EXP_J7_TASK_ID",
    "EXP_J8_TASK_ID",
    "EXP_J9_TASK_ID",
    "EXP_J10_TASK_ID",
    "TASK_ID",
    "register",
]
