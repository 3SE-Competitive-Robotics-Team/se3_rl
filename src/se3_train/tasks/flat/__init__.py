"""平地行走任务：GRU、单帧 MLP、五帧 History-MLP，以及抖动对照实验的 Exp-* 变体。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import (
    FLAT_ACTION_DELAY_RANGE_ONE_TO_THREE_STEPS_S,
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_BAD_TILT_LIMITS_TIGHT_DEG,
    FLAT_CMD_VEL_DEADBAND_WIDE,
    FLAT_MAX_ANG_VEL_YAW_LOW,
    FLAT_WHEEL_ACTION_SCALE,
    FLAT_WHEEL_CONTACT_WEIGHT_HEAVY,
    env_cfg,
    history_env_cfg,
)
from .rl_cfg import mlp_rl_cfg, rl_cfg

TASK_ID = "SE3-WheelLegged-Flat-GRU"
MLP_TASK_ID = "SE3-WheelLegged-Flat-MLP"
HISTORY_MLP_TASK_ID = "SE3-WheelLegged-Flat-History-MLP"

# 2026-09-03 抖动对照实验（单变量，全部以 Flat-MLP 为基线，仅换一个奖励/课程/延迟参数）。
# 基线本身用 SE3-WheelLegged-Flat-MLP 换随机种子重跑，用来量 run 间方差本底。
EXP_CMD_DEADBAND_TASK_ID = "SE3-WheelLegged-Flat-Exp-CmdDeadband"
EXP_WHEEL_CONTACT_TASK_ID = "SE3-WheelLegged-Flat-Exp-WheelContact"
EXP_TILT_BARRIER_TASK_ID = "SE3-WheelLegged-Flat-Exp-TiltBarrier"
EXP_ACTION_DELAY_TASK_ID = "SE3-WheelLegged-Flat-Exp-ActionDelay"
EXP_YAW_CURRICULUM_TASK_ID = "SE3-WheelLegged-Flat-Exp-YawCurriculum"

# Flat 三任务与 Exp-* 共享的基线契约：轮 scale 15 + 弹簧时代 action_smoothness 定价。
_FLAT_SPRING_BASE = {
    "wheel_action_scale": FLAT_WHEEL_ACTION_SCALE,
    "action_smoothness": FLAT_ACTION_SMOOTHNESS_SPRING,
}


def _register_flat_mlp_variant(task_id: str, **env_kwargs: object) -> None:
    """注册一个只改 env_cfg 单个旋钮的 Flat-MLP 变体；网络与 PPO 配置完全相同。"""
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg(**_FLAT_SPRING_BASE, **env_kwargs),  # type: ignore[arg-type]
        play_env_cfg=env_cfg(play=True, **_FLAT_SPRING_BASE, **env_kwargs),  # type: ignore[arg-type]
        rl_cfg=bind_task_name(mlp_rl_cfg(), task_id),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


def register() -> None:
    """注册平地 GRU、单帧 MLP、五帧 History-MLP 与 5 个 Exp-* 对照变体。"""
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
    # A1 放宽 command_velocity_error 死区：最大单项罚（占总罚 38%）且 2k 后不再下降，
    # 0.05 m/s 对轮式倒立摆物理不可达，等于在直接奖励高增益速度反馈。
    _register_flat_mlp_variant(
        EXP_CMD_DEADBAND_TASK_ID,
        command_velocity_deadband=FLAT_CMD_VEL_DEADBAND_WIDE,
    )
    # A2 加重轮离地罚：唯一一个越训越差的项（轮离地率 3%→6.6%），-10 拦不住策略去蹦。
    _register_flat_mlp_variant(
        EXP_WHEEL_CONTACT_TASK_ID,
        flat_wheel_contact_weight=FLAT_WHEEL_CONTACT_WEIGHT_HEAVY,
    )
    # B1 收紧倾角 barrier：实测平均倾角已在 12° 附近，soft 10° 起价太晚只收 0.056。
    _register_flat_mlp_variant(
        EXP_TILT_BARRIER_TASK_ID,
        bad_tilt_limits_deg=FLAT_BAD_TILT_LIMITS_TIGHT_DEG,
    )
    # B2 动作延迟拉到 1-3 个控制步：现在只有 4-6 ms，不足一个 20 ms 控制步，
    # 高频动作在训练里几乎没有真实代价。
    _register_flat_mlp_variant(
        EXP_ACTION_DELAY_TASK_ID,
        action_delay_range_s=FLAT_ACTION_DELAY_RANGE_ONE_TO_THREE_STEPS_S,
    )
    # B3 压 yaw 课程上限：课程已顶到 12 rad/s（688°/s，差速可行域上限），
    # 把策略逼进高增益区，零指令时用的是同一套权重。
    _register_flat_mlp_variant(
        EXP_YAW_CURRICULUM_TASK_ID,
        max_ang_vel_yaw=FLAT_MAX_ANG_VEL_YAW_LOW,
    )


__all__ = [
    "EXP_ACTION_DELAY_TASK_ID",
    "EXP_CMD_DEADBAND_TASK_ID",
    "EXP_TILT_BARRIER_TASK_ID",
    "EXP_WHEEL_CONTACT_TASK_ID",
    "EXP_YAW_CURRICULUM_TASK_ID",
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
