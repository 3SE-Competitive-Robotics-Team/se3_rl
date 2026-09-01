"""Recovery 任务的部署与 sim2sim 共享常量。"""

from __future__ import annotations

from .robot import JointGroup, RobotConfig

RECOVERY_COMMAND_HEIGHT_M = 0.26
"""Recovery 策略的默认站高指令，单位 m。"""

RECOVERY_COMMAND_HEIGHT_RANGE_M = (0.195, 0.390)
"""Recovery FineTune 最终课程覆盖的站高指令范围，单位 m。"""

RECOVERY_COMMAND_LIN_VEL_X_LIMIT_MPS = 1.50
"""Recovery FineTune 最终课程覆盖的前后速度上限，单位 m/s。"""

RECOVERY_COMMAND_YAW_RATE_LIMIT_RAD_S = 1.00
"""Recovery FineTune 最终课程覆盖的 yaw 角速度上限，单位 rad/s。"""

RECOVERY_WHEEL_ACTION_SCALE = 15.0
"""Recovery 策略训练时使用的左右轮 raw action -> 轮速 scale。

2026-09-01 45→15（σ-gate 诊断）：scale=45 时探索噪声的物理轮噪 ±σ×45 rad/s
冷启动期机械性挡死平衡学习，平衡校正 ±5 rad/s 只占 0.11 个动作单位、被 σ 淹没；
15 使校正信号 ≈0.33 a.u.、部署极速 ~56 rad/s 对应 |a|≈3.7（clip 100 无饱和）。
动作空间罚项（action_smoothness wheel_scale、wheel_action_rate）已按 ×9 同步补偿，
见 recovery_discovery/env_cfg.py；指令可行域预算另行解耦（见下一常量）。
"""

RECOVERY_DIFF_DRIVE_MAX_WHEEL_SPEED_RAD_S = 45.0
"""双轮差速指令可行域预算（物理 rad/s，M3508 能力量级），与动作 scale 解耦。

历史上与 RECOVERY_WHEEL_ACTION_SCALE 恰好同值同源；scale 改 15 后此预算保持 45，
否则 vx/yaw 指令组合会被错误压缩到 1/3。
"""

RECOVERY_LEG_ACTION_SCALE = 0.25
"""Recovery 对比实验使用的四个腿部 raw action -> joint target scale。"""

RECOVERY_ACTION_CLIP = 100.0
"""Recovery 策略训练时使用的 raw action clip。"""

RECOVERY_DEFAULT_COMMAND_8D = (
    0.0,
    0.0,
    0.0,
    0.0,
    RECOVERY_COMMAND_HEIGHT_M,
    0.0,
    0.0,
    0.0,
)
"""训练端 actor 使用的 8 维 recovery 默认 command。"""

RECOVERY_DEFAULT_STM_COMMAND_5D = RECOVERY_DEFAULT_COMMAND_8D[:5]
"""STM32 上行协议当前携带的 5 维 recovery 默认 command。"""


def recovery_action_scale(robot_cfg: RobotConfig | None = None) -> tuple[float, ...]:
    """返回 recovery policy 的 6D action scale，腿部固定 0.25，轮子固定 45rad/s。"""
    cfg = RobotConfig() if robot_cfg is None else robot_cfg
    action_scale = list(cfg.action_scale)
    for index in JointGroup.LEG_ACTUATORS:
        action_scale[index] = RECOVERY_LEG_ACTION_SCALE
    for index in JointGroup.WHEEL_ACTUATORS:
        action_scale[index] = RECOVERY_WHEEL_ACTION_SCALE
    return tuple(float(v) for v in action_scale)


def recovery_robot_config(robot_cfg: RobotConfig | None = None) -> RobotConfig:
    """返回与 recovery 训练 action contract 对齐的 RobotConfig 副本。"""
    cfg = RobotConfig() if robot_cfg is None else robot_cfg
    return cfg.model_copy(
        update={
            "action_scale": recovery_action_scale(cfg),
            "action_clip": RECOVERY_ACTION_CLIP,
        }
    )
