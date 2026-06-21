"""Recovery 任务的部署与 sim2sim 共享常量。"""

from __future__ import annotations

RECOVERY_COMMAND_HEIGHT_M = 0.26
"""Recovery 策略的默认站高指令，单位 m。"""

RECOVERY_COMMAND_HEIGHT_RANGE_M = (0.195, 0.390)
"""Recovery FineTune 最终课程覆盖的站高指令范围，单位 m。"""

RECOVERY_COMMAND_LIN_VEL_X_LIMIT_MPS = 1.50
"""Recovery FineTune 最终课程覆盖的前后速度上限，单位 m/s。"""

RECOVERY_COMMAND_YAW_RATE_LIMIT_RAD_S = 1.00
"""Recovery FineTune 最终课程覆盖的 yaw 角速度上限，单位 rad/s。"""

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
