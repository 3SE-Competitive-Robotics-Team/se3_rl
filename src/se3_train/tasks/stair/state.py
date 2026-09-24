"""CTBC 台阶爬升状态管理（实现已迁至 se3_train.mdp.ctbc_state，此处保留导入路径）。"""

from __future__ import annotations

from se3_train.mdp.ctbc_state import (
    _HIP_FEEDFORWARD_RATIO,
    _KNEE_FEEDFORWARD_RATIO,
    StairClimbState,
    _CtbcProfile,
)

__all__ = [
    "_HIP_FEEDFORWARD_RATIO",
    "_KNEE_FEEDFORWARD_RATIO",
    "StairClimbState",
    "_CtbcProfile",
]
