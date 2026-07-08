"""host 侧诊断日志的统一降频工具。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

DEFAULT_DIAGNOSTIC_LOG_INTERVAL_STEPS = 256
_GLOBAL_INTERVAL_ENV = "SE3_DIAG_LOG_INTERVAL_STEPS"
_DISABLE_ENV = "SE3_DISABLE_DIAG_LOG"
_GENERIC_INTERVAL_ATTR = "_se3_diag_log_interval_steps"


def _disabled_by_env() -> bool:
    value = os.environ.get(_DISABLE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_interval(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def diagnostic_log_interval(
    env: ManagerBasedRlEnv,
    default_interval: int = DEFAULT_DIAGNOSTIC_LOG_INTERVAL_STEPS,
    *,
    attr_name: str | None = None,
) -> int:
    """返回当前诊断日志间隔。

    设置 ``SE3_DIAG_LOG_INTERVAL_STEPS=0`` 或 ``SE3_DISABLE_DIAG_LOG=1`` 可以在
    benchmark 时关闭非关键诊断标量日志。
    """
    if _disabled_by_env():
        return 0

    interval = _coerce_interval(os.environ.get(_GLOBAL_INTERVAL_ENV))
    if interval is not None:
        return max(0, interval)

    if attr_name:
        interval = _coerce_interval(getattr(env, attr_name, None))
        if interval is not None:
            return max(0, interval)

    interval = _coerce_interval(getattr(env, _GENERIC_INTERVAL_ATTR, None))
    if interval is not None:
        return max(0, interval)

    return max(0, int(default_interval))


def should_log_diagnostics(
    env: ManagerBasedRlEnv,
    default_interval: int = DEFAULT_DIAGNOSTIC_LOG_INTERVAL_STEPS,
    *,
    attr_name: str | None = None,
) -> bool:
    """判断当前 policy step 是否需要写 host 侧诊断标量。"""
    interval = diagnostic_log_interval(env, default_interval, attr_name=attr_name)
    if interval <= 0:
        return False
    step = int(getattr(env, "common_step_counter", 0))
    return interval <= 1 or (step - 1) % interval == 0
