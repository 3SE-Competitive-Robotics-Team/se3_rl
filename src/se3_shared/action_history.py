"""policy action 历史差分工具。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

FRONT_ACTION_PERIOD = 2.0
FRONT_ACTION_INDICES = (0, 2)
FRONT_PHYSICAL_PERIOD = 2.0 * math.pi

FrontActionPeriod = float | Sequence[float] | np.ndarray | Literal[False]


def front_action_period_from_scale(action_scale: float) -> float:
    """由前杆 action scale 推出 action 空间内的整周周期。"""
    scale = float(action_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"front action scale must be positive and finite, got {action_scale}")
    return FRONT_PHYSICAL_PERIOD / scale


def front_action_periods_from_scales(
    action_scales: float | Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """从 4D/6D action scale 中读取 LF0/RF0 的 action 周期。"""
    scales = _as_flat_float_list(action_scales)
    if len(scales) == 1:
        period = front_action_period_from_scale(scales[0])
        return period, period
    periods = []
    for index in FRONT_ACTION_INDICES:
        if len(scales) <= index:
            raise ValueError(f"front action scales must contain index {index}, got {action_scales}")
        periods.append(front_action_period_from_scale(scales[index]))
    return periods[0], periods[1]


def periodic_policy_action_delta_np(
    action: np.ndarray,
    previous: np.ndarray,
    *,
    front_action_period: FrontActionPeriod | None = None,
) -> np.ndarray:
    """计算 6D policy action 差分；LF/RF 前杆按整周等价取最短差分。"""
    delta = np.asarray(action, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
    return wrap_front_action_delta_np(delta, front_action_period=front_action_period)


def periodic_policy_action_second_difference_np(
    action: np.ndarray,
    previous: np.ndarray,
    previous_previous: np.ndarray,
    *,
    front_action_period: FrontActionPeriod | None = None,
) -> np.ndarray:
    """计算 6D policy action 二阶差分；LF/RF 前杆使用相邻最短差分。"""
    current_delta = periodic_policy_action_delta_np(
        action,
        previous,
        front_action_period=front_action_period,
    )
    previous_delta = periodic_policy_action_delta_np(
        previous,
        previous_previous,
        front_action_period=front_action_period,
    )
    return current_delta - previous_delta


def wrap_front_action_delta_np(
    delta: np.ndarray,
    *,
    front_action_period: FrontActionPeriod | None = None,
) -> np.ndarray:
    """把 LF/RF 前杆 action 差分折回最短等价差分，其余维度保持线性。"""
    out = np.asarray(delta, dtype=np.float64).copy()
    if out.ndim == 0:
        return out
    if front_action_period is False:
        return out
    width = out.shape[-1]
    for order, index in enumerate(FRONT_ACTION_INDICES):
        if width > index:
            period = _front_period_at(front_action_period, order, index)
            out[..., index] = _wrap_period_np(out[..., index], period)
    return out


def wrap_front_action_value_delta_np(
    delta: np.ndarray,
    *,
    front_action_period: float | Literal[False] | None = None,
) -> np.ndarray:
    """把单个前杆 action 差值折回最短等价差分。"""
    if front_action_period is False:
        return np.asarray(delta, dtype=np.float64)
    period = (
        FRONT_ACTION_PERIOD
        if front_action_period is None
        else _validate_period(front_action_period)
    )
    return _wrap_period_np(np.asarray(delta, dtype=np.float64), period)


def periodic_policy_action_delta_torch(
    action: torch.Tensor,
    previous: torch.Tensor,
    *,
    front_action_period: FrontActionPeriod | None = None,
) -> torch.Tensor:
    """计算 6D policy action 差分；LF/RF 前杆按整周等价取最短差分。"""
    delta = action - previous
    return wrap_front_action_delta_torch(delta, front_action_period=front_action_period)


def periodic_policy_action_second_difference_torch(
    action: torch.Tensor,
    previous: torch.Tensor,
    previous_previous: torch.Tensor,
    *,
    front_action_period: FrontActionPeriod | None = None,
) -> torch.Tensor:
    """计算 6D policy action 二阶差分；LF/RF 前杆使用相邻最短差分。"""
    current_delta = periodic_policy_action_delta_torch(
        action,
        previous,
        front_action_period=front_action_period,
    )
    previous_delta = periodic_policy_action_delta_torch(
        previous,
        previous_previous,
        front_action_period=front_action_period,
    )
    return current_delta - previous_delta


def wrap_front_action_delta_torch(
    delta: torch.Tensor,
    *,
    front_action_period: FrontActionPeriod | None = None,
) -> torch.Tensor:
    """把 LF/RF 前杆 action 差分折回最短等价差分，其余维度保持线性。"""
    out = delta.clone()
    if out.ndim == 0:
        return out
    if front_action_period is False:
        return out
    width = out.shape[-1]
    for order, index in enumerate(FRONT_ACTION_INDICES):
        if width > index:
            period = _front_period_at(front_action_period, order, index)
            out[..., index] = _wrap_period_torch(out[..., index], period)
    return out


def wrap_front_action_value_delta_torch(
    delta: torch.Tensor,
    *,
    front_action_period: float | Literal[False] | None = None,
) -> torch.Tensor:
    """把单个前杆 action 差值折回最短等价差分。"""
    if front_action_period is False:
        return delta
    period = (
        FRONT_ACTION_PERIOD
        if front_action_period is None
        else _validate_period(front_action_period)
    )
    return _wrap_period_torch(delta, period)


def _front_period_at(
    front_action_period: FrontActionPeriod | None,
    front_order: int,
    action_index: int,
) -> float:
    if front_action_period is None:
        return FRONT_ACTION_PERIOD
    periods = _as_flat_float_list(front_action_period)
    if len(periods) == 1:
        return _validate_period(periods[0])
    if len(periods) == len(FRONT_ACTION_INDICES):
        return _validate_period(periods[front_order])
    if len(periods) > action_index:
        return _validate_period(periods[action_index])
    raise ValueError(
        f"front action period must be scalar, 2D front periods, or full action periods, got {front_action_period}"
    )


def _as_flat_float_list(value: float | Sequence[float] | np.ndarray) -> list[float]:
    if torch is not None and isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().reshape(-1).cpu().tolist()]
    return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def _validate_period(period: float) -> float:
    value = float(period)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"front action period must be positive and finite, got {period}")
    return value


def _wrap_period_np(value: np.ndarray, period: float) -> np.ndarray:
    half = 0.5 * float(period)
    return np.remainder(value + half, float(period)) - half


def _wrap_period_torch(value: torch.Tensor, period: float) -> torch.Tensor:
    half = 0.5 * float(period)
    return torch.remainder(value + half, float(period)) - half
