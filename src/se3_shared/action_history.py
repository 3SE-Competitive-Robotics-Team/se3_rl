"""policy action 历史差分工具。"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

FRONT_ACTION_PERIOD = 2.0
FRONT_ACTION_INDICES = (0, 2)


def periodic_policy_action_delta_np(action: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """计算 6D policy action 差分；LF/RF 前杆按整周等价取最短差分。"""
    delta = np.asarray(action, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
    return wrap_front_action_delta_np(delta)


def periodic_policy_action_second_difference_np(
    action: np.ndarray,
    previous: np.ndarray,
    previous_previous: np.ndarray,
) -> np.ndarray:
    """计算 6D policy action 二阶差分；LF/RF 前杆使用相邻最短差分。"""
    current_delta = periodic_policy_action_delta_np(action, previous)
    previous_delta = periodic_policy_action_delta_np(previous, previous_previous)
    return current_delta - previous_delta


def wrap_front_action_delta_np(delta: np.ndarray) -> np.ndarray:
    """把 LF/RF 前杆 action 差分折回 [-1, 1)，其余维度保持线性。"""
    out = np.asarray(delta, dtype=np.float64).copy()
    if out.ndim == 0:
        return out
    width = out.shape[-1]
    for index in FRONT_ACTION_INDICES:
        if width > index:
            out[..., index] = _wrap_period_np(out[..., index], FRONT_ACTION_PERIOD)
    return out


def wrap_front_action_value_delta_np(delta: np.ndarray) -> np.ndarray:
    """把单个前杆 action 差值折回 [-1, 1)。"""
    return _wrap_period_np(np.asarray(delta, dtype=np.float64), FRONT_ACTION_PERIOD)


def periodic_policy_action_delta_torch(
    action: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """计算 6D policy action 差分；LF/RF 前杆按整周等价取最短差分。"""
    delta = action - previous
    return wrap_front_action_delta_torch(delta)


def periodic_policy_action_second_difference_torch(
    action: torch.Tensor,
    previous: torch.Tensor,
    previous_previous: torch.Tensor,
) -> torch.Tensor:
    """计算 6D policy action 二阶差分；LF/RF 前杆使用相邻最短差分。"""
    current_delta = periodic_policy_action_delta_torch(action, previous)
    previous_delta = periodic_policy_action_delta_torch(previous, previous_previous)
    return current_delta - previous_delta


def wrap_front_action_delta_torch(delta: torch.Tensor) -> torch.Tensor:
    """把 LF/RF 前杆 action 差分折回 [-1, 1)，其余维度保持线性。"""
    out = delta.clone()
    if out.ndim == 0:
        return out
    width = out.shape[-1]
    for index in FRONT_ACTION_INDICES:
        if width > index:
            out[..., index] = _wrap_period_torch(out[..., index], FRONT_ACTION_PERIOD)
    return out


def wrap_front_action_value_delta_torch(delta: torch.Tensor) -> torch.Tensor:
    """把单个前杆 action 差值折回 [-1, 1)。"""
    return _wrap_period_torch(delta, FRONT_ACTION_PERIOD)


def _wrap_period_np(value: np.ndarray, period: float) -> np.ndarray:
    half = 0.5 * float(period)
    return np.remainder(value + half, float(period)) - half


def _wrap_period_torch(value: torch.Tensor, period: float) -> torch.Tensor:
    half = 0.5 * float(period)
    return torch.remainder(value + half, float(period)) - half
