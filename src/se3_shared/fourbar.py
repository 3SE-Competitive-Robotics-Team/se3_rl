"""SerialLeg 四连杆等效开树运动学。"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

FOURBAR_SURROGATE_MARKER = "fourbar_surrogate_marker"

# 这些常量来自闭链 MJCF 的左腿零位几何，单位为 m，投影在运动 x-z 平面。
_KNEE_X = -0.17993464
_KNEE_Z = 0.00489576
_CALF_X = 0.05003347
_CALF_Z = 0.04149627
_DRIVE_X = 0.04009536
_DRIVE_Z = 0.04530576
_COUPLER_LEN = math.hypot(-0.16999653, 0.00108627)
_CALF_LEN = math.hypot(_CALF_X, _CALF_Z)
_CALF_ZERO_ANGLE = math.atan2(_CALF_Z, _CALF_X)
_ACTIVE_LOWER = 0.0
_ACTIVE_UPPER = math.radians(129.95 - 43.46)
_LUT_SIZE = 8192
_TORCH_LUT_CACHE: dict[
    tuple[str, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_NP_LUT_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


def is_fourbar_surrogate_name_set(site_names: tuple[str, ...]) -> bool:
    """判断模型是否带有四连杆等效开树标记。"""
    return FOURBAR_SURROGATE_MARKER in set(site_names)


def output_knee_from_active_angle_torch(active_angle: torch.Tensor) -> torch.Tensor:
    """由主动杆夹角计算左腿输出膝关节角。"""
    alpha_grid, knee_grid, _, _, _ = _fourbar_lut(active_angle.device, active_angle.dtype)
    return _interp_lut(active_angle, alpha_grid, knee_grid)


def _output_knee_from_active_angle_analytic_torch(
    active_angle: torch.Tensor,
) -> torch.Tensor:
    """解析计算左腿输出膝关节角，作为 LUT 构建和精度校验的真值。"""
    alpha = active_angle.clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    beta = -alpha
    cos_b = torch.cos(beta)
    sin_b = torch.sin(beta)
    px = cos_b * _DRIVE_X + sin_b * _DRIVE_Z
    pz = -sin_b * _DRIVE_X + cos_b * _DRIVE_Z

    dx = px - _KNEE_X
    dz = pz - _KNEE_Z
    dist = torch.sqrt((dx * dx + dz * dz).clamp_min(1.0e-12))
    ex = dx / dist
    ez = dz / dist

    along = (_CALF_LEN**2 - _COUPLER_LEN**2 + dist * dist) / (2.0 * dist)
    height = torch.sqrt((_CALF_LEN**2 - along * along).clamp_min(0.0))
    cx = _KNEE_X + along * ex - height * ez
    cz = _KNEE_Z + along * ez + height * ex

    phi = torch.atan2(cz - _KNEE_Z, cx - _KNEE_X)
    zero = torch.as_tensor(_CALF_ZERO_ANGLE, device=phi.device, dtype=phi.dtype)
    return _wrap_angle_torch(zero - phi)


def policy_to_output_pos_torch(policy_pos: torch.Tensor) -> torch.Tensor:
    """把 policy 主动杆语义 [LF, LB, RF, RB] 映射为开树关节 [LF, LF1, RF, RF1]。"""
    out = policy_pos.clone()
    left_alpha = (policy_pos[:, 0] - policy_pos[:, 1]).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    right_alpha = (policy_pos[:, 3] - policy_pos[:, 2]).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    out[:, 1] = output_knee_from_active_angle_torch(left_alpha)
    out[:, 3] = -output_knee_from_active_angle_torch(right_alpha)
    return out


def output_to_policy_pos_torch(output_pos: torch.Tensor) -> torch.Tensor:
    """把开树输出膝角反解回虚拟主动杆语义。"""
    out = output_pos.clone()
    left_alpha = active_angle_from_output_knee_torch(output_pos[:, 1], right_side=False)
    right_alpha = active_angle_from_output_knee_torch(output_pos[:, 3], right_side=True)
    out[:, 1] = output_pos[:, 0] - left_alpha
    out[:, 3] = output_pos[:, 2] + right_alpha
    return out


def output_to_policy_vel_torch(output_pos: torch.Tensor, output_vel: torch.Tensor) -> torch.Tensor:
    """把开树输出速度反解回虚拟主动杆速度语义。"""
    out = output_vel.clone()
    left_alpha = active_angle_from_output_knee_torch(output_pos[:, 1], right_side=False)
    right_alpha = active_angle_from_output_knee_torch(output_pos[:, 3], right_side=True)
    left_j = output_knee_jacobian_torch(left_alpha, right_side=False)
    right_j = output_knee_jacobian_torch(right_alpha, right_side=True)
    left_alpha_dot = output_vel[:, 1] / _safe_denominator_torch(left_j)
    right_alpha_dot = output_vel[:, 3] / _safe_denominator_torch(right_j)
    out[:, 1] = output_vel[:, 0] - left_alpha_dot
    out[:, 3] = output_vel[:, 2] + right_alpha_dot
    return out


def policy_to_output_vel_torch(policy_pos: torch.Tensor, policy_vel: torch.Tensor) -> torch.Tensor:
    """把虚拟主动杆速度映射为开树输出关节速度。"""
    out = policy_vel.clone()
    left_alpha = (policy_pos[:, 0] - policy_pos[:, 1]).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    right_alpha = (policy_pos[:, 3] - policy_pos[:, 2]).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    left_j = output_knee_jacobian_torch(left_alpha, right_side=False)
    right_j = output_knee_jacobian_torch(right_alpha, right_side=True)
    out[:, 1] = left_j * (policy_vel[:, 0] - policy_vel[:, 1])
    out[:, 3] = right_j * (policy_vel[:, 3] - policy_vel[:, 2])
    return out


def active_angle_from_output_knee_torch(
    output_knee: torch.Tensor,
    *,
    right_side: bool,
) -> torch.Tensor:
    """用 LUT 把输出膝角反解为主动杆夹角。"""
    target = -output_knee if right_side else output_knee
    _, _, inverse_knee_grid, inverse_alpha_grid, _ = _fourbar_lut(target.device, target.dtype)
    return _interp_lut(target, inverse_knee_grid, inverse_alpha_grid)


def output_knee_jacobian_torch(active_angle: torch.Tensor, *, right_side: bool) -> torch.Tensor:
    """计算输出膝角对主动杆夹角的数值雅可比。"""
    alpha_grid, _, _, _, jacobian_grid = _fourbar_lut(active_angle.device, active_angle.dtype)
    value = _interp_lut(active_angle, alpha_grid, jacobian_grid)
    return -value if right_side else value


def policy_to_output_torque_torch(
    policy_pos: torch.Tensor, policy_torque: torch.Tensor
) -> torch.Tensor:
    """把虚拟主动杆力矩映射为开树输出关节力矩。"""
    out = policy_torque.clone()
    left_alpha = (policy_pos[:, 0] - policy_pos[:, 1]).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    right_alpha = (policy_pos[:, 3] - policy_pos[:, 2]).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    left_j = output_knee_jacobian_torch(left_alpha, right_side=False)
    right_j = output_knee_jacobian_torch(right_alpha, right_side=True)
    out[:, 0] = policy_torque[:, 0] + policy_torque[:, 1]
    out[:, 1] = -policy_torque[:, 1] / _safe_denominator_torch(left_j)
    out[:, 2] = policy_torque[:, 2] + policy_torque[:, 3]
    out[:, 3] = policy_torque[:, 3] / _safe_denominator_torch(right_j)
    return out


def policy_to_output_pos_np(policy_pos: np.ndarray) -> np.ndarray:
    """NumPy 版本：把 policy 主动杆语义映射为开树输出关节。"""
    arr = np.asarray(policy_pos, dtype=np.float64)
    original_shape = arr.shape
    out = arr.reshape(-1, 4).copy()
    left_alpha = np.clip(out[:, 0] - out[:, 1], _ACTIVE_LOWER, _ACTIVE_UPPER)
    right_alpha = np.clip(out[:, 3] - out[:, 2], _ACTIVE_LOWER, _ACTIVE_UPPER)
    out[:, 1] = output_knee_from_active_angle_np_array(left_alpha)
    out[:, 3] = -output_knee_from_active_angle_np_array(right_alpha)
    return out.reshape(original_shape)


def output_knee_from_active_angle_np(active_angle: float) -> float:
    """NumPy 版本：由主动杆夹角计算左腿输出膝关节角。"""
    alpha_grid, knee_grid, _, _, _ = _fourbar_lut_np()
    alpha = float(np.clip(active_angle, _ACTIVE_LOWER, _ACTIVE_UPPER))
    return float(np.interp(alpha, alpha_grid, knee_grid))


def _output_knee_from_active_angle_analytic_np_array(active_angle: np.ndarray) -> np.ndarray:
    """解析计算左腿输出膝角，作为 NumPy LUT 构建真值。"""
    alpha = np.clip(np.asarray(active_angle, dtype=np.float64), _ACTIVE_LOWER, _ACTIVE_UPPER)
    beta = -alpha
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    px = cos_b * _DRIVE_X + sin_b * _DRIVE_Z
    pz = -sin_b * _DRIVE_X + cos_b * _DRIVE_Z

    dx = px - _KNEE_X
    dz = pz - _KNEE_Z
    dist = np.sqrt(np.maximum(dx * dx + dz * dz, 1.0e-12))
    ex = dx / dist
    ez = dz / dist

    along = (_CALF_LEN**2 - _COUPLER_LEN**2 + dist * dist) / (2.0 * dist)
    height = np.sqrt(np.maximum(_CALF_LEN**2 - along * along, 0.0))
    cx = _KNEE_X + along * ex - height * ez
    cz = _KNEE_Z + along * ez + height * ex

    phi = np.arctan2(cz - _KNEE_Z, cx - _KNEE_X)
    return _wrap_angle_np_array(_CALF_ZERO_ANGLE - phi)


def output_knee_from_active_angle_np_array(active_angle: np.ndarray) -> np.ndarray:
    """NumPy 向量版本：由主动杆夹角计算左腿输出膝关节角。"""
    alpha_grid, knee_grid, _, _, _ = _fourbar_lut_np()
    return _interp_lut_np(np.asarray(active_angle, dtype=np.float64), alpha_grid, knee_grid)


def active_angle_from_output_knee_np(
    output_knee: np.ndarray,
    *,
    right_side: bool,
) -> np.ndarray:
    """NumPy 版本：用 LUT 把输出膝角反解为主动杆夹角。"""
    target = np.asarray(output_knee, dtype=np.float64)
    target = -target if right_side else target
    _, _, inverse_knee_grid, inverse_alpha_grid, _ = _fourbar_lut_np()
    return _interp_lut_np(target, inverse_knee_grid, inverse_alpha_grid)


def output_to_policy_pos_np(output_pos: np.ndarray) -> np.ndarray:
    """NumPy 版本：把开树输出关节反解回 policy 主动杆语义。"""
    arr = np.asarray(output_pos, dtype=np.float64)
    original_shape = arr.shape
    out = arr.reshape(-1, 4).copy()
    left_alpha = active_angle_from_output_knee_np(out[:, 1], right_side=False)
    right_alpha = active_angle_from_output_knee_np(out[:, 3], right_side=True)
    out[:, 1] = out[:, 0] - left_alpha
    out[:, 3] = out[:, 2] + right_alpha
    return out.reshape(original_shape)


def output_knee_jacobian_np(active_angle: np.ndarray, *, right_side: bool) -> np.ndarray:
    """NumPy 版本：计算输出膝角对主动杆夹角的数值雅可比。"""
    alpha_grid, _, _, _, jacobian_grid = _fourbar_lut_np()
    value = _interp_lut_np(np.asarray(active_angle, dtype=np.float64), alpha_grid, jacobian_grid)
    return -value if right_side else value


def output_to_policy_vel_np(output_pos: np.ndarray, output_vel: np.ndarray) -> np.ndarray:
    """NumPy 版本：把开树输出速度反解回 policy 主动杆速度语义。"""
    pos = np.asarray(output_pos, dtype=np.float64).reshape(-1, 4)
    vel_arr = np.asarray(output_vel, dtype=np.float64)
    original_shape = vel_arr.shape
    out = vel_arr.reshape(-1, 4).copy()
    left_alpha = active_angle_from_output_knee_np(pos[:, 1], right_side=False)
    right_alpha = active_angle_from_output_knee_np(pos[:, 3], right_side=True)
    left_j = output_knee_jacobian_np(left_alpha, right_side=False)
    right_j = output_knee_jacobian_np(right_alpha, right_side=True)
    left_alpha_dot = out[:, 1] / _safe_denominator_np(left_j)
    right_alpha_dot = out[:, 3] / _safe_denominator_np(right_j)
    out[:, 1] = out[:, 0] - left_alpha_dot
    out[:, 3] = out[:, 2] + right_alpha_dot
    return out.reshape(original_shape)


def policy_to_output_torque_np(policy_pos: np.ndarray, policy_torque: np.ndarray) -> np.ndarray:
    """NumPy 版本：把 policy 主动杆力矩映射为开树输出关节力矩。"""
    pos = np.asarray(policy_pos, dtype=np.float64).reshape(-1, 4)
    torque_arr = np.asarray(policy_torque, dtype=np.float64)
    original_shape = torque_arr.shape
    out = torque_arr.reshape(-1, 4).copy()
    torque_rows = torque_arr.reshape(-1, 4)
    left_alpha = np.clip(pos[:, 0] - pos[:, 1], _ACTIVE_LOWER, _ACTIVE_UPPER)
    right_alpha = np.clip(pos[:, 3] - pos[:, 2], _ACTIVE_LOWER, _ACTIVE_UPPER)
    left_j = output_knee_jacobian_np(left_alpha, right_side=False)
    right_j = output_knee_jacobian_np(right_alpha, right_side=True)
    out[:, 0] = torque_rows[:, 0] + torque_rows[:, 1]
    out[:, 1] = -torque_rows[:, 1] / _safe_denominator_np(left_j)
    out[:, 2] = torque_rows[:, 2] + torque_rows[:, 3]
    out[:, 3] = torque_rows[:, 3] / _safe_denominator_np(right_j)
    return out.reshape(original_shape)


def _fourbar_lut(
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回当前 device/dtype 上的四连杆转换查表。"""
    device_key = str(torch.device(device))
    cache_key = (device_key, dtype)
    cached = _TORCH_LUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    alpha_grid = torch.linspace(
        _ACTIVE_LOWER,
        _ACTIVE_UPPER,
        _LUT_SIZE,
        device=device,
        dtype=dtype,
    )
    knee_grid = _output_knee_from_active_angle_analytic_torch(alpha_grid)
    inverse_knee_grid, inverse_alpha_grid = _inverse_lut_grids(knee_grid, alpha_grid)

    eps = torch.as_tensor(1.0e-3, device=device, dtype=dtype)
    lo = (alpha_grid - eps).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    hi = (alpha_grid + eps).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    jacobian_grid = (
        _output_knee_from_active_angle_analytic_torch(hi)
        - _output_knee_from_active_angle_analytic_torch(lo)
    ) / (hi - lo).clamp_min(1.0e-6)

    cached = (alpha_grid, knee_grid, inverse_knee_grid, inverse_alpha_grid, jacobian_grid)
    _TORCH_LUT_CACHE[cache_key] = cached
    return cached


def _inverse_lut_grids(
    knee_grid: torch.Tensor,
    alpha_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把单调膝角表整理成 searchsorted 需要的升序表。"""
    increasing = bool(torch.all(knee_grid[1:] >= knee_grid[:-1]).item())
    decreasing = bool(torch.all(knee_grid[1:] <= knee_grid[:-1]).item())
    if increasing:
        return knee_grid, alpha_grid
    if decreasing:
        return torch.flip(knee_grid, dims=(0,)), torch.flip(alpha_grid, dims=(0,))
    raise RuntimeError("四连杆 LUT 的输出膝角表不是单调序列")


def _interp_lut(
    query: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
) -> torch.Tensor:
    """在一维升序 LUT 上做线性插值。"""
    q = query.clamp(x_grid[0], x_grid[-1])
    idx_hi = torch.searchsorted(x_grid, q.contiguous()).clamp(1, x_grid.numel() - 1)
    idx_lo = idx_hi - 1
    x0 = x_grid[idx_lo]
    x1 = x_grid[idx_hi]
    y0 = y_grid[idx_lo]
    y1 = y_grid[idx_hi]
    weight = (q - x0) / (x1 - x0).clamp_min(torch.finfo(x_grid.dtype).eps)
    return y0 + weight * (y1 - y0)


def _fourbar_lut_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 NumPy 侧共享的四连杆转换查表。"""
    global _NP_LUT_CACHE

    if _NP_LUT_CACHE is not None:
        return _NP_LUT_CACHE

    alpha_grid = np.linspace(_ACTIVE_LOWER, _ACTIVE_UPPER, _LUT_SIZE, dtype=np.float64)
    knee_grid = _output_knee_from_active_angle_analytic_np_array(alpha_grid)
    inverse_knee_grid, inverse_alpha_grid = _inverse_lut_grids_np(knee_grid, alpha_grid)

    eps = 1.0e-3
    lo = np.clip(alpha_grid - eps, _ACTIVE_LOWER, _ACTIVE_UPPER)
    hi = np.clip(alpha_grid + eps, _ACTIVE_LOWER, _ACTIVE_UPPER)
    jacobian_grid = (
        _output_knee_from_active_angle_analytic_np_array(hi)
        - _output_knee_from_active_angle_analytic_np_array(lo)
    ) / np.maximum(hi - lo, 1.0e-6)

    _NP_LUT_CACHE = (alpha_grid, knee_grid, inverse_knee_grid, inverse_alpha_grid, jacobian_grid)
    return _NP_LUT_CACHE


def _inverse_lut_grids_np(
    knee_grid: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把 NumPy 单调膝角表整理成升序插值表。"""
    increasing = bool(np.all(knee_grid[1:] >= knee_grid[:-1]))
    decreasing = bool(np.all(knee_grid[1:] <= knee_grid[:-1]))
    if increasing:
        return knee_grid, alpha_grid
    if decreasing:
        return np.flip(knee_grid), np.flip(alpha_grid)
    raise RuntimeError("四连杆 NumPy LUT 的输出膝角表不是单调序列")


def _interp_lut_np(
    query: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """在 NumPy 一维升序 LUT 上做线性插值。"""
    q = np.clip(query, x_grid[0], x_grid[-1])
    return np.interp(q, x_grid, y_grid)


def _wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _safe_denominator_torch(value: torch.Tensor) -> torch.Tensor:
    sign = torch.where(value < 0.0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(torch.abs(value) < 1.0e-6, sign * 1.0e-6, value)


def _safe_denominator_np(value: np.ndarray) -> np.ndarray:
    sign = np.where(value < 0.0, -np.ones_like(value), np.ones_like(value))
    return np.where(np.abs(value) < 1.0e-6, sign * 1.0e-6, value)


def _wrap_angle_np(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _wrap_angle_np_array(angle: np.ndarray) -> np.ndarray:
    return np.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
