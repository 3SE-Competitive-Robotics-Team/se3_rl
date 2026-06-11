"""按指令高度生成 SerialLeg 的默认腿部姿态。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .fourbar import output_knee_from_active_angle_np_array, output_knee_from_active_angle_torch
from .robot import FourbarRobotConfig

if TYPE_CHECKING:
    import torch

_ROBOT_CFG = FourbarRobotConfig()
_LUT_SIZE = 1024
_WHEEL_RADIUS = 0.06
_BASE_COM_X = -0.01780372
_LF1_BODY_XZ = (-0.12990117, 0.04639203)
_LF1_JOINT_XZ = (-0.05003347, -0.04149627)
_WHEEL_BODY_XZ = (-0.15699, -0.21049)
_NP_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
_TORCH_CACHE: dict[
    tuple[str, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def policy_default_from_height_torch(
    command_height: torch.Tensor,
    cfg: FourbarRobotConfig | None = None,
) -> torch.Tensor:
    """根据 base height command 返回 policy 语义下的 [LF, LB, RF, RB] 默认姿态。"""
    torch = _import_torch()
    _ = cfg
    height = command_height.reshape(-1).to(dtype=torch.float32)
    active_by_length, length_grid, active_grid, vec_x_grid, vec_z_grid = _height_default_lut_torch(
        height.device, height.dtype
    )
    target_x = torch.full_like(height, _BASE_COM_X)
    target_z = torch.as_tensor(_WHEEL_RADIUS, device=height.device, dtype=height.dtype) - height
    target_length = torch.clamp(
        torch.sqrt(target_x * target_x + target_z * target_z),
        min=length_grid[0],
        max=length_grid[-1],
    )
    active = _interp_monotonic_torch(target_length, length_grid, active_by_length)
    vec_x = _interp_monotonic_torch(active, active_grid, vec_x_grid)
    vec_z = _interp_monotonic_torch(active, active_grid, vec_z_grid)
    lf = torch.atan2(vec_x, -vec_z) - torch.atan2(target_x, -target_z)
    rf = -lf
    lb = lf - active
    rb = rf + active
    return torch.stack((lf, lb, rf, rb), dim=1)


def policy_default_from_height_np(
    command_height: float | np.ndarray,
    cfg: FourbarRobotConfig | None = None,
) -> np.ndarray:
    """NumPy 版本：根据 base height command 返回 [LF, LB, RF, RB] 默认姿态。"""
    robot_cfg = _ROBOT_CFG if cfg is None else cfg
    _ = robot_cfg
    height = np.asarray(command_height, dtype=np.float64)
    original_shape = height.shape
    flat_height = height.reshape(-1)
    active_by_length, length_grid, active_grid, vec_x_grid, vec_z_grid = _height_default_lut_np()
    target_x = np.full_like(flat_height, _BASE_COM_X)
    target_z = _WHEEL_RADIUS - flat_height
    target_length = np.clip(
        np.hypot(target_x, target_z),
        length_grid[0],
        length_grid[-1],
    )
    active = np.interp(target_length, length_grid, active_by_length)
    vec_x = np.interp(active, active_grid, vec_x_grid)
    vec_z = np.interp(active, active_grid, vec_z_grid)
    lf = np.arctan2(vec_x, -vec_z) - np.arctan2(target_x, -target_z)
    rf = -lf
    lb = lf - active
    rb = rf + active
    out = np.stack((lf, lb, rf, rb), axis=1)
    return out.reshape((*original_shape, 4))


def _height_default_lut_torch(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch = _import_torch()
    key = (str(device), dtype)
    cached = _TORCH_CACHE.get(key)
    if cached is not None:
        return cached

    lower, upper = _ROBOT_CFG.active_rod_angle_limits
    active = torch.linspace(float(lower), float(upper), _LUT_SIZE, device=device, dtype=dtype)
    output_knee = output_knee_from_active_angle_torch(active)
    vec_x, vec_z = _leg_vector_torch(output_knee)
    length = torch.sqrt(torch.clamp(vec_x * vec_x + vec_z * vec_z, min=1.0e-12))

    order = torch.argsort(length)
    cached = (active[order], length[order], active, vec_x, vec_z)
    _TORCH_CACHE[key] = cached
    return cached


def _height_default_lut_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _NP_CACHE
    if _NP_CACHE is not None:
        return _NP_CACHE

    lower, upper = _ROBOT_CFG.active_rod_angle_limits
    active = np.linspace(float(lower), float(upper), _LUT_SIZE, dtype=np.float64)
    output_knee = output_knee_from_active_angle_np_array(active)
    vec_x, vec_z = _leg_vector_np(output_knee)
    length = np.hypot(vec_x, vec_z)
    order = np.argsort(length)
    _NP_CACHE = (active[order], length[order], active, vec_x, vec_z)
    return _NP_CACHE


def _leg_vector_torch(output_knee: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch = _import_torch()
    body = torch.as_tensor(_LF1_BODY_XZ, device=output_knee.device, dtype=output_knee.dtype)
    joint = torch.as_tensor(_LF1_JOINT_XZ, device=output_knee.device, dtype=output_knee.dtype)
    wheel = torch.as_tensor(_WHEEL_BODY_XZ, device=output_knee.device, dtype=output_knee.dtype)
    cos_q = torch.cos(output_knee)
    sin_q = torch.sin(output_knee)

    rot_joint_x = cos_q * joint[0] + sin_q * joint[1]
    rot_joint_z = -sin_q * joint[0] + cos_q * joint[1]
    rot_wheel_x = cos_q * wheel[0] + sin_q * wheel[1]
    rot_wheel_z = -sin_q * wheel[0] + cos_q * wheel[1]

    x = body[0] + joint[0] - rot_joint_x + rot_wheel_x
    z = body[1] + joint[1] - rot_joint_z + rot_wheel_z
    return x, z


def _leg_vector_np(output_knee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body = np.asarray(_LF1_BODY_XZ, dtype=np.float64)
    joint = np.asarray(_LF1_JOINT_XZ, dtype=np.float64)
    wheel = np.asarray(_WHEEL_BODY_XZ, dtype=np.float64)
    cos_q = np.cos(output_knee)
    sin_q = np.sin(output_knee)
    rot_joint_x = cos_q * joint[0] + sin_q * joint[1]
    rot_joint_z = -sin_q * joint[0] + cos_q * joint[1]
    rot_wheel_x = cos_q * wheel[0] + sin_q * wheel[1]
    rot_wheel_z = -sin_q * wheel[0] + cos_q * wheel[1]
    x = body[0] + joint[0] - rot_joint_x + rot_wheel_x
    z = body[1] + joint[1] - rot_joint_z + rot_wheel_z
    return x, z


def _interp_monotonic_torch(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
) -> torch.Tensor:
    torch = _import_torch()
    x_clamped = torch.clamp(x, min=xp[0], max=xp[-1])
    idx = torch.searchsorted(xp, x_clamped, right=True)
    idx = torch.clamp(idx, 1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    t = (x_clamped - x0) / torch.clamp(x1 - x0, min=1.0e-12)
    return y0 + t * (y1 - y0)


def _import_torch():
    import torch

    return torch
