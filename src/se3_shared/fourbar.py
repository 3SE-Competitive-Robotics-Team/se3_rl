"""SerialLeg 四连杆等效开树运动学。"""

from __future__ import annotations

import math

import numpy as np
import torch

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
_ACTIVE_UPPER = 1.469449651507


def is_fourbar_surrogate_name_set(site_names: tuple[str, ...]) -> bool:
    """判断模型是否带有四连杆等效开树标记。"""
    return FOURBAR_SURROGATE_MARKER in set(site_names)


def output_knee_from_active_angle_torch(active_angle: torch.Tensor) -> torch.Tensor:
    """由主动杆夹角计算左腿输出膝关节角。"""
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
    return _wrap_angle_torch(torch.as_tensor(_CALF_ZERO_ANGLE, device=phi.device) - phi)


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


def active_angle_from_output_knee_torch(
    output_knee: torch.Tensor,
    *,
    right_side: bool,
) -> torch.Tensor:
    """用二分法把输出膝角反解为主动杆夹角。"""
    target = -output_knee if right_side else output_knee
    low = torch.zeros_like(target) + _ACTIVE_LOWER
    high = torch.zeros_like(target) + _ACTIVE_UPPER
    target = target.clamp(
        output_knee_from_active_angle_torch(high).min(),
        output_knee_from_active_angle_torch(low).max(),
    )
    for _ in range(16):
        mid = (low + high) * 0.5
        value = output_knee_from_active_angle_torch(mid)
        go_high = value > target
        low = torch.where(go_high, mid, low)
        high = torch.where(go_high, high, mid)
    return (low + high) * 0.5


def output_knee_jacobian_torch(active_angle: torch.Tensor, *, right_side: bool) -> torch.Tensor:
    """计算输出膝角对主动杆夹角的数值雅可比。"""
    eps = 1.0e-3
    lo = (active_angle - eps).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    hi = (active_angle + eps).clamp(_ACTIVE_LOWER, _ACTIVE_UPPER)
    value = (output_knee_from_active_angle_torch(hi) - output_knee_from_active_angle_torch(lo)) / (
        hi - lo
    ).clamp_min(1.0e-6)
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
    out = np.asarray(policy_pos, dtype=np.float64).copy()
    left_alpha = np.clip(out[0] - out[1], _ACTIVE_LOWER, _ACTIVE_UPPER)
    right_alpha = np.clip(out[3] - out[2], _ACTIVE_LOWER, _ACTIVE_UPPER)
    out[1] = output_knee_from_active_angle_np(left_alpha)
    out[3] = -output_knee_from_active_angle_np(right_alpha)
    return out


def output_knee_from_active_angle_np(active_angle: float) -> float:
    """NumPy 版本：由主动杆夹角计算左腿输出膝关节角。"""
    alpha = float(np.clip(active_angle, _ACTIVE_LOWER, _ACTIVE_UPPER))
    beta = -alpha
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    px = cos_b * _DRIVE_X + sin_b * _DRIVE_Z
    pz = -sin_b * _DRIVE_X + cos_b * _DRIVE_Z

    dx = px - _KNEE_X
    dz = pz - _KNEE_Z
    dist = math.sqrt(max(dx * dx + dz * dz, 1.0e-12))
    ex = dx / dist
    ez = dz / dist

    along = (_CALF_LEN**2 - _COUPLER_LEN**2 + dist * dist) / (2.0 * dist)
    height = math.sqrt(max(_CALF_LEN**2 - along * along, 0.0))
    cx = _KNEE_X + along * ex - height * ez
    cz = _KNEE_Z + along * ez + height * ex

    phi = math.atan2(cz - _KNEE_Z, cx - _KNEE_X)
    return _wrap_angle_np(_CALF_ZERO_ANGLE - phi)


def _wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _safe_denominator_torch(value: torch.Tensor) -> torch.Tensor:
    sign = torch.where(value < 0.0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(torch.abs(value) < 1.0e-6, sign * 1.0e-6, value)


def _wrap_angle_np(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
