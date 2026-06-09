"""CTBC 参考前馈到当前 SerialLeg action 语义的换算。"""

from __future__ import annotations

import math

import numpy as np
import torch

from .fourbar import (
    output_to_policy_pos_np,
    output_to_policy_pos_torch,
    policy_to_output_pos_np,
    policy_to_output_pos_torch,
)
from .height_default import (
    _height_default_lut_np,
    _height_default_lut_torch,
    _interp_monotonic_torch,
    _leg_vector_np,
    _leg_vector_torch,
)
from .robot import JointGroup, RobotConfig

REFERENCE_CTBC_CONTACT_WINDOW = 3
REFERENCE_CTBC_FORCE_THRESHOLD_N = 30.0
REFERENCE_CTBC_FF_AMPLITUDE = 1.2
REFERENCE_CTBC_FF_PERIOD_S = 0.6
REFERENCE_CTBC_LEG_SCALE = 0.25
REFERENCE_CTBC_HIP_RATIO = 1.5
REFERENCE_CTBC_KNEE_RATIO = 1.0
REFERENCE_CTBC_LEG_LENGTH_AMPLITUDE_M = 0.15
REFERENCE_CTBC_SWING_ANGLE_AMPLITUDE_RAD = math.radians(30.0)

_ROBOT_CFG = RobotConfig()


def reference_ctbc_bias_to_current_action_torch(
    leg_action: torch.Tensor,
    policy_default: torch.Tensor,
    side_pulse: torch.Tensor,
    *,
    leg_scales: torch.Tensor,
    height_conditioned_action_default: bool,
    reference_leg_scale: float = REFERENCE_CTBC_LEG_SCALE,
    hip_ratio: float = REFERENCE_CTBC_HIP_RATIO,
    knee_ratio: float = REFERENCE_CTBC_KNEE_RATIO,
    robot_cfg: RobotConfig | None = None,
) -> torch.Tensor:
    """把参考开链 hip/knee 前馈换算成当前 [front, active] raw action bias。"""

    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    current_target = _current_action_to_policy_target_torch(
        leg_action,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    target_output = policy_to_output_pos_torch(current_target)
    target_output = target_output + _reference_output_delta_torch(
        side_pulse,
        reference_leg_scale=reference_leg_scale,
        hip_ratio=hip_ratio,
        knee_ratio=knee_ratio,
    )
    target_policy = output_to_policy_pos_torch(target_output)
    target_action = _policy_target_to_current_action_torch(
        target_policy,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    return target_action - leg_action


def reference_ctbc_bias_to_current_action_np(
    leg_action: np.ndarray,
    policy_default: np.ndarray,
    side_pulse: np.ndarray,
    *,
    leg_scales: np.ndarray,
    height_conditioned_action_default: bool,
    reference_leg_scale: float = REFERENCE_CTBC_LEG_SCALE,
    hip_ratio: float = REFERENCE_CTBC_HIP_RATIO,
    knee_ratio: float = REFERENCE_CTBC_KNEE_RATIO,
    robot_cfg: RobotConfig | None = None,
) -> np.ndarray:
    """NumPy 版：供 sim2sim/probe 使用，语义与训练端完全一致。"""

    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    current_target = _current_action_to_policy_target_np(
        leg_action,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    target_output = policy_to_output_pos_np(current_target)
    target_output = target_output + _reference_output_delta_np(
        side_pulse,
        reference_leg_scale=reference_leg_scale,
        hip_ratio=hip_ratio,
        knee_ratio=knee_ratio,
    )
    target_policy = output_to_policy_pos_np(target_output)
    target_action = _policy_target_to_current_action_np(
        target_policy,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    return target_action - np.asarray(leg_action, dtype=np.float64)


def leg_length_ctbc_bias_to_current_action_torch(
    leg_action: torch.Tensor,
    policy_default: torch.Tensor,
    side_profile: torch.Tensor,
    *,
    leg_scales: torch.Tensor,
    height_conditioned_action_default: bool,
    amplitude_m: float = REFERENCE_CTBC_LEG_LENGTH_AMPLITUDE_M,
    swing_angle_rad: float = 0.0,
    robot_cfg: RobotConfig | None = None,
) -> torch.Tensor:
    """把腿长/摆角方向 CTBC 前馈反解成当前 [front, active] raw action bias。"""

    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    current_target = _current_action_to_policy_target_torch(
        leg_action,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    target_policy = current_target.clone()
    profile = side_profile.reshape(-1, 2).to(
        device=current_target.device, dtype=current_target.dtype
    )
    for side_idx, target_slice in enumerate((slice(0, 2), slice(2, 4))):
        if not bool((profile[:, side_idx] > 0.0).any()):
            continue
        target_policy[:, target_slice] = _leg_length_side_target_torch(
            current_target,
            side_idx,
            shortening_m=float(amplitude_m) * profile[:, side_idx],
            swing_angle_rad=float(swing_angle_rad) * profile[:, side_idx],
        )
    target_action = _policy_target_to_current_action_torch(
        target_policy,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    return target_action - leg_action.reshape(-1, 4)


def leg_length_ctbc_bias_to_current_action_np(
    leg_action: np.ndarray,
    policy_default: np.ndarray,
    side_profile: np.ndarray,
    *,
    leg_scales: np.ndarray,
    height_conditioned_action_default: bool,
    amplitude_m: float = REFERENCE_CTBC_LEG_LENGTH_AMPLITUDE_M,
    swing_angle_rad: float = 0.0,
    robot_cfg: RobotConfig | None = None,
) -> np.ndarray:
    """NumPy 版腿长/摆角 CTBC 前馈反解，供 probe/sim2sim 使用。"""

    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    current_target = _current_action_to_policy_target_np(
        leg_action,
        policy_default,
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    ).reshape(-1, 4)
    target_policy = current_target.copy()
    profile = np.asarray(side_profile, dtype=np.float64).reshape(-1, 2)
    for side_idx, target_slice in enumerate((slice(0, 2), slice(2, 4))):
        if not bool(np.any(profile[:, side_idx] > 0.0)):
            continue
        target_policy[:, target_slice] = _leg_length_side_target_np(
            current_target,
            side_idx,
            shortening_m=float(amplitude_m) * profile[:, side_idx],
            swing_angle_rad=float(swing_angle_rad) * profile[:, side_idx],
        )
    target_action = _policy_target_to_current_action_np(
        target_policy,
        np.asarray(policy_default, dtype=np.float64).reshape(-1, 4),
        leg_scales=leg_scales,
        height_conditioned_action_default=height_conditioned_action_default,
        robot_cfg=cfg,
    )
    return target_action - np.asarray(leg_action, dtype=np.float64).reshape(-1, 4)


def _leg_length_side_target_torch(
    current_policy: torch.Tensor,
    side_idx: int,
    *,
    shortening_m: torch.Tensor,
    swing_angle_rad: torch.Tensor,
) -> torch.Tensor:
    """按指定侧腿长缩短量和向前摆角量求目标腿型。"""

    output = policy_to_output_pos_torch(current_policy)
    if side_idx == 0:
        front_angle = current_policy[:, 0]
        output_knee = output[:, 1]
    else:
        front_angle = current_policy[:, 2]
        output_knee = -output[:, 3]

    vec_x, vec_z = _leg_vector_torch(output_knee)
    local_angle = torch.atan2(vec_x, -vec_z)
    direction = local_angle - front_angle if side_idx == 0 else local_angle + front_angle
    direction = direction + swing_angle_rad
    current_length = torch.sqrt(torch.clamp(vec_x * vec_x + vec_z * vec_z, min=1.0e-12))
    target_length = torch.clamp(current_length - shortening_m, min=0.0)
    target_x = torch.sin(direction) * target_length
    target_z = -torch.cos(direction) * target_length
    solved = _policy_from_leg_vector_torch(target_x, target_z)
    return solved[:, :2] if side_idx == 0 else solved[:, 2:4]


def _leg_length_side_target_np(
    current_policy: np.ndarray,
    side_idx: int,
    *,
    shortening_m: np.ndarray,
    swing_angle_rad: np.ndarray,
) -> np.ndarray:
    """NumPy 版：按指定侧腿长缩短量和向前摆角量求目标腿型。"""

    output = policy_to_output_pos_np(current_policy)
    if side_idx == 0:
        front_angle = current_policy[:, 0]
        output_knee = output[:, 1]
    else:
        front_angle = current_policy[:, 2]
        output_knee = -output[:, 3]

    vec_x, vec_z = _leg_vector_np(output_knee)
    local_angle = np.arctan2(vec_x, -vec_z)
    direction = local_angle - front_angle if side_idx == 0 else local_angle + front_angle
    direction = direction + np.asarray(swing_angle_rad, dtype=np.float64)
    current_length = np.hypot(vec_x, vec_z)
    target_length = np.maximum(current_length - np.asarray(shortening_m, dtype=np.float64), 0.0)
    target_x = np.sin(direction) * target_length
    target_z = -np.cos(direction) * target_length
    solved = _policy_from_leg_vector_np(target_x, target_z)
    return solved[:, :2] if side_idx == 0 else solved[:, 2:4]


def _policy_from_leg_vector_torch(target_x: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """由目标腿向量反解左右对称的 policy 腿型。"""

    active_by_length, length_grid, active_grid, vec_x_grid, vec_z_grid = _height_default_lut_torch(
        target_x.device, target_x.dtype
    )
    target_length = torch.clamp(
        torch.sqrt(torch.clamp(target_x * target_x + target_z * target_z, min=1.0e-12)),
        min=length_grid[0],
        max=length_grid[-1],
    )
    active = _interp_monotonic_torch(target_length, length_grid, active_by_length)
    vec_x = _interp_monotonic_torch(active, active_grid, vec_x_grid)
    vec_z = _interp_monotonic_torch(active, active_grid, vec_z_grid)
    lf = torch.atan2(vec_x, -vec_z) - torch.atan2(target_x, -target_z)
    rf = -lf
    return torch.stack((lf, lf - active, rf, rf + active), dim=1)


def _policy_from_leg_vector_np(target_x: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """NumPy 版：由目标腿向量反解左右对称的 policy 腿型。"""

    active_by_length, length_grid, active_grid, vec_x_grid, vec_z_grid = _height_default_lut_np()
    target_length = np.clip(np.hypot(target_x, target_z), length_grid[0], length_grid[-1])
    active = np.interp(target_length, length_grid, active_by_length)
    vec_x = np.interp(active, active_grid, vec_x_grid)
    vec_z = np.interp(active, active_grid, vec_z_grid)
    lf = np.arctan2(vec_x, -vec_z) - np.arctan2(target_x, -target_z)
    rf = -lf
    return np.stack((lf, lf - active, rf, rf + active), axis=1)


def _reference_output_delta_torch(
    side_pulse: torch.Tensor,
    *,
    reference_leg_scale: float,
    hip_ratio: float,
    knee_ratio: float,
) -> torch.Tensor:
    pulse = side_pulse.reshape(-1, 2)
    delta = torch.zeros((pulse.shape[0], 4), device=pulse.device, dtype=pulse.dtype)
    hip = pulse * float(reference_leg_scale) * float(hip_ratio)
    knee = pulse * float(reference_leg_scale) * float(knee_ratio)

    delta[:, 0] = hip[:, 0]
    delta[:, 1] = -knee[:, 0]
    delta[:, 2] = -hip[:, 1]
    delta[:, 3] = knee[:, 1]
    return delta


def _reference_output_delta_np(
    side_pulse: np.ndarray,
    *,
    reference_leg_scale: float,
    hip_ratio: float,
    knee_ratio: float,
) -> np.ndarray:
    pulse = np.asarray(side_pulse, dtype=np.float64).reshape(-1, 2)
    delta = np.zeros((pulse.shape[0], 4), dtype=np.float64)
    hip = pulse * float(reference_leg_scale) * float(hip_ratio)
    knee = pulse * float(reference_leg_scale) * float(knee_ratio)

    delta[:, 0] = hip[:, 0]
    delta[:, 1] = -knee[:, 0]
    delta[:, 2] = -hip[:, 1]
    delta[:, 3] = knee[:, 1]
    return delta.reshape((*np.asarray(side_pulse).shape[:-1], 4))


def _current_action_to_policy_target_torch(
    leg_action: torch.Tensor,
    policy_default: torch.Tensor,
    *,
    leg_scales: torch.Tensor,
    height_conditioned_action_default: bool,
    robot_cfg: RobotConfig,
) -> torch.Tensor:
    leg_action = leg_action.reshape(-1, 4)
    policy_default = policy_default.reshape(-1, 4)
    scales = leg_scales.to(device=leg_action.device, dtype=leg_action.dtype).reshape(4)
    target = torch.empty_like(policy_default)
    lower, upper = robot_cfg.active_rod_angle_limits
    target_lower = float(lower) - float(robot_cfg.active_rod_lower_target_overdrive)
    active_mid = 0.5 * (float(lower) + float(upper))
    lower_t = torch.as_tensor(target_lower, device=leg_action.device, dtype=leg_action.dtype)
    upper_t = torch.as_tensor(upper, device=leg_action.device, dtype=leg_action.dtype)
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = robot_cfg.active_rod_angle_coeffs[side_idx]
        front_target = policy_default[:, front_idx] + leg_action[:, front_idx] * scales[front_idx]
        if height_conditioned_action_default:
            active_default = (
                front_coef * policy_default[:, front_idx] + back_coef * policy_default[:, back_idx]
            )
        else:
            active_default = torch.full_like(front_target, active_mid)
        active_target = torch.clamp(
            active_default + leg_action[:, back_idx] * scales[back_idx],
            lower_t,
            upper_t,
        )
        target[:, front_idx] = front_target
        target[:, back_idx] = (active_target - front_coef * front_target) / back_coef
    return target


def _policy_target_to_current_action_torch(
    policy_target: torch.Tensor,
    policy_default: torch.Tensor,
    *,
    leg_scales: torch.Tensor,
    height_conditioned_action_default: bool,
    robot_cfg: RobotConfig,
) -> torch.Tensor:
    policy_target = policy_target.reshape(-1, 4)
    policy_default = policy_default.reshape(-1, 4)
    scales = leg_scales.to(device=policy_target.device, dtype=policy_target.dtype).reshape(4)
    action = torch.empty_like(policy_target)
    lower, upper = robot_cfg.active_rod_angle_limits
    active_mid = 0.5 * (float(lower) + float(upper))
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = robot_cfg.active_rod_angle_coeffs[side_idx]
        action[:, front_idx] = (
            policy_target[:, front_idx] - policy_default[:, front_idx]
        ) / scales[front_idx]
        active_target = (
            front_coef * policy_target[:, front_idx] + back_coef * policy_target[:, back_idx]
        )
        if height_conditioned_action_default:
            active_default = (
                front_coef * policy_default[:, front_idx] + back_coef * policy_default[:, back_idx]
            )
        else:
            active_default = torch.full_like(active_target, active_mid)
        action[:, back_idx] = (active_target - active_default) / scales[back_idx]
    return action


def _current_action_to_policy_target_np(
    leg_action: np.ndarray,
    policy_default: np.ndarray,
    *,
    leg_scales: np.ndarray,
    height_conditioned_action_default: bool,
    robot_cfg: RobotConfig,
) -> np.ndarray:
    leg_action = np.asarray(leg_action, dtype=np.float64).reshape(-1, 4)
    policy_default = np.asarray(policy_default, dtype=np.float64).reshape(-1, 4)
    scales = np.asarray(leg_scales, dtype=np.float64).reshape(4)
    target = np.empty_like(policy_default)
    lower, upper = robot_cfg.active_rod_angle_limits
    target_lower = float(lower) - float(robot_cfg.active_rod_lower_target_overdrive)
    active_mid = 0.5 * (float(lower) + float(upper))
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = robot_cfg.active_rod_angle_coeffs[side_idx]
        front_target = policy_default[:, front_idx] + leg_action[:, front_idx] * scales[front_idx]
        if height_conditioned_action_default:
            active_default = (
                front_coef * policy_default[:, front_idx] + back_coef * policy_default[:, back_idx]
            )
        else:
            active_default = np.full_like(front_target, active_mid)
        active_target = np.clip(
            active_default + leg_action[:, back_idx] * scales[back_idx],
            target_lower,
            upper,
        )
        target[:, front_idx] = front_target
        target[:, back_idx] = (active_target - front_coef * front_target) / back_coef
    return target.reshape(np.asarray(policy_default).shape)


def _policy_target_to_current_action_np(
    policy_target: np.ndarray,
    policy_default: np.ndarray,
    *,
    leg_scales: np.ndarray,
    height_conditioned_action_default: bool,
    robot_cfg: RobotConfig,
) -> np.ndarray:
    policy_target = np.asarray(policy_target, dtype=np.float64).reshape(-1, 4)
    policy_default = np.asarray(policy_default, dtype=np.float64).reshape(-1, 4)
    scales = np.asarray(leg_scales, dtype=np.float64).reshape(4)
    action = np.empty_like(policy_target)
    lower, upper = robot_cfg.active_rod_angle_limits
    active_mid = 0.5 * (float(lower) + float(upper))
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = robot_cfg.active_rod_angle_coeffs[side_idx]
        action[:, front_idx] = (
            policy_target[:, front_idx] - policy_default[:, front_idx]
        ) / scales[front_idx]
        active_target = (
            front_coef * policy_target[:, front_idx] + back_coef * policy_target[:, back_idx]
        )
        if height_conditioned_action_default:
            active_default = (
                front_coef * policy_default[:, front_idx] + back_coef * policy_default[:, back_idx]
            )
        else:
            active_default = np.full_like(active_target, active_mid)
        action[:, back_idx] = (active_target - active_default) / scales[back_idx]
    return action.reshape(np.asarray(policy_target).shape)


def current_leg_action_scales_np(robot_cfg: RobotConfig | None = None) -> np.ndarray:
    """返回当前前 4 维腿部 action scale。"""

    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    return np.asarray(cfg.action_scale, dtype=np.float64)[JointGroup.LEG_ACTUATORS]
