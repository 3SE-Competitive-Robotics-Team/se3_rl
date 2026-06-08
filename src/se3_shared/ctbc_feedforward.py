"""CTBC 参考前馈到当前 SerialLeg action 语义的换算。"""

from __future__ import annotations

import numpy as np
import torch

from .fourbar import (
    output_to_policy_pos_np,
    output_to_policy_pos_torch,
    policy_to_output_pos_np,
    policy_to_output_pos_torch,
)
from .robot import JointGroup, RobotConfig

REFERENCE_CTBC_CONTACT_WINDOW = 3
REFERENCE_CTBC_FORCE_THRESHOLD_N = 30.0
REFERENCE_CTBC_FF_AMPLITUDE = 1.2
REFERENCE_CTBC_FF_PERIOD_S = 0.6
REFERENCE_CTBC_LEG_SCALE = 0.25
REFERENCE_CTBC_HIP_RATIO = 1.5
REFERENCE_CTBC_KNEE_RATIO = 1.0

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
    active_mid = 0.5 * (float(lower) + float(upper))
    lower_t = torch.as_tensor(lower, device=leg_action.device, dtype=leg_action.dtype)
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
            lower,
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
