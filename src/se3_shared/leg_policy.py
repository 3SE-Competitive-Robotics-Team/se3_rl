"""SerialLeg 腿部 policy 坐标的相位观测与误差工具。"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from .robot import RobotConfig

_ROBOT_CFG = RobotConfig()


def policy_leg_phase_active_obs_np(
    policy_pos: np.ndarray,
    default_policy_pos: np.ndarray,
    *,
    robot_cfg: RobotConfig | None = None,
) -> np.ndarray:
    """把 4D 腿部 policy 位置转换为 6D 周期相位 + 主动杆夹角观测。"""
    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    pos = np.asarray(policy_pos, dtype=np.float64).reshape(-1, 4)
    default = np.asarray(default_policy_pos, dtype=np.float64).reshape(-1, 4)
    if default.shape[0] == 1 and pos.shape[0] != 1:
        default = np.broadcast_to(default, pos.shape)
    if pos.shape != default.shape:
        raise ValueError(f"policy leg shape mismatch: pos={pos.shape}, default={default.shape}")

    front_delta = pos[:, (0, 2)] - default[:, (0, 2)]
    active_delta = _active_rod_angles_np(pos, cfg) - _active_rod_angles_np(default, cfg)
    out = np.empty((pos.shape[0], 6), dtype=np.float64)
    out[:, 0] = np.sin(front_delta[:, 0])
    out[:, 1] = np.cos(front_delta[:, 0])
    out[:, 2] = active_delta[:, 0]
    out[:, 3] = np.sin(front_delta[:, 1])
    out[:, 4] = np.cos(front_delta[:, 1])
    out[:, 5] = active_delta[:, 1]
    return out


def policy_leg_phase_active_obs_torch(
    policy_pos: torch.Tensor,
    default_policy_pos: torch.Tensor,
    active_rod_angle_coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """把 4D 腿部 policy 位置转换为 6D 周期相位 + 主动杆夹角观测。"""
    pos = policy_pos.reshape(-1, 4)
    default = default_policy_pos.reshape(-1, 4)
    if default.shape[0] == 1 and pos.shape[0] != 1:
        default = default.expand_as(pos)
    if pos.shape != default.shape:
        raise ValueError(
            f"policy leg shape mismatch: pos={tuple(pos.shape)}, default={tuple(default.shape)}"
        )

    front_delta = pos[:, (0, 2)] - default[:, (0, 2)]
    active_delta = _active_rod_angles_torch(
        pos, active_rod_angle_coeffs
    ) - _active_rod_angles_torch(default, active_rod_angle_coeffs)
    return torch.stack(
        (
            torch.sin(front_delta[:, 0]),
            torch.cos(front_delta[:, 0]),
            active_delta[:, 0],
            torch.sin(front_delta[:, 1]),
            torch.cos(front_delta[:, 1]),
            active_delta[:, 1],
        ),
        dim=1,
    )


def policy_leg_position_error_np(
    policy_target: np.ndarray,
    policy_pos: np.ndarray,
    *,
    robot_cfg: RobotConfig | None = None,
) -> np.ndarray:
    """计算最近等价腿部 policy 位置误差；LF/RF 走最短角，主动杆夹角不 wrap。"""
    cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
    target = np.asarray(policy_target, dtype=np.float64).reshape(-1, 4)
    pos = np.asarray(policy_pos, dtype=np.float64).reshape(-1, 4)
    if target.shape != pos.shape:
        raise ValueError(f"policy leg shape mismatch: target={target.shape}, pos={pos.shape}")
    coeffs = np.asarray(cfg.active_rod_angle_coeffs, dtype=np.float64).reshape(2, 2)
    front_error = _wrap_angle_np(target[:, (0, 2)] - pos[:, (0, 2)])
    active_error = _active_rod_angles_np(target, cfg) - _active_rod_angles_np(pos, cfg)
    out = np.empty_like(target)
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = coeffs[side_idx]
        out[:, front_idx] = front_error[:, side_idx]
        out[:, back_idx] = (active_error[:, side_idx] - front_coef * out[:, front_idx]) / back_coef
    return out.reshape(np.asarray(policy_target, dtype=np.float64).shape)


def policy_leg_position_error_torch(
    policy_target: torch.Tensor,
    policy_pos: torch.Tensor,
    active_rod_angle_coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算最近等价腿部 policy 位置误差；LF/RF 走最短角，主动杆夹角不 wrap。"""
    target = policy_target.reshape(-1, 4)
    pos = policy_pos.reshape(-1, 4)
    if target.shape != pos.shape:
        raise ValueError(
            f"policy leg shape mismatch: target={tuple(target.shape)}, pos={tuple(pos.shape)}"
        )
    coeffs = _active_coeffs_torch(target, active_rod_angle_coeffs)
    front_error = _wrap_angle_torch(target[:, (0, 2)] - pos[:, (0, 2)])
    active_error = _active_rod_angles_torch(target, coeffs) - _active_rod_angles_torch(pos, coeffs)
    out = torch.empty_like(target)
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = coeffs[side_idx]
        out[:, front_idx] = front_error[:, side_idx]
        out[:, back_idx] = (active_error[:, side_idx] - front_coef * out[:, front_idx]) / back_coef
    return out.reshape_as(policy_target)


def _active_rod_angles_np(policy_pos: np.ndarray, robot_cfg: RobotConfig) -> np.ndarray:
    pos = np.asarray(policy_pos, dtype=np.float64).reshape(-1, 4)
    coeffs = np.asarray(robot_cfg.active_rod_angle_coeffs, dtype=np.float64).reshape(2, 2)
    return np.stack(
        (
            coeffs[0, 0] * pos[:, 0] + coeffs[0, 1] * pos[:, 1],
            coeffs[1, 0] * pos[:, 2] + coeffs[1, 1] * pos[:, 3],
        ),
        axis=1,
    )


def _active_rod_angles_torch(
    policy_pos: torch.Tensor,
    active_rod_angle_coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    pos = policy_pos.reshape(-1, 4)
    coeffs = _active_coeffs_torch(pos, active_rod_angle_coeffs)
    return torch.stack(
        (
            coeffs[0, 0] * pos[:, 0] + coeffs[0, 1] * pos[:, 1],
            coeffs[1, 0] * pos[:, 2] + coeffs[1, 1] * pos[:, 3],
        ),
        dim=1,
    )


def _active_coeffs_torch(
    reference: torch.Tensor,
    active_rod_angle_coeffs: torch.Tensor | None,
) -> torch.Tensor:
    if active_rod_angle_coeffs is not None:
        return active_rod_angle_coeffs.to(device=reference.device, dtype=reference.dtype).reshape(
            2, 2
        )
    return torch.tensor(
        _ROBOT_CFG.active_rod_angle_coeffs,
        device=reference.device,
        dtype=reference.dtype,
    )


def _wrap_angle_np(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_angle_torch(value: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return torch.remainder(value + math.pi, two_pi) - math.pi
