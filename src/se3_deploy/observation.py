"""真机 recovery policy 观测拼装。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from se3_shared import JointGroup, ObservationConfig, RobotConfig

from .protocol import PolicyStateFrame

_OBS_CFG = ObservationConfig()
_ROBOT_CFG = RobotConfig()


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """一次观测拼装结果。"""

    obs: np.ndarray
    had_nonfinite_input: bool


class RecoveryObservationBuilder:
    """按训练端 32 维 actor contract 拼装 recovery-only 观测。"""

    def __init__(self) -> None:
        self.default_dof_pos = np.asarray(_ROBOT_CFG.default_dof_pos, dtype=np.float32)
        self.command_scale = np.asarray(_OBS_CFG.command_scale, dtype=np.float32)
        self.command = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                _ROBOT_CFG.default_base_height,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def build(self, state: PolicyStateFrame, last_action: np.ndarray) -> ObservationResult:
        had_nonfinite = False
        base_ang_vel_body, bad = _finite_array(state.base_ang_vel_body, 3)
        had_nonfinite = had_nonfinite or bad
        projected_gravity, bad = _projected_gravity(state.projected_gravity)
        had_nonfinite = had_nonfinite or bad
        dof_pos, bad = _finite_array(state.dof_pos, 6)
        had_nonfinite = had_nonfinite or bad
        dof_vel, bad = _finite_array(state.dof_vel, 6)
        had_nonfinite = had_nonfinite or bad
        action_obs, bad = _finite_array(last_action, 6)
        had_nonfinite = had_nonfinite or bad

        obs: list[float] = []
        obs.extend((base_ang_vel_body * _OBS_CFG.ang_vel_scale).tolist())
        obs.extend(projected_gravity.tolist())
        obs.extend((self.command[:5] * self.command_scale).tolist())

        leg_pos_rel = dof_pos[JointGroup.CTRL_LEGS] - self.default_dof_pos[JointGroup.CTRL_LEGS]
        obs.extend(leg_pos_rel.tolist())

        leg_vel = dof_vel[JointGroup.CTRL_LEGS] * _OBS_CFG.leg_vel_scale
        obs.extend(leg_vel.tolist())

        obs.extend(dof_pos[JointGroup.CTRL_WHEELS].tolist())
        obs.extend((dof_vel[JointGroup.CTRL_WHEELS] * _OBS_CFG.wheel_vel_scale).tolist())
        obs.extend(action_obs.tolist())
        obs.extend(self.command[5:8].tolist())

        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape != (_OBS_CFG.num_obs,):
            raise RuntimeError(f"recovery obs shape mismatch: {arr.shape}")
        limit = float(_OBS_CFG.clip_value)
        arr = np.nan_to_num(arr, nan=0.0, posinf=limit, neginf=-limit)
        return ObservationResult(obs=np.clip(arr, -limit, limit), had_nonfinite_input=had_nonfinite)


def synthetic_recovery_state(seq: int = 0) -> PolicyStateFrame:
    """生成站立零速状态，用于 checkpoint dry-run。"""

    dof_pos = tuple(float(v) for v in _ROBOT_CFG.default_dof_pos)
    return PolicyStateFrame(
        seq=int(seq),
        timestamp_us=0,
        status_bits=0,
        base_ang_vel_body=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        dof_pos=dof_pos,
        dof_vel=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        motor_status=(0, 0, 0, 0, 0, 0),
    )


def _finite_array(values: object, size: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"array shape mismatch: expected {(size,)}, got {arr.shape}")
    finite = np.isfinite(arr)
    if finite.all():
        return arr, False
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), True


def _projected_gravity(values: object) -> tuple[np.ndarray, bool]:
    arr, bad = _finite_array(values, 3)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-6:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32), True
    normalized = arr / norm
    return np.clip(normalized, -1.0, 1.0), bad
