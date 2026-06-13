"""真机 recovery policy 观测拼装。"""

from __future__ import annotations

import numpy as np

from se3_shared import (
    ObservationConfig,
    PolicyObservationResult,
    RobotConfig,
    build_policy_observation_np,
)

from .protocol import PolicyStateFrame

_OBS_CFG = ObservationConfig()
_ROBOT_CFG = RobotConfig()
_COMMAND_VX_LIMIT_MPS = 1.50
_COMMAND_YAW_RATE_LIMIT_RAD_S = 6.00
_COMMAND_HEIGHT_MIN_M = 0.15
_COMMAND_HEIGHT_MAX_M = 0.33


class RecoveryObservationBuilder:
    """按训练端 34 维 actor contract 拼装 recovery-only 观测。"""

    def __init__(self) -> None:
        self.default_dof_pos = np.asarray(_ROBOT_CFG.default_dof_pos, dtype=np.float32)
        self.command_scale = np.asarray(_OBS_CFG.command_scale, dtype=np.float32)
        self.default_command = np.asarray(
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
        self.command = self.default_command

    def build(self, state: PolicyStateFrame, last_action: np.ndarray) -> PolicyObservationResult:
        return build_policy_observation_np(
            base_ang_vel_body=np.asarray(state.base_ang_vel_body, dtype=np.float32),
            projected_gravity=np.asarray(state.projected_gravity, dtype=np.float32),
            dof_pos=np.asarray(state.dof_pos, dtype=np.float32),
            dof_vel=np.asarray(state.dof_vel, dtype=np.float32),
            command=self.command_from_state(state),
            action_obs=np.asarray(last_action, dtype=np.float32),
            default_dof_pos=self.default_dof_pos,
            command_scale=self.command_scale,
            expected_num_obs=_OBS_CFG.num_obs,
            clip_value=_OBS_CFG.clip_value,
            normalize_projected_gravity=True,
        )

    def command_from_state(self, state: PolicyStateFrame) -> np.ndarray:
        command = np.asarray(state.policy_command, dtype=np.float32)
        if command.shape != self.default_command.shape or not np.isfinite(command).all():
            return self.default_command
        command = command.copy()
        command[0] = np.clip(command[0], -_COMMAND_VX_LIMIT_MPS, _COMMAND_VX_LIMIT_MPS)
        command[1] = np.clip(
            command[1],
            -_COMMAND_YAW_RATE_LIMIT_RAD_S,
            _COMMAND_YAW_RATE_LIMIT_RAD_S,
        )
        command[4] = np.clip(command[4], _COMMAND_HEIGHT_MIN_M, _COMMAND_HEIGHT_MAX_M)
        return command


def synthetic_recovery_state(seq: int = 0) -> PolicyStateFrame:
    """生成站立零速状态，用于 checkpoint dry-run。"""

    dof_pos = tuple(float(v) for v in _ROBOT_CFG.default_dof_pos)
    return PolicyStateFrame(
        seq=int(seq),
        tick_ms=int(seq) * int(_ROBOT_CFG.control_dt * 1000.0),
        target_seq=0,
        target_age_ms=0,
        target_valid=0,
        rc_switch_r=0,
        output_enabled=0,
        base_ang_vel_body=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        joint_pos=dof_pos[:4],
        joint_vel=(0.0, 0.0, 0.0, 0.0),
        wheel_pos=dof_pos[4:6],
        wheel_vel=(0.0, 0.0),
        target_joint_pos=(0.0, 0.0, 0.0, 0.0),
        hip_torque=(0.0, 0.0, 0.0, 0.0),
        wheel_torque=(0.0, 0.0),
        wheel_motor_torque=(0.0, 0.0),
        command=(
            0.0,
            0.0,
            0.0,
            0.0,
            float(_ROBOT_CFG.default_base_height),
        ),
    )
