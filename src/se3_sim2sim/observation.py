"""Observation assembly for the 29D joint-space policy input."""

from __future__ import annotations

import numpy as np

from se3_shared import JointGroup, ObservationConfig

from .config import RobotConfig
from .math_utils import rotate_inverse
from .runtime_spec import RuntimeSpec

_OBS_CFG = ObservationConfig()


class ObservationBuilder:
    def __init__(self, *, robot_cfg: RobotConfig, runtime: RuntimeSpec) -> None:
        self.robot_cfg = robot_cfg
        self.runtime = runtime
        self.commands_scale = np.asarray(robot_cfg.command_scale, dtype=np.float64)
        self.default_dof_pos = np.asarray(robot_cfg.default_dof_pos, dtype=np.float64)

    def build(
        self,
        *,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_world: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        command: np.ndarray,
        action_obs: np.ndarray,
    ) -> np.ndarray:
        obs: list[float] = []
        base_ang_vel_body = rotate_inverse(base_quat_wxyz, base_ang_vel_world)
        projected_gravity = rotate_inverse(base_quat_wxyz, np.asarray([0.0, 0.0, -1.0]))

        obs.extend((base_ang_vel_body * _OBS_CFG.ang_vel_scale).tolist())
        obs.extend(projected_gravity.tolist())
        obs.extend((np.asarray(command, dtype=np.float64) * self.commands_scale).tolist())

        leg_pos_rel = dof_pos[JointGroup.LEGS] - self.default_dof_pos[JointGroup.LEGS]
        obs.extend(leg_pos_rel.tolist())

        leg_vel = dof_vel[JointGroup.LEGS] * _OBS_CFG.leg_vel_scale
        obs.extend(leg_vel.tolist())

        obs.extend(dof_pos[JointGroup.WHEELS].tolist())
        obs.extend((dof_vel[JointGroup.WHEELS] * _OBS_CFG.wheel_vel_scale).tolist())

        obs.extend(np.asarray(action_obs, dtype=np.float64).tolist())

        arr = np.asarray(obs, dtype=np.float32)
        expected = int(self.runtime.policy.num_obs)
        if arr.shape != (expected,):
            raise RuntimeError(
                f"observation shape mismatch: expected {(expected,)}, got {arr.shape}"
            )
        return np.clip(arr, -self.runtime.clip_observations, self.runtime.clip_observations)
