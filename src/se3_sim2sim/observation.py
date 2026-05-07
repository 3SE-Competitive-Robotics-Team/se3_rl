"""Observation assembly for the 27D joint-space policy input."""

from __future__ import annotations

import numpy as np

from .config import RobotConfig
from .math_utils import rotate_inverse
from .runtime_spec import RuntimeSpec

_LEG_IDS = [0, 1, 3, 4]
_WHEEL_IDS = [2, 5]


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

        obs.extend((base_ang_vel_body * 0.25).tolist())
        obs.extend(projected_gravity.tolist())
        obs.extend((np.asarray(command, dtype=np.float64) * self.commands_scale).tolist())

        leg_pos_rel = dof_pos[_LEG_IDS] - self.default_dof_pos[_LEG_IDS]
        obs.extend(leg_pos_rel.tolist())

        leg_vel = dof_vel[_LEG_IDS] * 0.25
        obs.extend(leg_vel.tolist())

        obs.extend(dof_pos[_WHEEL_IDS].tolist())
        obs.extend((dof_vel[_WHEEL_IDS] * 0.05).tolist())

        obs.extend(np.asarray(action_obs, dtype=np.float64).tolist())

        arr = np.asarray(obs, dtype=np.float32)
        expected = int(self.runtime.policy.num_obs)
        if arr.shape != (expected,):
            raise RuntimeError(
                f"observation shape mismatch: expected {(expected,)}, got {arr.shape}"
            )
        return np.clip(arr, -self.runtime.clip_observations, self.runtime.clip_observations)
