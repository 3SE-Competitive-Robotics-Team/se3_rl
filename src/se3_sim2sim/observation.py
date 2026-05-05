"""Observation assembly for the 27D wheel-legged policy input."""

from __future__ import annotations

import numpy as np

from .config import RobotConfig
from .math_utils import rotate_inverse
from .runtime_spec import RuntimeSpec


class ObservationBuilder:
    def __init__(self, *, robot_cfg: RobotConfig, runtime: RuntimeSpec) -> None:
        self.robot_cfg = robot_cfg
        self.runtime = runtime
        self.commands_scale = np.asarray(robot_cfg.command_scale, dtype=np.float64)

    def build(
        self,
        *,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_world: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        command: np.ndarray,
        theta0: np.ndarray,
        theta0_dot: np.ndarray,
        l0: np.ndarray,
        l0_dot: np.ndarray,
        action_obs: np.ndarray,
    ) -> np.ndarray:
        obs: list[float] = []
        base_ang_vel_body = rotate_inverse(base_quat_wxyz, base_ang_vel_world)
        projected_gravity = rotate_inverse(base_quat_wxyz, np.asarray([0.0, 0.0, -1.0]))

        obs.extend((base_ang_vel_body * 0.25).tolist())
        obs.extend(projected_gravity.tolist())
        obs.extend((np.asarray(command, dtype=np.float64) * self.commands_scale).tolist())
        obs.extend(theta0.tolist())
        obs.extend((theta0_dot * 0.05).tolist())
        obs.extend((l0 * 5.0).tolist())
        obs.extend((l0_dot * 0.25).tolist())
        # Isaac Gym's wheel position observation has the opposite sign from the
        # MuJoCo hinge qpos for this MJCF.
        obs.extend((-dof_pos[[2, 5]]).tolist())
        obs.extend((dof_vel[[2, 5]] * 0.05).tolist())
        obs.extend(np.asarray(action_obs, dtype=np.float64).tolist())

        arr = np.asarray(obs, dtype=np.float32)
        expected = int(self.runtime.policy.num_obs)
        if arr.shape != (expected,):
            raise RuntimeError(
                f"observation shape mismatch: expected {(expected,)}, got {arr.shape}"
            )
        return np.clip(arr, -self.runtime.clip_observations, self.runtime.clip_observations)
