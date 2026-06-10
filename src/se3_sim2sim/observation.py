"""Observation assembly for the 31D joint-space policy input."""

from __future__ import annotations

import numpy as np

from se3_shared import build_policy_observation_np

from .config import RobotConfig
from .math_utils import rotate_inverse
from .runtime_spec import RuntimeSpec


class ObservationBuilder:
    def __init__(
        self,
        *,
        robot_cfg: RobotConfig,
        runtime: RuntimeSpec,
        default_dof_pos: np.ndarray,
        fourbar_surrogate: bool = False,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.runtime = runtime
        self.fourbar_surrogate = bool(fourbar_surrogate)
        self.commands_scale = np.asarray(robot_cfg.command_scale, dtype=np.float64)
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)

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
        base_ang_vel_body = rotate_inverse(base_quat_wxyz, base_ang_vel_world)
        projected_gravity = rotate_inverse(base_quat_wxyz, np.asarray([0.0, 0.0, -1.0]))
        expected = int(self.runtime.policy.num_obs)
        limit = float(self.runtime.clip_observations)
        result = build_policy_observation_np(
            base_ang_vel_body=base_ang_vel_body,
            projected_gravity=projected_gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            command=command,
            action_obs=action_obs,
            default_dof_pos=self.default_dof_pos,
            command_scale=self.commands_scale,
            expected_num_obs=expected,
            clip_value=limit,
            fourbar_surrogate=self.fourbar_surrogate,
        )
        return result.obs
