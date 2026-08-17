"""Observation assembly for the 34D joint-space policy input."""

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
    ) -> None:
        self.robot_cfg = robot_cfg
        self.runtime = runtime
        self.commands_scale = np.asarray(robot_cfg.command_scale, dtype=np.float64)
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64)
        self._history: dict[str, np.ndarray] | None = None
        self._last_frame_id: int | None = None

    def reset(self) -> None:
        """清空历史；下一帧会按训练端语义回填全部槽位。"""

        self._history = None
        self._last_frame_id = None

    def build(
        self,
        *,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_world: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        command: np.ndarray,
        action_obs: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        base_ang_vel_body = rotate_inverse(base_quat_wxyz, base_ang_vel_world)
        projected_gravity = rotate_inverse(base_quat_wxyz, np.asarray([0.0, 0.0, -1.0]))
        expected = int(self.runtime.base_num_obs)
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
        )
        current = result.obs.astype(np.float32, copy=False)
        history_length = int(self.runtime.observation_history_length)
        if history_length == 1:
            return current

        current_slices: dict[str, slice] = {}
        cursor = 0
        for term in self.runtime.observation_terms:
            current_slices[term.name] = slice(cursor, cursor + term.size)
            cursor += term.size

        if self._history is None:
            self._history = {
                name: np.repeat(current[sl][None, :], history_length, axis=0)
                for name, sl in current_slices.items()
            }
        elif self._last_frame_id == int(frame_id):
            for name, sl in current_slices.items():
                self._history[name][-1] = current[sl]
        else:
            for name, sl in current_slices.items():
                values = self._history[name]
                values[:-1] = values[1:]
                values[-1] = current[sl]
        self._last_frame_id = int(frame_id)
        return np.concatenate(
            [self._history[term.name].reshape(-1) for term in self.runtime.observation_terms]
        ).astype(np.float32, copy=False)
