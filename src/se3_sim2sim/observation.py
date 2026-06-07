"""Observation assembly for the joint-space policy input (32D or 42D)."""

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
        self._is_task_mode = runtime.policy.is_task_mode

    def build(
        self,
        *,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_world: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        command: np.ndarray,
        action_obs: np.ndarray,
        task_mode_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._is_task_mode:
            if task_mode_obs is None or len(task_mode_obs) != 13:
                raise ValueError(
                    f"task_mode policy 需要 13 维 task_mode_obs，实际得到 "
                    f"{None if task_mode_obs is None else len(task_mode_obs)} 维。"
                )
        else:
            if len(command) not in (7, 8):
                raise ValueError(
                    f"command 必须为 7 或 8 维 [vx, oz, pitch, roll, height, jump_flag, jump_target_height, (jump_phase)],"
                    f" 实际得到 {len(command)} 维。"
                )

        obs: list[float] = []
        base_ang_vel_body = rotate_inverse(base_quat_wxyz, base_ang_vel_world)
        projected_gravity = rotate_inverse(base_quat_wxyz, np.asarray([0.0, 0.0, -1.0]))

        obs.extend((base_ang_vel_body * _OBS_CFG.ang_vel_scale).tolist())
        obs.extend(projected_gravity.tolist())
        obs.extend((np.asarray(command[:5], dtype=np.float64) * self.commands_scale).tolist())

        leg_pos_rel = dof_pos[JointGroup.CTRL_LEGS] - self.default_dof_pos[JointGroup.CTRL_LEGS]
        obs.extend(leg_pos_rel.tolist())

        leg_vel = dof_vel[JointGroup.CTRL_LEGS] * _OBS_CFG.leg_vel_scale
        obs.extend(leg_vel.tolist())

        obs.extend(dof_pos[JointGroup.CTRL_WHEELS].tolist())
        obs.extend((dof_vel[JointGroup.CTRL_WHEELS] * _OBS_CFG.wheel_vel_scale).tolist())

        obs.extend(np.asarray(action_obs, dtype=np.float64).tolist())

        if self._is_task_mode:
            obs.extend(np.asarray(task_mode_obs, dtype=np.float64).tolist())
        else:
            # jump_commands: [jump_flag, jump_target_height, jump_phase]
            if len(command) >= 8:
                obs.extend(command[5:8].tolist())
            else:
                obs.extend(command[5:7].tolist())
                obs.append(0.0)

        arr = np.asarray(obs, dtype=np.float32)
        expected = int(self.runtime.policy.num_obs)
        if arr.shape != (expected,):
            raise RuntimeError(
                f"observation shape mismatch: expected {(expected,)}, got {arr.shape}"
            )
        return np.clip(arr, -self.runtime.clip_observations, self.runtime.clip_observations)
