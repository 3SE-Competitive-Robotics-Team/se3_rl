"""Observation assembly for the 31D joint-space policy input."""

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
        # 前 5 维乘以 commands_scale，后 2 维（jump_flag, jump_target_height）直接追加
        obs.extend((np.asarray(command[:5], dtype=np.float64) * self.commands_scale).tolist())

        leg_pos_rel = dof_pos[JointGroup.CTRL_LEGS] - self.default_dof_pos[JointGroup.CTRL_LEGS]
        obs.extend(leg_pos_rel.tolist())

        leg_vel = dof_vel[JointGroup.CTRL_LEGS] * _OBS_CFG.leg_vel_scale
        obs.extend(leg_vel.tolist())

        obs.extend(dof_pos[JointGroup.CTRL_WHEELS].tolist())
        obs.extend((dof_vel[JointGroup.CTRL_WHEELS] * _OBS_CFG.wheel_vel_scale).tolist())

        obs.extend(np.asarray(action_obs, dtype=np.float64).tolist())

        # jump_commands: [jump_flag, jump_target_height, jump_phase]，不缩放
        # jump_phase 由 workflow 计算并写入 command[7]
        if len(command) >= 8:
            obs.extend(command[5:8].tolist())
        else:
            obs.extend(command[5:7].tolist())
            obs.append(0.0)  # jump_phase 占位（向后兼容旧 7 维指令）

        arr = np.asarray(obs, dtype=np.float32)
        expected = int(self.runtime.policy.num_obs)
        if arr.shape != (expected,):
            raise RuntimeError(
                f"observation shape mismatch: expected {(expected,)}, got {arr.shape}"
            )
        limit = float(self.runtime.clip_observations)
        arr = np.nan_to_num(arr, nan=0.0, posinf=limit, neginf=-limit)
        return np.clip(arr, -limit, limit)
