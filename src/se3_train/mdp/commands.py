"""基础指令和速度 + 姿态指令生成器。

指令:(lin_vel_x, ang_vel_yaw, pitch, roll, height)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass
class BasicCommandCfg(CommandTermCfg):
    """基础指令配置，提供所有任务共享的速度、姿态和高度采样参数。"""

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    ang_vel_yaw_range: tuple[float, float] = (-3.0, 3.0)
    pitch_range: tuple[float, float] = (-0.2, 0.2)
    roll_range: tuple[float, float] = (-0.1, 0.1)
    height_range: tuple[float, float] = (0.20, 0.32)
    standing_height_range: tuple[float, float] = (0.20, 0.32)
    lin_vel_deadband: float = 0.1
    yaw_deadband: float = 0.1
    standing_ratio: float = 0.1
    resampling_time_range: tuple[float, float] = (5.0, 5.0)

    @dataclass
    class VizCfg:
        """速度跟踪调试可视化参数。"""

        z_offset: float = 0.25
        scale: float = 0.8
        command_color: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.65)
        actual_color: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.75)
        width: float = 0.015

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> BasicCommandTerm:
        return BasicCommandTerm(self, env)


class BasicCommandTerm(CommandTerm):
    """基础指令项。

    指令维度: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    """

    cfg: BasicCommandCfg

    def __init__(self, cfg: BasicCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 5 维指令: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._robot = env.scene["robot"]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样指令。"""
        n = len(env_ids)

        # 确定哪些环境处于站立状态。
        standing_count = int(n * self.cfg.standing_ratio)
        standing_ids = env_ids[:standing_count]
        moving_ids = env_ids[standing_count:]

        self._standing_mask[standing_ids] = True
        self._standing_mask[moving_ids] = False

        # 站立环境:零速度,默认姿态,按站立高度范围采样。
        self._command[standing_ids, 0] = 0.0
        self._command[standing_ids, 1] = 0.0
        self._command[standing_ids, 2] = 0.0  # pitch = 0
        self._command[standing_ids, 3] = 0.0  # roll = 0
        if len(standing_ids) > 0:
            standing_height = (
                torch.rand(len(standing_ids), device=self.device)
                * (self.cfg.standing_height_range[1] - self.cfg.standing_height_range[0])
                + self.cfg.standing_height_range[0]
            )
            self._command[standing_ids, 4] = standing_height

        # 运动环境:随机速度 + 随机姿态 + 随机高度。
        if len(moving_ids) > 0:
            lin_vel = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.lin_vel_x_range[1] - self.cfg.lin_vel_x_range[0])
                + self.cfg.lin_vel_x_range[0]
            )
            yaw_vel = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.ang_vel_yaw_range[1] - self.cfg.ang_vel_yaw_range[0])
                + self.cfg.ang_vel_yaw_range[0]
            )
            pitch = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.pitch_range[1] - self.cfg.pitch_range[0])
                + self.cfg.pitch_range[0]
            )
            roll = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.roll_range[1] - self.cfg.roll_range[0])
                + self.cfg.roll_range[0]
            )
            height = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.height_range[1] - self.cfg.height_range[0])
                + self.cfg.height_range[0]
            )

            self._command[moving_ids, 0] = lin_vel
            self._command[moving_ids, 1] = yaw_vel
            self._command[moving_ids, 2] = pitch
            self._command[moving_ids, 3] = roll
            self._command[moving_ids, 4] = height

    def _update_command(self) -> None:
        """对速度指令施加死区。"""
        moving = ~self._standing_mask
        lin_vel = self._command[:, 0]
        yaw_vel = self._command[:, 1]

        # 将小速度置零(死区)。
        lin_vel = torch.where(
            moving & (torch.abs(lin_vel) < self.cfg.lin_vel_deadband),
            torch.zeros_like(lin_vel),
            lin_vel,
        )
        yaw_vel = torch.where(
            moving & (torch.abs(yaw_vel) < self.cfg.yaw_deadband),
            torch.zeros_like(yaw_vel),
            yaw_vel,
        )

        self._command[:, 0] = lin_vel
        self._command[:, 1] = yaw_vel

    def _update_metrics(self) -> None:
        """更新指令指标。"""
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """绘制期望线速度和实际线速度箭头。"""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        commands = self.command.cpu().numpy()
        base_pos_ws = self._robot.data.root_link_pos_w.cpu().numpy()
        base_mat_ws = matrix_from_quat(self._robot.data.root_link_quat_w).cpu().numpy()
        lin_vel_bs = self._robot.data.root_link_lin_vel_b.cpu().numpy()

        local_offset = np.array([0.0, 0.0, self.cfg.viz.z_offset])
        for env_idx in env_indices:
            base_pos_w = base_pos_ws[env_idx]
            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            base_mat_w = base_mat_ws[env_idx]
            start = base_pos_w + base_mat_w @ local_offset
            command_vec_b = np.array([commands[env_idx, 0], 0.0, 0.0]) * self.cfg.viz.scale
            actual_vec_b = (
                np.array([lin_vel_bs[env_idx, 0], lin_vel_bs[env_idx, 1], 0.0]) * self.cfg.viz.scale
            )

            if np.linalg.norm(command_vec_b) > 1e-4:
                visualizer.add_arrow(
                    start,
                    start + base_mat_w @ command_vec_b,
                    color=self.cfg.viz.command_color,
                    width=self.cfg.viz.width,
                    label="期望速度",
                )
            if np.linalg.norm(actual_vec_b) > 1e-4:
                visualizer.add_arrow(
                    start,
                    start + base_mat_w @ actual_vec_b,
                    color=self.cfg.viz.actual_color,
                    width=self.cfg.viz.width,
                    label="实际速度",
                )


@dataclass
class VelocityHeightCommandCfg(BasicCommandCfg):
    """速度 + 姿态 + 高度指令生成器的配置。"""

    def build(self, env: ManagerBasedRlEnv) -> VelocityHeightCommandTerm:
        return VelocityHeightCommandTerm(self, env)


class VelocityHeightCommandTerm(BasicCommandTerm):
    """普通速度 + 姿态 + 高度指令项。"""

    cfg: VelocityHeightCommandCfg
