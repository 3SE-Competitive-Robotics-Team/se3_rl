"""速度 + 高度指令生成器。

指令:(lin_vel_x, ang_vel_yaw, height)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class VelocityHeightCommandCfg(CommandTermCfg):
    """速度 + 高度指令生成器的配置。"""

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    ang_vel_yaw_range: tuple[float, float] = (-6.0, 6.0)
    height: float = 0.28
    lin_vel_deadband: float = 0.1
    yaw_deadband: float = 0.1
    standing_ratio: float = 0.1
    resampling_time_range: tuple[float, float] = (5.0, 5.0)

    def build(self, env: ManagerBasedRlEnv) -> VelocityHeightCommandTerm:
        return VelocityHeightCommandTerm(self, env)


class VelocityHeightCommandTerm(CommandTerm):
    """速度 + 高度的指令项。"""

    cfg: VelocityHeightCommandCfg

    def __init__(self, cfg: VelocityHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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

        # 站立环境:零速度,固定高度。
        self._command[standing_ids, 0] = 0.0
        self._command[standing_ids, 1] = 0.0
        self._command[standing_ids, 2] = self.cfg.height

        # 运动环境:随机速度,固定高度。
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

            self._command[moving_ids, 0] = lin_vel
            self._command[moving_ids, 1] = yaw_vel
            self._command[moving_ids, 2] = self.cfg.height

    def _update_command(self) -> None:
        """对指令施加死区。"""
        # 对非站立环境施加死区。
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
