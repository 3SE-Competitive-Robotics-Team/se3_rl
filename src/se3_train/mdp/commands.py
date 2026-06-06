"""速度 + 姿态指令生成器。

指令:(lin_vel_x, ang_vel_yaw, pitch, roll, height)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from se3_train.mdp.height_default_cache import update_policy_default_from_height_cache

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class VelocityHeightCommandCfg(CommandTermCfg):
    """速度 + 姿态 + 高度指令生成器的配置。"""

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
    height_resample_on_reset_only: bool = False
    """是否只在 reset 时采样高度指令；普通重采样只更新速度和姿态指令。"""

    def build(self, env: ManagerBasedRlEnv) -> VelocityHeightCommandTerm:
        return VelocityHeightCommandTerm(self, env)


class VelocityHeightCommandTerm(CommandTerm):
    """速度 + 姿态 + 高度的指令项。

    指令维度: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    """

    cfg: VelocityHeightCommandCfg

    def __init__(self, cfg: VelocityHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 5 维指令: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._resampling_for_reset = False

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """重置指令项，并把跨 reset 缓存的终止诊断搬入日志。"""
        assert isinstance(env_ids, torch.Tensor)
        previous_resampling_for_reset = self._resampling_for_reset
        self._resampling_for_reset = True
        try:
            extras = super().reset(env_ids)
            self._log_bad_orientation_diagnostics(env_ids)
            return extras
        finally:
            self._resampling_for_reset = previous_resampling_for_reset

    def _log_bad_orientation_diagnostics(self, env_ids: torch.Tensor) -> None:
        """上报 bad_orientation 在恢复样本和平地样本中的拆分来源。"""
        diag = (
            self._env.extras.get("_bad_orientation_diag") if hasattr(self._env, "extras") else None
        )
        if not isinstance(diag, dict):
            return

        raw_bad = diag.get("raw_bad")
        counted_bad = diag.get("counted_bad")
        terminated = diag.get("terminated")
        recovery_mask = diag.get("recovery_mask")
        recovery_grace = diag.get("recovery_grace")
        tensors = (raw_bad, counted_bad, terminated, recovery_mask, recovery_grace)
        if not all(
            isinstance(item, torch.Tensor) and item.shape[0] == self.num_envs for item in tensors
        ):
            return

        reset_recovery = recovery_mask[env_ids]
        reset_flat = ~reset_recovery
        reset_terminated = terminated[env_ids]
        reset_raw_bad = raw_bad[env_ids]
        reset_counted_bad = counted_bad[env_ids]
        reset_grace = recovery_grace[env_ids]

        def _masked_rate(values: torch.Tensor, mask: torch.Tensor) -> float:
            if not mask.any():
                return 0.0
            return values[mask].float().mean().item()

        log = self._env.extras.setdefault("log", {})
        log.update(
            {
                "Episode_Termination/bad_orientation_recovery": (reset_terminated & reset_recovery)
                .float()
                .sum()
                .item(),
                "Episode_Termination/bad_orientation_flat": (reset_terminated & reset_flat)
                .float()
                .sum()
                .item(),
                "Recovery/bad_orientation_raw_rate": _masked_rate(reset_raw_bad, reset_recovery),
                "Recovery/bad_orientation_counted_rate": _masked_rate(
                    reset_counted_bad, reset_recovery
                ),
                "Recovery/bad_orientation_grace_rate": _masked_rate(reset_grace, reset_recovery),
                "Recovery/bad_orientation_termination_rate": _masked_rate(
                    reset_terminated, reset_recovery
                ),
            }
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样指令。"""
        n = len(env_ids)
        resample_height = not self.cfg.height_resample_on_reset_only or bool(
            getattr(self, "_resampling_for_reset", False)
        )

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
        if len(standing_ids) > 0 and resample_height:
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
            self._command[moving_ids, 0] = lin_vel
            self._command[moving_ids, 1] = yaw_vel
            self._command[moving_ids, 2] = pitch
            self._command[moving_ids, 3] = roll
            if resample_height:
                height = (
                    torch.rand(len(moving_ids), device=self.device)
                    * (self.cfg.height_range[1] - self.cfg.height_range[0])
                    + self.cfg.height_range[0]
                )
                self._command[moving_ids, 4] = height

        if resample_height:
            update_policy_default_from_height_cache(
                self._env,
                "velocity_height",
                env_ids=env_ids,
                command=self._command,
            )

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
