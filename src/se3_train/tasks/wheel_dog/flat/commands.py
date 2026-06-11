"""WheelDog 平地速度指令。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """计算掩码内均值；无样本时返回 0。"""
    if mask.any():
        return float(value[mask].mean().item())
    return 0.0


@dataclass
class DogVelocityCommandCfg(CommandTermCfg):
    """平地速度指令配置。

    指令维度: [lin_vel_x, lin_vel_y, ang_vel_yaw]。
    """

    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0)
    lin_vel_deadband: float = 0.08
    yaw_deadband: float = 0.08
    standing_ratio: float = 0.08

    def build(self, env: ManagerBasedRlEnv) -> DogVelocityCommandTerm:
        return DogVelocityCommandTerm(self, env)


class DogVelocityCommandTerm(CommandTerm):
    """WheelDog 速度指令项。"""

    cfg: DogVelocityCommandCfg

    def __init__(self, cfg: DogVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._robot = env.scene["robot"]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样速度指令。"""
        num_ids = len(env_ids)
        standing_count = int(num_ids * self.cfg.standing_ratio)
        standing_ids = env_ids[:standing_count]
        moving_ids = env_ids[standing_count:]

        self._standing_mask[standing_ids] = True
        self._standing_mask[moving_ids] = False
        self._command[standing_ids] = 0.0

        if len(moving_ids) == 0:
            return
        self._command[moving_ids, 0] = self._sample_range(len(moving_ids), self.cfg.lin_vel_x_range)
        self._command[moving_ids, 1] = self._sample_range(len(moving_ids), self.cfg.lin_vel_y_range)
        self._command[moving_ids, 2] = self._sample_range(
            len(moving_ids), self.cfg.ang_vel_yaw_range
        )

    def _update_command(self) -> None:
        """对很小的速度指令施加死区。"""
        moving = ~self._standing_mask
        vx = self._command[:, 0]
        vy = self._command[:, 1]
        yaw = self._command[:, 2]

        speed = torch.linalg.norm(self._command[:, :2], dim=1)
        small_speed = moving & (speed < self.cfg.lin_vel_deadband)
        vx = torch.where(small_speed, torch.zeros_like(vx), vx)
        vy = torch.where(small_speed, torch.zeros_like(vy), vy)
        yaw = torch.where(
            moving & (torch.abs(yaw) < self.cfg.yaw_deadband),
            torch.zeros_like(yaw),
            yaw,
        )

        self._command[:, 0] = vx
        self._command[:, 1] = vy
        self._command[:, 2] = yaw

    def _update_metrics(self) -> None:
        """写入速度跟随诊断指标。"""
        if not hasattr(self._env, "extras") or not isinstance(self._env.extras.get("log"), dict):
            return

        lin_vel_b = self._robot.data.root_link_lin_vel_b
        ang_vel_b = self._robot.data.root_link_ang_vel_b
        cmd = self._command
        active = torch.linalg.norm(cmd[:, :2], dim=1) > self.cfg.lin_vel_deadband

        vx_err = lin_vel_b[:, 0] - cmd[:, 0]
        vy_err = lin_vel_b[:, 1] - cmd[:, 1]
        yaw_err = ang_vel_b[:, 2] - cmd[:, 2]
        height = self._robot.data.root_link_pos_w[:, 2]

        self._env.extras["log"].update(
            {
                "WheelDog/diag_active_ratio": float(active.float().mean().item()),
                "WheelDog/diag_standing_ratio": float(self._standing_mask.float().mean().item()),
                "WheelDog/diag_cmd_vx_abs": float(torch.abs(cmd[:, 0]).mean().item()),
                "WheelDog/diag_cmd_vy_abs": float(torch.abs(cmd[:, 1]).mean().item()),
                "WheelDog/diag_actual_vx": float(lin_vel_b[:, 0].mean().item()),
                "WheelDog/diag_actual_vy": float(lin_vel_b[:, 1].mean().item()),
                "WheelDog/diag_vx_error_abs": float(torch.abs(vx_err).mean().item()),
                "WheelDog/diag_vy_error_abs": float(torch.abs(vy_err).mean().item()),
                "WheelDog/diag_vx_error_abs_active": _mean_on_mask(torch.abs(vx_err), active),
                "WheelDog/diag_vy_error_abs_active": _mean_on_mask(torch.abs(vy_err), active),
                "WheelDog/diag_yaw_error_abs": float(torch.abs(yaw_err).mean().item()),
                "WheelDog/diag_base_height": float(height.mean().item()),
                "WheelDog/diag_cmd_x_limit": float(abs(self.cfg.lin_vel_x_range[1])),
                "WheelDog/diag_cmd_y_limit": float(abs(self.cfg.lin_vel_y_range[1])),
            }
        )

    def _sample_range(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        """在闭区间内均匀采样。"""
        lo, hi = value_range
        return torch.rand(count, device=self.device) * (hi - lo) + lo


__all__ = ["DogVelocityCommandCfg", "DogVelocityCommandTerm"]
