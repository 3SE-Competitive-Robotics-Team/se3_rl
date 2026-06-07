"""CTBC 台阶前馈状态机。

状态机只负责根据轮子水平接触触发单侧动作 bias，不改变 observation 维度。
"""

from __future__ import annotations

import math

import torch

_BIAS_HEIGHTS = (0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36)
_FRONT_AMPS = (0.22, 0.18, 0.18, 0.16, 0.16, 0.15, 0.14, 0.14)
_ACTIVE_AMPS = (0.04, 0.10, 0.08, 0.10, 0.08, 0.08, 0.07, 0.07)


class StairClimbState:
    """每个 env 的 CTBC 接触触发前馈控制器。"""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        contact_window: int = 3,
        force_threshold_n: float = 30.0,
        ff_period_s: float = 0.6,
        control_dt: float = 0.02,
        cooldown_s: float = 0.3,
        ann_start_iter: int = 0,
        ann_end_iter: int = 1500,
        phantom_trigger_iter: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.contact_window = int(contact_window)
        self.force_threshold = float(force_threshold_n)
        self.ff_period_steps = max(1, round(float(ff_period_s) / float(control_dt)))
        self.cooldown_steps = max(1, round(float(cooldown_s) / float(control_dt)))
        self.ann_start_iter = int(ann_start_iter)
        self.ann_end_iter = int(ann_end_iter)
        self.phantom_trigger_iter = int(phantom_trigger_iter)

        self._contact_buf = torch.zeros(self.contact_window, self.num_envs, 2, device=self.device)
        self._stable = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._ff_phase = torch.full((self.num_envs, 2), -1, dtype=torch.long, device=self.device)
        self._cooldown = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self._complete_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._trigger_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_bias = torch.zeros(self.num_envs, 6, device=self.device)
        self._last_contact_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._iter = 0
        self._iter_origin: int | None = None
        self._kff = 1.0

        self._height_grid = torch.tensor(_BIAS_HEIGHTS, device=self.device)
        self._front_grid = torch.tensor(_FRONT_AMPS, device=self.device)
        self._active_grid = torch.tensor(_ACTIVE_AMPS, device=self.device)

    def update_iter(self, iteration: int) -> None:
        """更新相对 stair 阶段的退火进度。"""
        if self._iter_origin is None:
            self._iter_origin = int(iteration)
        local_iter = max(0, int(iteration) - self._iter_origin)
        self._iter = local_iter
        if local_iter < self.ann_start_iter:
            self._kff = 1.0
        elif local_iter >= self.ann_end_iter:
            self._kff = 0.0
        else:
            span = max(1, self.ann_end_iter - self.ann_start_iter)
            self._kff = max(0.0, 1.0 - (local_iter - self.ann_start_iter) / span)

    def step(self, wheel_contact_xy: torch.Tensor) -> None:
        """按左右轮水平接触力推进触发状态。"""
        force = wheel_contact_xy.to(device=self.device).reshape(self.num_envs, 2)
        self._last_contact_xy = force
        self._contact_buf[1:] = self._contact_buf[:-1].clone()
        self._contact_buf[0] = force

        self._stable = (self._contact_buf > self.force_threshold).all(dim=0)
        self._cooldown[self._cooldown > 0] -= 1

        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0)
        newly_triggered = self._stable & can_trigger
        self._ff_phase[newly_triggered] = 0
        self._trigger_count[newly_triggered.any(dim=1)] += 1

        if self._iter < self.phantom_trigger_iter:
            phantom = torch.rand(self.num_envs, 2, device=self.device) < 0.01
            phantom_trigger = phantom & can_trigger & (~newly_triggered)
            self._ff_phase[phantom_trigger] = 0
            self._trigger_count[phantom_trigger.any(dim=1)] += 1

        active = self._ff_phase >= 0
        self._ff_phase[active] += 1

        finished = self._ff_phase >= self.ff_period_steps
        self._cooldown[finished] = self.cooldown_steps
        self._ff_phase[finished] = -1
        self._complete_count[finished.any(dim=1)] += 1

    def ff_bias(self, command_height: torch.Tensor | None = None) -> torch.Tensor:
        """返回 6D action bias，幅值按 command height 插值。"""
        bias = torch.zeros(self.num_envs, 6, device=self.device)
        if self._kff <= 0.0:
            self._last_bias = bias
            return bias

        if command_height is None:
            height = torch.full((self.num_envs,), 0.26, device=self.device)
        else:
            height = command_height.to(device=self.device).reshape(self.num_envs)
        front_amp, active_amp = self._amps_from_height(height)

        for side in range(2):
            active_mask = self._ff_phase[:, side] >= 0
            if not active_mask.any():
                continue
            phase = self._ff_phase[:, side].float() / float(self.ff_period_steps)
            profile = 0.5 * (1.0 - torch.cos(2.0 * math.pi * phase))
            profile = profile * active_mask.float() * float(self._kff)
            if side == 0:
                bias[:, 0] = -front_amp * profile
                bias[:, 1] = active_amp * profile
            else:
                bias[:, 2] = front_amp * profile
                bias[:, 3] = active_amp * profile

        self._last_bias = bias
        return bias

    def active_mask(self) -> torch.Tensor:
        """返回任一侧 CTBC 前馈是否处于激活周期。"""
        return (self._ff_phase >= 0).any(dim=1)

    def diag(self) -> dict[str, float]:
        """返回训练日志诊断标量。"""
        active = self.active_mask()
        triggered_side = self._ff_phase >= 0
        return {
            "Stair/ctbc_trigger_rate": triggered_side.float().mean().item(),
            "Stair/ctbc_env_active_rate": active.float().mean().item(),
            "Stair/ctbc_complete_rate": (self._complete_count > 0).float().mean().item(),
            "Stair/ctbc_stable_contact_rate": self._stable.float().mean().item(),
            "Stair/ctbc_contact_xy_mean": self._last_contact_xy.mean().item(),
            "Stair/ctbc_contact_xy_max": self._last_contact_xy.max().item(),
            "Stair/ctbc_contact_xy_left_mean": self._last_contact_xy[:, 0].mean().item(),
            "Stair/ctbc_contact_xy_right_mean": self._last_contact_xy[:, 1].mean().item(),
            "Stair/ctbc_kff": float(self._kff),
            "Stair/ctbc_bias_abs_mean": torch.abs(self._last_bias[:, :4]).mean().item(),
            "Stair/ctbc_trigger_count_mean": self._trigger_count.float().mean().item(),
            "Stair/ctbc_complete_count_mean": self._complete_count.float().mean().item(),
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        """清空指定 env 的 CTBC 状态。"""
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self._contact_buf[:, ids] = 0.0
        self._stable[ids] = False
        self._ff_phase[ids] = -1
        self._cooldown[ids] = 0
        self._complete_count[ids] = 0
        self._trigger_count[ids] = 0
        self._last_bias[ids] = 0.0
        self._last_contact_xy[ids] = 0.0

    def _amps_from_height(self, height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按高度表线性插值得到 front/active 峰值 raw action。"""
        h = torch.clamp(height, self._height_grid[0], self._height_grid[-1])
        idx = torch.searchsorted(self._height_grid, h, right=True)
        idx = torch.clamp(idx, 1, self._height_grid.numel() - 1)
        h0 = self._height_grid[idx - 1]
        h1 = self._height_grid[idx]
        t = (h - h0) / torch.clamp(h1 - h0, min=1.0e-6)
        front = self._front_grid[idx - 1] + t * (self._front_grid[idx] - self._front_grid[idx - 1])
        active = self._active_grid[idx - 1] + t * (
            self._active_grid[idx] - self._active_grid[idx - 1]
        )
        return front, active
