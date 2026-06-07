"""CTBC 台阶前馈状态机。

状态机只负责根据轮子水平接触触发 retract+sweep 前馈，不改变 observation 维度。
"""

from __future__ import annotations

import torch


class StairClimbState:
    """每个 env 的 CTBC 接触触发前馈控制器。"""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        contact_window: int = 3,
        force_threshold_n: float = 30.0,
        ff_period_s: float = 2.0,
        control_dt: float = 0.02,
        cooldown_s: float = 0.4,
        retract_front_amp: float = 0.4,
        retract_active_amp: float = 0.0,
        sweep_front_amp: float = 0.2,
        sweep_active_amp: float = 0.0,
        wheel_amp: float = 1.5,
        retract_s: float = 0.3,
        sweep_s: float = 1.2,
        release_s: float = 0.5,
        ann_start_iter: int = 0,
        ann_end_iter: int = 1500,
        phantom_trigger_iter: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.contact_window = int(contact_window)
        self.force_threshold = float(force_threshold_n)
        self.control_dt = float(control_dt)
        self.retract_front_amp = float(retract_front_amp)
        self.retract_active_amp = float(retract_active_amp)
        self.sweep_front_amp = float(sweep_front_amp)
        self.sweep_active_amp = float(sweep_active_amp)
        self.wheel_amp = float(wheel_amp)
        self.retract_steps = max(1, round(float(retract_s) / self.control_dt))
        self.sweep_steps = max(1, round(float(sweep_s) / self.control_dt))
        self.release_steps = max(1, round(float(release_s) / self.control_dt))
        segment_steps = self.retract_steps + self.sweep_steps + self.release_steps
        self.ff_period_steps = max(1, round(float(ff_period_s) / self.control_dt), segment_steps)
        self.cooldown_steps = max(1, round(float(cooldown_s) / self.control_dt))
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

    def update_iter(self, iteration: int) -> None:
        """更新相对 stair 阶段的前馈退火进度。"""
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
        """按轮子水平接触力推进触发状态。"""
        force = wheel_contact_xy.to(device=self.device).reshape(self.num_envs, 2)
        self._last_contact_xy = force
        self._contact_buf[1:] = self._contact_buf[:-1].clone()
        self._contact_buf[0] = force

        self._stable = self._contact_buf.mean(dim=0) > self.force_threshold
        self._cooldown[self._cooldown > 0] -= 1

        can_trigger = (self._ff_phase < 0).all(dim=1) & (self._cooldown == 0).all(dim=1)
        newly_triggered_env = self._stable.any(dim=1) & can_trigger
        self._ff_phase[newly_triggered_env] = 0
        self._trigger_count[newly_triggered_env] += 1

        if self._iter < self.phantom_trigger_iter:
            phantom = torch.rand(self.num_envs, device=self.device) < 0.01
            phantom_trigger = phantom & can_trigger & (~newly_triggered_env)
            self._ff_phase[phantom_trigger] = 0
            self._trigger_count[phantom_trigger] += 1

        active = self._ff_phase >= 0
        self._ff_phase[active] += 1

        finished = self._ff_phase >= self.ff_period_steps
        self._cooldown[finished] = self.cooldown_steps
        self._ff_phase[finished] = -1
        self._complete_count[finished.any(dim=1)] += 1

    def ff_bias(self, command_height: torch.Tensor | None = None) -> torch.Tensor:
        """返回 6D action bias；当前版本使用离线验证的固定 retract+sweep 前馈。"""
        del command_height
        bias = torch.zeros(self.num_envs, 6, device=self.device)
        if self._kff <= 0.0:
            self._last_bias = bias
            return bias

        active_mask = (self._ff_phase >= 0).any(dim=1)
        if not active_mask.any():
            self._last_bias = bias
            return bias

        phase = torch.clamp(self._ff_phase[:, 0], min=0).float()
        retract = self._smooth01(phase / float(self.retract_steps))
        sweep_phase = (phase - float(self.retract_steps)) / float(self.sweep_steps)
        sweep = torch.sin(torch.pi * torch.clamp(sweep_phase, 0.0, 1.0))
        release_phase = (phase - float(self.retract_steps + self.sweep_steps)) / float(
            self.release_steps
        )
        retract = retract * (1.0 - self._smooth01(release_phase))
        profile_mask = active_mask.float() * float(self._kff)

        front = (self.retract_front_amp * retract + self.sweep_front_amp * sweep) * profile_mask
        active = (self.retract_active_amp * retract + self.sweep_active_amp * sweep) * profile_mask
        wheel = self.wheel_amp * sweep * profile_mask

        bias[:, 0] = front
        bias[:, 1] = active
        bias[:, 2] = -front
        bias[:, 3] = active
        bias[:, 4] = wheel
        bias[:, 5] = -wheel

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
            "Stair/ctbc_wheel_bias_abs_mean": torch.abs(self._last_bias[:, 4:6]).mean().item(),
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

    @staticmethod
    def _smooth01(x: torch.Tensor) -> torch.Tensor:
        """三次平滑阶跃，避免前馈相位边界突然跳变。"""
        y = torch.clamp(x, 0.0, 1.0)
        return y * y * (3.0 - 2.0 * y)
