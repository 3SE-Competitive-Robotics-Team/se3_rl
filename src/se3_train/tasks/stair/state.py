"""CTBC 台阶爬升状态管理。"""

from __future__ import annotations

import math
import os

import torch

_HIP_LEFT = 0
_KNEE_LEFT = 1
_HIP_RIGHT = 2
_KNEE_RIGHT = 3

_HIP_FEEDFORWARD_RATIO: float = 1.5
_KNEE_FEEDFORWARD_RATIO: float = 1.0


class StairClimbState:
    """逐 env 的 CTBC 状态机。

    `ff_bias()` 保持源 stair 任务的旧输出关节 action 语义；训练端动作项会在注入前
    将它等效换算到目标仓库的主动杆 action 语义。
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        contact_window: int = 3,
        force_threshold_n: float = 30.0,
        ff_amplitude_rad: float = 0.3,
        ff_period_s: float = 0.6,
        control_dt: float = 0.02,
        ff_start_iter: int = 0,
        ann_start_iter: int = 200,
        ann_end_iter: int = 800,
        phantom_trigger_iter: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.contact_window = contact_window
        self.force_threshold = force_threshold_n
        self.ff_amplitude = ff_amplitude_rad
        self.control_dt = control_dt
        self.ff_period_steps = max(1, round(ff_period_s / control_dt))
        self.ff_start = int(ff_start_iter)
        self.ann_start = max(int(ann_start_iter), self.ff_start)
        self.ann_end = max(int(ann_end_iter), self.ann_start)
        self.phantom_trigger_iter = phantom_trigger_iter

        self._contact_buf = torch.zeros(contact_window, num_envs, 2, device=device)
        self._stable = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self._riser_contact_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ff_phase = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)
        self._cooldown_steps = max(1, round(0.3 / control_dt))
        self._cooldown = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        self._complete_ff_cycle_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._max_height_gain = torch.zeros(num_envs, device=device)
        self._max_radial_progress = torch.zeros(num_envs, device=device)
        self._progress_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._kff: float = 1.0
        self._iter: int = 0
        self._iter_origin: int | None = None
        self._fixed_iter: int | None = None

    def update_iter(self, iteration: int) -> None:
        """更新当前 stair 阶段内的相对训练迭代数和前馈退火权重。"""
        if self._fixed_iter is not None:
            local_iter = self._fixed_iter
        else:
            if self._iter_origin is None:
                offset = int(os.environ.get("SE3_STAIR_LOCAL_ITER_OFFSET", "0"))
                self._iter_origin = int(iteration) - offset
            local_iter = max(0, int(iteration) - self._iter_origin)
        self._iter = local_iter
        if local_iter < self.ff_start:
            self._kff = 0.0
        elif local_iter < self.ann_start:
            self._kff = 1.0
        elif local_iter >= self.ann_end:
            self._kff = 0.0
        else:
            span = max(1, self.ann_end - self.ann_start)
            self._kff = max(0.0, 1.0 - (local_iter - self.ann_start) / span)

    def set_fixed_iteration(self, iteration: int | None) -> None:
        """将 play/watch 的 CTBC 课程固定到所选 checkpoint 轮数。"""
        self._fixed_iter = None if iteration is None else max(0, int(iteration))
        if self._fixed_iter is not None:
            self.update_iter(self._fixed_iter)

    def step(self, wheel_contact_xy: torch.Tensor) -> None:
        """根据当前轮子水平接触力更新触发窗口和前馈相位。"""
        self._contact_buf[1:] = self._contact_buf[:-1].clone()
        self._contact_buf[0] = wheel_contact_xy

        above = self._contact_buf > self.force_threshold
        self._stable = above.all(dim=0)
        stable_any = self._stable.any(dim=-1)
        self._riser_contact_steps[stable_any] += 1
        self._riser_contact_steps[~stable_any] = 0

        self._cooldown[self._cooldown > 0] -= 1
        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0)
        newly_triggered = self._stable & can_trigger
        self._ff_phase[newly_triggered] = 0

        if self._iter < self.phantom_trigger_iter:
            phantom_mask = torch.rand(self.num_envs, 2, device=self.device) < 0.01
            phantom_trigger = phantom_mask & can_trigger & (~newly_triggered)
            self._ff_phase[phantom_trigger] = 0

        active = self._ff_phase >= 0
        self._ff_phase[active] += 1

        finished = self._ff_phase >= self.ff_period_steps
        self._cooldown[finished] = self._cooldown_steps
        self._ff_phase[finished] = -1
        self._complete_ff_cycle_count[finished.any(dim=-1)] += 1

    def ff_bias(self) -> torch.Tensor:
        """返回旧输出关节 action 语义下的 6D CTBC bias。"""
        bias = torch.zeros(self.num_envs, 6, device=self.device)
        if self._kff == 0.0:
            return bias

        for side, (hip_idx, knee_idx) in enumerate(
            [(_HIP_LEFT, _KNEE_LEFT), (_HIP_RIGHT, _KNEE_RIGHT)]
        ):
            active_mask = self._ff_phase[:, side] >= 0
            if not active_mask.any():
                continue
            phase = self._ff_phase[:, side].float()
            t = phase / self.ff_period_steps
            cosine_val = self.ff_amplitude * (1.0 - torch.cos(2.0 * math.pi * t))
            cosine_val = cosine_val * active_mask.float()
            bias[:, hip_idx] = -cosine_val * _HIP_FEEDFORWARD_RATIO * self._kff
            bias[:, knee_idx] = cosine_val * _KNEE_FEEDFORWARD_RATIO * self._kff
        return bias

    def contact_triggered(self) -> torch.Tensor:
        return (self._ff_phase >= 0).any(dim=-1)

    def ctbc_trigger_weight(self) -> torch.Tensor:
        return self.contact_triggered().float() * float(self._kff)

    def climb_progress_delta(
        self,
        height_gain: torch.Tensor,
        radial_progress: torch.Tensor,
        *,
        max_height_gain: float,
        max_radial_progress: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_height = torch.clamp(height_gain, min=0.0, max=max_height_gain)
        clipped_radial = torch.clamp(radial_progress, min=0.0, max=max_radial_progress)
        height_delta = torch.clamp(clipped_height - self._max_height_gain, min=0.0)
        radial_delta = torch.clamp(clipped_radial - self._max_radial_progress, min=0.0)
        height_delta = torch.where(
            self._progress_initialized,
            height_delta,
            torch.zeros_like(height_delta),
        )
        radial_delta = torch.where(
            self._progress_initialized,
            radial_delta,
            torch.zeros_like(radial_delta),
        )
        self._max_height_gain = torch.maximum(self._max_height_gain, clipped_height)
        self._max_radial_progress = torch.maximum(self._max_radial_progress, clipped_radial)
        self._progress_initialized[:] = True
        return height_delta, radial_delta

    def riser_stall_active(self, min_duration_s: float) -> torch.Tensor:
        min_steps = max(1, round(float(min_duration_s) / self.control_dt))
        return self._riser_contact_steps >= min_steps

    def reset(self, env_ids: torch.Tensor) -> None:
        self._contact_buf[:, env_ids] = 0.0
        self._stable[env_ids] = False
        self._riser_contact_steps[env_ids] = 0
        self._ff_phase[env_ids] = -1
        self._cooldown[env_ids] = 0
        self._complete_ff_cycle_count[env_ids] = 0
        self._max_height_gain[env_ids] = 0.0
        self._max_radial_progress[env_ids] = 0.0
        self._progress_initialized[env_ids] = False

    @property
    def kff(self) -> float:
        return self._kff

    @property
    def complete_ff_cycle_count(self) -> torch.Tensor:
        return self._complete_ff_cycle_count.clone()

    @property
    def local_iteration(self) -> int:
        return self._iter

    @property
    def latest_contact_force(self) -> torch.Tensor:
        return self._contact_buf[0].clone()

    @property
    def stable_contact(self) -> torch.Tensor:
        return self._stable.clone()

    @property
    def ff_phase(self) -> torch.Tensor:
        return self._ff_phase.clone()

    @property
    def cooldown(self) -> torch.Tensor:
        return self._cooldown.clone()

    def diag(self) -> dict[str, float]:
        return {
            "Stair/diag_ctbc_trigger_rate": self.contact_triggered().float().mean().item(),
            "Stair/diag_ctbc_complete_ff_cycles": (
                self._complete_ff_cycle_count.float().mean().item()
            ),
            "Stair/diag_riser_stall_rate": self.riser_stall_active(0.25).float().mean().item(),
            "Stair/diag_ctbc_kff": self._kff,
            "Stair/diag_ctbc_local_iter": float(self._iter),
        }


__all__ = ["StairClimbState"]
