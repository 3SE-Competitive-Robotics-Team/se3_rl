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
        ff_x_m: float = 0.025,
        ff_lift_m: float = 0.085,
        ff_period_s: float = 0.6,
        ff_rise_ratio: float = 0.25,
        ff_hold_ratio: float = 0.45,
        ff_wheel_action: float = 0.0,
        control_dt: float = 0.02,
        ff_start_iter: int = 0,
        ann_start_iter: int = 200,
        ann_end_iter: int = 800,
        phantom_trigger_iter: int = 0,
        allow_bilateral_trigger: bool = True,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.contact_window = contact_window
        self.force_threshold = force_threshold_n
        self.ff_amplitude = ff_amplitude_rad
        self.ff_x_m = ff_x_m
        self.ff_lift_m = ff_lift_m
        self.ff_wheel_action = float(ff_wheel_action)
        self.control_dt = control_dt
        self.ff_period_steps = max(1, round(ff_period_s / control_dt))
        self.ff_rise_steps, self.ff_hold_steps, self.ff_return_steps = self._resolve_profile_steps(
            ff_rise_ratio, ff_hold_ratio
        )
        self.ff_start = int(ff_start_iter)
        self.ann_start = max(int(ann_start_iter), self.ff_start)
        self.ann_end = max(int(ann_end_iter), self.ann_start)
        self.phantom_trigger_iter = phantom_trigger_iter
        self.allow_bilateral_trigger = bool(allow_bilateral_trigger)

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
        self._initial_wheel_terrain_z = torch.full((num_envs, 2), float("nan"), device=device)
        self._max_wheel_supported_rise = torch.zeros(num_envs, 2, device=device)
        self._max_wheel_supported_both_rise = torch.zeros(num_envs, device=device)
        self._wheel_support_record_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self._wheel_supported_both_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._max_wheel_supported_both_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._kff: float = 1.0
        self._iter: int = 0
        self._iter_origin: int | None = None
        self._fixed_iter: int | None = None

    def _resolve_profile_steps(
        self,
        rise_ratio: float,
        hold_ratio: float,
    ) -> tuple[int, int, int]:
        """把前馈 profile 比例转换成 rise/hold/return 三段步数。"""
        rise_ratio = max(0.05, min(float(rise_ratio), 0.90))
        hold_ratio = max(0.0, min(float(hold_ratio), 0.90))
        if rise_ratio + hold_ratio >= 0.95:
            hold_ratio = max(0.0, 0.95 - rise_ratio)

        rise_steps = max(1, round(self.ff_period_steps * rise_ratio))
        hold_steps = max(0, round(self.ff_period_steps * hold_ratio))
        if rise_steps + hold_steps >= self.ff_period_steps:
            hold_steps = max(0, self.ff_period_steps - rise_steps - 1)
        return_steps = max(1, self.ff_period_steps - rise_steps - hold_steps)
        return rise_steps, hold_steps, return_steps

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
        wheel_contact_xy = torch.nan_to_num(
            wheel_contact_xy.to(device=self.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._contact_buf[1:] = self._contact_buf[:-1].clone()
        self._contact_buf[0] = wheel_contact_xy

        above = self._contact_buf > self.force_threshold
        self._stable = above.all(dim=0)
        stable_any = self._stable.any(dim=-1)
        self._riser_contact_steps[stable_any] += 1
        self._riser_contact_steps[~stable_any] = 0

        self._cooldown[self._cooldown > 0] -= 1
        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0)
        env_has_active_side = (self._ff_phase >= 0).any(dim=-1, keepdim=True)
        trigger_candidates = self._stable & can_trigger & (~env_has_active_side)
        if self.allow_bilateral_trigger:
            newly_triggered = can_trigger & trigger_candidates.any(dim=-1, keepdim=True)
        else:
            newly_triggered = self._select_single_trigger_side(
                trigger_candidates,
                wheel_contact_xy,
            )
        self._ff_phase[newly_triggered] = 0

        if self._iter < self.phantom_trigger_iter:
            env_has_active_side = (self._ff_phase >= 0).any(dim=-1, keepdim=True)
            phantom_mask = torch.rand(self.num_envs, 2, device=self.device) < 0.01
            phantom_candidates = phantom_mask & can_trigger & (~env_has_active_side)
            if self.allow_bilateral_trigger:
                phantom_trigger = can_trigger & phantom_candidates.any(dim=-1, keepdim=True)
            else:
                phantom_score = torch.rand(self.num_envs, 2, device=self.device)
                phantom_trigger = self._select_single_trigger_side(
                    phantom_candidates,
                    phantom_score,
                )
            self._ff_phase[phantom_trigger] = 0

        active = self._ff_phase >= 0
        self._ff_phase[active] += 1

        finished = self._ff_phase >= self.ff_period_steps
        self._cooldown[finished] = self._cooldown_steps
        self._ff_phase[finished] = -1
        self._complete_ff_cycle_count[finished.any(dim=-1)] += 1

    def _select_single_trigger_side(
        self,
        candidates: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """每个 env 最多选择一个 CTBC 触发侧。"""
        selected = torch.zeros_like(candidates, dtype=torch.bool)
        env_ids = candidates.any(dim=-1).nonzero().flatten()
        if env_ids.numel() == 0:
            return selected

        neg_inf = torch.full_like(score, -float("inf"))
        side_score = torch.where(candidates, score, neg_inf)
        side_ids = torch.argmax(side_score, dim=-1)
        selected[env_ids, side_ids[env_ids]] = True
        return selected

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
            cosine_val = 2.0 * self.ff_amplitude * self._ff_profile_envelope(phase)
            cosine_val = cosine_val * active_mask.float()
            bias[:, hip_idx] = -cosine_val * _HIP_FEEDFORWARD_RATIO * self._kff
            bias[:, knee_idx] = cosine_val * _KNEE_FEEDFORWARD_RATIO * self._kff
        return bias

    def _ff_profile_envelope(self, phase: torch.Tensor) -> torch.Tensor:
        """返回 rise-hold-return 三段式前馈包络。"""
        phase = torch.clamp(phase, min=0.0, max=float(self.ff_period_steps))
        rise_steps = float(self.ff_rise_steps)
        hold_end = float(self.ff_rise_steps + self.ff_hold_steps)
        return_steps = float(self.ff_return_steps)

        rise_t = torch.clamp(phase / rise_steps, 0.0, 1.0)
        rise = 0.5 * (1.0 - torch.cos(math.pi * rise_t))

        return_t = torch.clamp((phase - hold_end) / return_steps, 0.0, 1.0)
        returning = 0.5 * (1.0 + torch.cos(math.pi * return_t))

        envelope = torch.where(phase < rise_steps, rise, torch.ones_like(phase))
        envelope = torch.where(phase < hold_end, envelope, returning)
        return torch.clamp(envelope, 0.0, 1.0)

    def ff_wheel_delta_xz(self) -> torch.Tensor:
        """返回当前轮端 Cartesian 前馈位移，语义为 x 后退、z 抬升。"""
        delta = torch.zeros(self.num_envs, 2, 2, device=self.device)
        if self._kff == 0.0:
            return delta

        for side in range(2):
            active_mask = self._ff_phase[:, side] >= 0
            if not active_mask.any():
                continue
            phase = torch.clamp(self._ff_phase[:, side].float(), min=0.0)
            envelope = self._ff_profile_envelope(phase)
            envelope = envelope * active_mask.float() * float(self._kff)
            delta[:, side, 0] = -float(self.ff_x_m) * envelope
            delta[:, side, 1] = float(self.ff_lift_m) * envelope
        return delta

    def ff_wheel_action_delta(self) -> torch.Tensor:
        """返回 CTBC 期间叠加到左右轮 action 的前向轮速前馈。"""
        delta = torch.zeros(self.num_envs, 2, device=self.device)
        if self._kff == 0.0 or self.ff_wheel_action == 0.0:
            return delta

        active = self._ff_phase >= 0
        if not active.any():
            return delta
        phase = torch.clamp(self._ff_phase.float(), min=0.0)
        envelope = self._ff_profile_envelope(phase) * active.float()
        env_envelope = envelope.max(dim=1).values * float(self._kff)
        delta[:, :] = float(self.ff_wheel_action) * env_envelope.unsqueeze(-1)
        return delta

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
        height_gain = torch.nan_to_num(height_gain, nan=0.0, posinf=0.0, neginf=0.0)
        radial_progress = torch.nan_to_num(radial_progress, nan=0.0, posinf=0.0, neginf=0.0)
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
        self._max_height_gain = torch.nan_to_num(
            torch.maximum(self._max_height_gain, clipped_height),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._max_radial_progress = torch.nan_to_num(
            torch.maximum(self._max_radial_progress, clipped_radial),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._progress_initialized[:] = True
        return height_delta, radial_delta

    def wheel_terrain_rise(self, terrain_z: torch.Tensor) -> torch.Tensor:
        """返回左右轮下方地形相对本 episode 初始地形的抬升量。"""
        terrain_z = torch.nan_to_num(terrain_z, nan=0.0, posinf=0.0, neginf=0.0)
        if terrain_z.ndim == 1:
            terrain_z = terrain_z.unsqueeze(-1)
        if terrain_z.shape[1] < 2:
            terrain_z = terrain_z.expand(-1, 2)
        terrain_z = terrain_z[:, :2]
        uninitialized = ~torch.isfinite(self._initial_wheel_terrain_z).all(dim=1)
        if torch.any(uninitialized):
            self._initial_wheel_terrain_z[uninitialized] = terrain_z[uninitialized].detach()
        return torch.nan_to_num(
            terrain_z - self._initial_wheel_terrain_z,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def record_wheel_supported_rise(
        self,
        supported_rise: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        """记录 episode 内左右轮真实支撑过的最大地形抬升量。"""
        supported_rise = torch.nan_to_num(supported_rise, nan=0.0, posinf=0.0, neginf=0.0)
        if supported_rise.ndim == 1:
            supported_rise = supported_rise.unsqueeze(-1)
        if supported_rise.shape[1] < 2:
            supported_rise = supported_rise.expand(-1, 2)
        supported_rise = torch.clamp(supported_rise[:, :2], min=0.0)
        self._max_wheel_supported_rise = torch.maximum(
            self._max_wheel_supported_rise,
            supported_rise,
        )
        both_supported_rise = torch.min(supported_rise, dim=1).values
        self._max_wheel_supported_both_rise = torch.maximum(
            self._max_wheel_supported_both_rise,
            both_supported_rise,
        )
        if step_index is not None:
            step_value = int(step_index)
            new_step = self._wheel_support_record_step != step_value
            both_supported = both_supported_rise > 1.0e-4
            updated_steps = torch.where(
                both_supported,
                self._wheel_supported_both_steps + 1,
                torch.zeros_like(self._wheel_supported_both_steps),
            )
            self._wheel_supported_both_steps = torch.where(
                new_step,
                updated_steps,
                self._wheel_supported_both_steps,
            )
            self._max_wheel_supported_both_steps = torch.maximum(
                self._max_wheel_supported_both_steps,
                self._wheel_supported_both_steps,
            )
            self._wheel_support_record_step[new_step] = step_value
        return supported_rise

    def max_wheel_supported_rise(self) -> torch.Tensor:
        """返回 episode 内左右轮真实支撑过的最大地形抬升量。"""
        return self._max_wheel_supported_rise.clone()

    def max_wheel_supported_both_rise(self) -> torch.Tensor:
        """返回 episode 内双轮同时真实支撑过的最大地形抬升量。"""
        return self._max_wheel_supported_both_rise.clone()

    def max_wheel_supported_both_duration(self) -> torch.Tensor:
        """返回 episode 内双轮真实支撑地形抬升的最长持续时间。"""
        return self._max_wheel_supported_both_steps.float() * float(self.control_dt)

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
        self._initial_wheel_terrain_z[env_ids] = float("nan")
        self._max_wheel_supported_rise[env_ids] = 0.0
        self._max_wheel_supported_both_rise[env_ids] = 0.0
        self._wheel_support_record_step[env_ids] = -1
        self._wheel_supported_both_steps[env_ids] = 0
        self._max_wheel_supported_both_steps[env_ids] = 0

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
        active_side = self._ff_phase >= 0
        return {
            "Stair/diag_ctbc_trigger_rate": self.contact_triggered().float().mean().item(),
            "Stair/diag_ctbc_both_side_active_rate": (
                active_side.all(dim=-1).float().mean().item()
            ),
            "Stair/diag_ctbc_one_side_active_rate": (
                (active_side.sum(dim=-1) == 1).float().mean().item()
            ),
            "Stair/diag_ctbc_complete_ff_cycles": (
                self._complete_ff_cycle_count.float().mean().item()
            ),
            "Stair/diag_riser_stall_rate": self.riser_stall_active(0.25).float().mean().item(),
            "Stair/diag_ctbc_kff": self._kff,
            "Stair/diag_ctbc_local_iter": float(self._iter),
            "Stair/diag_ctbc_ff_rise_steps": float(self.ff_rise_steps),
            "Stair/diag_ctbc_ff_hold_steps": float(self.ff_hold_steps),
            "Stair/diag_ctbc_ff_return_steps": float(self.ff_return_steps),
            "Stair/diag_ctbc_allow_bilateral_trigger": float(self.allow_bilateral_trigger),
            "Stair/diag_ctbc_ff_wheel_action": float(self.ff_wheel_action),
        }


__all__ = ["StairClimbState"]
