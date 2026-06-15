"""CTBC 台阶爬升状态管理。"""

from __future__ import annotations

import os
import threading

import torch

_HIP_LEFT = 0
_KNEE_LEFT = 1
_HIP_RIGHT = 2
_KNEE_RIGHT = 3

_DEFAULT_HIP_FEEDFORWARD_RATIO: float = 1.5
_DEFAULT_KNEE_FEEDFORWARD_RATIO: float = 1.0


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
        ff_period_s: float = 0.3,
        ff_attack_s: float = 0.03,
        ff_hold_s: float = 0.08,
        hip_feedforward_ratio: float = _DEFAULT_HIP_FEEDFORWARD_RATIO,
        knee_feedforward_ratio: float = _DEFAULT_KNEE_FEEDFORWARD_RATIO,
        control_dt: float = 0.01,
        ff_start_iter: int = 0,
        ann_start_iter: int = 100,
        ann_end_iter: int = 300,
        phantom_trigger_iter: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.contact_window = contact_window
        self.force_threshold = force_threshold_n
        self.control_dt = float(control_dt)
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        self._ff_config_lock = threading.RLock()
        self.ff_amplitude = 0.0
        self.ff_period_steps = 0
        self.ff_period_s = 0.0
        self.ff_attack_s = 0.0
        self.ff_hold_s = 0.0
        self.ff_decay_s = 0.0
        self.hip_feedforward_ratio = _DEFAULT_HIP_FEEDFORWARD_RATIO
        self.knee_feedforward_ratio = _DEFAULT_KNEE_FEEDFORWARD_RATIO
        self.configure_feedforward(
            ff_amplitude_rad=ff_amplitude_rad,
            ff_period_s=ff_period_s,
            ff_attack_s=ff_attack_s,
            ff_hold_s=ff_hold_s,
            hip_feedforward_ratio=hip_feedforward_ratio,
            knee_feedforward_ratio=knee_feedforward_ratio,
        )
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

        with self._ff_config_lock:
            period_steps = self.ff_period_steps
        finished = self._ff_phase >= period_steps
        self._cooldown[finished] = self._cooldown_steps
        self._ff_phase[finished] = -1
        self._complete_ff_cycle_count[finished.any(dim=-1)] += 1

    def ff_bias(self) -> torch.Tensor:
        """返回旧输出关节 action 语义下的 6D CTBC bias。"""
        bias = torch.zeros(self.num_envs, 6, device=self.device)
        if self._kff == 0.0:
            return bias

        with self._ff_config_lock:
            ff_amplitude = self.ff_amplitude
            hip_ratio = self.hip_feedforward_ratio
            knee_ratio = self.knee_feedforward_ratio
        for side, (hip_idx, knee_idx) in enumerate(
            [(_HIP_LEFT, _KNEE_LEFT), (_HIP_RIGHT, _KNEE_RIGHT)]
        ):
            active_mask = self._ff_phase[:, side] >= 0
            if not active_mask.any():
                continue
            profile = self._ff_profile(self._ff_phase[:, side])
            lift = (2.0 * ff_amplitude) * profile * active_mask.float()
            bias[:, hip_idx] = -lift * hip_ratio * self._kff
            bias[:, knee_idx] = lift * knee_ratio * self._kff
        return bias

    def _ff_profile(self, phase_steps: torch.Tensor) -> torch.Tensor:
        """Single-shot CTBC envelope: fast attack, short hold, smooth decay."""
        with self._ff_config_lock:
            attack_end = self.ff_attack_s
            hold_end = self.ff_attack_s + self.ff_hold_s
            decay_s = self.ff_decay_s
        elapsed = torch.clamp(phase_steps.float(), min=0.0) * float(self.control_dt)

        attack_ratio = torch.clamp(elapsed / max(attack_end, 1.0e-6), 0.0, 1.0)
        decay_ratio = torch.clamp((elapsed - hold_end) / max(decay_s, 1.0e-6), 0.0, 1.0)
        attack_profile = _smoothstep(attack_ratio)
        decay_profile = 1.0 - _smoothstep(decay_ratio)
        return torch.where(
            elapsed < attack_end,
            attack_profile,
            torch.where(elapsed < hold_end, torch.ones_like(elapsed), decay_profile),
        )

    def configure_feedforward(
        self,
        *,
        ff_amplitude_rad: float,
        ff_period_s: float,
        ff_attack_s: float,
        ff_hold_s: float,
        hip_feedforward_ratio: float | None = None,
        knee_feedforward_ratio: float | None = None,
    ) -> dict[str, float | int]:
        """运行时更新 CTBC 脉冲，并对齐到整数控制步。"""
        period_steps = max(3, round(float(ff_period_s) / self.control_dt))
        attack_steps = min(
            max(1, round(float(ff_attack_s) / self.control_dt)),
            period_steps - 1,
        )
        hold_steps = min(
            max(0, round(float(ff_hold_s) / self.control_dt)),
            period_steps - attack_steps - 1,
        )
        decay_steps = period_steps - attack_steps - hold_steps

        with self._ff_config_lock:
            self.ff_amplitude = max(0.0, float(ff_amplitude_rad))
            self.ff_period_steps = period_steps
            self.ff_period_s = period_steps * self.control_dt
            self.ff_attack_s = attack_steps * self.control_dt
            self.ff_hold_s = hold_steps * self.control_dt
            self.ff_decay_s = decay_steps * self.control_dt
            if hip_feedforward_ratio is not None:
                self.hip_feedforward_ratio = max(0.0, float(hip_feedforward_ratio))
            if knee_feedforward_ratio is not None:
                self.knee_feedforward_ratio = max(0.0, float(knee_feedforward_ratio))
            return self.feedforward_config()

    def feedforward_config(self) -> dict[str, float | int]:
        """返回当前运行时 CTBC 脉冲配置。"""
        with self._ff_config_lock:
            return {
                "amplitude_rad": self.ff_amplitude,
                "period_steps": self.ff_period_steps,
                "period_s": self.ff_period_s,
                "attack_s": self.ff_attack_s,
                "hold_s": self.ff_hold_s,
                "hold_end_s": self.ff_attack_s + self.ff_hold_s,
                "decay_s": self.ff_decay_s,
                "hip_ratio": self.hip_feedforward_ratio,
                "knee_ratio": self.knee_feedforward_ratio,
            }

    def trigger_feedforward(self, env_idx: int, side: str = "both") -> None:
        """手动触发指定 viewer env 的左、右或双侧 CTBC 脉冲。"""
        env_idx = int(env_idx)
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx out of range: {env_idx}")
        side_indices = {
            "left": (0,),
            "right": (1,),
            "both": (0, 1),
        }
        if side not in side_indices:
            raise ValueError(f"unknown CTBC side: {side}")
        for side_idx in side_indices[side]:
            self._cooldown[env_idx, side_idx] = 0
            self._ff_phase[env_idx, side_idx] = 0

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


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


__all__ = ["StairClimbState"]
