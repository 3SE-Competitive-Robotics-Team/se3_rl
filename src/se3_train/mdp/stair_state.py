"""CTBC 台阶前馈状态机。"""

from __future__ import annotations

import math

import torch

from se3_shared import (
    REFERENCE_CTBC_FF_AMPLITUDE,
    REFERENCE_CTBC_FF_PERIOD_S,
    REFERENCE_CTBC_FORCE_THRESHOLD_N,
    REFERENCE_CTBC_HIP_RATIO,
    REFERENCE_CTBC_KNEE_RATIO,
    REFERENCE_CTBC_LEG_SCALE,
    RobotConfig,
    policy_default_from_height_torch,
    reference_ctbc_bias_to_current_action_torch,
)

_SHARED_ROBOT = RobotConfig()
_DEFAULT_LEG_SCALES = torch.tensor(
    _SHARED_ROBOT.action_scale[:4],
    dtype=torch.float32,
)


class StairClimbState:
    """每个 env 的 CTBC 接触触发前馈控制器。"""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        contact_window: int = 3,
        force_threshold_n: float = REFERENCE_CTBC_FORCE_THRESHOLD_N,
        ff_amplitude_rad: float = REFERENCE_CTBC_FF_AMPLITUDE,
        ff_period_s: float = REFERENCE_CTBC_FF_PERIOD_S,
        control_dt: float = 0.02,
        cooldown_s: float = 0.4,
        reference_leg_scale: float = REFERENCE_CTBC_LEG_SCALE,
        hip_ratio: float = REFERENCE_CTBC_HIP_RATIO,
        knee_ratio: float = REFERENCE_CTBC_KNEE_RATIO,
        ann_start_iter: int = 0,
        ann_end_iter: int = 1500,
        phantom_trigger_iter: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.contact_window = int(contact_window)
        self.force_threshold = float(force_threshold_n)
        self.ff_amplitude = float(ff_amplitude_rad)
        self.control_dt = float(control_dt)
        self.ff_period_steps = max(1, round(float(ff_period_s) / self.control_dt))
        self.cooldown_steps = max(1, round(float(cooldown_s) / self.control_dt))
        self.reference_leg_scale = float(reference_leg_scale)
        self.hip_ratio = float(hip_ratio)
        self.knee_ratio = float(knee_ratio)
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
        self._last_pulse = torch.zeros(self.num_envs, 2, device=self.device)
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

    def step(
        self,
        wheel_contact_xy: torch.Tensor,
        trigger_mask: torch.Tensor | None = None,
    ) -> None:
        """按轮子水平接触力推进触发状态。"""
        force = wheel_contact_xy.to(device=self.device).reshape(self.num_envs, 2)
        if trigger_mask is None:
            allowed = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            allowed = trigger_mask.to(device=self.device, dtype=torch.bool).reshape(self.num_envs)
            self._ff_phase[~allowed] = -1
            self._cooldown[~allowed] = 0
            force = force * allowed.unsqueeze(-1)
        self._last_contact_xy = force
        self._contact_buf[1:] = self._contact_buf[:-1].clone()
        self._contact_buf[0] = force

        self._stable = (self._contact_buf > self.force_threshold).all(dim=0)
        self._cooldown[self._cooldown > 0] -= 1

        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0) & allowed.unsqueeze(-1)
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

    def ff_bias(
        self,
        command_height: torch.Tensor | None = None,
        *,
        current_leg_action: torch.Tensor | None = None,
        policy_default: torch.Tensor | None = None,
        leg_action_scales: torch.Tensor | None = None,
        height_conditioned_action_default: bool = True,
    ) -> torch.Tensor:
        """返回当前 action 语义下的 6D raw action bias。"""
        bias = torch.zeros(self.num_envs, 6, device=self.device)
        side_pulse = self._reference_side_pulse()
        self._last_pulse = side_pulse
        if self._kff <= 0.0 or not bool((side_pulse > 0.0).any()):
            self._last_bias = bias
            return bias

        if current_leg_action is None:
            current_leg_action = torch.zeros(self.num_envs, 4, device=self.device)
        else:
            current_leg_action = current_leg_action.to(device=self.device).reshape(self.num_envs, 4)

        if policy_default is None:
            policy_default = self._fallback_policy_default(command_height)
        else:
            policy_default = policy_default.to(device=self.device).reshape(self.num_envs, 4)

        if leg_action_scales is None:
            leg_action_scales = _DEFAULT_LEG_SCALES.to(device=self.device)
        else:
            leg_action_scales = leg_action_scales.to(device=self.device)

        bias[:, :4] = reference_ctbc_bias_to_current_action_torch(
            current_leg_action,
            policy_default,
            side_pulse,
            leg_scales=leg_action_scales,
            height_conditioned_action_default=height_conditioned_action_default,
            reference_leg_scale=self.reference_leg_scale,
            hip_ratio=self.hip_ratio,
            knee_ratio=self.knee_ratio,
        )
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
            "Stair/ctbc_reference_pulse_mean": self._last_pulse.mean().item(),
            "Stair/ctbc_reference_pulse_max": self._last_pulse.max().item(),
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
        self._last_pulse[ids] = 0.0
        self._last_contact_xy[ids] = 0.0

    def _reference_side_pulse(self) -> torch.Tensor:
        """返回参考代码的 amplitude * (1 - cos) 侧向脉冲。"""
        pulse = torch.zeros(self.num_envs, 2, device=self.device)
        active = self._ff_phase >= 0
        if not bool(active.any()):
            return pulse
        phase = torch.clamp(self._ff_phase, min=0).to(dtype=torch.float32)
        t = phase / float(self.ff_period_steps)
        pulse = self.ff_amplitude * (1.0 - torch.cos(2.0 * math.pi * t))
        pulse = pulse * active.float() * float(self._kff)
        return pulse

    def _fallback_policy_default(self, command_height: torch.Tensor | None) -> torch.Tensor:
        """缺少 action term 上下文时，用 height command 或共享默认姿态兜底。"""
        if command_height is not None:
            return policy_default_from_height_torch(
                command_height.to(device=self.device).reshape(self.num_envs),
                _SHARED_ROBOT,
            )
        default = torch.as_tensor(
            _SHARED_ROBOT.default_dof_pos[:4],
            device=self.device,
            dtype=torch.float32,
        )
        return default.reshape(1, 4).repeat(self.num_envs, 1)
