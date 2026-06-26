"""原生 MuJoCo sim2sim 的台阶 CTBC 前馈注入器。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from se3_shared import (
    JointGroup,
    output_leg_wheel_xz_np,
    output_to_policy_pos_np,
    policy_to_output_pos_np,
    wheel_xz_to_output_pos_np,
)

if TYPE_CHECKING:
    from .config import StairCtbcConfig
    from .robot import WheelLeggedRobot


class StairCtbcRuntime:
    """单环境 CTBC 状态机，保持与训练端 stair 任务的前馈语义一致。"""

    def __init__(self, cfg: StairCtbcConfig, *, control_dt: float) -> None:
        self.cfg = cfg
        self.control_dt = float(control_dt)
        self.contact_window = max(1, int(cfg.contact_window))
        self.force_threshold = float(cfg.force_threshold_n)
        self.ff_period_steps = max(1, round(float(cfg.ff_period_s) / self.control_dt))
        self.ff_rise_steps, self.ff_hold_steps, self.ff_return_steps = self._resolve_profile_steps(
            float(cfg.ff_rise_ratio),
            float(cfg.ff_hold_ratio),
        )
        self._contact_buf = np.zeros((self.contact_window, 2), dtype=np.float64)
        self._ff_phase = np.full(2, -1, dtype=np.int64)
        self._cooldown = np.zeros(2, dtype=np.int64)
        self._cooldown_steps = max(1, round(0.3 / self.control_dt))
        self._kff = 1.0
        self._local_iter = 0
        self._complete_ff_cycles = 0
        self._last_contact_score = np.zeros(2, dtype=np.float64)
        self._last_action_delta = np.zeros(6, dtype=np.float64)

    @property
    def kff(self) -> float:
        return float(self._kff)

    @property
    def action_delta(self) -> np.ndarray:
        return self._last_action_delta.copy()

    @property
    def active(self) -> np.ndarray:
        return self._ff_phase >= 0

    def reset(self) -> None:
        """清空 CTBC 接触窗口、相位和上一帧注入量。"""

        self._contact_buf.fill(0.0)
        self._ff_phase.fill(-1)
        self._cooldown.fill(0)
        self._complete_ff_cycles = 0
        self._last_contact_score.fill(0.0)
        self._last_action_delta.fill(0.0)

    def update(self, wheel_contact_score: np.ndarray, *, iteration: int | None) -> None:
        """根据当前轮-台阶立面接触更新 CTBC 触发相位。"""

        self._update_iter(iteration)
        score = np.nan_to_num(
            np.asarray(wheel_contact_score, dtype=np.float64).reshape(2),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._last_contact_score[:] = score
        self._contact_buf[1:] = self._contact_buf[:-1].copy()
        self._contact_buf[0] = score

        stable = np.all(self._contact_buf > self.force_threshold, axis=0)
        self._cooldown[self._cooldown > 0] -= 1
        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0)
        trigger_candidates = stable & can_trigger & (not np.any(self._ff_phase >= 0))
        newly_triggered = np.zeros(2, dtype=bool)
        if np.any(trigger_candidates):
            if bool(self.cfg.allow_bilateral_trigger):
                newly_triggered[:] = can_trigger & np.any(trigger_candidates)
            else:
                side_score = np.where(trigger_candidates, score, -np.inf)
                newly_triggered[int(np.argmax(side_score))] = True
        self._ff_phase[newly_triggered] = 0

        active = self._ff_phase >= 0
        self._ff_phase[active] += 1
        finished = self._ff_phase >= self.ff_period_steps
        if np.any(finished):
            self._cooldown[finished] = self._cooldown_steps
            self._ff_phase[finished] = -1
            self._complete_ff_cycles += 1

    def obs(self) -> np.ndarray:
        """返回训练端 ctbc_phase_obs 对齐的 3 维观测槽。"""

        out = np.zeros(3, dtype=np.float32)
        active = self._ff_phase >= 0
        phase = np.where(
            active,
            np.clip(
                (self._ff_phase.astype(np.float64) + 1.0) / float(self.ff_period_steps), 0.0, 1.0
            ),
            0.0,
        )
        scale = float(self.cfg.obs_scale)
        out[:2] = (phase[:2] * scale).astype(np.float32)
        out[2] = scale if np.any(active) else 0.0
        return out

    def apply(self, robot: WheelLeggedRobot, action: np.ndarray) -> np.ndarray:
        """把 CTBC 前馈叠加到最终执行 action，返回裁剪后的执行 action。"""

        clipped = np.asarray(action, dtype=np.float64).reshape(6).copy()
        self._last_action_delta.fill(0.0)
        if self._kff == 0.0 or not np.any(self._ff_phase >= 0):
            return clipped

        desired = clipped.copy()
        desired[:4] = self._apply_leg_delta(robot, desired[:4])
        desired[4:6] = self._apply_wheel_delta(desired[4:6])
        if robot.cfg.action_clip is not None:
            clip = float(robot.cfg.action_clip)
            desired = np.clip(desired, -clip, clip)
        self._last_action_delta[:] = desired - clipped
        return desired

    def telemetry(self) -> dict[str, object]:
        """返回 Viser/Rerun 日志用的 CTBC 诊断量。"""

        active = self._ff_phase >= 0
        return {
            "ctbc_enabled": True,
            "ctbc_trigger": float(np.any(active)),
            "ctbc_left_active": float(active[0]),
            "ctbc_right_active": float(active[1]),
            "ctbc_phase_left": float(self.obs()[0]),
            "ctbc_phase_right": float(self.obs()[1]),
            "ctbc_kff": float(self._kff),
            "ctbc_local_iter": int(self._local_iter),
            "ctbc_contact_left": float(self._last_contact_score[0]),
            "ctbc_contact_right": float(self._last_contact_score[1]),
            "ctbc_complete_ff_cycles": int(self._complete_ff_cycles),
            "ctbc_action_delta": self._last_action_delta.copy().tolist(),
        }

    def _update_iter(self, iteration: int | None) -> None:
        local_iter = self.cfg.fixed_iter if self.cfg.fixed_iter is not None else iteration
        self._local_iter = max(0, int(0 if local_iter is None else local_iter))
        if self._local_iter < int(self.cfg.ann_start_iter):
            self._kff = 1.0
            return
        if self._local_iter >= int(self.cfg.ann_end_iter):
            self._kff = 0.0
            return
        span = max(1, int(self.cfg.ann_end_iter) - int(self.cfg.ann_start_iter))
        self._kff = max(0.0, 1.0 - (self._local_iter - int(self.cfg.ann_start_iter)) / span)

    def _resolve_profile_steps(self, rise_ratio: float, hold_ratio: float) -> tuple[int, int, int]:
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

    def _ff_profile_envelope(self, phase: np.ndarray) -> np.ndarray:
        phase = np.clip(np.asarray(phase, dtype=np.float64), 0.0, float(self.ff_period_steps))
        rise_steps = float(self.ff_rise_steps)
        hold_end = float(self.ff_rise_steps + self.ff_hold_steps)
        return_steps = float(self.ff_return_steps)
        rise_t = np.clip(phase / rise_steps, 0.0, 1.0)
        rise = 0.5 * (1.0 - np.cos(math.pi * rise_t))
        return_t = np.clip((phase - hold_end) / return_steps, 0.0, 1.0)
        returning = 0.5 * (1.0 + np.cos(math.pi * return_t))
        envelope = np.where(phase < rise_steps, rise, 1.0)
        envelope = np.where(phase < hold_end, envelope, returning)
        return np.clip(envelope, 0.0, 1.0)

    def _ff_wheel_delta_xz(self) -> np.ndarray:
        delta = np.zeros((2, 2), dtype=np.float64)
        if self._kff == 0.0:
            return delta
        active = self._ff_phase >= 0
        if not np.any(active):
            return delta
        phase = np.maximum(self._ff_phase.astype(np.float64), 0.0)
        envelope = self._ff_profile_envelope(phase) * active.astype(np.float64) * float(self._kff)
        delta[:, 0] = -float(self.cfg.ff_x_m) * envelope
        delta[:, 1] = float(self.cfg.ff_lift_m) * envelope
        return delta

    def _ff_wheel_action_delta(self) -> np.ndarray:
        if self._kff == 0.0 or float(self.cfg.ff_wheel_action) == 0.0:
            return np.zeros(2, dtype=np.float64)
        active = self._ff_phase >= 0
        if not np.any(active):
            return np.zeros(2, dtype=np.float64)
        phase = np.maximum(self._ff_phase.astype(np.float64), 0.0)
        envelope = self._ff_profile_envelope(phase) * active.astype(np.float64)
        env_envelope = float(np.max(envelope)) * float(self._kff)
        return np.full(2, float(self.cfg.ff_wheel_action) * env_envelope, dtype=np.float64)

    def _apply_leg_delta(self, robot: WheelLeggedRobot, leg_action: np.ndarray) -> np.ndarray:
        wheel_delta_xz = self._ff_wheel_delta_xz()
        if not np.any(np.abs(wheel_delta_xz) > 0.0):
            return leg_action
        if not robot.active_rod_action_semantics:
            return leg_action

        policy_default = robot.action_decoder.policy_default(
            command_height=float(robot.command[4]),
            fallback_default=robot.default_dof_pos[JointGroup.CTRL_LEGS],
        )
        current_policy_pos = robot._current_policy_leg_pos()
        current_policy = robot.action_decoder.leg_target(
            leg_action,
            policy_default,
            current_policy_pos=current_policy_pos,
        )
        current_output = policy_to_output_pos_np(current_policy)
        current_wheel_xz = output_leg_wheel_xz_np(current_output)
        desired_wheel_xz = current_wheel_xz + wheel_delta_xz
        desired_output = current_output.copy()
        cartesian_desired_output = wheel_xz_to_output_pos_np(desired_wheel_xz)
        active_side = np.any(np.abs(wheel_delta_xz) > 0.0, axis=1)
        for side_idx in range(2):
            if active_side[side_idx]:
                sl = slice(2 * side_idx, 2 * side_idx + 2)
                desired_output[sl] = cartesian_desired_output[sl]
        desired_policy = output_to_policy_pos_np(desired_output)
        action_reference = policy_default.copy()
        if robot.cfg.leg_action_reference == "front_current":
            action_reference[[0, 2]] = current_policy_pos[[0, 2]]

        desired_action = np.asarray(leg_action, dtype=np.float64).copy()
        leg_scale = robot.action_scale[JointGroup.LEG_ACTUATORS]
        coeffs = robot.action_decoder.active_rod_angle_coeffs
        active_mid = float(robot.action_decoder.active_rod_angle_mid)
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            if not active_side[side_idx]:
                continue
            front_coef, back_coef = coeffs[side_idx]
            desired_action[front_idx] = (
                desired_policy[front_idx] - action_reference[front_idx]
            ) / leg_scale[front_idx]
            if robot.cfg.height_conditioned_action_default:
                active_default = (
                    front_coef * action_reference[front_idx]
                    + back_coef * action_reference[back_idx]
                )
            else:
                active_default = active_mid
            active_desired = (
                front_coef * desired_policy[front_idx] + back_coef * desired_policy[back_idx]
            )
            desired_action[back_idx] = (active_desired - active_default) / leg_scale[back_idx]

        return np.nan_to_num(desired_action, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_wheel_delta(self, wheel_action: np.ndarray) -> np.ndarray:
        return np.asarray(wheel_action, dtype=np.float64).reshape(2) + self._ff_wheel_action_delta()


__all__ = ["StairCtbcRuntime"]
