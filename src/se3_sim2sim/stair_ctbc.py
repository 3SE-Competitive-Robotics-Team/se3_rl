"""原生 MuJoCo sim2sim 的台阶 CTBC 前馈注入器。"""

from __future__ import annotations

import json
import math
from pathlib import Path
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

_CTBC_SOURCE_OUTPUT_LEG_SCALE = 0.25
_CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN = np.asarray((-1.0, -1.0, 1.0, 1.0), dtype=np.float64)
_BASE_BODY_NAME = "base_link"
_WHEEL_BODY_NAMES = ("l_wheel_Link", "r_wheel_Link")
_HIP_OFFSETS_BODY = np.asarray(
    ((0.0, 0.16885, 0.0), (0.0, -0.16885, 0.0)),
    dtype=np.float64,
)
_HIP_FEEDFORWARD_RATIO = 1.2
_KNEE_FEEDFORWARD_RATIO = 2.0


def _polar_to_xz_np(length_m: np.ndarray, swing_rad: np.ndarray) -> np.ndarray:
    length = np.asarray(length_m, dtype=np.float64)
    swing = np.asarray(swing_rad, dtype=np.float64)
    return np.stack((length * np.sin(swing), -length * np.cos(swing)), axis=-1)


def _profile_coordinate_mode(payload: dict[str, object]) -> str:
    metadata = payload.get("metadata")
    mode = metadata.get("coordinate_mode") if isinstance(metadata, dict) else None
    if (
        mode is None
        and isinstance(metadata, dict)
        and metadata.get("type") == "training_duration_polar"
    ):
        mode = "body_polar"
    return "body_polar" if mode == "body_polar" else "body_cartesian"


class _CtbcProfile:
    """从 JSON 加载的分段线性 CTBC 前馈轨迹。"""

    def __init__(
        self,
        *,
        path: Path,
        times_s: np.ndarray,
        values: np.ndarray,
        period_s: float,
        coordinate_mode: str,
    ) -> None:
        self.path = path
        self.times_s = times_s
        self.values = values
        self.period_s = float(period_s)
        self.coordinate_mode = coordinate_mode

    @classmethod
    def load(cls, path: Path | str) -> _CtbcProfile:
        profile_path = Path(path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        points = payload.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"CTBC profile {profile_path} must contain at least two points")

        coordinate_mode = _profile_coordinate_mode(payload)
        rows: list[tuple[float, float, float, float, float]] = []
        for idx, point in enumerate(points):
            if not isinstance(point, dict):
                raise ValueError(f"CTBC profile point {idx} must be an object")
            t = cls._number(point, "t", "time_s", required=True)
            if coordinate_mode == "body_polar":
                x = cls._number(
                    point,
                    "leg_length_m",
                    ("length_m", "r_m", "radius_m"),
                    default=0.24,
                )
                raw_swing = point.get("swing_angle_rad")
                if raw_swing is None:
                    for key in ("theta_rad", "angle_rad"):
                        if key in point:
                            raw_swing = point[key]
                            break
                if raw_swing is None:
                    swing_deg = cls._number(
                        point,
                        "swing_angle_deg",
                        ("theta_deg", "angle_deg"),
                        default=10.0,
                    )
                    swing_rad = math.radians(swing_deg)
                else:
                    try:
                        swing_rad = float(raw_swing)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "CTBC profile field 'swing_angle_rad' must be numeric"
                        ) from exc
                    if not math.isfinite(swing_rad):
                        raise ValueError("CTBC profile field 'swing_angle_rad' must be finite")
                z = swing_rad
            else:
                x = cls._number(point, "body_x_m", ("x_m", "x"), default=0.0)
                z = cls._number(point, "body_z_m", ("z_m", "z"), default=0.0)
            amp = cls._number(point, "amp", "amp_scale", default=0.0)
            wheel_action = cls._number(
                point,
                "wheel_action",
                "wheel_action_delta",
                default=0.0,
            )
            if t < 0.0:
                raise ValueError(f"CTBC profile point {idx} has negative time {t}")
            rows.append((t, x, z, amp, wheel_action))

        rows.sort(key=lambda row: row[0])
        times_s = np.asarray([row[0] for row in rows], dtype=np.float64)
        if np.any(np.diff(times_s) <= 0.0):
            raise ValueError(f"CTBC profile {profile_path} times must be strictly increasing")
        values = np.asarray([row[1:] for row in rows], dtype=np.float64)
        if not np.isfinite(times_s).all() or not np.isfinite(values).all():
            raise ValueError(f"CTBC profile {profile_path} contains non-finite values")

        period_s = float(payload.get("period_s", times_s[-1]))
        if not math.isfinite(period_s) or period_s <= 0.0:
            raise ValueError(f"CTBC profile {profile_path} has invalid period_s={period_s!r}")
        if period_s < float(times_s[-1]):
            period_s = float(times_s[-1])
        return cls(
            path=profile_path,
            times_s=times_s,
            values=values,
            period_s=period_s,
            coordinate_mode=coordinate_mode,
        )

    @staticmethod
    def _number(
        point: dict[str, object],
        primary: str,
        alias: str | tuple[str, ...],
        *,
        default: float | None = None,
        required: bool = False,
    ) -> float:
        aliases = (alias,) if isinstance(alias, str) else alias
        raw = point.get(primary)
        if raw is None:
            for key in aliases:
                if key in point:
                    raw = point[key]
                    break
        if raw is None:
            raw = default
        if raw is None and required:
            raise ValueError(f"missing required CTBC profile field {primary!r}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CTBC profile field {primary!r} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"CTBC profile field {primary!r} must be finite")
        return value

    def sample(self, phase_steps: np.ndarray, *, control_dt: float) -> np.ndarray:
        t = np.clip(
            np.asarray(phase_steps, dtype=np.float64) * float(control_dt), 0.0, self.period_s
        )
        out = np.zeros((t.shape[0], self.values.shape[1]), dtype=np.float64)
        for col in range(self.values.shape[1]):
            out[:, col] = np.interp(t, self.times_s, self.values[:, col])
        return out


class StairCtbcRuntime:
    """单环境 CTBC 状态机，保持与训练端 stair 任务的前馈语义一致。"""

    def __init__(self, cfg: StairCtbcConfig, *, control_dt: float) -> None:
        self.cfg = cfg
        self.control_dt = float(control_dt)
        self._profile = (
            _CtbcProfile.load(cfg.profile_path) if cfg.profile_path is not None else None
        )
        self.trigger_mode = str(cfg.trigger_mode)
        self.contact_window = max(1, int(cfg.contact_window))
        self.force_threshold = float(cfg.force_threshold_n)
        self.pitch_window = max(1, int(cfg.pitch_window))
        self.pitch_threshold_rad = float(cfg.pitch_threshold_rad)
        duration_s = (
            self._profile.period_s if self._profile is not None else float(cfg.ff_duration_s)
        )
        self.ff_duration_steps = max(1, round(float(duration_s) / self.control_dt))
        self.ff_period_steps = self.ff_duration_steps
        self.ff_rise_steps = self.ff_duration_steps
        self.ff_hold_steps = 0
        self.ff_return_steps = 0
        self._contact_buf = np.zeros((self.contact_window, 2), dtype=np.float64)
        self._pitch_buf = np.zeros(self.pitch_window, dtype=np.float64)
        self._ff_phase = np.full(2, -1, dtype=np.int64)
        self._cooldown = np.zeros(2, dtype=np.int64)
        self._cooldown_steps = max(1, round(0.3 / self.control_dt))
        self._kff = 1.0
        self._local_iter = 0
        self._complete_ff_cycles = 0
        self._last_contact_score = np.zeros(2, dtype=np.float64)
        self._last_trigger_score = np.zeros(2, dtype=np.float64)
        self._last_stable = np.zeros(2, dtype=bool)
        self._last_pitch_rad = 0.0
        self._last_pitch_abs_deg = 0.0
        self._last_action_delta = np.zeros(6, dtype=np.float64)
        self._ff_anchor_wheel_xz = np.full((2, 2), np.nan, dtype=np.float64)

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
        self._pitch_buf.fill(0.0)
        self._ff_phase.fill(-1)
        self._cooldown.fill(0)
        self._complete_ff_cycles = 0
        self._last_contact_score.fill(0.0)
        self._last_trigger_score.fill(0.0)
        self._last_stable.fill(False)
        self._last_pitch_rad = 0.0
        self._last_pitch_abs_deg = 0.0
        self._last_action_delta.fill(0.0)
        self._ff_anchor_wheel_xz.fill(np.nan)

    def update(
        self,
        wheel_contact_score: np.ndarray,
        *,
        pitch_rad: float | None = None,
        iteration: int | None,
    ) -> None:
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

        pitch_value = 0.0 if pitch_rad is None else float(pitch_rad)
        if not math.isfinite(pitch_value):
            pitch_value = 0.0
        self._last_pitch_rad = pitch_value
        pitch_abs_rad = abs(pitch_value)
        self._last_pitch_abs_deg = math.degrees(pitch_abs_rad)
        self._pitch_buf[1:] = self._pitch_buf[:-1].copy()
        self._pitch_buf[0] = pitch_abs_rad

        if self.trigger_mode == "pitch":
            stable_scalar = bool(np.all(self._pitch_buf > self.pitch_threshold_rad))
            stable = np.full(2, stable_scalar, dtype=bool)
            trigger_score = np.full(2, self._last_pitch_abs_deg, dtype=np.float64)
        else:
            stable = np.all(self._contact_buf > self.force_threshold, axis=0)
            trigger_score = score
        self._last_stable[:] = stable
        self._last_trigger_score[:] = trigger_score
        self._cooldown[self._cooldown > 0] -= 1
        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0)
        trigger_candidates = stable & can_trigger
        newly_triggered = np.zeros(2, dtype=bool)
        if np.any(trigger_candidates):
            if self.trigger_mode == "pitch" or bool(self.cfg.allow_bilateral_trigger):
                newly_triggered[:] = can_trigger & np.any(trigger_candidates)
            else:
                side_score = np.where(trigger_candidates, trigger_score, -np.inf)
                newly_triggered[int(np.argmax(side_score))] = True
        self._ff_phase[newly_triggered] = 0

        active = self._ff_phase >= 0
        self._ff_phase[active] += 1
        finished = self._ff_phase >= self.ff_duration_steps
        if np.any(finished):
            self._cooldown[finished] = self._cooldown_steps
            self._ff_phase[finished] = -1
            self._ff_anchor_wheel_xz[finished] = np.nan
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
        desired[:4] = self._apply_output_bias_delta(robot, desired[:4])
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
            "ctbc_trigger_mode": str(self.trigger_mode),
            "ctbc_trigger_mode_pitch": float(self.trigger_mode == "pitch"),
            "ctbc_trigger_score_left": float(self._last_trigger_score[0]),
            "ctbc_trigger_score_right": float(self._last_trigger_score[1]),
            "ctbc_stable_left": float(self._last_stable[0]),
            "ctbc_stable_right": float(self._last_stable[1]),
            "ctbc_contact_left": float(self._last_contact_score[0]),
            "ctbc_contact_right": float(self._last_contact_score[1]),
            "ctbc_force_threshold_n": float(self.force_threshold),
            "ctbc_contact_window": int(self.contact_window),
            "ctbc_pitch_rad": float(self._last_pitch_rad),
            "ctbc_pitch_abs_deg": float(self._last_pitch_abs_deg),
            "ctbc_pitch_threshold_deg": float(math.degrees(self.pitch_threshold_rad)),
            "ctbc_pitch_window": int(self.pitch_window),
            "ctbc_complete_ff_cycles": int(self._complete_ff_cycles),
            "ctbc_action_delta": self._last_action_delta.copy().tolist(),
            "ctbc_profile_path": (None if self._profile is None else str(self._profile.path)),
            "ctbc_cartesian_frame": "body",
            "ctbc_coordinate_mode": str(self.cfg.coordinate_mode),
            "ctbc_duration_s": float(self.ff_duration_steps * self.control_dt),
            "ctbc_duration_steps": int(self.ff_duration_steps),
            "ctbc_leg_length_m": float(self.cfg.leg_length_m),
            "ctbc_swing_angle_rad": float(self.cfg.swing_angle_rad),
            "ctbc_swing_angle_deg": float(math.degrees(float(self.cfg.swing_angle_rad))),
            "ctbc_body_x_m_legacy": float(self.cfg.body_x_m),
            "ctbc_body_z_m_legacy": float(self.cfg.body_z_m),
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

    def _ff_profile_envelope(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=np.float64)
        return np.where(phase >= 0.0, 1.0, 0.0)

    def _ff_output_bias(self) -> np.ndarray:
        bias = np.zeros(4, dtype=np.float64)
        if self._kff == 0.0:
            return bias
        active = self._ff_phase >= 0
        if not np.any(active):
            return bias
        phase = np.maximum(self._ff_phase.astype(np.float64), 0.0)
        if self._profile is None:
            envelope = self._ff_profile_envelope(phase) * active.astype(np.float64)
        else:
            envelope = self._profile.sample(phase, control_dt=self.control_dt)[:, 2]
            envelope = envelope * active.astype(np.float64)
        feedforward = 2.0 * float(self.cfg.ff_amplitude_rad) * envelope * float(self._kff)
        for side_idx, (hip_idx, knee_idx) in enumerate(((0, 1), (2, 3))):
            if not active[side_idx]:
                continue
            bias[hip_idx] = -feedforward[side_idx] * _HIP_FEEDFORWARD_RATIO
            bias[knee_idx] = feedforward[side_idx] * _KNEE_FEEDFORWARD_RATIO
        return bias

    def _ff_wheel_target_delta_xz(self, anchor_wheel_xz: np.ndarray) -> np.ndarray:
        delta = np.zeros((2, 2), dtype=np.float64)
        if self._kff == 0.0:
            return delta
        active = self._ff_phase >= 0
        if not np.any(active):
            return delta
        phase = np.maximum(self._ff_phase.astype(np.float64), 0.0)
        if self._profile is None:
            envelope = (
                self._ff_profile_envelope(phase) * active.astype(np.float64) * float(self._kff)
            )
            if self.cfg.coordinate_mode == "body_polar":
                length = np.full(2, float(self.cfg.leg_length_m), dtype=np.float64)
                swing = np.full(2, float(self.cfg.swing_angle_rad), dtype=np.float64)
                target = _polar_to_xz_np(length, swing)
                delta = (target - anchor_wheel_xz) * envelope[:, None]
            else:
                delta[:, 0] = -float(self.cfg.body_x_m) * envelope
                delta[:, 1] = float(self.cfg.body_z_m) * envelope
        else:
            samples = self._profile.sample(phase, control_dt=self.control_dt)
            envelope = active.astype(np.float64) * float(self._kff)
            if self._profile.coordinate_mode == "body_polar":
                target = _polar_to_xz_np(samples[:, 0], samples[:, 1])
                delta = (target - anchor_wheel_xz) * envelope[:, None]
            else:
                delta[:, 0] = samples[:, 0] * envelope
                delta[:, 1] = samples[:, 1] * envelope
        return delta

    def _output_bias_wheel_delta_xz(
        self,
        robot: WheelLeggedRobot,
        output_bias: np.ndarray,
    ) -> np.ndarray:
        delta = np.zeros((2, 2), dtype=np.float64)
        has_output_bias = np.abs(output_bias.reshape(2, 2)).max(axis=1) > 0.0
        if np.any(has_output_bias):
            output_delta = (
                output_bias * _CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN * _CTBC_SOURCE_OUTPUT_LEG_SCALE
            )
            policy_default = robot.action_decoder.policy_default(
                command_height=float(robot.command[4]),
                fallback_default=robot.default_dof_pos[JointGroup.CTRL_LEGS],
            )
            default_output = policy_to_output_pos_np(policy_default)
            nominal_requested_output = default_output + output_delta
            nominal_realizable_output = policy_to_output_pos_np(
                output_to_policy_pos_np(nominal_requested_output)
            )
            delta = output_leg_wheel_xz_np(nominal_realizable_output) - output_leg_wheel_xz_np(
                default_output
            )
        return delta

    def _ff_wheel_action_delta(self) -> np.ndarray:
        if self._kff == 0.0:
            return np.zeros(2, dtype=np.float64)
        if self._profile is None and float(self.cfg.ff_wheel_action) == 0.0:
            return np.zeros(2, dtype=np.float64)
        active = self._ff_phase >= 0
        if not np.any(active):
            return np.zeros(2, dtype=np.float64)
        phase = np.maximum(self._ff_phase.astype(np.float64), 0.0)
        if self._profile is None:
            envelope = self._ff_profile_envelope(phase) * active.astype(np.float64)
            env_envelope = float(np.max(envelope)) * float(self._kff)
            return np.full(2, float(self.cfg.ff_wheel_action) * env_envelope, dtype=np.float64)
        samples = self._profile.sample(phase, control_dt=self.control_dt)
        envelope = samples[:, 3] * active.astype(np.float64)
        env_envelope = float(np.max(envelope)) * float(self._kff)
        return np.full(2, env_envelope, dtype=np.float64)

    def _apply_output_bias_delta(
        self,
        robot: WheelLeggedRobot,
        leg_action: np.ndarray,
    ) -> np.ndarray:
        output_bias = self._ff_output_bias()
        has_output_bias = bool(np.any(np.abs(output_bias) > 0.0))
        output_delta = (
            output_bias * _CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN * _CTBC_SOURCE_OUTPUT_LEG_SCALE
        )
        if not robot.active_rod_action_semantics:
            if not has_output_bias:
                return leg_action
            leg_scale = robot.action_scale[JointGroup.LEG_ACTUATORS]
            return np.asarray(leg_action, dtype=np.float64).reshape(4) + output_delta / leg_scale

        legacy_delta_xz = self._output_bias_wheel_delta_xz(robot, output_bias)
        ff_active = self._ff_phase >= 0
        anchor_side = (np.abs(legacy_delta_xz).max(axis=1) > 0.0) | ff_active
        if not np.any(anchor_side):
            return leg_action
        anchor_wheel_xz = self._anchored_wheel_xz_body(robot, anchor_side)
        wheel_delta_xz = legacy_delta_xz + self._ff_wheel_target_delta_xz(anchor_wheel_xz)
        active_side = np.any(np.abs(wheel_delta_xz) > 0.0, axis=1)
        if not np.any(active_side):
            return leg_action

        policy_default = robot.action_decoder.policy_default(
            command_height=float(robot.command[4]),
            fallback_default=robot.default_dof_pos[JointGroup.CTRL_LEGS],
        )
        current_policy = robot.action_decoder.leg_target(leg_action, policy_default)
        current_output = policy_to_output_pos_np(current_policy)
        desired_wheel_xz = anchor_wheel_xz + wheel_delta_xz
        desired_output = current_output.copy()
        cartesian_desired_output = wheel_xz_to_output_pos_np(desired_wheel_xz)
        for side_idx in range(2):
            if active_side[side_idx]:
                sl = slice(2 * side_idx, 2 * side_idx + 2)
                desired_output[sl] = cartesian_desired_output[sl]
        desired_policy = output_to_policy_pos_np(desired_output)

        desired_action = np.asarray(leg_action, dtype=np.float64).copy()
        leg_scale = robot.action_scale[JointGroup.LEG_ACTUATORS]
        coeffs = robot.action_decoder.active_rod_angle_coeffs
        active_mid = float(robot.action_decoder.active_rod_angle_mid)
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            if not active_side[side_idx]:
                continue
            front_coef, back_coef = coeffs[side_idx]
            desired_action[front_idx] = (
                desired_policy[front_idx] - policy_default[front_idx]
            ) / leg_scale[front_idx]
            if robot.cfg.height_conditioned_action_default:
                active_default = (
                    front_coef * policy_default[front_idx] + back_coef * policy_default[back_idx]
                )
            else:
                active_default = active_mid
            active_desired = (
                front_coef * desired_policy[front_idx] + back_coef * desired_policy[back_idx]
            )
            desired_action[back_idx] = (active_desired - active_default) / leg_scale[back_idx]

        return np.nan_to_num(desired_action, nan=0.0, posinf=0.0, neginf=0.0)

    def _anchored_wheel_xz_body(
        self,
        robot: WheelLeggedRobot,
        active_side: np.ndarray,
    ) -> np.ndarray:
        """返回触发瞬间轮心 body-frame 锚点；新触发侧按当前实际轮心初始化。"""

        actual = self._actual_wheel_xz_body(robot)
        active = np.asarray(active_side, dtype=bool).reshape(2)
        missing = active & ~np.isfinite(self._ff_anchor_wheel_xz).all(axis=1)
        if np.any(missing):
            self._ff_anchor_wheel_xz[missing] = actual[missing]
        anchored = actual.copy()
        anchored[active] = self._ff_anchor_wheel_xz[active]
        return anchored

    def _actual_wheel_xz_body(self, robot: WheelLeggedRobot) -> np.ndarray:
        """读取 MuJoCo 当前轮心，并转成随 base_link 旋转的髋轴局部 XZ 坐标。"""

        base = robot.data.body(_BASE_BODY_NAME)
        base_pos = np.asarray(base.xpos, dtype=np.float64).reshape(3)
        base_rot = np.asarray(base.xmat, dtype=np.float64).reshape(3, 3)
        wheel_xz = np.zeros((2, 2), dtype=np.float64)
        for side_idx, body_name in enumerate(_WHEEL_BODY_NAMES):
            wheel_pos = np.asarray(robot.data.body(body_name).xpos, dtype=np.float64).reshape(3)
            wheel_body = base_rot.T @ (wheel_pos - base_pos) - _HIP_OFFSETS_BODY[side_idx]
            wheel_xz[side_idx, 0] = float(wheel_body[0])
            wheel_xz[side_idx, 1] = float(wheel_body[2])
        return wheel_xz

    def _apply_wheel_delta(self, wheel_action: np.ndarray) -> np.ndarray:
        return np.asarray(wheel_action, dtype=np.float64).reshape(2) + self._ff_wheel_action_delta()


__all__ = ["StairCtbcRuntime"]
