"""CTBC 台阶爬升状态管理。"""

from __future__ import annotations

import json
import math
import os
from itertools import pairwise
from pathlib import Path

import torch

_HIP_LEFT = 0
_KNEE_LEFT = 1
_HIP_RIGHT = 2
_KNEE_RIGHT = 3

_HIP_FEEDFORWARD_RATIO: float = 1.2
_KNEE_FEEDFORWARD_RATIO: float = 2.0


def _polar_to_xz_torch(length_m: torch.Tensor, swing_rad: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (length_m * torch.sin(swing_rad), -length_m * torch.cos(swing_rad)),
        dim=-1,
    )


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
        times_s: torch.Tensor,
        values: torch.Tensor,
        period_s: float,
        coordinate_mode: str,
    ) -> None:
        self.path = path
        self.times_s = times_s
        self.values = values
        self.period_s = float(period_s)
        self.coordinate_mode = coordinate_mode

    @classmethod
    def load(cls, path: Path | str, *, device: torch.device | str) -> _CtbcProfile:
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
                    z = math.radians(swing_deg)
                else:
                    z = float(raw_swing)
                    if not math.isfinite(z):
                        raise ValueError("CTBC profile field 'swing_angle_rad' must be finite")
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
        times = [row[0] for row in rows]
        if any(curr <= prev for prev, curr in pairwise(times)):
            raise ValueError(f"CTBC profile {profile_path} times must be strictly increasing")
        period_s = float(payload.get("period_s", times[-1]))
        if not math.isfinite(period_s) or period_s <= 0.0:
            raise ValueError(f"CTBC profile {profile_path} has invalid period_s={period_s!r}")
        if period_s < float(times[-1]):
            period_s = float(times[-1])

        values = [row[1:] for row in rows]
        times_t = torch.tensor(times, device=device, dtype=torch.float32)
        values_t = torch.tensor(values, device=device, dtype=torch.float32)
        if not torch.isfinite(times_t).all() or not torch.isfinite(values_t).all():
            raise ValueError(f"CTBC profile {profile_path} contains non-finite values")
        return cls(
            path=profile_path,
            times_s=times_t,
            values=values_t,
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

    def sample(self, phase_steps: torch.Tensor, *, control_dt: float) -> torch.Tensor:
        phase_shape = phase_steps.shape
        t = torch.clamp(
            phase_steps.to(device=self.times_s.device, dtype=self.times_s.dtype).reshape(-1)
            * float(control_dt),
            min=0.0,
            max=float(self.period_s),
        )
        right = torch.searchsorted(self.times_s, t, right=False)
        right = torch.clamp(right, min=1, max=self.times_s.numel() - 1)
        left = right - 1
        t0 = self.times_s[left]
        t1 = self.times_s[right]
        v0 = self.values[left]
        v1 = self.values[right]
        alpha = ((t - t0) / torch.clamp(t1 - t0, min=1.0e-6)).unsqueeze(-1)
        sampled = v0 + (v1 - v0) * alpha
        return sampled.reshape(*phase_shape, self.values.shape[-1])


class StairClimbState:
    """逐 env 的 CTBC 状态机。

    `ff_bias()` 保持源 stair 任务的旧输出关节 action 语义；训练端动作项会在注入前
    将它等效换算到目标仓库的主动杆 action 语义。
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        trigger_mode: str = "pitch",
        contact_window: int = 3,
        force_threshold_n: float = 10.0,
        pitch_threshold_rad: float = math.radians(6.0),
        pitch_threshold_deg: float | None = None,
        pitch_window: int = 3,
        ff_amplitude_rad: float = 1.70,
        coordinate_mode: str = "body_polar",
        leg_length_m: float = 0.24,
        swing_angle_rad: float = math.radians(10.0),
        swing_angle_deg: float | None = None,
        ff_x_m: float = 0.02,
        ff_lift_m: float = 0.02,
        ff_duration_s: float | None = None,
        ff_period_s: float = 0.6,
        ff_rise_ratio: float = 0.35,
        ff_hold_ratio: float = 0.0,
        ff_wheel_action: float = 0.0,
        control_dt: float = 0.02,
        ff_start_iter: int = 0,
        ann_start_iter: int = 900,
        ann_end_iter: int = 1800,
        phantom_trigger_iter: int = 0,
        allow_bilateral_trigger: bool = False,
        profile_path: Path | str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self._profile = (
            _CtbcProfile.load(profile_path, device=device) if profile_path is not None else None
        )
        self.contact_window = contact_window
        self.force_threshold = force_threshold_n
        self.trigger_mode = "pitch" if trigger_mode == "pitch" else "force"
        self.pitch_threshold_rad = (
            float(pitch_threshold_rad)
            if pitch_threshold_deg is None
            else math.radians(float(pitch_threshold_deg))
        )
        self.pitch_window = max(1, int(pitch_window))
        self.ff_amplitude = ff_amplitude_rad
        self.coordinate_mode = "body_polar" if coordinate_mode == "body_polar" else "body_cartesian"
        self.leg_length_m = float(leg_length_m)
        self.swing_angle_rad = (
            float(swing_angle_rad)
            if swing_angle_deg is None
            else math.radians(float(swing_angle_deg))
        )
        self.ff_x_m = ff_x_m
        self.ff_lift_m = ff_lift_m
        self.ff_wheel_action = float(ff_wheel_action)
        self.control_dt = control_dt
        del ff_rise_ratio, ff_hold_ratio
        duration_s = (
            self._profile.period_s
            if self._profile is not None
            else float(ff_period_s if ff_duration_s is None else ff_duration_s)
        )
        self.ff_duration_steps = max(1, round(duration_s / control_dt))
        self.ff_period_steps = self.ff_duration_steps
        self.ff_rise_steps = self.ff_duration_steps
        self.ff_hold_steps = 0
        self.ff_return_steps = 0
        self.ff_start = int(ff_start_iter)
        self.ann_start = max(int(ann_start_iter), self.ff_start)
        self.ann_end = max(int(ann_end_iter), self.ann_start)
        self.phantom_trigger_iter = phantom_trigger_iter
        self.allow_bilateral_trigger = bool(allow_bilateral_trigger)

        self._contact_buf = torch.zeros(contact_window, num_envs, 2, device=device)
        self._pitch_buf = torch.zeros(self.pitch_window, num_envs, device=device)
        self._stable = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self._pitch_stable = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._last_pitch_rad = torch.zeros(num_envs, device=device)
        self._last_pitch_abs_deg = torch.zeros(num_envs, device=device)
        self._trigger_score = torch.zeros(num_envs, 2, device=device)
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
        self._stair_success_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._max_stair_success_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._stair_success_record_step = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=device,
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
        """保留旧诊断字段的 duration 步数映射。"""
        del rise_ratio, hold_ratio
        return self.ff_duration_steps, 0, 0

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

    def step(self, wheel_contact_xy: torch.Tensor, pitch_rad: torch.Tensor | None = None) -> None:
        """根据当前轮子水平接触力更新触发窗口和前馈相位。"""
        wheel_contact_xy = torch.nan_to_num(
            wheel_contact_xy.to(device=self.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._contact_buf[1:] = self._contact_buf[:-1].clone()
        self._contact_buf[0] = wheel_contact_xy

        if pitch_rad is None:
            pitch_value = torch.zeros(self.num_envs, device=self.device)
        else:
            pitch_value = torch.nan_to_num(
                pitch_rad.to(device=self.device).reshape(self.num_envs),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        self._last_pitch_rad[:] = pitch_value
        pitch_abs_rad = torch.abs(pitch_value)
        self._last_pitch_abs_deg[:] = torch.rad2deg(pitch_abs_rad)
        self._pitch_buf[1:] = self._pitch_buf[:-1].clone()
        self._pitch_buf[0] = pitch_abs_rad

        if self.trigger_mode == "pitch":
            self._pitch_stable = (self._pitch_buf > float(self.pitch_threshold_rad)).all(dim=0)
            self._stable = self._pitch_stable.unsqueeze(-1).expand(-1, 2).clone()
            self._trigger_score[:] = self._last_pitch_abs_deg.unsqueeze(-1).expand(-1, 2)
        else:
            above = self._contact_buf > self.force_threshold
            self._stable = above.all(dim=0)
            self._trigger_score[:] = wheel_contact_xy
        stable_any = self._stable.any(dim=-1)
        self._riser_contact_steps[stable_any] += 1
        self._riser_contact_steps[~stable_any] = 0

        self._cooldown[self._cooldown > 0] -= 1
        can_trigger = (self._ff_phase == -1) & (self._cooldown == 0)
        trigger_candidates = self._stable & can_trigger
        if self.trigger_mode == "pitch" or self.allow_bilateral_trigger:
            newly_triggered = can_trigger & trigger_candidates.any(dim=-1, keepdim=True)
        else:
            newly_triggered = self._select_single_trigger_side(
                trigger_candidates,
                self._trigger_score,
            )
        self._ff_phase[newly_triggered] = 0

        if self._iter < self.phantom_trigger_iter:
            phantom_mask = torch.rand(self.num_envs, 2, device=self.device) < 0.01
            phantom_candidates = phantom_mask & can_trigger
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

        finished = self._ff_phase >= self.ff_duration_steps
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
            if self._profile is None:
                envelope = self._ff_profile_envelope(phase)
            else:
                envelope = self._profile.sample(phase, control_dt=self.control_dt)[:, 2]
            bias_value = 2.0 * self.ff_amplitude * envelope
            bias_value = bias_value * active_mask.float()
            bias[:, hip_idx] = -bias_value * _HIP_FEEDFORWARD_RATIO * self._kff
            bias[:, knee_idx] = bias_value * _KNEE_FEEDFORWARD_RATIO * self._kff
        return bias

    def _ff_profile_envelope(self, phase: torch.Tensor) -> torch.Tensor:
        """返回 duration 内恒定为 1 的前馈包络。"""
        return torch.where(phase >= 0.0, torch.ones_like(phase), torch.zeros_like(phase))

    def ff_wheel_target_xz(self) -> torch.Tensor:
        """返回机体系 hip-relative 极坐标目标转换出的绝对轮心 XZ。"""
        target = torch.full((self.num_envs, 2, 2), float("nan"), device=self.device)
        if self._kff == 0.0:
            return target
        active = self._ff_phase >= 0
        if not active.any():
            return target
        phase = torch.clamp(self._ff_phase.float(), min=0.0)
        if self._profile is None:
            if self.coordinate_mode != "body_polar":
                return target
            length = torch.full_like(phase, float(self.leg_length_m))
            swing = torch.full_like(phase, float(self.swing_angle_rad))
            polar_target = _polar_to_xz_torch(length, swing)
        elif self._profile.coordinate_mode == "body_polar":
            samples = self._profile.sample(phase, control_dt=self.control_dt)
            polar_target = _polar_to_xz_torch(samples[:, :, 0], samples[:, :, 1])
        else:
            return target
        target[active] = polar_target[active]
        return target

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
            if self._profile is None:
                if self.coordinate_mode == "body_polar":
                    continue
                envelope = self._ff_profile_envelope(phase)
                envelope = envelope * active_mask.float() * float(self._kff)
                delta[:, side, 0] = -float(self.ff_x_m) * envelope
                delta[:, side, 1] = float(self.ff_lift_m) * envelope
            else:
                if self._profile.coordinate_mode == "body_polar":
                    continue
                samples = self._profile.sample(phase, control_dt=self.control_dt)
                active = active_mask.float() * float(self._kff)
                delta[:, side, 0] = samples[:, 0] * active
                delta[:, side, 1] = samples[:, 1] * active
        return delta

    def ff_wheel_action_delta(self) -> torch.Tensor:
        """返回 CTBC 期间叠加到左右轮 action 的前向轮速前馈。"""
        delta = torch.zeros(self.num_envs, 2, device=self.device)
        if self._kff == 0.0:
            return delta
        if self._profile is None and self.ff_wheel_action == 0.0:
            return delta

        active = self._ff_phase >= 0
        if not active.any():
            return delta
        phase = torch.clamp(self._ff_phase.float(), min=0.0)
        if self._profile is None:
            envelope = self._ff_profile_envelope(phase) * active.float()
            env_envelope = envelope.max(dim=1).values * float(self._kff)
            delta[:, :] = float(self.ff_wheel_action) * env_envelope.unsqueeze(-1)
            return delta

        samples = self._profile.sample(phase, control_dt=self.control_dt)
        wheel_action = samples[:, :, 3] * active.float()
        env_wheel_action = wheel_action.max(dim=1).values * float(self._kff)
        delta[:, :] = env_wheel_action.unsqueeze(-1)
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

    def record_stair_success_candidate(
        self,
        success_candidate: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        """记录 strict stair success 候选条件的连续满足时长。"""
        success_candidate = success_candidate.to(device=self.device).bool()
        if success_candidate.ndim != 1 or success_candidate.shape[0] != self.num_envs:
            success_candidate = success_candidate.reshape(self.num_envs)
        if step_index is None:
            new_step = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            step_value = int(step_index)
            new_step = self._stair_success_record_step != step_value

        updated_steps = torch.where(
            success_candidate,
            self._stair_success_steps + 1,
            torch.zeros_like(self._stair_success_steps),
        )
        self._stair_success_steps = torch.where(
            new_step,
            updated_steps,
            self._stair_success_steps,
        )
        self._max_stair_success_steps = torch.maximum(
            self._max_stair_success_steps,
            self._stair_success_steps,
        )
        if step_index is not None:
            self._stair_success_record_step[new_step] = int(step_index)
        return self.stair_success_duration()

    def stair_success_duration(self) -> torch.Tensor:
        """返回当前 strict stair success 连续满足时长。"""
        return self._stair_success_steps.float() * float(self.control_dt)

    def max_stair_success_duration(self) -> torch.Tensor:
        """返回 episode 内 strict stair success 最长连续满足时长。"""
        return self._max_stair_success_steps.float() * float(self.control_dt)

    def riser_stall_active(self, min_duration_s: float) -> torch.Tensor:
        min_steps = max(1, round(float(min_duration_s) / self.control_dt))
        return self._riser_contact_steps >= min_steps

    def reset(self, env_ids: torch.Tensor) -> None:
        self._contact_buf[:, env_ids] = 0.0
        self._pitch_buf[:, env_ids] = 0.0
        self._stable[env_ids] = False
        self._pitch_stable[env_ids] = False
        self._last_pitch_rad[env_ids] = 0.0
        self._last_pitch_abs_deg[env_ids] = 0.0
        self._trigger_score[env_ids] = 0.0
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
        self._stair_success_steps[env_ids] = 0
        self._max_stair_success_steps[env_ids] = 0
        self._stair_success_record_step[env_ids] = -1

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
    def latest_pitch_rad(self) -> torch.Tensor:
        return self._last_pitch_rad.clone()

    @property
    def latest_pitch_abs_deg(self) -> torch.Tensor:
        return self._last_pitch_abs_deg.clone()

    @property
    def trigger_score(self) -> torch.Tensor:
        return self._trigger_score.clone()

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
            "Stair/diag_strict_success_duration_s": self.stair_success_duration().mean().item(),
            "Stair/diag_strict_success_max_duration_s": (
                self.max_stair_success_duration().mean().item()
            ),
            "Stair/diag_ctbc_kff": self._kff,
            "Stair/diag_ctbc_local_iter": float(self._iter),
            "Stair/diag_ctbc_trigger_mode_pitch": float(self.trigger_mode == "pitch"),
            "Stair/diag_ctbc_force_threshold_n": float(self.force_threshold),
            "Stair/diag_ctbc_contact_window": float(self.contact_window),
            "Stair/diag_ctbc_pitch_abs_deg": float(self._last_pitch_abs_deg.mean().item()),
            "Stair/diag_ctbc_pitch_threshold_deg": float(math.degrees(self.pitch_threshold_rad)),
            "Stair/diag_ctbc_pitch_window": float(self.pitch_window),
            "Stair/diag_ctbc_coordinate_mode_polar": float(self.coordinate_mode == "body_polar"),
            "Stair/diag_ctbc_leg_length_m": float(self.leg_length_m),
            "Stair/diag_ctbc_swing_angle_deg": float(math.degrees(self.swing_angle_rad)),
            "Stair/diag_ctbc_ff_duration_steps": float(self.ff_duration_steps),
            "Stair/diag_ctbc_ff_rise_steps": float(self.ff_rise_steps),
            "Stair/diag_ctbc_ff_hold_steps": float(self.ff_hold_steps),
            "Stair/diag_ctbc_ff_return_steps": float(self.ff_return_steps),
            "Stair/diag_ctbc_allow_bilateral_trigger": float(self.allow_bilateral_trigger),
            "Stair/diag_ctbc_ff_wheel_action": float(self.ff_wheel_action),
            "Stair/diag_ctbc_profile_enabled": float(self._profile is not None),
        }


__all__ = ["StairClimbState"]
