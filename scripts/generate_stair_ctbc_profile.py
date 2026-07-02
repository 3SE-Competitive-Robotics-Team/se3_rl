"""生成台阶任务使用的 CTBC 轮心轨迹 profile。

输出 JSON 会被训练端 StairClimbState 和原生 MuJoCo sim2sim 的
StairCtbcRuntime 通过同一个 ``profile_path`` 字段读取。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _half_cosine(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return 0.5 * (1.0 - math.cos(math.pi * x))


def _sample_path(
    *,
    period_s: float,
    samples: int,
    step_height_m: float,
    clearance_m: float,
    retreat_m: float,
    hold_ratio: float,
    return_start_ratio: float,
    peak_time_ratio: float,
    wheel_action: float,
) -> list[dict[str, float]]:
    peak_z = max(0.0, float(step_height_m) + float(clearance_m))
    retreat_m = max(0.0, float(retreat_m))
    samples = max(3, int(samples))
    hold_ratio = max(0.0, min(0.45, float(hold_ratio)))
    peak_time_ratio = max(0.12, min(0.70, float(peak_time_ratio)))
    return_start_ratio = max(peak_time_ratio + hold_ratio, min(0.95, float(return_start_ratio)))
    if return_start_ratio >= 0.98:
        return_start_ratio = 0.98

    points: list[dict[str, float]] = []
    for idx in range(samples):
        u = idx / float(samples - 1)
        if u <= peak_time_ratio:
            s = _half_cosine(u / peak_time_ratio)
            x_m = -retreat_m * s
            z_m = peak_z * s
            amp = s
        elif u <= return_start_ratio:
            denom = max(1.0e-6, return_start_ratio - peak_time_ratio)
            s = (u - peak_time_ratio) / denom
            hold = _smoothstep(s)
            # Keep high clearance while letting the wheel drift slightly toward neutral.
            x_m = -retreat_m * (1.0 - 0.25 * hold)
            z_m = peak_z
            amp = 1.0 - 0.15 * hold
        else:
            s = _half_cosine((u - return_start_ratio) / max(1.0e-6, 1.0 - return_start_ratio))
            x_m = -retreat_m * 0.75 * (1.0 - s)
            z_m = peak_z * (1.0 - s)
            amp = 0.85 * (1.0 - s)

        rounded_x = round(x_m, 6)
        rounded_z = round(z_m, 6)
        points.append(
            {
                "t": round(float(period_s) * u, 6),
                "body_x_m": rounded_x,
                "body_z_m": rounded_z,
                "x_m": rounded_x,
                "z_m": rounded_z,
                "amp": round(max(0.0, min(1.0, amp)), 6),
                "wheel_action": round(float(wheel_action), 6),
            }
        )

    neutral = {"body_x_m": 0.0, "body_z_m": 0.0, "x_m": 0.0, "z_m": 0.0}
    points[0].update({**neutral, "amp": 0.0, "wheel_action": 0.0})
    points[-1].update({**neutral, "amp": 0.0, "wheel_action": 0.0})
    return points


def _ctbc_cosine_envelope(
    t: float,
    *,
    period_s: float,
    rise_ratio: float,
    hold_ratio: float,
) -> float:
    period_s = max(1.0e-6, float(period_s))
    rise_ratio = max(0.05, min(float(rise_ratio), 0.90))
    hold_ratio = max(0.0, min(float(hold_ratio), 0.90))
    if rise_ratio + hold_ratio >= 0.95:
        hold_ratio = max(0.0, 0.95 - rise_ratio)

    phase = max(0.0, min(float(t), period_s))
    rise_s = max(1.0e-6, period_s * rise_ratio)
    hold_end_s = period_s * (rise_ratio + hold_ratio)
    return_s = max(1.0e-6, period_s - hold_end_s)

    if phase < rise_s:
        return 0.5 * (1.0 - math.cos(math.pi * phase / rise_s))
    if phase < hold_end_s:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (phase - hold_end_s) / return_s))


def _sample_cosine_profile(
    *,
    period_s: float,
    control_dt: float,
    ff_x_m: float,
    ff_lift_m: float,
    ff_rise_ratio: float,
    ff_hold_ratio: float,
    ff_wheel_action: float,
) -> list[dict[str, float]]:
    """按训练端 CTBC 余弦 envelope 生成 profile points。"""
    period_s = max(1.0e-6, float(period_s))
    control_dt = max(1.0e-4, float(control_dt))
    steps = max(1, round(period_s / control_dt))
    points: list[dict[str, float]] = []
    for idx in range(steps + 1):
        t = min(period_s, idx * control_dt)
        envelope = _ctbc_cosine_envelope(
            t,
            period_s=period_s,
            rise_ratio=ff_rise_ratio,
            hold_ratio=ff_hold_ratio,
        )
        rounded_x = round(-float(ff_x_m) * envelope, 6)
        rounded_z = round(float(ff_lift_m) * envelope, 6)
        points.append(
            {
                "t": round(t, 6),
                "body_x_m": rounded_x,
                "body_z_m": rounded_z,
                "x_m": rounded_x,
                "z_m": rounded_z,
                "amp": round(envelope, 6),
                "wheel_action": round(float(ff_wheel_action) * envelope, 6),
            }
        )
    neutral = {"body_x_m": 0.0, "body_z_m": 0.0, "x_m": 0.0, "z_m": 0.0}
    points[0].update({**neutral, "amp": 0.0, "wheel_action": 0.0})
    points[-1].update({**neutral, "amp": 0.0, "wheel_action": 0.0})
    return points


def _sample_duration_profile(
    *,
    duration_s: float,
    control_dt: float,
    ff_x_m: float,
    ff_lift_m: float,
    ff_wheel_action: float,
) -> list[dict[str, float]]:
    duration_s = max(1.0e-6, float(duration_s))
    control_dt = max(1.0e-4, float(control_dt))
    steps = max(1, round(duration_s / control_dt))
    x_m = round(-max(0.0, float(ff_x_m)), 6)
    z_m = round(max(0.0, float(ff_lift_m)), 6)
    wheel_action = round(float(ff_wheel_action), 6)
    points: list[dict[str, float]] = []
    for idx in range(steps + 1):
        t = duration_s * idx / steps
        points.append(
            {
                "t": round(t, 6),
                "body_x_m": x_m,
                "body_z_m": z_m,
                "x_m": x_m,
                "z_m": z_m,
                "amp": 1.0,
                "wheel_action": wheel_action,
            }
        )
    return points


def _sample_polar_duration_profile(
    *,
    duration_s: float,
    control_dt: float,
    leg_length_m: float,
    swing_angle_rad: float,
    ff_wheel_action: float,
) -> list[dict[str, float]]:
    duration_s = max(1.0e-6, float(duration_s))
    control_dt = max(1.0e-4, float(control_dt))
    leg_length_m = max(1.0e-6, float(leg_length_m))
    swing_angle_rad = float(swing_angle_rad)
    steps = max(1, round(duration_s / control_dt))
    x_m = round(leg_length_m * math.sin(swing_angle_rad), 6)
    z_m = round(-leg_length_m * math.cos(swing_angle_rad), 6)
    wheel_action = round(float(ff_wheel_action), 6)
    points: list[dict[str, float]] = []
    for idx in range(steps + 1):
        t = duration_s * idx / steps
        points.append(
            {
                "t": round(t, 6),
                "leg_length_m": round(leg_length_m, 6),
                "swing_angle_rad": round(swing_angle_rad, 6),
                "swing_angle_deg": round(math.degrees(swing_angle_rad), 6),
                "body_x_m": x_m,
                "body_z_m": z_m,
                "x_m": x_m,
                "z_m": z_m,
                "amp": 1.0,
                "wheel_action": wheel_action,
            }
        )
    return points


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/ctbc_profiles/stair_gmr_style_12cm.json"),
    )
    parser.add_argument("--step-height-m", type=float, default=0.12)
    parser.add_argument("--clearance-m", type=float, default=0.045)
    parser.add_argument("--retreat-m", type=float, default=0.16)
    parser.add_argument("--period-s", type=float, default=0.72)
    parser.add_argument("--samples", type=int, default=13)
    parser.add_argument("--peak-time-ratio", type=float, default=0.32)
    parser.add_argument("--hold-ratio", type=float, default=0.24)
    parser.add_argument("--return-start-ratio", type=float, default=0.66)
    parser.add_argument("--wheel-action", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.period_s <= 0.0:
        raise ValueError("--period-s must be positive")
    if args.step_height_m < 0.0:
        raise ValueError("--step-height-m must be non-negative")
    if args.clearance_m < 0.0:
        raise ValueError("--clearance-m must be non-negative")

    payload = {
        "version": 1,
        "generator": "scripts/generate_stair_ctbc_profile.py",
        "description": (
            "GMR-style task-space CTBC profile: define the desired wheel-center "
            "trajectory first, then let existing CTBC IK map it to leg actions."
        ),
        "period_s": round(float(args.period_s), 6),
        "metadata": {
            "cartesian_frame": "body",
            "step_height_m": float(args.step_height_m),
            "clearance_m": float(args.clearance_m),
            "body_x_m": float(args.retreat_m),
            "body_z_m": float(args.step_height_m) + float(args.clearance_m),
            "retreat_m": float(args.retreat_m),
            "peak_time_ratio": float(args.peak_time_ratio),
            "hold_ratio": float(args.hold_ratio),
            "return_start_ratio": float(args.return_start_ratio),
        },
        "points": _sample_path(
            period_s=float(args.period_s),
            samples=int(args.samples),
            step_height_m=float(args.step_height_m),
            clearance_m=float(args.clearance_m),
            retreat_m=float(args.retreat_m),
            hold_ratio=float(args.hold_ratio),
            return_start_ratio=float(args.return_start_ratio),
            peak_time_ratio=float(args.peak_time_ratio),
            wheel_action=float(args.wheel_action),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
