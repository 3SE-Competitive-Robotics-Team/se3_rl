"""把逐条结果汇总成头条指标（W&B 曲线与 compare 用）。缺项（quick 档没跑的评测）不出现在结果里。"""

from __future__ import annotations

import math

import numpy as np

T90_CAP_S = 5.0
"""起步 5 s 内没跟到 90% 记为 5 s，便于求均值与画曲线。"""


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _rate(flags) -> float | None:
    flags = [bool(f) for f in flags]
    return float(np.mean(flags)) if flags else None


def _ok(r: dict) -> bool:
    return "error" not in r and "skipped" not in r


def headline(results: list[dict]) -> dict[str, float]:
    out: dict[str, float | None] = {}
    ok = [r for r in results if _ok(r)]
    by_suite = {s: [r for r in ok if r["suite"] == s] for s in "ABCDE"}

    # A 平地跟踪
    flat = by_suite["A"]
    vx_cases = [r for r in flat if not r["fell"] and "vx_ratio" in r and "wz_ratio" not in r]
    out["A/vx_err_mean"] = _mean(r["vx_err"] for r in vx_cases)
    out["A/vx_err_max"] = max((r["vx_err"] for r in vx_cases), default=None)
    top = [
        r
        for r in flat
        if not r["fell"]
        and "wz_ratio" in r
        and r["name"].startswith("flat_yaw")
        and "12" in r["name"]
    ]
    out["A/yaw_top_ratio"] = _mean(r["wz_ratio"] for r in top)
    yaw_cases = [r for r in flat if not r["fell"] and "wz_ratio" in r]
    out["A/yaw_err_mean"] = _mean(r["wz_err"] for r in yaw_cases)
    out["A/height_abs_err_mean"] = _mean(abs(r["height_err"]) for r in flat if not r["fell"])
    out["A/pitch_rate_band_rms_max"] = max(
        (
            r["pitch_rate_band_rms"]
            for r in flat
            if not r["fell"] and np.isfinite(r["pitch_rate_band_rms"])
        ),
        default=None,
    )
    out["A/falls"] = float(sum(r["fell"] for r in flat)) if flat else None

    # B 静站与起步
    stand = by_suite["B"]
    stand30 = [r for r in stand if r["name"].startswith("stand30")]
    out["B/drift_speed_max"] = max(
        (r["drift_speed"] for r in stand30 if "drift_speed" in r), default=None
    )
    out["B/drift_m_max"] = max((r["drift_m"] for r in stand30 if "drift_m" in r), default=None)
    out["B/yaw_drift_deg_max"] = max(
        (r["yaw_drift_deg"] for r in stand30 if "yaw_drift_deg" in r), default=None
    )

    def t90(r: dict) -> float:
        return min(float(r.get("t90_s", math.inf)), T90_CAP_S)

    out["B/t90_after_stand_mean_s"] = _mean(t90(r) for r in stand30 if not r["fell"])
    pairs = []
    for r in stand30:
        twin = next((c for c in stand if c["name"] == r["name"].replace("stand30", "cold")), None)
        if twin and not r["fell"] and not twin["fell"]:
            pairs.append(t90(r) - t90(twin))
    out["B/t90_degrade_mean_s"] = _mean(pairs)
    out["B/falls"] = float(sum(r["fell"] for r in stand)) if stand else None

    # C 台阶
    stairs = by_suite["C"]
    det = [r for r in stairs if "_noise" not in r["name"]]
    up = [r for r in det if r["name"].startswith("stairs_up_")]
    levels_cleared = []
    for row in sorted({int(r["name"].split("_r")[1].split("_")[0]) for r in up}):
        running = [r for r in up if f"_r{row}_" in r["name"] and r["name"].endswith("running")]
        rate = _rate(r["cleared"] for r in running)
        out[f"C/stairs_up_r{row}_clear_rate"] = rate
        if rate is not None and rate >= 0.5:
            levels_cleared.append(row)
    out["C/stairs_up_max_level"] = (
        float(max(levels_cleared)) if levels_cleared else (0.0 if up else None)
    )
    up_cols = [r for r in det if r["name"].startswith(("stairs_up_", "stairs_two_step_up_"))]
    run_rate = _rate(r["cleared"] for r in up_cols if r["name"].endswith("running"))
    static_rate = _rate(r["cleared"] for r in up_cols if r["name"].endswith("static"))
    out["C/up_clear_rate_running"] = run_rate
    out["C/up_clear_rate_static"] = static_rate
    if run_rate is not None and static_rate is not None:
        out["C/static_start_gap"] = run_rate - static_rate
    out["C/two_step_up_clear_rate"] = _rate(
        r["cleared"] for r in det if r["name"].startswith("stairs_two_step_up_")
    )
    events = [e for r in up_cols for e in r.get("riser_events", [])]
    out["C/target_style_rate"] = _rate(e["target_style"] for e in events)
    out["C/jump_onto_rate"] = _rate(e["jumped"] for e in events)
    out["C/body_first_rate"] = _rate(e["body_first"] for e in events)
    out["C/lift_dt_mean_s"] = _mean(e["lift_dt_s"] for e in events)
    down = [r for r in det if r["name"].startswith(("stairs_down_", "stairs_two_step_down_"))]
    out["C/down_fall_rate"] = _rate(r["fell"] for r in down)
    out["C/down_clear_rate"] = _rate(r["cleared"] for r in down)
    noisy = [r for r in stairs if "_noise" in r["name"]]
    out["C/noise_up_clear_rate"] = _rate(r["cleared"] for r in noisy)
    out["C/up_fall_rate"] = _rate(r["fell"] for r in up_cols)

    # D 跳跃
    jump = by_suite["D"]
    out["D/success_rate"] = _rate(r.get("success", False) for r in jump)
    out["D/clearance_err_mean_m"] = _mean(
        r["clearance_err_m"] for r in jump if "clearance_err_m" in r
    )
    out["D/falls"] = float(sum(r.get("fell", False) for r in jump)) if jump else None

    # E 鲁棒
    robust = by_suite["E"]
    if robust:
        out["E/falls"] = float(sum(r.get("fell", False) for r in robust))
        out["E/flat_vx_err_mean"] = _mean(
            r["vx_err"] for r in robust if r["kind"] == "flat" and not r["fell"]
        )
        out["E/stand_drift_m_max"] = max(
            (r["drift_m"] for r in robust if r["kind"] == "standing_drift" and "drift_m" in r),
            default=None,
        )
        out["E/push_tilt_max_deg"] = max(
            (r["tilt_max_deg"] for r in robust if r["kind"] == "push" and "tilt_max_deg" in r),
            default=None,
        )
        out["E/stairs_r6_clear_rate"] = _rate(r["cleared"] for r in robust if r["kind"] == "stairs")

    out["meta/errors"] = float(sum("error" in r for r in results))
    return {k: float(v) for k, v in out.items() if v is not None}


__all__ = ["headline"]
