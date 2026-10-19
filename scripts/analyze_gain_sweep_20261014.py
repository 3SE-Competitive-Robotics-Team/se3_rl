"""2026-10-14 实车增益扫描（A–E）对比：逐段主频、幅值、策略传递、通道功率与高频成分；只读既有采集，不连接硬件。

口径沿用 scripts/analyze_control_sync_20261012.py：谐波回归求主频与复幅值，功率是该频率分量的
指令层功率 0.5·Re(F·conj(v))，不是输出轴实测功率；带内 RMS 用 Welch（fs 1000，nperseg 900，noverlap 450）。
E 组使能过短没有稳态段，只能和其它组在"使能沿"那段（起身过程）上比较。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_control_sync_20261012 import harmonic, load
from scipy.optimize import minimize_scalar
from scipy.signal import butter, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = Path("D:/RoboMaster/serialleg2026/build/control-sync-captures")
SWEEP = CAPTURE / "gain_sweep_20261013"
REFERENCE_1012 = CAPTURE / "standing_20261012_reconnected/burst_0003_1052941.npz"
OUT = ROOT / ".scratch/gain_sweep_analysis_20261014"
VARIANTS = "ABCDE"
LEGS = ["LF", "RF", "LB", "RB"]  # 记录器电机顺序
WHEEL_SPEEDS = np.array(
    [0.0, 32.93, 49.07, 57.82, 63.65, 65.29, 65.70, 67.43, 69.28, 70.90, 71.37, 71.81]
)
WHEEL_TORQUES = np.array([3.71, 3.71, 3.61, 3.54, 3.46, 3.39, 3.32, 2.95, 2.21, 1.47, 0.74, 0.0])


def ratio(num, den):
    """复数比值的增益与相位（度）；相位为正表示分子超前。"""
    value = np.asarray(num) / np.asarray(den)
    return {"gain": np.abs(value).tolist(), "phase_deg": np.rad2deg(np.angle(value)).tolist()}


def dominant_frequency(t, y, lo=3.0, hi=14.0):
    """在 lo–hi Hz 内找单频谐波回归残差最小的频率。"""
    freqs = np.linspace(lo, hi, 1101)
    score = [np.sum(harmonic(t, y, f)[3] ** 2) for f in freqs]
    best = int(np.argmin(score))
    opt = minimize_scalar(
        lambda f: np.sum(harmonic(t, y, f)[3] ** 2),
        bounds=(freqs[max(0, best - 2)], freqs[min(len(freqs) - 1, best + 2)]),
        method="bounded",
    )
    return float(opt.x)


def welch_band_rms(t, x, lo, hi):
    """与 2026-10-12 分析相同的 Welch 口径。"""
    fs = 1.0 / float(np.median(np.diff(t)))
    nperseg = min(900, len(x))
    freqs, power = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=0)
    band = (freqs >= lo) & (freqs <= hi)
    return np.sqrt(np.sum(power[band], axis=0) * (freqs[1] - freqs[0]))


def controller_joint(data, mon):
    """与固件 LegMit_JointPosition 相同的 wrap 后关节角，用于逐帧复算 PD。"""
    return np.angle(np.exp(1j * (data["position"] + np.asarray(mon["offset"]))))


def formula_check(data, mon):
    """用该段监控里的增益复算主控力矩；能对上说明板上确实跑的是该组增益。"""
    kp, kd = mon["leg_kp"], mon["leg_kd"]
    wheel_gain = mon.get("wheel_gain", 0.2)
    err = data["target"][:, :4] - controller_joint(data, mon)
    err[:, :2] = np.angle(np.exp(1j * err[:, :2]))
    leg = np.clip(kp * err - kd * data["velocity"] + data["spring"], -40, 40)
    wheel = np.clip(wheel_gain * (data["target"][:, 4:] - data["wheel_velocity"]), -2.21, 2.21)
    limit = np.interp(np.abs(data["wheel_limit_velocity"]), WHEEL_SPEEDS, WHEEL_TORQUES, right=0.0)
    wheel = np.clip(wheel, -limit, limit)
    predicted = np.column_stack([leg, wheel])
    predicted[(data["flags"] & 8) == 0] = 0
    return np.max(np.abs(predicted - data["torque"]), axis=0).tolist()


def steady_metrics(data, meta):
    """稳态段：主频、各通道复幅值、策略传递、反馈滞后与指令层功率。"""
    mon = meta["monitor_before_export"]
    wheel_gain = mon.get("wheel_gain", 0.2)
    t = data["input_time"]
    pitch = data["gyro"][:, 0]  # 原始 BMI088 x 轴即部署机身俯仰角速度
    f = dominant_frequency(t, pitch)
    omega = 2 * np.pi * f
    hg, r2, _, _ = harmonic(t, pitch, f)
    hq = np.array([harmonic(data["rx_sof_time"][:, j], data["q"][:, j], f, 3)[0] for j in range(4)])
    hv = np.array(
        [harmonic(data["rx_sof_time"][:, j], data["velocity"][:, j], f, 3)[0] for j in range(4)]
    )
    hwq = np.array(
        [harmonic(data["rx_sof_time"][:, 4 + j], data["wheel_q"][:, j], f, 3)[0] for j in range(2)]
    )
    hwv_fb = np.array(
        [
            harmonic(data["rx_sof_time"][:, 4 + j], data["wheel_feedback_velocity"][:, j], f, 3)[0]
            for j in range(2)
        ]
    )
    h = {
        key: harmonic(t, data[key], f, 3)[0]
        for key in ("P_target", "P_position", "D", "spring", "target", "torque", "wheel_velocity")
    }
    hcurrent = np.array(
        [
            harmonic(data["rx_sof_time"][:, j], data["current_torque"][:, j], f, 3)[0]
            for j in range(6)
        ]
    )
    v_leg = 1j * omega * harmonic(t, data["q"], f, 3)[0]
    v_wheel = 1j * omega * hwq
    power = {
        key: (0.5 * np.real(h[key] * np.conj(v_leg))).tolist()
        for key in ("P_target", "P_position", "D", "spring")
    }
    power["wheel_target_term"] = (
        0.5 * np.real(wheel_gain * h["target"][4:] * np.conj(v_wheel))
    ).tolist()
    power["wheel_feedback_term"] = (
        0.5 * np.real(-wheel_gain * h["wheel_velocity"] * np.conj(v_wheel))
    ).tolist()
    return {
        "frequency_hz": f,
        "pitch_fit_r2": float(r2),
        "amplitude": {
            "pitch_rate_rad_s": float(abs(hg)),
            "leg_deg": np.rad2deg(np.abs(hq)).tolist(),
            "wheel_velocity_rad_s": np.abs(h["wheel_velocity"]).tolist(),
            "leg_target_deg": np.rad2deg(np.abs(h["target"][:4])).tolist(),
            "wheel_target_rad_s": np.abs(h["target"][4:]).tolist(),
            "leg_torque_nm": np.abs(h["torque"][:4]).tolist(),
            "wheel_torque_nm": np.abs(h["torque"][4:]).tolist(),
            "leg_current_torque_nm": np.abs(hcurrent[:4]).tolist(),
        },
        "band_rms_5_15": {
            "pitch_rate_rad_s": float(welch_band_rms(t, pitch, 5, 15)),
            "leg_deg": np.rad2deg(welch_band_rms(t, data["q"], 5, 15)).tolist(),
            "wheel_velocity_rad_s": welch_band_rms(t, data["wheel_velocity"], 5, 15).tolist(),
        },
        "band_rms_15_100": {
            "leg_torque_nm": welch_band_rms(t, data["torque"][:, :4], 15, 100).tolist(),
            "leg_velocity_rad_s": welch_band_rms(t, data["velocity"], 15, 100).tolist(),
            "pitch_rate_rad_s": float(welch_band_rms(t, pitch, 15, 100)),
        },
        "policy_pitch_to_wheel_target": ratio(h["target"][4:], hg),
        "policy_pitch_to_leg_target": ratio(h["target"][:4], hg),
        "wheel_torque_vs_wheel_target": ratio(h["torque"][4:], h["target"][4:]),
        "wheel_rpm_feedback_vs_angle_derivative": ratio(hwv_fb, v_wheel),
        "leg_velocity_feedback_vs_position_derivative": ratio(hv, 1j * omega * hq),
        "command_power_w": power,
        "leg_torque_saturation_fraction": np.mean(
            np.abs(data["torque"][:, :4]) >= 39.999, axis=0
        ).tolist(),
        "wheel_torque_saturation_fraction": np.mean(
            np.abs(data["torque"][:, 4:]) >= 2.20999, axis=0
        ).tolist(),
        "formula_max_error_nm": formula_check(data, mon),
    }


def edge_metrics(data):
    """使能沿段（起身过程）：0.2 s 窗的俯仰 5–15 Hz 与腿力矩/速度 15–100 Hz 带通 RMS。"""
    t = data["input_time"]
    fs = 1.0 / float(np.median(np.diff(t)))
    low = butter(4, [5, 15], btype="band", fs=fs, output="sos")
    high = butter(4, [15, 100], btype="band", fs=fs, output="sos")
    pitch = sosfiltfilt(low, data["gyro"][:, 0])
    torque = sosfiltfilt(high, data["torque"][:, :4], axis=0)
    velocity = sosfiltfilt(high, data["velocity"], axis=0)
    rows = []
    for start in np.arange(t[0], t[-1] - 0.2 + 1e-9, 0.1):
        mask = (t >= start) & (t < start + 0.2)
        rows.append(
            {
                "t0_s": float(start - t[0]),
                "pitch_5_15_rms_rad_s": float(np.sqrt(np.mean(pitch[mask] ** 2))),
                "leg_torque_15_100_rms_nm_max": float(
                    np.max(np.sqrt(np.mean(torque[mask] ** 2, axis=0)))
                ),
                "leg_velocity_15_100_rms_rad_s_max": float(
                    np.max(np.sqrt(np.mean(velocity[mask] ** 2, axis=0)))
                ),
            }
        )
    tail = t >= t[-1] - 0.4
    tail_freq = dominant_frequency(t[tail], data["gyro"][tail, 0], 3.0, 40.0)
    hg, r2, _, _ = harmonic(t[tail], data["gyro"][tail, 0], tail_freq)
    return {
        "windows": rows,
        "tail_0p4s_pitch_frequency_hz": tail_freq,
        "tail_0p4s_pitch_amplitude_rad_s": float(abs(hg)),
        "tail_0p4s_pitch_fit_r2": float(r2),
        "end_leg_position_deg": np.rad2deg(data["q"][-50:].mean(axis=0)).tolist(),
        "leg_torque_saturation_fraction": np.mean(
            np.abs(data["torque"][:, :4]) >= 39.999, axis=0
        ).tolist(),
    }


def classify(data):
    """全程使能且输出放行才算稳态；使能但未全程放行的是起身沿。"""
    flags = data["flags"].astype(np.uint32)
    enabled, output = (flags & 1) != 0, (flags & 8) != 0
    if enabled.all() and output.all():
        return "steady"
    if enabled.all():
        return "enable_edge"
    if enabled.any():
        return "partial_enabled"
    return "disabled"


def analyze_run(folder: Path) -> dict:
    """一个变体的一次运行：按段分类并分别统计。"""
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    segments = []
    for path in sorted(folder.glob("burst_*.npz")):
        data, meta = load(path)
        kind = classify(data)
        item = {
            "file": path.name,
            "kind": kind,
            "start_tick_ms": meta["monitor_before_export"]["start_tick"],
        }
        if kind == "steady":
            item.update(steady_metrics(data, meta))
        elif kind == "enable_edge" or (kind == "partial_enabled"):
            item.update(edge_metrics(data))
        segments.append(item)
    return {"run": folder.name, "gains": run["gains"], "segments": segments}


def summarize(results: dict) -> list[dict]:
    """每组取稳态段的均值与范围，供表格与作图。"""
    rows = []
    for name, result in results.items():
        steady = [s for s in result["segments"] if s["kind"] == "steady"]
        row = {"variant": name, "gains": result["gains"], "steady_segments": len(steady)}
        if steady:

            def collect(getter, segments=steady):
                values = np.array([getter(s) for s in segments], dtype=float)
                return {
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }

            row["frequency_hz"] = collect(lambda s: s["frequency_hz"])
            row["pitch_rate_amp"] = collect(lambda s: s["amplitude"]["pitch_rate_rad_s"])
            row["pitch_rate_band_rms"] = collect(lambda s: s["band_rms_5_15"]["pitch_rate_rad_s"])
            row["leg_amp_deg_max"] = collect(lambda s: max(s["amplitude"]["leg_deg"]))
            row["leg_band_rms_deg_max"] = collect(lambda s: max(s["band_rms_5_15"]["leg_deg"]))
            row["wheel_band_rms_max"] = collect(
                lambda s: max(s["band_rms_5_15"]["wheel_velocity_rad_s"])
            )
            row["wheel_target_amp_max"] = collect(
                lambda s: max(s["amplitude"]["wheel_target_rad_s"])
            )
            row["policy_wheel_gain_mean"] = collect(
                lambda s: np.mean(s["policy_pitch_to_wheel_target"]["gain"])
            )
            row["leg_D_power_sum_w"] = collect(lambda s: sum(s["command_power_w"]["D"]))
            row["leg_policy_power_sum_w"] = collect(lambda s: sum(s["command_power_w"]["P_target"]))
            row["wheel_target_power_sum_w"] = collect(
                lambda s: sum(s["command_power_w"]["wheel_target_term"])
            )
            row["wheel_feedback_power_sum_w"] = collect(
                lambda s: sum(s["command_power_w"]["wheel_feedback_term"])
            )
            row["leg_torque_hf_rms_max"] = collect(
                lambda s: max(s["band_rms_15_100"]["leg_torque_nm"])
            )
            row["formula_error_max_nm"] = max(max(s["formula_max_error_nm"]) for s in steady)
        edges = [s for s in result["segments"] if "windows" in s]
        if edges:
            edge = edges[0]
            late = [w for w in edge["windows"] if w["t0_s"] >= 0.5]
            row["edge_late_pitch_5_15_rms"] = float(
                np.mean([w["pitch_5_15_rms_rad_s"] for w in late])
            )
            row["edge_late_leg_torque_hf_rms"] = float(
                np.mean([w["leg_torque_15_100_rms_nm_max"] for w in late])
            )
            row["edge_peak_leg_torque_hf_rms"] = float(
                max(w["leg_torque_15_100_rms_nm_max"] for w in edge["windows"])
            )
            row["edge_tail_frequency_hz"] = edge["tail_0p4s_pitch_frequency_hz"]
            row["edge_tail_pitch_amp"] = edge["tail_0p4s_pitch_amplitude_rad_s"]
            row["edge_tail_fit_r2"] = edge["tail_0p4s_pitch_fit_r2"]
        rows.append(row)
    return rows


def plot(results: dict, summary: list[dict]) -> None:
    """三张图：稳态幅值/频率对比、各组最后一个稳态段时域、使能沿窗内 RMS 随时间。"""
    order = [r["variant"] for r in summary]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = []
    for r in summary:
        g = r["gains"]
        labels.append(f"{r['variant']}\n{g['leg_kp']:g}/{g['leg_kd']:g}/{g['wheel_gain']:g}")
    for ax, key, title in (
        (axes[0, 0], "pitch_rate_band_rms", "pitch rate 5–15 Hz RMS (rad/s)"),
        (axes[0, 1], "leg_band_rms_deg_max", "leg 5–15 Hz RMS, worst joint (deg)"),
        (axes[1, 0], "wheel_band_rms_max", "wheel speed 5–15 Hz RMS, worst wheel (rad/s)"),
        (axes[1, 1], "frequency_hz", "oscillation frequency (Hz)"),
    ):
        for i, r in enumerate(summary):
            if key in r:
                v = r[key]
                ax.errorbar(
                    i,
                    v["mean"],
                    yerr=[[v["mean"] - v["min"]], [v["max"] - v["mean"]]],
                    fmt="o",
                    capsize=6,
                    color="C0",
                )
            else:
                ax.text(
                    i, 0, "no steady\nsegment", ha="center", va="bottom", fontsize=8, color="C3"
                )
        ax.set_xticks(range(len(order)), labels)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Gain sweep 2026-10-14 (leg Kp/Kd/wheel gain), steady segments min–mean–max")
    fig.savefig(OUT / "steady_comparison.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(order), 1, figsize=(12, 2.4 * len(order)), constrained_layout=True)
    for ax, name in zip(axes, order, strict=True):
        folder = next((SWEEP / name).glob("run_*"))
        files = sorted(folder.glob("burst_*.npz"))
        chosen = None
        for path in files:
            data, _ = load(path)
            if classify(data) == "steady":
                chosen = (path, data)
        if chosen is None:
            data, _ = load(files[0])
            chosen = (files[0], data)
        path, data = chosen
        t = data["input_time"] - data["input_time"][0]
        ax.plot(t, data["gyro"][:, 0], color="C0", label="pitch rate (rad/s)")
        ax2 = ax.twinx()
        ax2.plot(t, data["target"][:, 4], color="C1", alpha=0.7, label="LW target (rad/s)")
        ax.set_ylabel("rad/s")
        ax.set_title(f"{name}: {path.name} ({classify(data)})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("s")
    fig.savefig(OUT / "time_series.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for name in order:
        edges = [s for s in results[name]["segments"] if "windows" in s]
        if not edges:
            continue
        w = edges[0]["windows"]
        t0 = [x["t0_s"] + 0.1 for x in w]
        axes[0].plot(t0, [x["pitch_5_15_rms_rad_s"] for x in w], marker="o", label=name)
        axes[1].plot(t0, [x["leg_torque_15_100_rms_nm_max"] for x in w], marker="o", label=name)
    axes[0].set(
        title="enable edge: pitch rate 5–15 Hz RMS (0.2 s windows)",
        xlabel="s after enable",
        ylabel="rad/s",
    )
    axes[1].set(
        title="enable edge: leg torque 15–100 Hz RMS, worst joint",
        xlabel="s after enable",
        ylabel="N·m",
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(OUT / "enable_edge.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in VARIANTS:
        runs = sorted((SWEEP / name).glob("run_*"))
        if runs:
            results[name] = analyze_run(runs[-1])
    reference_data, reference_meta = load(REFERENCE_1012)
    reference = {
        "file": str(REFERENCE_1012),
        "note": "2026-10-12 同策略 60/3/0.2 稳态段，用于核对今天 A 组是否复现",
        **steady_metrics(reference_data, reference_meta),
    }
    summary = summarize(results)
    (OUT / "analysis.json").write_text(
        json.dumps(
            {"variants": results, "summary": summary, "reference_20261012": reference},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot(results, summary)
    print(
        json.dumps(
            {
                "summary": summary,
                "reference_20261012_frequency_hz": reference["frequency_hz"],
                "reference_20261012_pitch_band_rms": reference["band_rms_5_15"]["pitch_rate_rad_s"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
