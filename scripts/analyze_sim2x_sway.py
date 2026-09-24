"""分析 record_sim2x_video.py 导出的逐 tick CSV：站立/行进阶段的腿部摆动频率、幅值与平衡分工。

用法：
    uv run python scripts/analyze_sim2x_sway.py --csv 标签=路径 [--csv 标签=路径 ...] --phase stand \
        --window 2,14 [--png out.png]

输出每个标签在指定时间窗内：俯仰角/腿关节角/腿目标/轮速的标准差、峰峰值、主频与主频幅值，
以及腿目标与俯仰角、轮目标与俯仰角的相关系数（判断策略用腿还是用轮做平衡修正）。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

LEG_NAMES = ("LF0", "LB", "RF0", "RB")


def _load(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            out[key] = np.array(
                [float(r[key]) if r[key] not in ("", "nan") else np.nan for r in rows]
            )
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def _spectrum(x: np.ndarray, fs: float) -> tuple[float, float]:
    """去均值后的单边幅值谱峰：返回（主频 Hz，主频正弦幅值）。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    n = x.size
    if n < 16:
        return float("nan"), float("nan")
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window)) * 2.0 / window.sum()
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return float(freqs[k]), float(spec[k])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def analyze(
    tag: str, data: dict[str, np.ndarray], t0: float, t1: float, fs: float
) -> dict[str, float]:
    m = (data["t"] >= t0) & (data["t"] <= t1)
    pitch = data["pitch_deg"][m]
    q = np.stack([np.degrees(data[f"q{i}"][m]) for i in range(4)], axis=1)
    tgt = np.stack([np.degrees(data[f"leg_target{i}"][m]) for i in range(4)], axis=1)
    wl, wr = data["wheel_l"][m], data["wheel_r"][m]
    wt = np.stack([data[f"wheel_target{i}"][m] for i in range(2)], axis=1)
    f_pitch, a_pitch = _spectrum(pitch, fs)
    f_q, a_q = _spectrum(q[:, 0], fs)
    f_tgt, a_tgt = _spectrum(tgt[:, 0], fs)
    f_w, a_w = _spectrum(wl, fs)
    out = {
        "pitch_mean": float(pitch.mean()),
        "pitch_std": float(pitch.std()),
        "pitch_pp": float(np.ptp(pitch)),
        "pitch_f": f_pitch,
        "pitch_amp": a_pitch,
        "q_std_mean": float(q.std(axis=0).mean()),
        "q_pp_max": float(np.ptp(q, axis=0).max()),
        "q_f": f_q,
        "q_amp": a_q,
        "tgt_std_mean": float(tgt.std(axis=0).mean()),
        "tgt_step_mean": float(np.abs(np.diff(tgt, axis=0)).mean()),
        "tgt_f": f_tgt,
        "tgt_amp": a_tgt,
        "wheel_std": float(np.concatenate([wl, wr]).std()),
        "wheel_f": f_w,
        "wheel_amp": a_w,
        "wheel_tgt_step_mean": float(np.abs(np.diff(wt, axis=0)).mean()),
        "corr_pitch_legtgt": float(np.mean([abs(_corr(pitch, tgt[:, i])) for i in range(4)])),
        "corr_pitch_wheeltgt": float(np.mean([abs(_corr(pitch, wt[:, i])) for i in range(2)])),
        "base_z_std_mm": float(data["base_z"][m].std() * 1000.0),
    }
    print(
        f"{tag:6s} 俯仰 {out['pitch_mean']:5.2f}±{out['pitch_std']:.2f}° 峰峰 {out['pitch_pp']:.2f}° 主频 {out['pitch_f']:.2f} Hz 幅 {out['pitch_amp']:.2f}° | "
        f"腿角 σ {out['q_std_mean']:.2f}° 峰峰 {out['q_pp_max']:.2f}° 主频 {out['q_f']:.2f} Hz 幅 {out['q_amp']:.2f}° | "
        f"腿目标 σ {out['tgt_std_mean']:.2f}° 每步 {out['tgt_step_mean']:.2f}° 主频 {out['tgt_f']:.2f} Hz | "
        f"轮速 σ {out['wheel_std']:.2f} 主频 {out['wheel_f']:.2f} Hz 幅 {out['wheel_amp']:.2f} | "
        f"|corr| 俯仰-腿目标 {out['corr_pitch_legtgt']:.2f} 俯仰-轮目标 {out['corr_pitch_wheeltgt']:.2f} | z σ {out['base_z_std_mm']:.1f} mm"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", action="append", required=True, help="标签=CSV 路径")
    parser.add_argument("--window", default="2,14", help="分析时间窗 秒,秒")
    parser.add_argument("--fs", type=float, default=50.0, help="policy tick 频率 Hz")
    parser.add_argument("--png", type=Path, default=None, help="时间序列与频谱图输出")
    parser.add_argument("--plot-seconds", type=float, default=6.0, help="时间序列图显示的秒数")
    args = parser.parse_args()
    t0, t1 = (float(x) for x in args.window.split(","))
    datasets = {}
    for item in args.csv:
        tag, path = item.split("=", 1)
        datasets[tag] = _load(Path(path))
    for tag, data in datasets.items():
        analyze(tag, data, t0, t1, args.fs)

    if args.png is None:
        return 0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 11))
    rows = [
        ("pitch_deg", "pitch [deg]", 1.0),
        ("q0", "LF0 joint [deg]", 180.0 / math.pi),
        ("leg_target0", "LF0 target [deg]", 180.0 / math.pi),
        ("wheel_l", "left wheel [rad/s]", 1.0),
    ]
    for r, (key, label, scale) in enumerate(rows):
        for tag, data in datasets.items():
            m = (data["t"] >= t0) & (data["t"] <= min(t1, t0 + args.plot_seconds))
            axes[r, 0].plot(data["t"][m], data[key][m] * scale, label=tag, linewidth=0.9)
            mm = (data["t"] >= t0) & (data["t"] <= t1)
            x = (data[key][mm] - data[key][mm].mean()) * scale
            n = x.size
            window = np.hanning(n)
            spec = np.abs(np.fft.rfft(x * window)) * 2.0 / window.sum()
            freqs = np.fft.rfftfreq(n, d=1.0 / args.fs)
            axes[r, 1].plot(freqs[1:], spec[1:], label=tag, linewidth=0.9)
        axes[r, 0].set_ylabel(label)
        axes[r, 1].set_xlim(0, args.fs / 2)
        axes[r, 1].set_ylabel("amplitude")
        axes[r, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    fig.suptitle(
        f"sim2x stand window {t0:.0f}-{t1:.0f} s: time series (left) and amplitude spectra (right)"
    )
    fig.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=110)
    print(f"图: {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
