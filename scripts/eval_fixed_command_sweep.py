"""固定平地指令扫描，保存原生 runtime 时序与高度振荡指标。"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from se3_runtime import PolicyControlLoop, PolicyRuntime
from se3_runtime_mujoco import MujocoPolicyAdapter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".scratch/command_sweep400"
HEIGHTS = [0.22, 0.30, 0.34, 0.36, 0.38]
SPEEDS = [0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4]
MODELS = {
    "A22": ROOT
    / "logs/rsl_rl/SE3-WheelLegged-Rough-A22-StairCommand/2026-09-12_19-18-15_rough-A22-stopgate-seed42-3x8192-1k/onnx/model_400.onnx",
    "A23": ROOT
    / "logs/rsl_rl/SE3-WheelLegged-Rough-A23-StairNoGate/2026-09-12_19-26-09_rough-A23-nogate-seed42-3x8192-1k/onnx/model_400.onnx",
}


def spectrum(values, dt):
    """去线性趋势后估计频率，谱峰只作周期线索。"""
    t = np.arange(len(values)) * dt
    x = values - np.polyval(np.polyfit(t, values, 1), t)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), dt)
    valid = (freq >= 0.3) & (freq <= 10)
    idx = np.flatnonzero(valid)[np.argmax(power[valid])]
    total = power[valid].sum()
    return float(freq[idx]), float(
        power[max(1, idx - 1) : idx + 2].sum() / total
    ) if total > 1e-20 else 0


def run_case(case):
    """每个组合独立复位，指令在目标阶段保持不变。"""
    tag, height, speed = case
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    path = MODELS[tag]
    rt = PolicyRuntime.load(path, session_options=opts, action_delay_random_seed=42)
    ad = MujocoPolicyAdapter(
        rt.contract,
        artifact_path=path,
        model_path=ROOT
        / "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml",
        solver_dt_s=0.005,
    )
    loop = PolicyControlLoop(rt, ad)
    loop.reset()
    dt = rt.contract.timing.policy_dt_s
    base = mujoco.mj_name2id(ad.model, mujoco.mjtObj.mjOBJ_BODY, rt.contract.robot.base_link)
    ad.set_command_field("velocity_height", "height", height)
    ad.set_command_field("velocity_height", "lin_vel_x", 0)
    for _ in range(round(8 / dt)):
        loop.policy_step()
    ad.set_command_field("velocity_height", "lin_vel_x", speed)
    rows = []
    failed = False
    for step in range(round(25 / dt)):
        result = loop.policy_step()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(ad.model, ad.data, mujoco.mjtObj.mjOBJ_XBODY, base, vel, 1)
        rot = ad.data.xmat[base].reshape(3, 3)
        pitch = np.degrees(np.arcsin(np.clip(-rot[2, 0], -1, 1)))
        row = [(step + 1) * dt, ad.data.xpos[base, 2], vel[3], pitch, *result.inference.raw_action]
        rows.append(row)
        if not np.isfinite(row).all() or rot[2, 2] < 0.5:
            failed = True
            break
    a = np.asarray(rows)
    tail = a[a[:, 0] > 15]
    if len(tail) < 100:
        tail = a
    z = tail[:, 1]
    vx = tail[:, 2]
    pitch = tail[:, 3]
    freq, concentration = spectrum(z, dt)
    hz = [spectrum(tail[:, j], dt)[0] for j in range(4, 8)]
    name = f"{tag}_h{height:.2f}_v{speed:.1f}"
    np.savez_compressed(OUT / (name + ".npz"), trace=a)
    summary = dict(
        model=tag,
        height=height,
        speed=speed,
        failed=failed,
        seconds=len(a) * dt,
        height_mean=float(z.mean()),
        height_error=float(z.mean() - height),
        height_std=float(z.std()),
        height_p95_p05=float(np.percentile(z, 95) - np.percentile(z, 5)),
        height_peak_to_peak=float(np.ptp(z)),
        vx_mean=float(vx.mean()),
        vx_mae=float(np.abs(vx - speed).mean()),
        vx_std=float(vx.std()),
        pitch_std=float(pitch.std()),
        height_frequency=freq,
        height_spectral_concentration=concentration,
        leg_action_frequency=hz,
        trace=name + ".npz",
    )
    return summary


def main():
    """并行扫描并输出同尺度热图。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [(tag, h, v) for tag in MODELS for h in HEIGHTS for v in SPEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_case, c) for c in cases]
        for f in as_completed(futures):
            r = f.result()
            rows.append(r)
            print(
                f"{len(rows)}/{len(cases)} {r['model']} h={r['height']} v={r['speed']} zrange={r['height_p95_p05']:.4f} vx={r['vx_mean']:.3f}",
                flush=True,
            )
    rows.sort(key=lambda r: (r["model"], r["height"], r["speed"]))
    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    maximum = max(r["height_p95_p05"] * 100 for r in rows)
    for ax, tag in zip(axes, MODELS, strict=True):
        matrix = np.array(
            [
                [
                    next(
                        r["height_p95_p05"] * 100
                        for r in rows
                        if r["model"] == tag and r["height"] == h and r["speed"] == v
                    )
                    for v in SPEEDS
                ]
                for h in HEIGHTS
            ]
        )
        im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=maximum, cmap="YlOrRd")
        ax.set_xticks(range(len(SPEEDS)), SPEEDS)
        ax.set_yticks(range(len(HEIGHTS)), HEIGHTS)
        ax.set_xlabel("Fixed vx command (m/s)")
        ax.set_ylabel("Fixed height command (m)")
        ax.set_title(tag + " / model 400")
        for i in range(len(HEIGHTS)):
            for j in range(len(SPEEDS)):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes, label="Height P95-P05 (cm), final 10 s")
    fig.savefig(OUT / "height_sweep.png", dpi=160)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
