"""汇总固定指令扫描结果并绘制代表性时序。"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.cwd()
out = root / ".scratch/command_sweep400_500hz"
rows = json.loads((out / "results.json").read_text())
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
for col, h in enumerate([0.38, 0.30]):
    for tag in ["A22", "A23"]:
        a = np.load(out / f"{tag}_h{h:.2f}_v1.6.npz")["trace"]
        a = a[a[:, 0] > 15]
        for ax, idx in zip(axes[:, col], [1, 2, 3], strict=True):
            ax.plot(a[:, 0], a[:, idx], label=tag, lw=1.4)
    axes[0, col].axhline(h, color="black", ls="--", lw=1, label="command")
    axes[1, col].axhline(1.6, color="black", ls="--", lw=1)
    axes[0, col].set_title(f"Height command {h:.2f} m / vx command 1.6 m/s")
    axes[0, col].set_ylabel("Actual height (m)")
    axes[1, col].set_ylabel("Forward velocity (m/s)")
    axes[2, col].set_ylabel("Pitch (deg)")
    axes[2, col].set_xlabel("Time after speed command (s)")
    for ax in axes[:, col]:
        ax.grid(alpha=0.25)
    axes[0, col].legend()
fig.savefig(out / "representative_traces.png", dpi=160)
lines = [
    "# A22/A23 model400 固定平地指令扫描",
    "",
    "两组各35个组合，分别运行200 Hz与500 Hz积分，共140条独立轨迹。策略频率50 Hz，控制tick沿用契约200 Hz。使用原生MuJoCo与纯Python se3_runtime，不运行viewer。场景为serialleg_closed_chain_v3_train_obb_trim.xml无限平面；height=[.22,.30,.34,.36,.38]，vx=[0,.2,.4,.8,1.2,1.6,2.4]。每次独立reset，在目标高度零速8秒，然后保持目标速度25秒；统计最后10秒。未进行低姿运动后升高路径或域随机化鲁棒性评测。",
    "",
    "幅度定义为高度P95−P05，不是正弦单边振幅；频率由末10秒去趋势频谱估计，分辨率0.1Hz。保存全部时序：时间、实际高度、机身系前向速度、俯仰、6维raw action。",
    "",
    "## 主要结果（500 Hz，与当前viewer启动参数一致）",
    "",
    "| 模型 | 高度指令m | 速度指令m/s | 实际均高m | 高度变化cm | 实际均速m/s | 高度主频Hz |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for h, v in [(0.38, 1.6), (0.38, 1.2), (0.36, 1.6), (0.30, 0.8), (0.30, 1.6), (0.30, 2.4)]:
    for tag in ["A22", "A23"]:
        r = next(x for x in rows if x["model"] == tag and x["height"] == h and x["speed"] == v)
        lines.append(
            f"| {tag} | {h:.2f} | {v:.1f} | {r['height_mean']:.4f} | {r['height_p95_p05'] * 100:.3f} | {r['vx_mean']:.3f} | {r['height_frequency']:.1f} |"
        )
lines += [
    "",
    "A23 .38/1.6高度主频.8Hz，四个腿动作主频也是.8Hz；200Hz该组合幅度1.476cm、500Hz1.511cm。A23 .30/.8–2.4在两种频率下均持续大幅蹲伸并多数组合超速。支持闭环周期运动，不证明某个奖励项或门控是唯一根因。A22本次高姿扫描未出现同等幅度，但低姿速度跟踪仍差。两组均没有倾角超过60度或非有限值提前退出。",
    "",
    "静站后启动路径不覆盖用户所有手动切换历史；不能将未复现的组合解释为绝对稳定。高度稳定也不等于跟踪正确，需要同时看均高与速度误差。",
    "",
    "200Hz数据：../command_sweep400/results.json；500Hz数据：results.json；各组合原始时序：同目录npz。未修改训练、runtime或用户viewer，未提交代码。",
]
(out / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[8:24]))
