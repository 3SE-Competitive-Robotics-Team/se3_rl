"""绘制共享配置中的 M3508+C620 14:1 输出轴 T-N 包络。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from se3_shared.motor import M3508_C620_14


def main() -> None:
    """生成当前训练和 sim2sim 共用的 T-N 曲线图。"""

    out_dir = Path("logs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m3508_c620_14to1_tn_curve.png"

    speed = np.linspace(0.0, M3508_C620_14.no_load_speed, 400)
    torque = M3508_C620_14.torque_limit_np(speed)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(torque, speed, linewidth=2.6, color="black")
    ax.set_title("M3508 + C620 Output-Shaft T-N Envelope (14:1)")
    ax.set_xlabel("Output torque (N m)")
    ax.set_ylabel("Output angular velocity (rad/s)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.03,
        "Digitized from the C620 current-loop load curve and converted from 19:1 to 14:1.",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
