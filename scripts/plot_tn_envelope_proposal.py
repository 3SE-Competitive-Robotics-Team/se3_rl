"""DM-8009P 腿部电机 T-N 包络：当前训练口径与拟改口径对比（四象限，同一幅图）。

当前：mjlab DcMotor 限矩 min(stall·(1−v/v0), effort_limit)，stall 填峰值 40、effort_limit 填额定 20，
恒扭矩区被额定 20 N·m 削平。
拟改：同一公式，参数按物理含义取——stall 取电压限零速截距 V·Kt·N/R（约 132 N·m），effort_limit 取峰值 40（电流限），
即 40 N·m 平台到转折速度后沿反电动势线下降、空载速度处过零。
拟改 80%：拟改包络力矩整体 ×0.8（stall、effort_limit 同乘），平台 32 N·m，转折与空载速度不变——仍是同一公式。

用法:
    uv run python scripts/plot_tn_envelope_proposal.py [--out .scratch/tn_envelope_proposal.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from se3_shared.motor import DM8009P

# 默认站姿下 300 N 膝气弹簧在每根主动杆上的等效力矩（se3_shared.fourbar，M52 文档）
_SPRING_TORQUE_AT_DEFAULT_POSE = 13.63
# 达妙 2025 年产品选型手册 p18 性能曲线：24 V、定速 100 rpm 扫负载，测到 25 N·m 转速仍保持（该图是唯一公开的负载数据）
_OFFICIAL_TORQUE_AT_100RPM = 25.0
# 用户定的安全系数：拟改包络力矩整体打八折（2026-09-30）
_PROPOSAL_SCALE = 0.8


def dc_motor_envelope(
    vel: np.ndarray, stall: float, no_load: float, effort_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """与 mjlab dc_motor_clip 同式的上下界。"""
    vel_at_limit = no_load * (1.0 + effort_limit / stall)
    v = np.clip(vel, -vel_at_limit, vel_at_limit)
    top = np.minimum(stall * (1.0 - v / no_load), effort_limit)
    bottom = np.maximum(stall * (-1.0 - v / no_load), -effort_limit)
    return top, bottom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path(".scratch/tn_envelope_proposal.png"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    m = DM8009P
    v0 = m.no_load_speed
    peak = 40.0
    # 电压限零速截距 N·Kt·V/R，与训练端 robot_cfg 同一来源
    t_v_kt = m.voltage_limited_stall_torque
    k = _PROPOSAL_SCALE

    vel = np.linspace(-1.65 * v0, 1.65 * v0, 2001)
    cur_top, cur_bot = dc_motor_envelope(vel, m.stall_torque, v0, m.rated_torque)
    new_top, new_bot = dc_motor_envelope(vel, t_v_kt, v0, peak)
    alt_top, alt_bot = dc_motor_envelope(vel, k * t_v_kt, v0, k * peak)

    cur_corner = v0 * (1.0 - m.rated_torque / m.stall_torque)
    new_corner = v0 * (1.0 - peak / t_v_kt)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.fill_between(vel, new_bot, new_top, color="#e4572e", alpha=0.10, lw=0)
    ax.fill_between(vel, cur_bot, cur_top, color="#2e86ab", alpha=0.22, lw=0)
    ax.plot(
        vel,
        cur_top,
        color="#2e86ab",
        lw=2.4,
        label=f"当前：stall {m.stall_torque:g}、effort_limit {m.rated_torque:g}（额定）",
    )
    ax.plot(vel, cur_bot, color="#2e86ab", lw=2.4)
    ax.plot(
        vel,
        new_top,
        color="#e4572e",
        lw=2.4,
        label=f"拟改：stall {t_v_kt:.0f}（V·Kt/R）、effort_limit {peak:g}（峰值）",
    )
    ax.plot(vel, new_bot, color="#e4572e", lw=2.4)
    ax.plot(
        vel,
        alt_top,
        color="#e4572e",
        lw=2.4,
        ls="--",
        label=f"拟改 ×{k:g}：stall {k * t_v_kt:.0f}、effort_limit {k * peak:g}（平台 {k * peak:g}）",
    )
    ax.plot(vel, alt_bot, color="#e4572e", lw=2.4, ls="--")

    ax.axhline(0, color="black", lw=1.0)
    ax.axvline(0, color="black", lw=1.0)
    for y, text in ((m.rated_torque, "额定 20"), (k * peak, f"{k * peak:g}"), (peak, "峰值 40")):
        for s in (1, -1):
            ax.axhline(s * y, color="gray", lw=0.7, ls=":")
        ax.text(vel[-1], y + 0.6, text, ha="right", va="bottom", fontsize=9, color="gray")
    ax.axhspan(
        -_SPRING_TORQUE_AT_DEFAULT_POSE,
        _SPRING_TORQUE_AT_DEFAULT_POSE,
        color="#6a994e",
        alpha=0.07,
        lw=0,
    )
    ax.text(
        -13.0,
        _SPRING_TORQUE_AT_DEFAULT_POSE + 0.5,
        f"±{_SPRING_TORQUE_AT_DEFAULT_POSE} 默认站姿气弹簧等效力矩",
        fontsize=9,
        color="#6a994e",
    )
    for x in (v0, -v0):
        ax.axvline(x, color="gray", lw=0.7, ls=":")
    ax.text(v0 + 0.2, -47, f"空载 {v0:.2f} rad/s", fontsize=9, color="gray")

    for corner, color, y, dy in (
        (cur_corner, "#2e86ab", m.rated_torque, -6.0),
        (new_corner, "#e4572e", peak, 4.5),
        (new_corner, "#e4572e", k * peak, -7.0),
    ):
        for s in (1, -1):
            ax.plot([s * corner], [s * y], "o", color=color, ms=6, zorder=5)
        ax.annotate(
            f"转折 {corner:.1f} rad/s",
            (corner, y),
            xytext=(corner + 1.0, y + dy),
            fontsize=9,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )

    # 达妙 2025 选型手册 p18：24 V 定速 100 rpm 扫负载到 25 N·m，转速全程保持——该转速下至少 25 N·m 可用
    official_speed = 100.0 * 2.0 * np.pi / 60.0
    ax.plot(
        [official_speed], [_OFFICIAL_TORQUE_AT_100RPM], marker="*", color="#222222", ms=16, zorder=6
    )
    ax.annotate(
        f"官方实测：100 rpm（{official_speed:.2f} rad/s）下至少 {_OFFICIAL_TORQUE_AT_100RPM:g} N·m",
        (official_speed, _OFFICIAL_TORQUE_AT_100RPM),
        xytext=(official_speed - 16.0, _OFFICIAL_TORQUE_AT_100RPM + 4.0),
        fontsize=9,
        color="#222222",
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 0.9},
    )

    quad = [
        (0.66, 0.93, "I  正转驱动"),
        (0.34, 0.93, "II  反转制动"),
        (0.34, 0.05, "III  反转驱动"),
        (0.66, 0.05, "IV  正转制动"),
    ]
    for x, y, text in quad:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            color="#444",
            weight="bold",
        )

    ax.set_xlim(vel[0], vel[-1])
    ax.set_ylim(-52, 52)
    ax.set_xlabel("输出轴转速 ω (rad/s)")
    ax.set_ylabel("输出轴力矩 τ (N·m)")
    ax.set_title(
        "DM-J8009P V1.0 @24V 腿部电机 T-N 包络：当前 / 拟改 / 拟改 ×0.8（mjlab DcMotor 同一公式）"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.11), fontsize=10, framealpha=0.95)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(
        f"已写入 {args.out}；T_v={t_v_kt:.1f}、×{k:g} 后 {k * t_v_kt:.1f}；转折 当前 {cur_corner:.2f} / 拟改 {new_corner:.2f} rad/s"
    )


if __name__ == "__main__":
    main()
