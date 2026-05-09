"""可视化 M3508-Hexroll 和 DM-8009P 的 T-N 包络线。

用法:
    uv run python scripts/plot_tn_envelope.py
"""

from __future__ import annotations

import numpy as np

from se3_shared.motor import DM8009P, M3508_HEXROLL, MotorSpec


def _tn_envelope(
    motor: MotorSpec,
    vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 T-N 包络线上下界 — 与 DcMotorActuatorCfg._clip_effort 一致。"""
    sat = motor.stall_torque
    vlim = motor.no_load_speed
    elim = motor.rated_torque

    vel_at_effort_lim = vlim * (1.0 + elim / sat)
    vel_clipped = np.clip(vel, -vel_at_effort_lim, vel_at_effort_lim)

    top = sat * (1.0 - vel_clipped / vlim)
    bottom = sat * (-1.0 - vel_clipped / vlim)

    max_eff = np.minimum(top, elim)
    min_eff = np.maximum(bottom, -elim)
    return max_eff, min_eff


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, motor in zip(axes, [M3508_HEXROLL, DM8009P], strict=True):
        vel = np.linspace(-motor.no_load_speed * 1.2, motor.no_load_speed * 1.2, 500)
        top, bottom = _tn_envelope(motor, vel)

        ax.fill_between(vel, bottom, top, alpha=0.25, color="steelblue", label="T-N envelope")
        ax.plot(vel, top, "b-", linewidth=1.5)
        ax.plot(vel, bottom, "b-", linewidth=1.5)

        ax.axhline(
            motor.rated_torque,
            color="green",
            linestyle="--",
            linewidth=1,
            label=f"effort_limit = {motor.rated_torque} Nm",
        )
        ax.axhline(-motor.rated_torque, color="green", linestyle="--", linewidth=1)
        ax.axhline(
            motor.stall_torque,
            color="red",
            linestyle=":",
            linewidth=1,
            label=f"stall_torque = {motor.stall_torque} Nm",
        )
        ax.axhline(-motor.stall_torque, color="red", linestyle=":", linewidth=1)
        ax.axvline(
            motor.no_load_speed,
            color="orange",
            linestyle=":",
            linewidth=1,
            label=f"no_load_speed = {motor.no_load_speed:.1f} rad/s",
        )
        ax.axvline(-motor.no_load_speed, color="orange", linestyle=":", linewidth=1)

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        ax.set_xlabel("Joint velocity (rad/s)")
        ax.set_ylabel("Torque (Nm)")
        ax.set_title(f"{motor.name}\ngear={motor.gear_ratio:.2f}:1")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scripts/tn_envelope.png", dpi=150)
    print("Saved: scripts/tn_envelope.png")


if __name__ == "__main__":
    main()
