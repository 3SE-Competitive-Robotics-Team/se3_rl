"""膝关节弹簧与四连杆传动机构可视化。

用 MuJoCo FK 计算真实关节位置，保证姿态精确。
四连杆 ABCD 和弹簧挂点 P₁/P₂ 均直接读取正式 OBB 闭链 MJCF。

用法:
    uv run python scripts/plot_spring_geometry.py
    uv run python scripts/plot_spring_geometry.py --theta-hip 0.6171 --theta-knee 0.207
"""

from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from se3_shared import (
    JointGroup,
    output_to_policy_pos_np,
    policy_to_closedchain_passive_pos_np,
)

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "PingFang HK",
    "Heiti TC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
]
mpl.rcParams["axes.unicode_minus"] = False

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
WHEEL_RADIUS = 0.060
DEFAULT_HIP = 0.6171
DEFAULT_KNEE = 0.2070
KNEE_RANGE = (-0.6, 0.8)


def _set_joint_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float
) -> None:
    """按关节名写入 qpos，避免受四连杆虚拟关节数量影响。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"未找到关节: {joint_name}")
    data.qpos[model.jnt_qposadr[jid]] = value


def _mujoco_fk(theta_hip: float, theta_knee: float) -> dict[str, np.ndarray]:
    """把旧输出轴输入映射到闭链模型，再读取真实 XZ 几何。"""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    output_pos = np.asarray(
        [[-theta_hip, -theta_knee, theta_hip, theta_knee]],
        dtype=np.float64,
    )
    policy_pos = output_to_policy_pos_np(output_pos)[0]
    passive_pos = policy_to_closedchain_passive_pos_np(policy_pos)
    for joint_name, value in zip(JointGroup.POLICY_LEG_NAMES, policy_pos, strict=True):
        _set_joint_qpos(model, data, joint_name, float(value))
    for joint_name, value in zip(
        JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES,
        passive_pos,
        strict=True,
    ):
        _set_joint_qpos(model, data, joint_name, float(value))
    for joint_name in JointGroup.WHEEL_NAMES:
        _set_joint_qpos(model, data, joint_name, 0.0)
    mujoco.mj_forward(model, data)

    def xz(value: np.ndarray) -> np.ndarray:
        return np.asarray([value[0], value[2]], dtype=np.float64)

    a = xz(data.xanchor[model.joint("lf0_Joint").id])
    b = xz(data.xpos[model.body("l_coupler_Link").id])
    c = xz(data.site_xpos[model.site("l_coupler_end").id])
    d = xz(data.xanchor[model.joint("lf1_Joint").id])
    w = xz(data.xpos[model.body("l_wheel_Link").id])
    p1 = xz(data.site_xpos[model.site("l_spring_p1").id])
    p2 = xz(data.site_xpos[model.site("l_spring_p2").id])
    closure = xz(data.site_xpos[model.site("lf_coupler_closure").id])
    closure_error = float(np.linalg.norm(c - closure))
    if closure_error > 1.0e-5:
        raise RuntimeError(f"闭链投影失败: closure_error={closure_error:.6e} m")

    return {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "P1": p1,
        "P2": p2,
        "wheel": w,
        "closure_error": np.asarray(closure_error),
    }


def draw(theta_hip: float, theta_knee: float) -> None:
    """单图：整腿侧视图 + 四连杆 + 弹簧。"""
    pts = _mujoco_fk(theta_hip, theta_knee)
    a, b, c, d = pts["A"], pts["B"], pts["C"], pts["D"]
    p1, p2, w = pts["P1"], pts["P2"], pts["wheel"]

    dp = p2 - p1
    slen = np.linalg.norm(dp)
    closure_error = float(pts["closure_error"])

    _fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(
        f"SerialLeg 膝关节传动与弹簧 | "
        f"hip={np.rad2deg(theta_hip):.1f}\u00b0 knee={np.rad2deg(theta_knee):.1f}\u00b0 | "
        f"挂点距离={slen * 1000:.1f} mm",
        fontsize=12,
    )

    # 大腿
    ax.plot(
        [a[0], d[0]],
        [a[1], d[1]],
        "k-",
        linewidth=6,
        solid_capstyle="round",
        label="大腿",
        zorder=2,
    )
    # 小腿
    ax.plot(
        [d[0], w[0]],
        [d[1], w[1]],
        color="0.3",
        linewidth=5,
        solid_capstyle="round",
        label="小腿",
        zorder=2,
    )
    # 驱动杆 (AB)
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color="darkorange",
        linewidth=3,
        solid_capstyle="round",
        label="驱动杆 (AB)",
        zorder=3,
    )
    # 连杆 (BC)
    ax.plot(
        [b[0], c[0]],
        [b[1], c[1]],
        color="purple",
        linewidth=3,
        solid_capstyle="round",
        label="连杆 (BC)",
        zorder=3,
    )
    # 小腿上段 (CD)
    ax.plot(
        [d[0], c[0]],
        [d[1], c[1]],
        color="teal",
        linewidth=3,
        solid_capstyle="round",
        label="小腿上段 (CD)",
        zorder=3,
    )

    # 弹簧锯齿线
    if slen > 1e-6:
        n_coils = 12
        s_dir = dp / slen
        s_perp = np.array([-s_dir[1], s_dir[0]])
        amp = 0.005
        sp_s = p1
        sp_e = p2
        seg = np.linalg.norm(sp_e - sp_s)
        coil_pts = [p1, sp_s]
        for i in range(1, n_coils * 2 + 1):
            t = i / (n_coils * 2 + 1)
            ct = sp_s + s_dir * seg * t
            sign = 1 if i % 2 == 1 else -1
            coil_pts.append(ct + sign * s_perp * amp)
        coil_pts.extend([sp_e, p2])
        carr = np.array(coil_pts)
        ax.plot(carr[:, 0], carr[:, 1], "b-", linewidth=2, label="弹簧", zorder=4)

    # 轮子
    wc = mpatches.Circle(w, WHEEL_RADIUS, fill=False, edgecolor="0.5", linewidth=1.5, zorder=2)
    ax.add_patch(wc)
    ax.plot(*w, "+", color="0.5", markersize=8, markeredgewidth=1.5)

    # 地面
    ground_y = w[1] - WHEEL_RADIUS
    xlims = [min(a[0], d[0], w[0]) - 0.06, max(a[0], d[0], w[0]) + 0.1]
    ax.plot(xlims, [ground_y, ground_y], "k-", linewidth=2)
    ax.fill_between(xlims, ground_y - 0.01, ground_y, color="0.85")

    # 铰接点
    for pt, clr, sz in [(a, "red", 140), (d, "red", 140), (b, "darkorange", 90), (c, "purple", 90)]:
        ax.scatter(*pt, s=sz, c=clr, zorder=10, edgecolors="k", linewidths=1)
    ax.scatter(*p1, s=70, c="blue", marker="s", zorder=10, edgecolors="k", linewidths=1)
    ax.scatter(*p2, s=70, c="green", marker="s", zorder=10, edgecolors="k", linewidths=1)

    # 标注
    ax.annotate(
        "A (髋轴)",
        a,
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=9,
        color="red",
        fontweight="bold",
    )
    ax.annotate(
        "D (膝轴)",
        d,
        xytext=(-60, -5),
        textcoords="offset points",
        fontsize=9,
        color="red",
        fontweight="bold",
    )
    ax.annotate(
        "B",
        b,
        xytext=(8, 5),
        textcoords="offset points",
        fontsize=9,
        color="darkorange",
        fontweight="bold",
    )
    ax.annotate(
        "C",
        c,
        xytext=(8, 5),
        textcoords="offset points",
        fontsize=9,
        color="purple",
        fontweight="bold",
    )
    ax.annotate(
        "P1",
        p1,
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=9,
        color="blue",
        fontweight="bold",
    )
    ax.annotate(
        "P2",
        p2,
        xytext=(-20, -12),
        textcoords="offset points",
        fontsize=9,
        color="green",
        fontweight="bold",
    )
    ax.annotate(
        f"轮 R={WHEEL_RADIUS * 1000:.0f}mm",
        w,
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color="0.5",
    )

    # 尺寸标注
    thigh_len = np.linalg.norm(d - a)
    shank_len = np.linalg.norm(w - d)
    ax.annotate(
        f"{thigh_len * 1000:.0f}mm",
        (a + d) / 2,
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=8,
        color="0.4",
    )
    ax.annotate(
        f"{shank_len * 1000:.0f}mm",
        (d + w) / 2,
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=8,
        color="0.4",
    )

    # 参数框
    info = (
        f"大腿 = {thigh_len * 1000:.0f} mm\n"
        f"小腿 = {shank_len * 1000:.0f} mm\n"
        f"轮 R = {WHEEL_RADIUS * 1000:.0f} mm\n"
        f"hip0 = {np.rad2deg(DEFAULT_HIP):.1f}\u00b0\n"
        f"knee0 = {np.rad2deg(DEFAULT_KNEE):.1f}\u00b0\n"
        f"膝范围 [{np.rad2deg(KNEE_RANGE[0]):.0f}\u00b0, {np.rad2deg(KNEE_RANGE[1]):.0f}\u00b0]\n"
        "\n"
        "正式 OBB 闭链几何:\n"
        f"  AB = {np.linalg.norm(b - a) * 1000:.1f} mm\n"
        f"  BC = {np.linalg.norm(c - b) * 1000:.1f} mm\n"
        f"  CD = {np.linalg.norm(c - d) * 1000:.1f} mm\n"
        f"  AD = {np.linalg.norm(d - a) * 1000:.1f} mm\n"
        f"  closure error = {closure_error * 1e6:.2f} μm\n"
        "\n"
        "MJCF 弹簧挂点:\n"
        f"  |P1P2| = {slen * 1000:.1f} mm"
    )
    ax.text(
        0.98,
        0.98,
        info,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9},
    )

    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_xlim(xlims)
    ax.set_ylim(ground_y - 0.02, max(a[1], b[1]) + 0.04)

    plt.tight_layout()
    out_path = "scripts/spring_geometry.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存到 {out_path}")
    if mpl.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="膝关节传动机构与弹簧可视化")
    parser.add_argument("--theta-hip", type=float, default=DEFAULT_HIP)
    parser.add_argument("--theta-knee", type=float, default=DEFAULT_KNEE)
    args = parser.parse_args()
    draw(args.theta_hip, args.theta_knee)


if __name__ == "__main__":
    main()
