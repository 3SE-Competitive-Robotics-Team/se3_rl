"""膝关节弹簧与四连杆传动机构可视化。

用 MuJoCo FK 计算真实关节位置，保证姿态精确。
四连杆 ABCD 通过 Freudenstein 闭环方程精确求解，
P₁/P₂ 分别固连在驱动杆 AB 和小腿上段 CD 上。

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

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml"
WHEEL_RADIUS = 0.059
DEFAULT_HIP = 0.6171
DEFAULT_KNEE = 0.2070
KNEE_RANGE = (-0.6, 0.8)

# 四连杆参数（占位，需从 CAD 提供）
L_AB = 0.06
L_BC = 0.08
L_CD = 0.04
L_AD = 0.18

# CD 杆件相对小腿方向的固定偏角（占位，需从 CAD 提供）
# CD 从膝轴 D 指向连杆铰接点 C，位于小腿"后侧"（朝机身方向）
# 此角定义为 CD 方向相对于 DW（小腿方向）的偏移，正值 = 逆时针
PSI_OFFSET = np.deg2rad(-90)

# 四连杆装配构型：+1 或 -1，对应两种闭合模式
FOURBAR_MODE = +1

# 弹簧挂点（占位）
L_P1 = 0.014
L_P2 = 0.015
ALPHA_EXT = np.deg2rad(5)
BETA_EXT = np.deg2rad(35)

# 弹簧力学（占位）
K_SPRING = 900.0
S0 = 0.06
DELTA_0 = 0.004
DELTA_1 = 0.0095


def _fourbar_solve_phi(
    psi: float,
    l_ab: float,
    l_bc: float,
    l_cd: float,
    l_ad: float,
    mode: int = +1,
) -> float:
    """Freudenstein 半角代换法求解驱动杆角 φ。

    在大腿局部坐标系中（A 在原点，D 在 [L_AD, 0]），
    已知小腿上段角 ψ，求驱动杆角 φ。
    所有角度相对大腿方向（A→D = +x）逆时针为正。
    """
    p = 2 * l_ab * (l_cd * np.cos(psi) - l_ad)
    q = 2 * l_ab * l_cd * np.sin(psi)
    r = l_bc**2 - l_ab**2 - l_ad**2 - l_cd**2 + 2 * l_ad * l_cd * np.cos(psi)

    disc = p**2 + q**2 - r**2
    if disc < -1e-3:
        raise ValueError(f"四连杆超出运动范围: disc={disc:.6f}, psi={np.rad2deg(psi):.1f}°")
    disc = max(disc, 0.0)

    return float(2 * np.arctan2(q + mode * np.sqrt(disc), p + r))


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    """按关节名写入 qpos，避免受四连杆虚拟关节数量影响。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"未找到关节: {joint_name}")
    data.qpos[model.jnt_qposadr[jid]] = value


def _mujoco_fk(theta_hip: float, theta_knee: float) -> dict[str, np.ndarray]:
    """用 MuJoCo 计算精确 body 世界坐标，取 XZ 侧视图。"""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for joint_name, value in (
        ("lf0_Joint", theta_hip),
        ("lf1_Joint", theta_knee),
        ("l_wheel_Joint", 0.0),
        ("rf0_Joint", theta_hip),
        ("rf1_Joint", theta_knee),
        ("r_wheel_Joint", 0.0),
    ):
        _set_joint_qpos(model, data, joint_name, value)
    mujoco.mj_forward(model, data)

    def xz(body_name: str) -> np.ndarray:
        bid = model.body(body_name).id
        pos = data.xpos[bid]
        return np.array([pos[0], pos[2]])

    a = xz("lf0_Link")
    d = xz("lf1_Link")
    w = xz("l_wheel_Link")

    thigh_vec = d - a
    thigh_angle = np.arctan2(thigh_vec[1], thigh_vec[0])
    shank_vec = w - d
    shank_angle = np.arctan2(shank_vec[1], shank_vec[0])

    # ψ：CD 杆件在大腿局部坐标系中的方位角
    # CD 固连在小腿上，与小腿方向有固定偏角 PSI_OFFSET
    psi_world = shank_angle + PSI_OFFSET
    psi_local = psi_world - thigh_angle

    # φ：通过 Freudenstein 闭环方程精确求解驱动杆角
    phi_local = _fourbar_solve_phi(psi_local, L_AB, L_BC, L_CD, L_AD, FOURBAR_MODE)
    phi_world = phi_local + thigh_angle

    # B（驱动杆末端）固连在 AB 杆件上
    b = a + L_AB * np.array([np.cos(phi_world), np.sin(phi_world)])
    # C（从动铰）固连在 CD 杆件上
    c = d + L_CD * np.array([np.cos(psi_world), np.sin(psi_world)])

    # P₁ 固连在驱动杆 AB 上，A 侧反向延伸（B 的对侧）
    p1 = a + L_P1 * np.array([-np.cos(phi_world + ALPHA_EXT), -np.sin(phi_world + ALPHA_EXT)])
    # P₂ 固连在小腿上段 CD 上，D 侧反向延伸（C 的对侧）
    p2 = d + L_P2 * np.array([-np.cos(psi_world + BETA_EXT), -np.sin(psi_world + BETA_EXT)])

    # 验证：BC 连杆长度应等于 L_BC
    bc_err = abs(np.linalg.norm(c - b) - L_BC)
    if bc_err > 1e-4:
        print(f"四连杆闭合误差: |BC|={np.linalg.norm(c - b):.4f}, 期望={L_BC}, 误差={bc_err:.6f}")

    return {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "P1": p1,
        "P2": p2,
        "wheel": w,
        "thigh_angle": thigh_angle,
        "shank_angle": shank_angle,
    }


def draw(theta_hip: float, theta_knee: float) -> None:
    """单图：整腿侧视图 + 四连杆 + 弹簧。"""
    pts = _mujoco_fk(theta_hip, theta_knee)
    a, b, c, d = pts["A"], pts["B"], pts["C"], pts["D"]
    p1, p2, w = pts["P1"], pts["P2"], pts["wheel"]

    dp = p2 - p1
    slen = np.linalg.norm(dp)
    s_eff = slen - DELTA_0 - DELTA_1
    force = K_SPRING * (S0 - s_eff) if slen > 1e-6 else 0.0

    _fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(
        f"SerialLeg 膝关节传动与弹簧 | "
        f"hip={np.rad2deg(theta_hip):.1f}\u00b0 knee={np.rad2deg(theta_knee):.1f}\u00b0 | "
        f"F={force:.1f} N",
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
        sp_s = p1 + s_dir * DELTA_0
        sp_e = p2 - s_dir * DELTA_1
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
        "四连杆 (需 CAD):\n"
        f"  AB = {L_AB * 1000:.0f} mm\n"
        f"  BC = {L_BC * 1000:.0f} mm\n"
        f"  CD = {L_CD * 1000:.0f} mm\n"
        f"  AD = {L_AD * 1000:.0f} mm\n"
        f"  ψ_off = {np.rad2deg(PSI_OFFSET):.0f}\u00b0\n"
        "\n"
        "弹簧 (需 CAD):\n"
        f"  a(P1) = {L_P1 * 1000:.1f} mm\n"
        f"  b(P2) = {L_P2 * 1000:.1f} mm\n"
        f"  k = {K_SPRING:.0f} N/m\n"
        f"  s0 = {S0 * 1000:.1f} mm"
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
