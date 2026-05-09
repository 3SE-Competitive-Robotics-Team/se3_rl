"""膝关节弹簧几何关系可视化 — 展示弹簧挂点、连杆和需要提供的参数。

在机器人初始姿态（default_dof_pos）下绘制单腿侧视图，标注弹簧相关的
9 个几何/力学参数，帮助从 CAD 或实物中定位测量对象。

用法:
    uv run python scripts/plot_spring_geometry.py
    uv run python scripts/plot_spring_geometry.py --theta 0.3  # 指定膝关节角度
"""

from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.sans-serif"] = [
    "PingFang HK",
    "Heiti TC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
]
mpl.rcParams["axes.unicode_minus"] = False


# ============================================================
# 弹簧几何参数（占位默认值，需要用户从 CAD 提供实际数据）
# ============================================================
PARAMS = {
    "a": 0.014,  # 大腿侧弹簧挂点到坐标原点距离 (m)
    "alpha": np.deg2rad(5),  # 大腿侧挂点方位角 (rad)
    "b": 0.015,  # 小腿侧弹簧挂点到膝关节轴距离 (m)
    "beta": np.deg2rad(35),  # 小腿侧挂点基准角偏移 (rad)
    "link_l": 0.07,  # 大腿根轴到膝关节轴距离 (m) — 注意：参考实现用 0.07，实际 MJCF 大腿长 0.18
    "k": 900.0,  # 弹簧刚度 (N/m)
    "s0": 0.06,  # 弹簧自然长度 (m)
    "delta_0": 0.004,  # 大腿侧铰链固定长度 (m)
    "delta_1": 0.0095,  # 小腿侧铰链固定长度 (m)
}

# 连杆长度（从 MJCF）
THIGH_LENGTH = 0.18  # lf1_Link pos x
CALF_LENGTH = 0.2  # l_wheel_Link pos z (approx)


def compute_spring_points(
    theta: float,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    link_l: float,
    **_: float,
) -> dict[str, np.ndarray]:
    """计算弹簧几何中的三个关键点坐标。

    坐标系：原点在大腿根（hip 轴），x 向下（重力方向），y 沿腿展开方向。
    这里用弹簧模型的局部坐标系（原点 = 弹簧参考点，非 hip 轴）。
    """
    p0 = np.array([link_l, 0.0])  # 膝关节轴
    p1 = np.array([a * np.cos(alpha), a * np.sin(alpha)])  # 大腿侧挂点
    p2 = np.array(
        [
            link_l - b * np.sin(theta + beta),
            -b * np.cos(theta + beta),
        ]
    )  # 小腿侧挂点
    return {"P0": p0, "P1": p1, "P2": p2}


def draw_spring_geometry(theta: float) -> None:
    """绘制单腿侧视图：连杆 + 弹簧 + 参数标注。"""
    pts = compute_spring_points(theta, **PARAMS)
    p0 = pts["P0"]
    p1 = pts["P1"]
    p2 = pts["P2"]

    # 弹簧有效长度
    dp = p2 - p1
    spring_len = np.linalg.norm(dp)
    s_eff = spring_len - PARAMS["delta_0"] - PARAMS["delta_1"]
    force = PARAMS["k"] * (PARAMS["s0"] - s_eff)

    _fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_title(
        f"膝关节弹簧几何示意图\n"
        f"θ = {np.rad2deg(theta):.1f}° | 弹簧有效长度 s = {s_eff * 1000:.2f} mm | "
        f"弹簧力 F = {force:.1f} N",
        fontsize=12,
    )

    origin = np.array([0.0, 0.0])

    # --- 绘制连杆 ---
    # 大腿杆件：从原点到膝关节轴 P0
    ax.plot(
        [origin[0], p0[0]],
        [origin[1], p0[1]],
        "k-",
        linewidth=4,
        solid_capstyle="round",
        label="大腿杆件",
    )

    # 小腿杆件：从膝关节轴 P0 向下延伸（根据 theta 旋转）
    calf_end = p0 + np.array(
        [
            -0.04 * np.sin(theta + PARAMS["beta"]),
            -0.04 * np.cos(theta + PARAMS["beta"]),
        ]
    )
    ax.plot(
        [p0[0], calf_end[0]],
        [p0[1], calf_end[1]],
        color="0.3",
        linewidth=4,
        solid_capstyle="round",
        label="小腿杆件（局部）",  # noqa: RUF001
    )

    # --- 绘制弹簧（锯齿线） ---
    n_coils = 8
    spring_dir = dp / spring_len
    spring_perp = np.array([-spring_dir[1], spring_dir[0]])
    coil_amp = 0.003  # 锯齿幅度

    # 弹簧起点（跳过 delta_0）和终点（跳过 delta_1）
    spring_start = p1 + spring_dir * PARAMS["delta_0"]
    spring_end = p2 - spring_dir * PARAMS["delta_1"]
    coil_len = np.linalg.norm(spring_end - spring_start)

    coil_pts = [spring_start]
    for i in range(1, n_coils * 2 + 1):
        t = i / (n_coils * 2 + 1)
        center = spring_start + spring_dir * coil_len * t
        sign = 1 if i % 2 == 1 else -1
        coil_pts.append(center + sign * spring_perp * coil_amp)
    coil_pts.append(spring_end)
    coil_arr = np.array(coil_pts)

    # 铰链段（直线）
    ax.plot(
        [p1[0], spring_start[0]],
        [p1[1], spring_start[1]],
        "b-",
        linewidth=1.5,
    )
    ax.plot(
        [spring_end[0], p2[0]],
        [spring_end[1], p2[1]],
        "b-",
        linewidth=1.5,
    )
    # 弹簧锯齿
    ax.plot(coil_arr[:, 0], coil_arr[:, 1], "b-", linewidth=1.5, label="弹簧")

    # --- 绘制关键点 ---
    point_style = {"zorder": 5, "edgecolors": "k", "linewidths": 1.0}
    ax.scatter(*origin, s=100, c="black", marker="o", **point_style)
    ax.scatter(*p0, s=120, c="red", marker="o", **point_style)
    ax.scatter(*p1, s=80, c="blue", marker="s", **point_style)
    ax.scatter(*p2, s=80, c="green", marker="s", **point_style)

    # --- 标注点名称 ---
    ax.annotate(
        "原点\n(弹簧坐标系)",
        origin,
        textcoords="offset points",
        xytext=(-50, -20),
        fontsize=9,
        color="black",
    )
    ax.annotate(
        "P₀ 膝关节轴",
        p0,
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        color="red",
        fontweight="bold",
    )
    ax.annotate(
        "P₁ 大腿侧挂点",
        p1,
        textcoords="offset points",
        xytext=(10, 15),
        fontsize=9,
        color="blue",
        fontweight="bold",
    )
    ax.annotate(
        "P₂ 小腿侧挂点",
        p2,
        textcoords="offset points",
        xytext=(10, -20),
        fontsize=9,
        color="green",
        fontweight="bold",
    )

    # --- 标注参数（带尺寸线） ---
    # 参数 l：原点到 P0
    _draw_dim_line(ax, origin, p0, "l", offset_y=-0.008, color="red")

    # 参数 a：原点到 P1
    _draw_dim_line(ax, origin, p1, "a", offset_y=0.006, color="blue")

    # 参数 b：P0 到 P2
    _draw_dim_line(ax, p0, p2, "b", offset_y=-0.006, color="green")

    # 参数 alpha 角度弧
    _draw_angle_arc(ax, origin, 0, PARAMS["alpha"], 0.01, "\u03b1", color="blue")

    # 参数 beta 角度弧（从膝关节轴）
    _draw_angle_arc(
        ax,
        p0,
        -(np.pi / 2 - theta),
        -(np.pi / 2 - theta - PARAMS["beta"]),
        0.012,
        "β",
        color="green",
    )

    # --- 参数表格文字框 ---
    param_text = (
        "需要从 CAD/实物提供的参数:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  a     = {PARAMS['a'] * 1000:.1f} mm    大腿侧挂点距离\n"
        f"  \u03b1     = {np.rad2deg(PARAMS['alpha']):.1f}\u00b0      大腿侧挂点角\n"
        f"  b     = {PARAMS['b'] * 1000:.1f} mm   小腿侧挂点距离\n"
        f"  \u03b2     = {np.rad2deg(PARAMS['beta']):.1f}\u00b0     小腿侧挂点角\n"
        f"  l     = {PARAMS['link_l'] * 1000:.1f} mm   关节轴间距\n"
        f"  k     = {PARAMS['k']:.0f} N/m   弹簧刚度\n"
        f"  s\u2080    = {PARAMS['s0'] * 1000:.1f} mm   弹簧自然长度\n"
        f"  \u03b4\u2080    = {PARAMS['delta_0'] * 1000:.1f} mm    大腿侧铰链长\n"
        f"  \u03b4\u2081    = {PARAMS['delta_1'] * 1000:.1f} mm    小腿侧铰链长\n"
    )
    props = {"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.9}
    ax.text(
        0.98,
        0.02,
        param_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        family="sans-serif",
    )

    # --- 图例 ---
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.10)
    ax.set_ylim(-0.04, 0.03)

    plt.tight_layout()
    out_path = "scripts/spring_geometry.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存到 {out_path}")
    plt.show()


def _draw_dim_line(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
    offset_y: float = 0.005,
    color: str = "gray",
) -> None:
    """绘制尺寸标注线。"""
    mid = (start + end) / 2
    ax.annotate(
        "",
        xy=end + np.array([0, offset_y]),
        xytext=start + np.array([0, offset_y]),
        arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.2},
    )
    ax.text(
        mid[0],
        mid[1] + offset_y + 0.002 * np.sign(offset_y),
        label,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        fontweight="bold",
        fontstyle="italic",
    )


def _draw_angle_arc(
    ax: plt.Axes,
    center: np.ndarray,
    angle_start: float,
    angle_end: float,
    radius: float,
    label: str,
    color: str = "gray",
) -> None:
    """绘制角度标注弧。"""
    arc = mpatches.Arc(
        center,
        2 * radius,
        2 * radius,
        angle=0,
        theta1=np.rad2deg(min(angle_start, angle_end)),
        theta2=np.rad2deg(max(angle_start, angle_end)),
        color=color,
        linewidth=1.5,
    )
    ax.add_patch(arc)
    mid_angle = (angle_start + angle_end) / 2
    label_pos = center + radius * 1.3 * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    ax.text(
        label_pos[0],
        label_pos[1],
        label,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        fontweight="bold",
        fontstyle="italic",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="膝关节弹簧几何可视化")
    parser.add_argument(
        "--theta",
        type=float,
        default=0.207,
        help="膝关节角度 (rad), 默认使用 default_dof_pos 中 lf1 的值 0.207",
    )
    args = parser.parse_args()
    draw_spring_geometry(args.theta)


if __name__ == "__main__":
    main()
