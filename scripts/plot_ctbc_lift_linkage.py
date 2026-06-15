"""绘制新版四连杆 action 语义下的 CTBC 抬腿姿态。

图中同时展示：
1. 默认姿态；
2. 源 CTBC 在旧输出关节空间请求的姿态；
3. 经过四连杆反解后，目标训练模型实际可实现的姿态。

用法：
    uv run python scripts/plot_ctbc_lift_linkage.py --no-show
    uv run python scripts/plot_ctbc_lift_linkage.py --phase 0.35 --kff 0.8 --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from se3_shared import RobotConfig
from se3_shared.fourbar import output_to_policy_pos_np, policy_to_output_pos_np
from se3_shared.height_default import policy_default_from_height_np
from se3_train.mdp.actions import (
    _CTBC_SOURCE_OUTPUT_LEG_SCALE,
    _CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN,
)
from se3_train.tasks.stair.state import (
    _HIP_FEEDFORWARD_RATIO,
    _KNEE_FEEDFORWARD_RATIO,
)

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
mpl.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_fourbar_surrogate_train.xml"
)

DEFAULT_FF_AMPLITUDE = 1.2
DEFAULT_PHASE = 0.5
DEFAULT_COMMAND_HEIGHT = 0.30
SOURCE_OUTPUT_LEG_SCALE = _CTBC_SOURCE_OUTPUT_LEG_SCALE
SOURCE_TO_TARGET_OUTPUT_SIGN = np.asarray(_CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN)
WHEEL_RADIUS = 0.060

SIDE_NAMES = ("左腿", "右腿")
FRONT_INDICES = (0, 2)
KNEE_INDICES = (1, 3)
ACTIVE_COEFFICIENTS = ((1.0, -1.0), (-1.0, 1.0))


@dataclass(frozen=True)
class CtbcPose:
    """一次 CTBC 相位对应的新旧 action 和输出关节目标。"""

    phase: float
    command_height: float
    cosine_value: float
    old_action_bias: np.ndarray
    output_delta: np.ndarray
    default_policy: np.ndarray
    default_output: np.ndarray
    requested_output: np.ndarray
    realizable_policy: np.ndarray
    realizable_output: np.ndarray
    action_delta: np.ndarray
    active_default: np.ndarray
    active_target: np.ndarray
    active_clamped: np.ndarray


@dataclass(frozen=True)
class LegPoints:
    hip: np.ndarray
    knee: np.ndarray
    wheel: np.ndarray


class SurrogateFk:
    """目标 stair surrogate MJCF 的左右腿侧视图 FK。"""

    def __init__(self, mjcf_path: Path = MJCF_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self._joint_qpos = {
            name: self._joint_qpos_address(name)
            for name in (
                "lf0_Joint",
                "lf1_Joint",
                "rf0_Joint",
                "rf1_Joint",
                "l_wheel_Joint",
                "r_wheel_Joint",
            )
        }

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"MJCF 中不存在 joint: {name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _body_xz(self, name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"MJCF 中不存在 body: {name}")
        pos = self.data.xpos[body_id]
        return np.array([pos[0], pos[2]], dtype=np.float64)

    def _site_xz(self, name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"MJCF 中不存在 site: {name}")
        pos = self.data.site_xpos[site_id]
        return np.array([pos[0], pos[2]], dtype=np.float64)

    def __call__(
        self,
        output_pos: np.ndarray,
        *,
        base_height: float,
    ) -> tuple[LegPoints, LegPoints]:
        output = np.asarray(output_pos, dtype=np.float64).reshape(4)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:7] = np.array([0.0, 0.0, float(base_height), 1.0, 0.0, 0.0, 0.0])
        for name, value in zip(
            ("lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"),
            output,
            strict=True,
        ):
            self.data.qpos[self._joint_qpos[name]] = value
        self.data.qpos[self._joint_qpos["l_wheel_Joint"]] = 0.0
        self.data.qpos[self._joint_qpos["r_wheel_Joint"]] = 0.0
        mujoco.mj_forward(self.model, self.data)

        return (
            LegPoints(
                hip=self._body_xz("lf0_Link"),
                knee=self._site_xz("lf_thigh_end"),
                wheel=self._body_xz("l_wheel_Link"),
            ),
            LegPoints(
                hip=self._body_xz("rf0_Link"),
                knee=self._site_xz("rf_thigh_end"),
                wheel=self._body_xz("r_wheel_Link"),
            ),
        )


def _active_angles(policy_pos: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy_pos, dtype=np.float64).reshape(4)
    return np.array(
        [
            front_coef * policy[front_idx] + back_coef * policy[knee_idx]
            for (front_coef, back_coef), front_idx, knee_idx in zip(
                ACTIVE_COEFFICIENTS,
                FRONT_INDICES,
                KNEE_INDICES,
                strict=True,
            )
        ],
        dtype=np.float64,
    )


def _ctbc_pose(
    *,
    phase: float,
    command_height: float,
    ff_amplitude: float,
    kff: float,
    robot_cfg: RobotConfig,
) -> CtbcPose:
    phase = float(np.clip(phase, 0.0, 1.0))
    cosine_value = float(ff_amplitude) * (1.0 - float(np.cos(2.0 * np.pi * phase)))
    hip_bias = -cosine_value * _HIP_FEEDFORWARD_RATIO * float(kff)
    knee_bias = cosine_value * _KNEE_FEEDFORWARD_RATIO * float(kff)
    old_action_bias = np.array(
        [hip_bias, knee_bias, hip_bias, knee_bias],
        dtype=np.float64,
    )
    output_delta = old_action_bias * SOURCE_TO_TARGET_OUTPUT_SIGN * SOURCE_OUTPUT_LEG_SCALE

    default_policy = policy_default_from_height_np(
        float(command_height),
        robot_cfg,
    ).reshape(4)
    default_output = policy_to_output_pos_np(default_policy[None, :])[0]
    requested_output = default_output + output_delta
    realizable_policy = output_to_policy_pos_np(requested_output[None, :])[0]
    realizable_output = policy_to_output_pos_np(realizable_policy[None, :])[0]

    active_default = _active_angles(default_policy)
    active_target = _active_angles(realizable_policy)
    lower, upper = robot_cfg.active_rod_angle_limits
    active_clamped = (active_target <= lower + 1.0e-8) | (active_target >= upper - 1.0e-8)

    action_delta = np.zeros(4, dtype=np.float64)
    leg_scales = np.asarray(robot_cfg.action_scale[:4], dtype=np.float64)
    for side_idx, (front_idx, active_idx) in enumerate(
        zip(FRONT_INDICES, KNEE_INDICES, strict=True)
    ):
        action_delta[front_idx] = (
            realizable_policy[front_idx] - default_policy[front_idx]
        ) / leg_scales[front_idx]
        action_delta[active_idx] = (
            active_target[side_idx] - active_default[side_idx]
        ) / leg_scales[active_idx]

    return CtbcPose(
        phase=phase,
        command_height=float(command_height),
        cosine_value=cosine_value,
        old_action_bias=old_action_bias,
        output_delta=output_delta,
        default_policy=default_policy,
        default_output=default_output,
        requested_output=requested_output,
        realizable_policy=realizable_policy,
        realizable_output=realizable_output,
        action_delta=action_delta,
        active_default=active_default,
        active_target=active_target,
        active_clamped=active_clamped,
    )


def _draw_leg(
    ax: plt.Axes,
    points: LegPoints,
    *,
    color: str,
    alpha: float,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    coords = np.stack([points.hip, points.knee, points.wheel])
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=(38, 30, 24),
        color=color,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.6,
        zorder=zorder + 1,
    )
    ax.add_patch(
        mpatches.Circle(
            points.wheel,
            WHEEL_RADIUS,
            fill=False,
            edgecolor=color,
            linewidth=max(1.0, 0.55 * linewidth),
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _draw_stair_hint(ax: plt.Axes, default_points: LegPoints) -> None:
    ground_z = default_points.wheel[1] - WHEEL_RADIUS
    riser_x = default_points.wheel[0] + WHEEL_RADIUS
    step_height = 0.08
    step_width = 0.25
    ax.plot(
        [riser_x - 0.18, riser_x, riser_x, riser_x + step_width],
        [ground_z, ground_z, ground_z + step_height, ground_z + step_height],
        color="0.32",
        linewidth=1.2,
        zorder=1,
    )
    ax.fill_between(
        [riser_x, riser_x + step_width],
        ground_z - 0.02,
        ground_z + step_height,
        color="0.91",
        zorder=0,
    )
    ax.text(
        riser_x + 0.012,
        ground_z + step_height + 0.008,
        "8 cm 台阶",
        fontsize=8,
        color="0.28",
    )


def _draw_forward_direction(ax: plt.Axes) -> None:
    """在图内标注机器人前进正方向。"""
    ax.annotate(
        "+X 车前进正方向",
        xy=(0.88, 0.11),
        xytext=(0.52, 0.11),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=9,
        color="#0b7a3b",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#0b7a3b",
            "linewidth": 2.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.86, "lw": 0.0},
        zorder=12,
    )


def _side_info(pose: CtbcPose, side_idx: int, robot_cfg: RobotConfig) -> str:
    front_idx = FRONT_INDICES[side_idx]
    knee_idx = KNEE_INDICES[side_idx]
    upper = float(robot_cfg.active_rod_angle_limits[1])
    residual = pose.realizable_output[knee_idx] - pose.requested_output[knee_idx]
    lines = [
        "源旧 action bias:",
        f"  front = {pose.old_action_bias[front_idx]:+.3f}",
        f"  knee  = {pose.old_action_bias[knee_idx]:+.3f}",
        "",
        "目标输出增量（轴向等效）:",
        f"  Δfront = {pose.output_delta[front_idx]:+.3f} rad",
        f"  Δknee  = {pose.output_delta[knee_idx]:+.3f} rad",
        "",
        "新版 action 等效增量:",
        f"  Δa_front  = {pose.action_delta[front_idx]:+.3f}",
        f"  Δa_active = {pose.action_delta[knee_idx]:+.3f}",
        "",
        f"active: {pose.active_default[side_idx]:.3f} -> {pose.active_target[side_idx]:.3f} rad",
        f"active 上限: {upper:.3f} rad",
        f"knee 请求/实现: {pose.requested_output[knee_idx]:+.3f} / "
        f"{pose.realizable_output[knee_idx]:+.3f} rad",
    ]
    if pose.active_clamped[side_idx]:
        lines.extend(["", f"触发主动杆上限，knee 残差 {residual:+.3f} rad"])
    else:
        lines.extend(["", "未触发主动杆限位"])
    return "\n".join(lines)


def _all_points(*poses: tuple[LegPoints, LegPoints]) -> np.ndarray:
    points: list[np.ndarray] = []
    for pose in poses:
        for leg in pose:
            points.extend([leg.hip, leg.knee, leg.wheel])
    return np.stack(points)


def plot_ctbc_linkage(
    *,
    phase: float,
    command_height: float,
    ff_amplitude: float,
    kff: float,
    samples: int,
    out_path: Path,
    show: bool,
) -> None:
    """生成目标四连杆 action 语义下的 CTBC 抬腿示意图。"""
    robot_cfg = RobotConfig()
    fk = SurrogateFk()
    pose = _ctbc_pose(
        phase=phase,
        command_height=command_height,
        ff_amplitude=ff_amplitude,
        kff=kff,
        robot_cfg=robot_cfg,
    )
    default_points = fk(pose.default_output, base_height=command_height)
    requested_points = fk(pose.requested_output, base_height=command_height)
    realizable_points = fk(pose.realizable_output, base_height=command_height)

    sweep_points: list[tuple[LegPoints, LegPoints]] = []
    for sample_phase in np.linspace(0.0, 1.0, max(2, int(samples))):
        sample = _ctbc_pose(
            phase=float(sample_phase),
            command_height=command_height,
            ff_amplitude=ff_amplitude,
            kff=kff,
            robot_cfg=robot_cfg,
        )
        sweep_points.append(
            fk(
                sample.realizable_output,
                base_height=command_height,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharex=True, sharey=True)
    fig.suptitle(
        "新版 CTBC 抬腿示意：旧输出关节 bias -> active-rod action（surrogate 等效开树）",
        fontsize=14,
    )

    point_cloud = _all_points(default_points, requested_points, realizable_points, *sweep_points)
    x_min, z_min = point_cloud.min(axis=0)
    x_max, z_max = point_cloud.max(axis=0)
    x_pad = max(0.09, 0.18 * (x_max - x_min))
    z_pad = max(0.08, 0.18 * (z_max - z_min))

    for side_idx, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_title(SIDE_NAMES[side_idx], fontsize=12)
        _draw_stair_hint(ax, default_points[side_idx])

        for sample_points in sweep_points:
            _draw_leg(
                ax,
                sample_points[side_idx],
                color="#2f6fba",
                alpha=0.055,
                linewidth=1.0,
                zorder=2,
            )
        _draw_leg(
            ax,
            default_points[side_idx],
            color="0.35",
            alpha=0.48,
            linewidth=3.2,
            zorder=3,
        )
        _draw_leg(
            ax,
            requested_points[side_idx],
            color="#d95f02",
            alpha=0.90,
            linewidth=5.2,
            linestyle="--",
            zorder=4,
        )
        _draw_leg(
            ax,
            realizable_points[side_idx],
            color="#1769aa",
            alpha=1.0,
            linewidth=3.2,
            zorder=5,
        )

        ax.annotate(
            "",
            xy=realizable_points[side_idx].wheel,
            xytext=default_points[side_idx].wheel,
            arrowprops={"arrowstyle": "->", "color": "#1769aa", "linewidth": 1.8},
            zorder=8,
        )
        ax.text(
            0.03,
            0.97,
            _side_info(pose, side_idx, robot_cfg),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.3,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.93},
            zorder=10,
        )
        ax.grid(True, alpha=0.20)
        ax.set_xlabel("X (m)")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(z_min - WHEEL_RADIUS - z_pad, z_max + z_pad)
        _draw_forward_direction(ax)

    axes[0].set_ylabel("Z (m)")
    legend_handles = [
        mlines.Line2D([], [], color="0.35", alpha=0.48, linewidth=4, label="默认姿态"),
        mlines.Line2D(
            [],
            [],
            color="#d95f02",
            linewidth=5,
            linestyle="--",
            label="源 CTBC 经轴向变换的输出请求",
        ),
        mlines.Line2D([], [], color="#1769aa", linewidth=3, label="新版四连杆可实现目标"),
        mlines.Line2D(
            [],
            [],
            color="#2f6fba",
            alpha=0.18,
            linewidth=3,
            label="新版 CTBC 周期轨迹",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )
    fig.text(
        0.5,
        0.055,
        (
            f"phase={pose.phase:.2f}, kff={kff:.2f}, "
            f"command_height={pose.command_height:.2f}m, "
            f"ff_amplitude(action)={ff_amplitude:.2f}, "
            f"旧输出关节 scale={SOURCE_OUTPUT_LEG_SCALE:.2f} rad/action；"
            "源->目标轴向符号=[-1,-1,+1,+1]"
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"default_output={np.array2string(pose.default_output, precision=6)}")
    print(f"requested_output={np.array2string(pose.requested_output, precision=6)}")
    print(f"realizable_output={np.array2string(pose.realizable_output, precision=6)}")
    print(f"equivalent_action_delta={np.array2string(pose.action_delta, precision=6)}")
    print(f"active_target={np.array2string(pose.active_target, precision=6)}")
    print(f"active_clamped={pose.active_clamped.tolist()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制新版四连杆 action 语义下的 CTBC 抬腿姿态。")
    parser.add_argument(
        "--phase",
        type=float,
        default=DEFAULT_PHASE,
        help="CTBC 相位，0.5 为前馈峰值。",
    )
    parser.add_argument("--ff-amplitude", type=float, default=DEFAULT_FF_AMPLITUDE)
    parser.add_argument("--kff", type=float, default=1.0, help="CTBC 退火权重。")
    parser.add_argument(
        "--command-height",
        type=float,
        default=DEFAULT_COMMAND_HEIGHT,
        help="height-conditioned action default 使用的机体高度指令。",
    )
    parser.add_argument("--samples", type=int, default=32, help="周期轨迹采样数。")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("scripts/ctbc_lift_linkage.png"),
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plot_ctbc_linkage(
        phase=args.phase,
        command_height=args.command_height,
        ff_amplitude=args.ff_amplitude,
        kff=args.kff,
        samples=args.samples,
        out_path=args.out_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
