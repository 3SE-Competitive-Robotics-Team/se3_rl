"""可视化 CTBC bias 对轮心轨迹和机器人姿态的影响。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from se3_shared.fourbar import policy_to_output_pos_np
from se3_shared.height_default import policy_default_from_height_np
from se3_shared.robot import RobotConfig

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = (
    ROOT / "assets" / "robots" / "serialleg" / "mjcf" / "serialleg_fourbar_surrogate_train.xml"
)
OUT_DIR = ROOT / "tmp" / "ctbc_bias_visual"
JOINT_NAMES = (
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
)
WHEEL_BODY_NAMES = ("l_wheel_Link", "r_wheel_Link")


@dataclass(frozen=True)
class BiasCase:
    """单个 CTBC bias 可视化案例。"""

    name: str
    front_amp: float
    active_amp: float
    color: str


@dataclass(frozen=True)
class ScaleCase:
    """单个动作缩放来源。"""

    name: str
    scales: tuple[float, float, float, float]


CASES = (
    BiasCase("current CTBC (-0.14, +0.07)", 0.14, 0.07, "#0072B2"),
    BiasCase("medium teacher (-0.50, +0.50)", 0.50, 0.50, "#E69F00"),
    BiasCase("strong teacher (-0.80, +0.80)", 0.80, 0.80, "#D55E00"),
)

SCALE_CASES = (
    ScaleCase("code_scale", tuple(RobotConfig().action_scale[:4])),
    ScaleCase("agent_note_scale", (0.35, 0.25, 0.35, 0.25)),
)


class CtbcVisualizer:
    """用 MuJoCo FK 渲染 CTBC bias 的侧视轨迹和视频。"""

    def __init__(self, height: float, scale_case: ScaleCase) -> None:
        self.height = float(height)
        self.scale_case = scale_case
        self.robot_cfg = RobotConfig()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_addr = {
            name: int(
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in JOINT_NAMES
        }
        self.wheel_body_ids = {
            name: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in WHEEL_BODY_NAMES
        }
        self.base_policy = np.asarray(policy_default_from_height_np(self.height), dtype=np.float64)
        self.scales = np.asarray(scale_case.scales, dtype=np.float64)

    def biased_policy(self, case: BiasCase, profile: float) -> np.ndarray:
        """按左腿触发生成当前相位的 policy joint target。"""
        policy = self.base_policy.copy()
        policy[0] = self.base_policy[0] - case.front_amp * profile * self.scales[0]
        active = self._active_angle(self.base_policy) + case.active_amp * profile * self.scales[1]
        policy[1] = policy[0] - active
        return policy

    def wheel_path(self, case: BiasCase, samples: int = 81) -> np.ndarray:
        """返回左轮轮心随余弦 bias 变化的世界坐标轨迹。"""
        points: list[np.ndarray] = []
        for phase in np.linspace(0.0, 1.0, samples):
            profile = 0.5 * (1.0 - math.cos(2.0 * math.pi * float(phase)))
            self.set_policy(self.biased_policy(case, profile))
            points.append(self.data.xpos[self.wheel_body_ids["l_wheel_Link"]].copy())
        return np.stack(points)

    def set_policy(self, policy: np.ndarray) -> None:
        """把 policy 语义关节角写入 MuJoCo 模型。"""
        output = policy_to_output_pos_np(policy)
        full = np.asarray((output[0], output[1], 0.0, output[2], output[3], 0.0))
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = (0.0, 0.0, self.height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, value in zip(JOINT_NAMES, full, strict=True):
            self.data.qpos[self.joint_qpos_addr[name]] = float(value)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def render_video(
        self, case: BiasCase, out_path: Path, seconds: float = 1.6, fps: int = 40
    ) -> None:
        """渲染 visual-only 的 MuJoCo 侧视动画。"""
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        data = mujoco.MjData(model)
        joint_qpos_addr = {
            name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in JOINT_NAMES
        }
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if "collision" in name:
                model.geom_rgba[geom_id, 3] = 0.0

        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = 1
        opt.geomgroup[1] = 1
        opt.sitegroup[:] = 0

        renderer = mujoco.Renderer(model, height=540, width=960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.0, 0.16, 0.18)
        cam.distance = 0.85
        cam.elevation = -12
        cam.azimuth = 90

        frames = []
        for frame_idx in range(int(seconds * fps)):
            phase = frame_idx / max(1, int(seconds * fps) - 1)
            profile = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
            output = policy_to_output_pos_np(self.biased_policy(case, profile))
            full = np.asarray((output[0], output[1], 0.0, output[2], output[3], 0.0))
            data.qpos[:] = 0.0
            data.qpos[0:3] = (0.0, 0.0, self.height)
            data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            for name, value in zip(JOINT_NAMES, full, strict=True):
                data.qpos[joint_qpos_addr[name]] = float(value)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(renderer.render())

        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=fps, quality=8)
        renderer.close()

    @staticmethod
    def _active_angle(policy: np.ndarray) -> float:
        return float(policy[0] - policy[1])


def write_plot(visualizer: CtbcVisualizer, out_path: Path) -> None:
    """写出轮心轨迹对比图。"""
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=180)
    base_path = visualizer.wheel_path(CASES[0])
    origin = base_path[0]
    radius = 0.06
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            radius,
            edgecolor="#777777",
            facecolor="none",
            linestyle="--",
            linewidth=1.1,
            label="wheel radius 60mm",
        )
    )
    ax.scatter([0.0], [0.0], s=28, c="#222222", label="initial wheel center")

    summary_lines: list[str] = []
    for case in CASES:
        path = visualizer.wheel_path(case) - origin
        ax.plot(path[:, 0], path[:, 2], color=case.color, linewidth=2.2, label=case.name)
        ax.scatter([path[len(path) // 2, 0]], [path[len(path) // 2, 2]], color=case.color, s=28)
        peak = path[len(path) // 2]
        summary_lines.append(
            f"{case.name}: peak dx={peak[0] * 1000:+.1f}mm, dz={peak[2] * 1000:+.1f}mm"
        )

    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("wheel-center dx / m, forward positive")
    ax.set_ylabel("wheel-center dz / m, upward positive")
    ax.set_title(
        f"CTBC left-trigger bias path, base height={visualizer.height:.2f}m, "
        f"{visualizer.scale_case.name}, scales={tuple(round(float(v), 3) for v in visualizer.scales)}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_html(plot_paths: list[Path], video_paths: list[Path], out_path: Path) -> None:
    """写出可直接浏览的对比页面。"""
    video_cards = []
    for video in video_paths:
        title = video.stem.replace("_", " ")
        video_cards.append(
            f"""
            <section>
              <h2>{title}</h2>
              <video src="{video.name}" controls loop muted autoplay></video>
            </section>
            """
        )
    out_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>CTBC bias visual</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f8; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .plot {{ width: 100%; background: white; border: 1px solid #d9dee5; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 18px; }}
    section {{ background: white; border: 1px solid #d9dee5; padding: 12px; }}
    video {{ width: 100%; display: block; }}
    p {{ line-height: 1.55; }}
  </style>
</head>
<body>
<main>
  <h1>CTBC bias 可视化：当前 vs teacher 候选</h1>
  <p>轨迹图使用真实 MuJoCo FK。第一张是代码真实 action scale；第二张是 AGENTS 里记录的预期 scale，用来对照语义差异。动画只显示 visual mesh 和地面，不显示 collision。</p>
  {"".join(f'<img class="plot" src="{plot.name}" alt="CTBC wheel center path">' for plot in plot_paths)}
  <div class="grid">
    {"".join(video_cards)}
  </div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    visualizer = CtbcVisualizer(height=0.34, scale_case=SCALE_CASES[0])
    for scale_case in SCALE_CASES:
        plot_visualizer = CtbcVisualizer(height=0.34, scale_case=scale_case)
        plot_path = OUT_DIR / f"ctbc_bias_paths_{scale_case.name}.png"
        write_plot(plot_visualizer, plot_path)
        plot_paths.append(plot_path)
    video_paths: list[Path] = []
    for case in CASES:
        safe_name = (
            case.name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace("+", "p")
            .replace("-", "m")
        )
        video_path = OUT_DIR / f"{safe_name}.mp4"
        visualizer.render_video(case, video_path)
        video_paths.append(video_path)
    html_path = OUT_DIR / "index.html"
    write_html(plot_paths, video_paths, html_path)
    print(html_path)


if __name__ == "__main__":
    main()
