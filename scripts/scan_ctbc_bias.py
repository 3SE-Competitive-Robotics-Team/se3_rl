"""扫描当前闭链动作语义下的 CTBC bias 候选。

脚本做两件事：
1. 按不同 base height 离线扫描 front_amp / active_amp 网格，评估轮心位移。
2. 对每个高度的优选候选生成 visual-only MuJoCo 开环视频。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from se3_shared.fourbar import policy_to_output_pos_np
from se3_shared.height_default import policy_default_from_height_np
from se3_shared.robot import RobotConfig

ROOT = Path(__file__).resolve().parents[1]
XML_PATH = (
    ROOT / "assets" / "robots" / "serialleg" / "mjcf" / "serialleg_fourbar_surrogate_train.xml"
)
OUT_DIR = ROOT / "tmp" / "ctbc_bias_scan"
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
class ScanRow:
    """单个 bias 候选的 FK 评分结果。"""

    height_m: float
    side: str
    front_amp: float
    active_amp: float
    max_dx_m: float
    max_dz_m: float
    dz_at_max_dx_m: float
    dx_at_max_dz_m: float
    min_active_margin_rad: float
    max_active_rad: float
    soft_upper_rad: float
    score: float
    accepted: bool
    reject_reason: str


def parse_float_list(value: str) -> list[float]:
    """解析逗号分隔的浮点数列表。"""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def frange(start: float, stop: float, step: float) -> list[float]:
    """生成包含 stop 附近端点的浮点网格。"""
    values: list[float] = []
    current = start
    while current <= stop + 1.0e-9:
        values.append(round(current, 6))
        current += step
    return values


class CtbcBiasScanner:
    """使用 MuJoCo FK 评估当前动作空间中的 CTBC bias。"""

    def __init__(self, xml_path: Path) -> None:
        self.robot_cfg = RobotConfig()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
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

    def scan(
        self,
        heights: list[float],
        front_values: list[float],
        active_values: list[float],
        num_phase_samples: int,
        min_dx: float,
        min_dz: float,
        active_margin: float,
    ) -> list[ScanRow]:
        """扫描所有高度和幅值组合。"""
        rows: list[ScanRow] = []
        for height in heights:
            base_policy = policy_default_from_height_np(height)
            base_pos = self._wheel_positions(height, base_policy)
            for side in ("left", "right"):
                side_idx = 0 if side == "left" else 1
                for front_amp in front_values:
                    for active_amp in active_values:
                        row = self._score_candidate(
                            height=height,
                            side=side,
                            side_idx=side_idx,
                            base_policy=base_policy,
                            base_wheel_pos=base_pos[side_idx],
                            front_amp=front_amp,
                            active_amp=active_amp,
                            num_phase_samples=num_phase_samples,
                            min_dx=min_dx,
                            min_dz=min_dz,
                            active_margin=active_margin,
                        )
                        rows.append(row)
        return rows

    def _score_candidate(
        self,
        *,
        height: float,
        side: str,
        side_idx: int,
        base_policy: np.ndarray,
        base_wheel_pos: np.ndarray,
        front_amp: float,
        active_amp: float,
        num_phase_samples: int,
        min_dx: float,
        min_dz: float,
        active_margin: float,
    ) -> ScanRow:
        del side_idx
        soft_lower, soft_upper = self.robot_cfg.active_rod_soft_angle_limits
        phases = np.linspace(0.0, 1.0, num_phase_samples)
        dx_values: list[float] = []
        dz_values: list[float] = []
        active_values: list[float] = []

        for phase in phases:
            profile = 0.5 * (1.0 - math.cos(2.0 * math.pi * float(phase)))
            policy = self._biased_policy(
                base_policy, side, front_amp * profile, active_amp * profile
            )
            wheel_pos = self._wheel_positions(height, policy)[0 if side == "left" else 1]
            delta = wheel_pos - base_wheel_pos
            dx_values.append(float(delta[0]))
            dz_values.append(float(delta[2]))
            active_values.append(float(self._active_angle(policy, side)))

        dx_arr = np.asarray(dx_values)
        dz_arr = np.asarray(dz_values)
        active_arr = np.asarray(active_values)
        max_dx_idx = int(np.argmax(dx_arr))
        max_dz_idx = int(np.argmax(dz_arr))
        max_dx = float(dx_arr[max_dx_idx])
        max_dz = float(dz_arr[max_dz_idx])
        dz_at_max_dx = float(dz_arr[max_dx_idx])
        dx_at_max_dz = float(dx_arr[max_dz_idx])
        min_active_margin = float(
            np.min(np.minimum(active_arr - soft_lower, soft_upper - active_arr))
        )
        max_active = float(np.max(active_arr))

        reject_reason = ""
        if max_dx < min_dx:
            reject_reason = "forward_dx_too_small"
        elif max_dz < min_dz:
            reject_reason = "lift_dz_too_small"
        elif dx_at_max_dz < 0.0:
            reject_reason = "peak_lift_moves_backward"
        elif min_active_margin < active_margin:
            reject_reason = "active_rod_near_soft_limit"

        accepted = reject_reason == ""
        # 分数偏好：向前和抬高都好，但主动杆限位余量越小惩罚越重。
        score = max_dx + 1.5 * max_dz + 0.2 * dz_at_max_dx + 0.05 * min_active_margin
        if not accepted:
            score -= 1.0
        return ScanRow(
            height_m=height,
            side=side,
            front_amp=front_amp,
            active_amp=active_amp,
            max_dx_m=max_dx,
            max_dz_m=max_dz,
            dz_at_max_dx_m=dz_at_max_dx,
            dx_at_max_dz_m=dx_at_max_dz,
            min_active_margin_rad=min_active_margin,
            max_active_rad=max_active,
            soft_upper_rad=float(soft_upper),
            score=float(score),
            accepted=accepted,
            reject_reason=reject_reason,
        )

    def _biased_policy(
        self,
        base_policy: np.ndarray,
        side: str,
        front_amp: float,
        active_amp: float,
    ) -> np.ndarray:
        policy = np.asarray(base_policy, dtype=np.float64).copy()
        scales = np.asarray(self.robot_cfg.action_scale[:4], dtype=np.float64)
        if side == "left":
            policy[0] = base_policy[0] - front_amp * scales[0]
            active = self._active_angle(base_policy, "left") + active_amp * scales[1]
            policy[1] = policy[0] - active
        else:
            policy[2] = base_policy[2] + front_amp * scales[2]
            active = self._active_angle(base_policy, "right") + active_amp * scales[3]
            policy[3] = policy[2] + active
        return policy

    def _active_angle(self, policy: np.ndarray, side: str) -> float:
        if side == "left":
            return float(policy[0] - policy[1])
        return float(policy[3] - policy[2])

    def _wheel_positions(self, height: float, policy_pos: np.ndarray) -> np.ndarray:
        output = policy_to_output_pos_np(policy_pos)
        full = np.asarray((output[0], output[1], 0.0, output[2], output[3], 0.0), dtype=np.float64)
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = (0.0, 0.0, height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, value in zip(JOINT_NAMES, full, strict=True):
            self.data.qpos[self.joint_qpos_addr[name]] = float(value)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return np.stack(
            (
                self.data.xpos[self.wheel_body_ids["l_wheel_Link"]].copy(),
                self.data.xpos[self.wheel_body_ids["r_wheel_Link"]].copy(),
            )
        )


def best_rows(rows: list[ScanRow], top_per_height: int) -> list[ScanRow]:
    """按最高评分选出优选候选。"""
    result: list[ScanRow] = []
    keys = sorted({(row.height_m, row.side) for row in rows})
    for height, side in keys:
        group = [
            row for row in rows if row.height_m == height and row.side == side and row.accepted
        ]
        if not group:
            group = [row for row in rows if row.height_m == height and row.side == side]
        result.extend(sorted(group, key=lambda item: item.score, reverse=True)[:top_per_height])
    return result


def conservative_rows(
    rows: list[ScanRow], top_per_height: int, min_dx: float, min_dz: float
) -> list[ScanRow]:
    """按最小有效幅值选出保守候选。"""
    result: list[ScanRow] = []
    keys = sorted({(row.height_m, row.side) for row in rows})
    for height, side in keys:
        group = [
            row
            for row in rows
            if row.height_m == height
            and row.side == side
            and row.accepted
            and row.max_dx_m >= min_dx
            and row.max_dz_m >= min_dz
        ]
        if not group:
            group = [row for row in rows if row.height_m == height and row.side == side]
        result.extend(
            sorted(
                group,
                key=lambda item: (
                    math.hypot(item.front_amp, item.active_amp),
                    abs(item.front_amp - item.active_amp),
                    -item.min_active_margin_rad,
                ),
            )[:top_per_height]
        )
    return result


def write_csv(path: Path, rows: list[ScanRow]) -> None:
    """写出扫描结果 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[ScanRow]) -> None:
    """写出优选候选 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_candidate_video(row: ScanRow, out_dir: Path, seconds: float, fps: int) -> Path:
    """生成单个候选的 visual-only 开环视频。"""
    robot_cfg = RobotConfig()
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 隐藏碰撞几何，保留地面和视觉 mesh。
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "collision" in name:
            model.geom_rgba[geom_id, 3] = 0.0

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    opt.sitegroup[:] = 0

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 0.13)
    cam.distance = 1.0
    cam.elevation = -15
    cam.azimuth = 90 if row.side == "left" else -90

    joint_qpos_addr = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    }
    base_policy = policy_default_from_height_np(row.height_m)
    scales = np.asarray(robot_cfg.action_scale[:4], dtype=np.float64)

    def active_angle(policy: np.ndarray, side: str) -> float:
        return float(policy[0] - policy[1]) if side == "left" else float(policy[3] - policy[2])

    def biased_policy(profile: float) -> np.ndarray:
        policy = base_policy.copy()
        if row.side == "left":
            policy[0] = base_policy[0] - row.front_amp * profile * scales[0]
            active = active_angle(base_policy, "left") + row.active_amp * profile * scales[1]
            policy[1] = policy[0] - active
        else:
            policy[2] = base_policy[2] + row.front_amp * profile * scales[2]
            active = active_angle(base_policy, "right") + row.active_amp * profile * scales[3]
            policy[3] = policy[2] + active
        return policy

    frames = []
    for frame_idx in range(int(seconds * fps)):
        phase = frame_idx / max(1, int(seconds * fps) - 1)
        profile = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        output = policy_to_output_pos_np(biased_policy(profile))
        full = np.asarray((output[0], output[1], 0.0, output[2], output[3], 0.0), dtype=np.float64)
        data.qpos[:] = 0.0
        data.qpos[0:3] = (0.0, 0.0, row.height_m)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for name, value in zip(JOINT_NAMES, full, strict=True):
            data.qpos[joint_qpos_addr[name]] = float(value)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        frames.append(renderer.render())

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"h{row.height_m:.2f}_{row.side}_front{row.front_amp:.2f}_active{row.active_amp:.2f}.mp4"
    )
    imageio.mimsave(out, frames, fps=fps, quality=8)
    renderer.close()
    return out


def print_table(rows: list[ScanRow]) -> None:
    """打印精简优选表。"""
    print("height side  front active  max_dx  max_dz  dz@dx  dx@dz  margin  score")
    for row in rows:
        print(
            f"{row.height_m:5.2f} {row.side:5s} "
            f"{row.front_amp:5.2f} {row.active_amp:6.2f} "
            f"{row.max_dx_m:7.4f} {row.max_dz_m:7.4f} "
            f"{row.dz_at_max_dx_m:6.4f} {row.dx_at_max_dz_m:6.4f} "
            f"{row.min_active_margin_rad:7.4f} {row.score:7.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="扫描 CTBC bias 并生成 visual-only MuJoCo 开环视频。"
    )
    parser.add_argument("--heights", default="0.22,0.24,0.26,0.28,0.30")
    parser.add_argument("--front-start", type=float, default=0.04)
    parser.add_argument("--front-stop", type=float, default=0.22)
    parser.add_argument("--front-step", type=float, default=0.02)
    parser.add_argument("--active-start", type=float, default=0.04)
    parser.add_argument("--active-stop", type=float, default=0.24)
    parser.add_argument("--active-step", type=float, default=0.02)
    parser.add_argument("--num-phase-samples", type=int, default=41)
    parser.add_argument("--min-dx", type=float, default=0.035)
    parser.add_argument("--min-dz", type=float, default=0.012)
    parser.add_argument("--active-margin", type=float, default=0.03)
    parser.add_argument("--top-per-height-side", type=int, default=3)
    parser.add_argument("--render-top", type=int, default=1)
    parser.add_argument("--selection-mode", choices=("score", "conservative"), default="score")
    parser.add_argument("--video-seconds", type=float, default=1.2)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    heights = parse_float_list(args.heights)
    front_values = frange(args.front_start, args.front_stop, args.front_step)
    active_values = frange(args.active_start, args.active_stop, args.active_step)

    scanner = CtbcBiasScanner(XML_PATH)
    rows = scanner.scan(
        heights=heights,
        front_values=front_values,
        active_values=active_values,
        num_phase_samples=args.num_phase_samples,
        min_dx=args.min_dx,
        min_dz=args.min_dz,
        active_margin=args.active_margin,
    )
    if args.selection_mode == "conservative":
        selected = conservative_rows(rows, args.top_per_height_side, args.min_dx, args.min_dz)
    else:
        selected = best_rows(rows, args.top_per_height_side)
    write_csv(OUT_DIR / "scan.csv", rows)
    write_summary(OUT_DIR / f"selected_{args.selection_mode}.json", selected)
    print_table(selected)
    print(f"wrote {OUT_DIR / 'scan.csv'}")
    print(f"wrote {OUT_DIR / f'selected_{args.selection_mode}.json'}")

    videos: list[str] = []
    if args.render_top > 0:
        if args.selection_mode == "conservative":
            render_rows = conservative_rows(rows, args.render_top, args.min_dx, args.min_dz)
        else:
            render_rows = best_rows(rows, args.render_top)
        for row in render_rows:
            path = render_candidate_video(
                row, OUT_DIR / f"videos_{args.selection_mode}", args.video_seconds, args.fps
            )
            videos.append(str(path))
            print(f"wrote {path}")
    if videos:
        (OUT_DIR / f"videos_{args.selection_mode}.json").write_text(
            json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
