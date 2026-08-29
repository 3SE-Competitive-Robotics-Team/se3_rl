"""终版 checkpoint 的原生 MuJoCo 分姿态验收（sim2x runtime，2026-07-18 协议镜像）。

镜像 evaluate_recovery_discovery_fixed_poses.py 的口径：
- 五姿态（standing/left_side/right_side/prone/supine），stage-0 抖动（roll/pitch ±5 度、
  全角 yaw、clearance 1-5 mm、height offset 0-20 mm、零初速、关节标称）。
- 成功判据：tilt < 15 度 且 |height-0.26| < 0.02，连续保持 0.5 s；episode 5 s。
- 追加 zero_hold 20 s（vx/净位移/累积偏航）与指令跟踪 smoke。
姿态数学直接复用训练端 events 的同名工具，保证几何一致。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submodules" / "se3-sim2x" / "src"))

import mujoco  # noqa: E402
import torch  # noqa: E402

from se3_runtime import PolicyControlLoop, PolicyRuntime  # noqa: E402
from se3_runtime._serialleg_v1 import (  # noqa: E402
    closedchain_passive_position,
    height_conditioned_policy_default,
)
from se3_runtime_mujoco.adapter import MujocoPolicyAdapter  # noqa: E402
from se3_train.mdp.events import (  # noqa: E402
    _FULL_ANGLE_RESET_BBOX_MAX,
    _FULL_ANGLE_RESET_BBOX_MIN,
    _quat_z_row,
    quat_from_euler_xyz,
)

POSES: dict[str, tuple[float, float]] = {
    # name -> (roll, pitch) 基准；yaw 与 ±5 度抖动逐 episode 采样
    "standing": (0.0, 0.0),
    "left_side": (0.5 * math.pi, 0.0),
    "right_side": (-0.5 * math.pi, 0.0),
    "prone": (0.0, math.pi),
    "supine": (0.0, -math.pi),
}
PASSIVE_JOINT_NAMES = ("lf1_Joint", "l_coupler_Joint", "rf1_Joint", "r_coupler_Joint")
H_CMD = 0.26
SUCCESS_TILT_DEG = 15.0
SUCCESS_HEIGHT_TOL = 0.02
HOLD_S = 0.5
EPISODE_S = 5.0
STEP_DT = 0.02


def _quat_wxyz_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    q = quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw]))
    return q[0].numpy().astype(np.float64)


def _safe_height(quat_wxyz: np.ndarray, clearance: float) -> float:
    z_row = _quat_z_row(torch.tensor(quat_wxyz, dtype=torch.float64).unsqueeze(0))[0]
    bbox_min = torch.tensor(_FULL_ANGLE_RESET_BBOX_MIN, dtype=torch.float64)
    bbox_max = torch.tensor(_FULL_ANGLE_RESET_BBOX_MAX, dtype=torch.float64)
    min_z = torch.minimum(z_row * bbox_min, z_row * bbox_max).sum()
    return float(-min_z) + clearance


class NativeHarness:
    def __init__(self, onnx_path: Path, seed: int) -> None:
        self.runtime = PolicyRuntime.load(onnx_path, action_delay_random_seed=seed)
        self.adapter = MujocoPolicyAdapter(
            self.runtime.contract, artifact_path=self.runtime.bundle.path
        )
        self.loop = PolicyControlLoop(self.runtime, self.adapter)
        self.model = self.adapter.model
        self.data = self.adapter.data
        self.base_qpos = self.adapter._base_qpos_address
        names = self.runtime.contract.robot.policy_joint_names
        self.leg_names = list(names[:4])
        self.nominal = np.asarray(
            self.runtime.contract.robot.default_joint_position, dtype=np.float64
        )
        self.renderer: mujoco.Renderer | None = None

    def set_command(self, **fields: float) -> None:
        base = {
            "lin_vel_x": 0.0,
            "ang_vel_yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "height": H_CMD,
            "jump_flag": 0.0,
            "jump_target_height": 0.0,
            "jump_phase": 0.0,
        }
        base.update(fields)
        for k, v in base.items():
            self.adapter.set_command_field("velocity_height", k, float(v))

    def _write_state(self, quat_wxyz: np.ndarray, z: float, joints4: np.ndarray) -> None:
        d, m, a = self.data, self.model, self.base_qpos
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[a : a + 3] = (0.0, 0.0, z)
        d.qpos[a + 3 : a + 7] = quat_wxyz
        passive = np.asarray(closedchain_passive_position(joints4)).reshape(4)
        for name, val in zip(self.leg_names, joints4, strict=True):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[jid]] = val
        for name, val in zip(PASSIVE_JOINT_NAMES, passive, strict=True):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[jid]] = val
        mujoco.mj_forward(m, d)

    def reset_pose(self, quat_wxyz: np.ndarray, z: float, joints4: np.ndarray) -> None:
        self.loop.reset()
        self._write_state(quat_wxyz, z, joints4)
        self.runtime.reset()

    def reset_standing_exact(self) -> None:
        default4 = np.asarray(
            height_conditioned_policy_default(
                H_CMD, tuple(self.runtime.contract.action.active_rod_angle_limits)
            )
        ).reshape(4)
        quat = np.asarray((1.0, 0.0, 0.0, 0.0))
        self.reset_pose(quat, H_CMD, default4)
        # 轮子贴地微调
        clearance = (
            min(
                float(self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)][2])
                for n in ("l_wheel_Link", "r_wheel_Link")
            )
            - 0.06
        )
        self.data.qpos[self.base_qpos + 2] -= clearance - 0.001
        mujoco.mj_forward(self.model, self.data)
        self.runtime.reset()

    def state(self) -> tuple[float, float, float, float, float, float]:
        d, a = self.data, self.base_qpos
        x, y, z = (float(v) for v in d.qpos[a : a + 3])
        w, qx, qy, qz = (float(v) for v in d.qpos[a + 3 : a + 7])
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, r22))))
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return x, y, z, tilt, yaw, float(d.qvel[0])

    def ensure_renderer(self) -> mujoco.Renderer | None:
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, 480, 640)
            except Exception as error:
                print(f"[video] renderer unavailable: {error}", flush=True)
                self.renderer = None
        return self.renderer

    def render_frame(self, cam: mujoco.MjvCamera) -> np.ndarray | None:
        renderer = self.ensure_renderer()
        if renderer is None:
            return None
        a = self.base_qpos
        cam.lookat[:] = self.data.qpos[a : a + 3]
        cam.lookat[2] = max(0.15, float(self.data.qpos[a + 2]))
        renderer.update_scene(self.data, camera=cam)
        return renderer.render()


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.2
    cam.azimuth = 135.0
    cam.elevation = -18.0
    return cam


def run_pose_episode(h: NativeHarness, quat: np.ndarray, z: float, record: bool):
    frames: list[np.ndarray] = []
    cam = make_camera()
    hold_steps = max(1, math.ceil(HOLD_S / STEP_DT))
    n_steps = math.ceil(EPISODE_S / STEP_DT)
    streak = 0
    success_time: float | None = None
    tilt = height = float("nan")
    for step in range(n_steps):
        h.loop.policy_step()
        _, _, height, tilt, _, _ = h.state()
        ok = (tilt < SUCCESS_TILT_DEG) and (abs(height - H_CMD) < SUCCESS_HEIGHT_TOL)
        streak = streak + 1 if ok else 0
        if success_time is None and streak >= hold_steps:
            success_time = (step - hold_steps + 1 + 1) * STEP_DT
        if record and step % 2 == 0:
            frame = h.render_frame(cam)
            if frame is not None:
                frames.append(frame)
    return success_time, tilt, height, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    results: dict[str, object] = {
        "onnx": str(args.onnx),
        "protocol": "native-sim2x-20260718-mirror",
    }

    videos: dict[str, list[np.ndarray]] = {}
    pose_rows = []
    for pose, (roll0, pitch0) in POSES.items():
        harness = NativeHarness(args.onnx, seed=1000)
        harness.set_command()
        times, tilts, heights, succ = [], [], [], 0
        for ep in range(args.episodes):
            roll = roll0 + math.radians(rng.uniform(-5.0, 5.0))
            pitch = pitch0 + math.radians(rng.uniform(-5.0, 5.0))
            yaw = rng.uniform(-math.pi, math.pi)
            quat = _quat_wxyz_from_euler(roll, pitch, yaw)
            z = _safe_height(quat, rng.uniform(0.001, 0.005)) + rng.uniform(0.0, 0.02)
            harness.reset_pose(quat, z, harness.nominal[:4].copy())
            record = args.video and ep == 0
            t_up, tilt, height, frames = run_pose_episode(harness, quat, z, record)
            if record and frames:
                videos[f"{pose}_standup"] = frames
            if t_up is not None:
                succ += 1
                times.append(t_up)
            tilts.append(tilt)
            heights.append(abs(height - H_CMD))
        row = {
            "pose": pose,
            "episodes": args.episodes,
            "success_rate": succ / args.episodes,
            "mean_standup_time_s": (sum(times) / len(times)) if times else None,
            "final_tilt_deg_mean": float(np.mean(tilts)),
            "final_height_err_m_mean": float(np.mean(heights)),
        }
        pose_rows.append(row)
        print(
            f"[pose] {pose:11s} success={row['success_rate']:.3f} "
            f"rise={row['mean_standup_time_s'] if row['mean_standup_time_s'] is not None else float('nan'):.3f}s "
            f"tilt={row['final_tilt_deg_mean']:.2f} deg",
            flush=True,
        )
    results["fixed_poses"] = pose_rows

    # zero_hold 20 s
    h = NativeHarness(args.onnx, seed=7)
    h.set_command()
    h.reset_standing_exact()
    cam = make_camera()
    xs, vxs, yaws, tilts_hold = [], [], [], []
    frames = []
    prev_yaw = None
    accum_yaw = 0.0
    for step in range(int(20.0 / STEP_DT)):
        h.loop.policy_step()
        x, y, z, tilt, yaw, vx = h.state()
        if prev_yaw is not None:
            d = yaw - prev_yaw
            d = (d + math.pi) % (2.0 * math.pi) - math.pi
            accum_yaw += d
        prev_yaw = yaw
        xs.append((x, y))
        vxs.append(vx)
        yaws.append(yaw)
        tilts_hold.append(tilt)
        if args.video and step % 4 == 0:
            frame = h.render_frame(cam)
            if frame is not None:
                frames.append(frame)
    if args.video and frames:
        videos["zero_hold"] = frames
    zero = {
        "seconds": 20.0,
        "vx_mean": float(np.mean(vxs)),
        "net_dx_m": xs[-1][0] - xs[0][0],
        "net_dy_m": xs[-1][1] - xs[0][1],
        "accum_yaw_deg": math.degrees(accum_yaw),
        "tilt_deg_mean": float(np.mean(tilts_hold)),
        "tilt_deg_final": tilts_hold[-1],
    }
    results["zero_hold_20s"] = zero
    print(
        f"[zero] vx_mean={zero['vx_mean']:+.4f} net_dx={zero['net_dx_m']:+.3f} "
        f"accum_yaw={zero['accum_yaw_deg']:+.2f} deg tilt_mean={zero['tilt_deg_mean']:.2f}",
        flush=True,
    )

    # 指令跟踪 smoke（各 3 s，从干净站立起步）
    smoke_rows = []
    for name, fields, probe in (
        ("vx_+1.0", {"lin_vel_x": 1.0}, "vx"),
        ("vx_-1.0", {"lin_vel_x": -1.0}, "vx"),
        ("yaw_+2.5", {"ang_vel_yaw": 2.5}, "yaw_rate"),
        # runtime 契约硬夹 height ∈ [0.20, 0.32]（policy_descriptor.py），
        # 训练部署范围 0.195-0.39 暂不可达 —— 用可达边界做 smoke。
        ("height_0.20", {"height": 0.20}, "height"),
        ("height_0.32", {"height": 0.32}, "height"),
    ):
        h = NativeHarness(args.onnx, seed=11)
        h.set_command()
        h.reset_standing_exact()
        h.set_command(**fields)
        vals, tilts_s = [], []
        prev_yaw = None
        for _step in range(int(3.0 / STEP_DT)):
            h.loop.policy_step()
            x, y, z, tilt, yaw, vx = h.state()
            tilts_s.append(tilt)
            if probe == "vx":
                vals.append(vx)
            elif probe == "height":
                vals.append(z)
            else:
                if prev_yaw is not None:
                    d = (yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
                    vals.append(d / STEP_DT)
                prev_yaw = yaw
        tail = vals[len(vals) // 2 :]
        smoke_rows.append(
            {
                "case": name,
                "target": next(iter(fields.values())),
                "tail_mean": float(np.mean(tail)),
                "tilt_deg_max": float(np.max(tilts_s)),
            }
        )
        print(
            f"[smoke] {name:12s} target={smoke_rows[-1]['target']:+.3f} "
            f"tail_mean={smoke_rows[-1]['tail_mean']:+.3f} tilt_max={smoke_rows[-1]['tilt_deg_max']:.2f}",
            flush=True,
        )
    results["command_smoke"] = smoke_rows

    out_json = args.out_dir / "native_acceptance.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_json}", flush=True)

    if videos:
        import mediapy

        for name, frames in videos.items():
            path = args.out_dir / f"{name}.mp4"
            fps = 25.0 if "standup" in name else 12.5
            mediapy.write_video(str(path), frames, fps=fps)
            print(f"[video] {path} ({len(frames)} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
