"""用原生 MuJoCo sim2x 跑确定性策略、录制 MP4，并按指令阶段输出跟踪与抖动指标。

与 `se3-sim2x` 共用 PolicyRuntime / MujocoPolicyAdapter / PolicyControlLoop，不复制任何观测或
动作数学；只是把 Viser 换成离屏 mujoco.Renderer + imageio 写 MP4，并在每个 policy tick 采样
机身速度、倾角、轮速与动作目标的变化量。

用法：
    uv run python scripts/record_sim2x_video.py --onnx <model.onnx> --output <out.mp4> [--physics-hz 1000]

指令阶段用 --phase "秒,vx,yaw,标签" 重复给出；默认为站立 → 前进 1.0 → 停 → 原地转 2.0 → 前进 1.5 → 停。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from se3_runtime import PolicyControlLoop, PolicyRuntime
from se3_runtime_mujoco.adapter import MujocoPolicyAdapter

DEFAULT_PHASES = [
    "3,0,0,stand",
    "5,1.0,0,forward 1.0 m/s",
    "3,0,0,stop",
    "4,0,2.0,turn 2.0 rad/s",
    "5,1.5,0,forward 1.5 m/s",
    "4,0,0,stop",
]


@dataclass
class Phase:
    duration_s: float
    lin_vel_x: float
    yaw_rate: float
    label: str


def _parse_phase(text: str) -> Phase:
    parts = [p.strip() for p in text.split(",", 3)]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"阶段格式应为 '秒,vx,yaw,标签'，收到 {text!r}")
    return Phase(float(parts[0]), float(parts[1]), float(parts[2]), parts[3])


def _decoded_vector(action, name_hint: str) -> np.ndarray | None:
    """从 DecodedPolicyAction 里按字段名关键字取出 ndarray（腿目标 / 轮目标）。"""
    for f in fields(action):
        if name_hint in f.name:
            value = getattr(action, f.name)
            if isinstance(value, np.ndarray):
                return value.astype(np.float64).ravel()
    return None


def _hide_collision_geoms(model: mujoco.MjModel) -> None:
    """机器人碰撞 geom 全透明，只留视觉网格与地面。"""
    for geom_id in range(model.ngeom):
        is_collision = model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
        if is_collision and model.geom_bodyid[geom_id] != 0:
            model.geom_rgba[geom_id, 3] = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="MP4 输出路径")
    parser.add_argument("--physics-hz", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0, help="动作延迟采样种子")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--phase", action="append", type=_parse_phase, default=None)
    parser.add_argument(
        "--settle-s", type=float, default=1.0, help="每阶段统计时跳过的起始过渡秒数"
    )
    args = parser.parse_args()
    phases = args.phase or [_parse_phase(p) for p in DEFAULT_PHASES]

    runtime = PolicyRuntime.load(args.onnx, action_delay_random_seed=args.seed)
    adapter = MujocoPolicyAdapter(
        runtime.contract,
        artifact_path=runtime.bundle.path,
        solver_dt_s=1.0 / args.physics_hz,
    )
    loop = PolicyControlLoop(runtime, adapter)
    model, data = adapter.model, adapter.data
    policy_dt = float(runtime.contract.timing.policy_dt_s)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.contract.robot.base_link)

    _hide_collision_geoms(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = base_id
    camera.distance = 1.6
    camera.azimuth = 140.0
    camera.elevation = -18.0
    scene_option = mujoco.MjvOption()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264", quality=8, macro_block_size=1
    )

    loop.reset()
    records: list[dict[str, float]] = []
    prev_leg = prev_wheel = None
    frame_period = 1.0 / args.fps
    next_frame_t = 0.0
    t = 0.0
    fallen = False
    for phase_index, phase in enumerate(phases):
        adapter.set_command_field("velocity_height", "lin_vel_x", phase.lin_vel_x)
        adapter.set_command_field("velocity_height", "ang_vel_yaw", phase.yaw_rate)
        phase_start = t
        steps = round(phase.duration_s / policy_dt)
        for _ in range(steps):
            result = loop.policy_step()
            t = float(data.time)
            rot = data.xmat[base_id].reshape(3, 3)
            v_body = rot.T @ data.qvel[0:3]
            yaw_rate = float(data.qvel[5])
            gravity = np.asarray(
                adapter.read_policy_input().robot.projected_gravity_body, dtype=np.float64
            ).ravel()
            tilt_deg = math.degrees(math.acos(float(np.clip(-gravity[2], -1.0, 1.0))))
            joint_vel = np.asarray(adapter.policy_joint_velocity, dtype=np.float64).ravel()
            leg = _decoded_vector(result.last_decoded_action, "leg")
            wheel = _decoded_vector(result.last_decoded_action, "wheel")
            leg_rate = (
                float(np.abs(leg - prev_leg).mean())
                if leg is not None and prev_leg is not None
                else float("nan")
            )
            wheel_rate = (
                float(np.abs(wheel - prev_wheel).mean())
                if wheel is not None and prev_wheel is not None
                else float("nan")
            )
            prev_leg, prev_wheel = leg, wheel
            base_z = float(data.xpos[base_id][2])
            if tilt_deg > 60.0 or base_z < 0.08:
                fallen = True
            records.append(
                {
                    "t": t,
                    "phase": phase.label,
                    "phase_index": float(phase_index),
                    "in_window": float(t - phase_start >= args.settle_s),
                    "cmd_vx": phase.lin_vel_x,
                    "cmd_yaw": phase.yaw_rate,
                    "vx": float(v_body[0]),
                    "vy": float(v_body[1]),
                    "yaw_rate": yaw_rate,
                    "tilt_deg": tilt_deg,
                    "base_z": base_z,
                    "wheel_l": float(joint_vel[4]),
                    "wheel_r": float(joint_vel[5]),
                    "leg_target_rate_rad": leg_rate,
                    "wheel_target_rate_rad_s": wheel_rate,
                }
            )
            if t + 1e-9 >= next_frame_t:
                next_frame_t += frame_period
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                lines = [
                    f"{args.onnx.name}   t = {t:5.2f} s   phase: {phase.label}",
                    f"cmd vx {phase.lin_vel_x:+.2f} m/s  yaw {phase.yaw_rate:+.2f} rad/s",
                    f"vx {v_body[0]:+.2f} m/s  yaw {yaw_rate:+.2f} rad/s  tilt {tilt_deg:4.1f} deg  z {base_z:.3f} m",
                    f"wheel L/R {joint_vel[4]:+6.1f} / {joint_vel[5]:+6.1f} rad/s",
                ]
                for i, line in enumerate(lines):
                    draw.text((12, 10 + 18 * i), line, fill=(255, 255, 255))
                writer.append_data(np.asarray(frame))
    writer.close()

    # 分阶段指标（跳过每阶段起始过渡）
    summary = []
    for phase_index, phase in enumerate(phases):
        rows = [r for r in records if r["phase_index"] == phase_index and r["in_window"] > 0]
        if not rows:
            continue
        arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0] if k != "phase"}
        summary.append(
            {
                "phase": phase.label,
                "cmd_vx": phase.lin_vel_x,
                "cmd_yaw": phase.yaw_rate,
                "vx_mean": float(arr["vx"].mean()),
                "vx_abs_err_mean": float(np.abs(arr["vx"] - phase.lin_vel_x).mean()),
                "yaw_mean": float(arr["yaw_rate"].mean()),
                "yaw_abs_err_mean": float(np.abs(arr["yaw_rate"] - phase.yaw_rate).mean()),
                "tilt_mean_deg": float(arr["tilt_deg"].mean()),
                "tilt_std_deg": float(arr["tilt_deg"].std()),
                "base_z_mean": float(arr["base_z"].mean()),
                "wheel_vel_std": float(np.std(np.concatenate([arr["wheel_l"], arr["wheel_r"]]))),
                "wheel_vel_hf_rms": float(
                    np.sqrt(
                        np.mean(np.diff(arr["wheel_l"]) ** 2 + np.diff(arr["wheel_r"]) ** 2) / 2.0
                    )
                ),
                "leg_target_rate_deg_per_step": float(
                    np.degrees(np.nanmean(arr["leg_target_rate_rad"]))
                ),
                "wheel_target_rate_rad_s_per_step": float(
                    np.nanmean(arr["wheel_target_rate_rad_s"])
                ),
            }
        )
    report = {
        "onnx": str(args.onnx),
        "physics_hz": args.physics_hz,
        "policy_dt_s": policy_dt,
        "fallen": fallen,
        "phases": summary,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"视频: {args.output}  帧数 {int(next_frame_t * args.fps)}  倒地: {fallen}")
    print(
        f"{'phase':18s} {'vx_cmd':>6} {'vx':>6} {'|err|':>6} {'yaw_cmd':>7} {'yaw':>6} {'|err|':>6} {'tilt':>5} {'tiltσ':>5} {'z':>6} {'wheelΔ':>7} {'legΔ°':>6} {'whlΔ':>6}"
    )
    for s in summary:
        print(
            f"{s['phase']:18s} {s['cmd_vx']:6.2f} {s['vx_mean']:6.2f} {s['vx_abs_err_mean']:6.2f} {s['cmd_yaw']:7.2f} {s['yaw_mean']:6.2f} "
            f"{s['yaw_abs_err_mean']:6.2f} {s['tilt_mean_deg']:5.1f} {s['tilt_std_deg']:5.2f} {s['base_z_mean']:6.3f} {s['wheel_vel_hf_rms']:7.2f} "
            f"{s['leg_target_rate_deg_per_step']:6.2f} {s['wheel_target_rate_rad_s_per_step']:6.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
