"""录制原生 MuJoCo 下倒置 SerialLeg 的机身外部翻转扭矩扫频视频。

实验不加载策略，也不向六个 policy 关节施加电机力矩；MJCF 内的 300 N 膝气弹簧、
关节被动参数、闭链约束和碰撞均保持生效。外部扭矩通过 ``xfrc_applied`` 施加到
``base_link`` 质心，方向为机身坐标系 +Y（pitch）轴。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_closed_chain_v3_train_obb_trim.xml"
)
DEFAULT_TORQUES_NM = (2.0, 4.0, 6.0, 8.0, 10.0)
INVERTED_QUATERNION_WXYZ = np.asarray((0.0, 0.0, 1.0, 0.0), dtype=np.float64)


@dataclass
class TorqueTrial:
    """一档外部扭矩的独立 MuJoCo 状态与诊断。"""

    torque_nm: float
    data: mujoco.MjData
    renderer: mujoco.Renderer | None
    camera: mujoco.MjvCamera
    released: bool = False
    release_time_s: float | None = None
    first_upright_time_s: float | None = None
    trajectory: dict[str, list[float]] = field(
        default_factory=lambda: {
            "time_s": [],
            "applied_torque_nm": [],
            "tilt_deg": [],
            "base_x_m": [],
            "base_y_m": [],
            "base_z_m": [],
            "base_ang_speed_rad_s": [],
        }
    )


def _font(size: int) -> ImageFont.ImageFont:
    """优先加载跨平台 TrueType 字体，失败时回退 Pillow 默认字体。"""

    candidates = (
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """计算单个碰撞 geom 的世界系最低 z，算法与 sim2x adapter 保持一致。"""

    geom_type = model.geom_type[geom_id]
    position = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vertex_address : vertex_address + vertex_count]
        return float(np.min(vertices @ rotation[2, :]) + position[2])
    if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        axis_z = abs(float(rotation[2, 2]))
        radial_z = math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        half_length = float(size[1])
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half_length += float(size[0])
        extent = half_length * axis_z + float(size[0]) * radial_z
        return float(position[2] - extent)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        extent = float(np.dot(np.abs(rotation[2, :]), size[:3]))
        return float(position[2] - extent)
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = float(np.linalg.norm(rotation[2, :] * size[:3]))
        return float(position[2] - extent)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(position[2] - size[0])
    return float(position[2] - model.geom_rbound[geom_id])


def _minimum_robot_collision_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """返回机器人全部可碰撞 geom 的最低世界 z，忽略 world floor。"""

    values: list[float] = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        values.append(_geom_min_z(model, data, geom_id))
    if not values:
        raise RuntimeError("SerialLeg 模型未找到机器人碰撞 geom")
    return min(values)


def _tilt_deg(data: mujoco.MjData, base_body_id: int) -> float:
    """返回机身 z 轴相对世界 +z 的夹角，0° 直立、180° 倒置。"""

    rotation = data.xmat[base_body_id].reshape(3, 3)
    return math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))


def _base_ang_speed(data: mujoco.MjData, base_body_id: int) -> float:
    """返回机身空间角速度模长。"""

    return float(np.linalg.norm(data.cvel[base_body_id, :3]))


def _prepare_model(model_path: Path) -> mujoco.MjModel:
    """加载原生 MuJoCo 模型，并仅在渲染层隐藏机器人碰撞网格。"""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != 0 and model.geom_group[geom_id] == 0:
            model.geom_rgba[geom_id, 3] = 0.0
    return model


def _make_camera() -> mujoco.MjvCamera:
    """生成便于观察 pitch 翻转的侧视相机。"""

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.35
    camera.azimuth = 90.0
    camera.elevation = -12.0
    return camera


def _reset_inverted(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_qpos_address: int,
    standing_keyframe_id: int,
    floor_clearance_m: float,
) -> None:
    """从 standing keyframe 保留关节姿态，把机身精确重置为 pitch 倒置。"""

    mujoco.mj_resetDataKeyframe(model, data, standing_keyframe_id)
    data.qvel[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    data.qpos[base_qpos_address : base_qpos_address + 3] = (0.0, 0.0, 0.16)
    data.qpos[base_qpos_address + 3 : base_qpos_address + 7] = INVERTED_QUATERNION_WXYZ
    mujoco.mj_forward(model, data)
    minimum_z = _minimum_robot_collision_z(model, data)
    data.qpos[base_qpos_address + 2] += float(floor_clearance_m) - minimum_z
    mujoco.mj_forward(model, data)


def _apply_trial_step(
    model: mujoco.MjModel,
    trial: TorqueTrial,
    *,
    base_body_id: int,
    settle_s: float,
    assist_max_s: float,
    release_tilt_deg: float,
) -> None:
    """推进一个 solver step，并按近直立门控撤掉外部扭矩。"""

    data = trial.data
    tilt = _tilt_deg(data, base_body_id)
    assist_deadline = settle_s + assist_max_s
    if (
        not trial.released
        and data.time >= settle_s
        and (tilt <= release_tilt_deg or data.time >= assist_deadline)
    ):
        trial.released = True
        trial.release_time_s = float(data.time)

    assist_active = data.time >= settle_s and not trial.released
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    if assist_active:
        rotation = data.xmat[base_body_id].reshape(3, 3)
        body_pitch_axis_world = rotation[:, 1]
        data.xfrc_applied[base_body_id, 3:6] = body_pitch_axis_world * float(trial.torque_nm)

    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError(f"{trial.torque_nm:g} N·m 试验出现非有限状态")

    tilt = _tilt_deg(data, base_body_id)
    if trial.first_upright_time_s is None and tilt <= 15.0:
        trial.first_upright_time_s = float(data.time)
    base_position = data.xpos[base_body_id]
    values = {
        "time_s": float(data.time),
        "applied_torque_nm": float(trial.torque_nm if assist_active else 0.0),
        "tilt_deg": tilt,
        "base_x_m": float(base_position[0]),
        "base_y_m": float(base_position[1]),
        "base_z_m": float(base_position[2]),
        "base_ang_speed_rad_s": _base_ang_speed(data, base_body_id),
    }
    for name, value in values.items():
        trial.trajectory[name].append(value)


def _annotate_frame(
    frame: np.ndarray,
    trial: TorqueTrial,
    *,
    base_body_id: int,
    settle_s: float,
) -> np.ndarray:
    """给单档视频叠加扭矩、时间、倾角与辅助开关。"""

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    title_font = _font(max(20, image.width // 28))
    detail_font = _font(max(16, image.width // 38))
    tilt = _tilt_deg(trial.data, base_body_id)
    assist_on = trial.data.time >= settle_s and not trial.released
    draw.rectangle((0, 0, image.width, 78), fill=(12, 18, 28))
    draw.text(
        (16, 8),
        f"External pitch torque: {trial.torque_nm:+g} N m",
        font=title_font,
        fill=(255, 220, 70),
    )
    draw.text(
        (16, 46),
        f"t={trial.data.time:5.2f}s   tilt={tilt:6.1f}deg   assist={'ON' if assist_on else 'OFF'}",
        font=detail_font,
        fill=(235, 240, 248),
    )
    return np.asarray(image)


def _render_trial_frame(
    trial: TorqueTrial,
    *,
    base_body_id: int,
    settle_s: float,
) -> np.ndarray:
    """渲染一档试验并添加诊断文字。"""

    if trial.renderer is None:
        raise RuntimeError("请求视频帧时 renderer 未初始化")
    base_position = trial.data.xpos[base_body_id]
    trial.camera.lookat[:] = base_position
    trial.camera.lookat[2] = max(0.15, float(base_position[2]))
    trial.renderer.update_scene(trial.data, camera=trial.camera)
    frame = np.asarray(trial.renderer.render()).copy()
    return _annotate_frame(
        frame,
        trial,
        base_body_id=base_body_id,
        settle_s=settle_s,
    )


def _compose_grid(
    frames: list[np.ndarray],
    *,
    time_s: float,
    tile_width: int = 480,
    tile_height: int = 360,
) -> np.ndarray:
    """把多档扭矩画面排成同步对比视频。"""

    if len(frames) == 8:
        canvas = Image.new("RGB", (tile_width * 4, tile_height * 2), color=(12, 18, 28))
        for index, frame in enumerate(frames):
            tile = Image.fromarray(frame).resize(
                (tile_width, tile_height),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(tile, ((index % 4) * tile_width, (index // 4) * tile_height))
        return np.asarray(canvas)

    canvas = Image.new("RGB", (tile_width * 3, tile_height * 2), color=(12, 18, 28))
    for index, frame in enumerate(frames):
        tile = Image.fromarray(frame).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        canvas.paste(tile, ((index % 3) * tile_width, (index // 3) * tile_height))

    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    detail_font = _font(20)
    x0 = 2 * tile_width + 24
    y0 = tile_height + 28
    draw.text((x0, y0), "Native MuJoCo torque sweep", font=title_font, fill=(255, 220, 70))
    lines = (
        f"t = {time_s:5.2f} s",
        "motors = 0 N m",
        "gas springs = 300 N",
        "axis = signed base Y (pitch)",
        "release at tilt <= 30 deg",
    )
    for line_index, line in enumerate(lines):
        draw.text(
            (x0, y0 + 48 + 34 * line_index),
            line,
            font=detail_font,
            fill=(232, 238, 247),
        )
    return np.asarray(canvas)


def _video_writer(path: Path, fps: int):
    """创建使用 imageio-ffmpeg 的流式 H.264 writer。"""

    return imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="warning",
    )


def _parse_torques(value: str) -> tuple[float, ...]:
    """解析逗号分隔扭矩并拒绝零、非有限值与重复项。"""

    torques = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not torques or any(not math.isfinite(item) or item == 0.0 for item in torques):
        raise argparse.ArgumentTypeError("扭矩列表必须包含有限非零值")
    if len(set(torques)) != len(torques):
        raise argparse.ArgumentTypeError("扭矩列表不得重复")
    return torques


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--torques",
        type=_parse_torques,
        default=DEFAULT_TORQUES_NM,
        help="逗号分隔的 body-Y 外部扭矩，单位 N·m",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--assist-max-s", type=float, default=3.0)
    parser.add_argument("--total-s", type=float, default=5.0)
    parser.add_argument("--release-tilt-deg", type=float, default=30.0)
    parser.add_argument("--floor-clearance-m", type=float, default=0.003)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.settle_s < 0.0 or args.assist_max_s <= 0.0:
        raise ValueError("settle-s 必须非负且 assist-max-s 必须为正")
    if args.total_s <= args.settle_s:
        raise ValueError("total-s 必须大于 settle-s")
    if not 0.0 < args.release_tilt_deg < 180.0:
        raise ValueError("release-tilt-deg 必须位于 (0, 180)")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("fps 与画面尺寸必须为正数")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _prepare_model(model_path)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    standing_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing")
    if base_body_id < 0 or standing_keyframe_id < 0:
        raise RuntimeError("MJCF 缺少 base_link 或 standing keyframe")
    base_joint_id = int(model.body_jntadr[base_body_id])
    base_qpos_address = int(model.jnt_qposadr[base_joint_id])

    trials: list[TorqueTrial] = []
    for torque_nm in args.torques:
        data = mujoco.MjData(model)
        _reset_inverted(
            model,
            data,
            base_qpos_address=base_qpos_address,
            standing_keyframe_id=standing_keyframe_id,
            floor_clearance_m=args.floor_clearance_m,
        )
        renderer = None if args.no_video else mujoco.Renderer(model, args.height, args.width)
        trials.append(
            TorqueTrial(
                torque_nm=float(torque_nm),
                data=data,
                renderer=renderer,
                camera=_make_camera(),
            )
        )

    individual_writers: dict[float, object] = {}
    grid_writer = None
    if not args.no_video:
        for trial in trials:
            path = out_dir / f"inverted_pitch_torque_{trial.torque_nm:g}Nm.mp4"
            individual_writers[trial.torque_nm] = _video_writer(path, args.fps)
        grid_writer = _video_writer(out_dir / "inverted_pitch_torque_comparison.mp4", args.fps)

    try:
        frame_times = np.arange(0.0, args.total_s + 0.5 / args.fps, 1.0 / args.fps)
        for frame_time in frame_times:
            while trials[0].data.time + 0.5 * model.opt.timestep < frame_time:
                for trial in trials:
                    _apply_trial_step(
                        model,
                        trial,
                        base_body_id=base_body_id,
                        settle_s=args.settle_s,
                        assist_max_s=args.assist_max_s,
                        release_tilt_deg=args.release_tilt_deg,
                    )
            if args.no_video:
                continue
            frames = [
                _render_trial_frame(
                    trial,
                    base_body_id=base_body_id,
                    settle_s=args.settle_s,
                )
                for trial in trials
            ]
            for trial, frame in zip(trials, frames, strict=True):
                individual_writers[trial.torque_nm].append_data(frame)
            assert grid_writer is not None
            grid_writer.append_data(_compose_grid(frames, time_s=float(frame_time)))
    finally:
        for writer in individual_writers.values():
            writer.close()
        if grid_writer is not None:
            grid_writer.close()
        for trial in trials:
            if trial.renderer is not None:
                trial.renderer.close()

    summaries: list[dict[str, float | None]] = []
    for trial in trials:
        arrays = {
            name: np.asarray(values, dtype=np.float64) for name, values in trial.trajectory.items()
        }
        np.savez_compressed(
            out_dir / f"inverted_pitch_torque_{trial.torque_nm:g}Nm_trajectory.npz",
            **arrays,
        )
        tilt = arrays["tilt_deg"]
        angular_speed = arrays["base_ang_speed_rad_s"]
        base_z = arrays["base_z_m"]
        summary = {
            "torque_nm": trial.torque_nm,
            "release_time_s": trial.release_time_s,
            "first_upright_time_s": trial.first_upright_time_s,
            "minimum_tilt_deg": float(np.min(tilt)),
            "final_tilt_deg": float(tilt[-1]),
            "maximum_base_ang_speed_rad_s": float(np.max(angular_speed)),
            "maximum_base_height_m": float(np.max(base_z)),
            "final_base_height_m": float(base_z[-1]),
        }
        summaries.append(summary)
        print(
            f"[torque] {trial.torque_nm:>4g} N·m | "
            f"release={trial.release_time_s!s:>6} s | "
            f"upright={trial.first_upright_time_s!s:>6} s | "
            f"min_tilt={summary['minimum_tilt_deg']:.1f}° | "
            f"final_tilt={summary['final_tilt_deg']:.1f}° | "
            f"max_omega={summary['maximum_base_ang_speed_rad_s']:.2f} rad/s",
            flush=True,
        )

    manifest = {
        "engine": f"MuJoCo {mujoco.__version__}",
        "model": str(model_path),
        "model_timestep_s": float(model.opt.timestep),
        "motors_enabled": False,
        "knee_gas_spring_force_n": 300.0,
        "external_torque_frame": "signed base_link body Y axis",
        "settle_s": float(args.settle_s),
        "assist_max_s": float(args.assist_max_s),
        "total_s": float(args.total_s),
        "release_tilt_deg": float(args.release_tilt_deg),
        "floor_clearance_m": float(args.floor_clearance_m),
        "fps": int(args.fps),
        "trials": summaries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
