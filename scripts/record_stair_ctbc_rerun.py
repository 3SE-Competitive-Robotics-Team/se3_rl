"""录制 stair warm-start rollout 的 CTBC 前馈 Rerun 回放。"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

TASK_NAME = "SE3-WheelLegged-Stair-GRU"
WATCH_USE_TRAIN_ENV_ENV = "SE3_WATCH_USE_TRAIN_ENV"
WATCH_ITER_ENV = "SE3_WATCH_ITER"
WATCH_TERRAIN_LEVEL_ENV = "SE3_WATCH_TERRAIN_LEVEL"
WATCH_COMMAND_HEIGHT_ENV = "SE3_WATCH_COMMAND_HEIGHT"
TRAIN_VIEW_ITER_ENV = "SE3_TRAIN_VIEW_ITER"
TRAIN_VIEW_TERRAIN_LEVEL_ENV = "SE3_TRAIN_VIEW_TERRAIN_LEVEL"
TRAIN_VIEW_COMMAND_HEIGHT_ENV = "SE3_TRAIN_VIEW_COMMAND_HEIGHT"

ACTION_LABELS = (
    "left_front",
    "left_drive",
    "right_front",
    "right_drive",
    "left_wheel",
    "right_wheel",
)
SIDE_LABELS = ("left", "right")
_WHEEL_RADIUS_M = 0.060


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless MuJoCo/MJLab rollout，录制 stair CTBC 前馈到 Rerun .rrd。"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--record-env-id", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument(
        "--iteration", type=int, default=0, help="固定 CTBC 课程迭代，0 表示 kff=1。"
    )
    parser.add_argument(
        "--no-fixed-iteration",
        action="store_true",
        help="不固定 CTBC 课程迭代，让 local iter 像诊断脚本一样随 rollout 自然推进。",
    )
    parser.add_argument("--terrain-level", type=int, default=0)
    parser.add_argument("--command-vx", type=float, default=0.6)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-height", type=float, default=0.39)
    parser.add_argument("--ff-x-m", type=float, default=None)
    parser.add_argument("--ff-lift-m", type=float, default=None)
    parser.add_argument("--ff-period-s", type=float, default=None)
    parser.add_argument("--ff-rise-ratio", type=float, default=None)
    parser.add_argument("--ff-hold-ratio", type=float, default=None)
    parser.add_argument("--ff-wheel-action", type=float, default=None)
    parser.add_argument("--ctbc-profile", type=Path, default=None)
    parser.add_argument("--trigger-mode", choices=("force", "pitch"), default=None)
    parser.add_argument("--force-threshold-n", type=float, default=None)
    parser.add_argument("--contact-window", type=int, default=None)
    parser.add_argument("--pitch-threshold-deg", type=float, default=None)
    parser.add_argument("--pitch-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-x-offset-m", type=float, default=None)
    parser.add_argument("--start-y-offset-m", type=float, default=None)
    parser.add_argument("--start-yaw-deg", type=float, default=None)
    parser.add_argument("--pass-support-ratio", type=float, default=0.7)
    parser.add_argument("--pass-wheel-contact-n", type=float, default=1.0)
    parser.add_argument("--pass-wheel-clearance-tol-m", type=float, default=0.035)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/rerun/stair_ctbc/base_model_4999_ctbc_kff1.rrd"),
    )
    parser.add_argument("--spawn", action="store_true", help="录制同时打开 Rerun viewer。")
    parser.add_argument("--memory-limit", default="1GB")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--camera-distance", type=float, default=1.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--manual-trigger-time",
        type=float,
        default=None,
        help="可选：在指定秒数强制触发 CTBC，用于单独观察前馈形状。",
    )
    parser.add_argument(
        "--manual-trigger-side",
        choices=("left", "right", "both"),
        default="left",
        help="手动触发时激活的侧别。",
    )
    return parser.parse_args()


def _set_watch_env(args: argparse.Namespace) -> None:
    os.environ[WATCH_USE_TRAIN_ENV_ENV] = "1"
    if args.no_fixed_iteration:
        for name in (WATCH_ITER_ENV, TRAIN_VIEW_ITER_ENV):
            os.environ.pop(name, None)
    else:
        for name in (WATCH_ITER_ENV, TRAIN_VIEW_ITER_ENV):
            os.environ[name] = str(int(args.iteration))
    for name in (WATCH_TERRAIN_LEVEL_ENV, TRAIN_VIEW_TERRAIN_LEVEL_ENV):
        os.environ[name] = str(int(args.terrain_level))
    for name in (WATCH_COMMAND_HEIGHT_ENV, TRAIN_VIEW_COMMAND_HEIGHT_ENV):
        os.environ[name] = str(float(args.command_height))


def _load_policy(env, checkpoint: Path, task: str, device: str):
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    agent_cfg = load_rl_cfg(task)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()
    return policy


def _tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(torch.nan_to_num(value.float()).mean().item())
    return float(value)


def _finite_np(value: torch.Tensor) -> np.ndarray:
    return np.nan_to_num(_tensor_np(value), nan=0.0, posinf=0.0, neginf=0.0)


def _wheel_body_ids(base_env) -> list[int]:
    attr_name = "_record_stair_wheel_body_ids"
    cached = getattr(base_env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = base_env.scene["robot"]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    setattr(base_env, attr_name, body_ids)
    return body_ids


def _wheel_terrain_measurements(base_env, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    robot = base_env.scene["robot"]
    sensor = base_env.scene["wheel_height_sensor"]
    heights = torch.nan_to_num(sensor.data.heights, nan=0.0, posinf=0.0, neginf=0.0)
    if heights.ndim == 1:
        heights = heights.unsqueeze(-1)
    if heights.shape[1] < len(body_ids):
        heights = heights.expand(-1, len(body_ids))
    heights = heights[:, : len(body_ids)]
    wheel_pos_w = torch.nan_to_num(
        robot.data.body_link_pos_w[:, body_ids, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    terrain_z = wheel_pos_w[:, :, 2] - heights
    return terrain_z, heights


def _wheel_contact_force(base_env) -> torch.Tensor:
    sensor = base_env.scene["wheel_sensor"]
    data = sensor.data
    if data.force is None:
        return torch.zeros(base_env.num_envs, 2, device=base_env.device)
    force = torch.nan_to_num(data.force, nan=0.0, posinf=0.0, neginf=0.0)
    force_mag = torch.linalg.vector_norm(force, dim=-1)
    if force_mag.ndim == 3:
        force_mag = force_mag.amax(dim=-1)
    if force_mag.ndim == 1:
        force_mag = force_mag.unsqueeze(-1)
    if force_mag.shape[1] < 2:
        force_mag = force_mag.expand(-1, 2)
    return force_mag[:, :2]


def _configure_cfg(args: argparse.Namespace):
    from mjlab.tasks.registry import load_env_cfg

    import se3_train  # noqa: F401

    cfg = load_env_cfg(args.task, play=False)
    cfg.seed = int(args.seed)
    cfg.scene.num_envs = int(args.num_envs)

    command_cfg = cfg.commands.get("velocity_height")
    if command_cfg is not None:
        command_cfg.lin_vel_x_range = (float(args.command_vx), float(args.command_vx))
        command_cfg.ang_vel_yaw_range = (float(args.command_yaw), float(args.command_yaw))
        command_cfg.height_range = (float(args.command_height), float(args.command_height))
        command_cfg.standing_height_range = (float(args.command_height), float(args.command_height))
        command_cfg.height_resample_on_reset_only = True

    # 诊断录制用固定场景，去掉随机扰动，避免把 CTBC 效果和域随机化混在一起。
    cfg.curriculum = dict(getattr(cfg, "curriculum", {}) or {})
    for name in ("command_vel", "command_height", "terrain_levels", "push_disturbance"):
        cfg.curriculum.pop(name, None)

    cfg.events = dict(getattr(cfg, "events", {}) or {})
    for name in (
        "friction",
        "restitution",
        "base_mass",
        "inertia",
        "com",
        "pd_gains",
        "default_dof_pos",
        "push_robots",
    ):
        cfg.events.pop(name, None)

    # 不用 termination 自动 reset，才能把摔倒/卡住过程完整录下来。
    cfg.terminations = {}

    if "init_stair_climb_state" in cfg.events:
        term = cfg.events["init_stair_climb_state"]
        params = dict(term.params or {})
        overrides = {
            "ff_x_m": args.ff_x_m,
            "ff_lift_m": args.ff_lift_m,
            "ff_period_s": args.ff_period_s,
            "ff_rise_ratio": args.ff_rise_ratio,
            "ff_hold_ratio": args.ff_hold_ratio,
            "ff_wheel_action": args.ff_wheel_action,
            "profile_path": args.ctbc_profile,
            "trigger_mode": args.trigger_mode,
            "force_threshold_n": args.force_threshold_n,
            "contact_window": args.contact_window,
            "pitch_threshold_deg": args.pitch_threshold_deg,
            "pitch_window": args.pitch_window,
        }
        for name, value in overrides.items():
            if value is not None:
                if name in {"contact_window", "pitch_window"}:
                    params[name] = int(value)
                elif name == "profile_path":
                    params[name] = value
                elif name == "trigger_mode":
                    params[name] = str(value)
                else:
                    params[name] = float(value)

        cfg.events["init_stair_climb_state"] = replace(term, params=params)
    return cfg


def _maybe_set_start_pose(base_env, args: argparse.Namespace) -> None:
    if (
        args.start_x_offset_m is None
        and args.start_y_offset_m is None
        and args.start_yaw_deg is None
    ):
        return
    robot = base_env.scene["robot"]
    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    pos = torch.nan_to_num(robot.data.root_link_pos_w, nan=0.0).clone()
    quat = torch.nan_to_num(robot.data.root_link_quat_w, nan=0.0).clone()
    origins = base_env.scene.env_origins
    if origins is not None:
        if args.start_x_offset_m is not None:
            pos[:, 0] = origins[:, 0] + float(args.start_x_offset_m)
        if args.start_y_offset_m is not None:
            pos[:, 1] = origins[:, 1] + float(args.start_y_offset_m)
    else:
        if args.start_x_offset_m is not None:
            pos[:, 0] += float(args.start_x_offset_m)
        if args.start_y_offset_m is not None:
            pos[:, 1] += float(args.start_y_offset_m)
    if args.start_yaw_deg is not None:
        yaw = torch.full(
            (base_env.num_envs,),
            math.radians(float(args.start_yaw_deg)),
            device=base_env.device,
        )
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
    vel = torch.zeros(base_env.num_envs, 6, device=base_env.device)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    base_env.sim.forward()


class _MujocoMirror:
    """把 MJLab robot 状态同步到训练 env 的原生 MuJoCo model。"""

    def __init__(self, model: mujoco.MjModel, robot) -> None:
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.root_qpos_addr = self._joint_qpos_addr(
            "floating_base_joint",
            "robot/floating_base_joint",
        )
        self.joint_qpos_addr: dict[int, int] = {}
        for robot_joint_id, name in enumerate(robot.joint_names):
            qpos_addr = self._joint_qpos_addr(name, f"robot/{name}")
            if qpos_addr is not None:
                self.joint_qpos_addr[robot_joint_id] = qpos_addr

    def _joint_qpos_addr(self, *names: str) -> int | None:
        for name in names:
            mj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if mj_joint_id >= 0:
                return int(self.model.jnt_qposadr[mj_joint_id])
        return None

    def sync(self, robot, time_s: float, env_id: int = 0) -> None:
        root_pos = _finite_np(robot.data.root_link_pos_w[env_id]).reshape(3)
        root_quat = _finite_np(robot.data.root_link_quat_w[env_id]).reshape(4)
        norm = float(np.linalg.norm(root_quat))
        if norm > 1.0e-8:
            root_quat = root_quat / norm
        qpos_addr = 0 if self.root_qpos_addr is None else self.root_qpos_addr
        self.data.qpos[qpos_addr : qpos_addr + 3] = root_pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = root_quat
        joint_pos = _finite_np(robot.data.joint_pos[env_id]).reshape(-1)
        for robot_joint_id, qpos_addr in self.joint_qpos_addr.items():
            self.data.qpos[qpos_addr] = joint_pos[robot_joint_id]
        self.data.time = float(time_s)
        mujoco.mj_forward(self.model, self.data)


class _MujocoRgbRenderer:
    """用原生 mujoco.Renderer 渲染训练 env 的 MJCF + terrain。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        width: int,
        height: int,
        distance: float,
        azimuth: float,
        elevation: float,
    ) -> None:
        self.renderer = mujoco.Renderer(model, height=int(height), width=int(width))
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(distance)
        self.camera.azimuth = float(azimuth)
        self.camera.elevation = float(elevation)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        if self.base_body_id < 0:
            self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    def render(self, data: mujoco.MjData) -> np.ndarray:
        if self.base_body_id >= 0:
            base_pos = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
            self.camera.lookat[:] = base_pos + np.asarray([0.25, 0.0, 0.04])
        self.renderer.update_scene(data, camera=self.camera)
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        self.renderer.close()


def _send_blueprint(rr) -> None:
    import rerun.blueprint as rrb

    time_series_view = rrb.TimeSeriesView
    layout = rrb.Horizontal(
        rrb.Spatial2DView(origin="/mujoco_render", name="MuJoCo MJCF render"),
        rrb.Vertical(
            rrb.Spatial3DView(origin="/ctbc_side", name="CTBC side view"),
            rrb.Grid(
                time_series_view(origin="/plots/ctbc", contents="/plots/ctbc/**", name="CTBC"),
                time_series_view(origin="/plots/state", contents="/plots/state/**", name="State"),
                time_series_view(
                    origin="/plots/action", contents="/plots/action/**", name="Action"
                ),
                grid_columns=1,
                name="Signals",
            ),
            row_shares=[0.35, 0.65],
        ),
        column_shares=[0.56, 0.44],
        name="Stair CTBC",
    )
    rr.send_blueprint(
        rrb.Blueprint(
            layout,
            rrb.TimePanel(state="collapsed"),
            collapse_panels=True,
            auto_layout=False,
            auto_views=False,
        ),
        make_active=True,
        make_default=True,
    )


def _log_rgb_frame(rr, image: np.ndarray, *, jpeg_quality: int) -> None:
    rr.log("/mujoco_render/rgb", rr.Image(image).compress(jpeg_quality=int(jpeg_quality)))


def _log_static_stair(rr, terrain_origin: np.ndarray, step_height: float = 0.05) -> None:
    step_depth = 0.80
    half_width = 6.0
    for idx in range(6):
        height = step_height * float(idx + 1)
        center = [
            float(terrain_origin[0]) + 1.0 + idx * step_depth + 0.5 * step_depth,
            float(terrain_origin[1]),
            float(terrain_origin[2]) + 0.5 * height,
        ]
        rr.log(
            f"/world/stair/positive_x_step_{idx}",
            rr.Boxes3D(
                half_sizes=[0.5 * step_depth, half_width, 0.5 * height],
                colors=[145, 145, 145, 120],
            ),
            static=True,
        )
        rr.log(
            f"/world/stair/positive_x_step_{idx}",
            rr.Transform3D(translation=center),
            static=True,
        )


def _log_model_state(viewer, mirror: _MujocoMirror, robot) -> None:
    rr = viewer.rr
    if not viewer.body_paths:
        viewer.log_model(mirror.model)
    for body_id, path in enumerate(viewer.body_paths):
        rr.log(
            path,
            rr.Transform3D(
                translation=np.asarray(mirror.data.xpos[body_id], dtype=np.float32),
                quaternion=_quat_wxyz_to_xyzw(mirror.data.xquat[body_id]),
            ),
        )
    _log_robot_overlay(rr, robot)
    base_id = viewer.follow_body_id
    if base_id >= 0:
        rr.log(
            "/world/base_path",
            rr.Points3D(
                positions=[np.asarray(mirror.data.xpos[base_id], dtype=np.float32)],
                colors=[255, 220, 0, 255],
                radii=0.02,
            ),
        )


def _log_robot_overlay(rr, robot) -> None:
    body_pos = _finite_np(robot.data.body_link_pos_w[0]).astype(np.float32)
    labels = list(robot.body_names)
    colors = []
    for name in labels:
        if name == "base_link":
            colors.append([255, 230, 40, 255])
        elif name.startswith("l_") or name.startswith("lf"):
            colors.append([45, 155, 255, 255])
        else:
            colors.append([255, 120, 50, 255])
    rr.log(
        "/world/robot_overlay/body_points",
        rr.Points3D(
            positions=body_pos,
            labels=labels,
            colors=colors,
            radii=0.04,
        ),
    )
    name_to_id = {name: idx for idx, name in enumerate(labels)}
    strips = []
    for chain in (
        ("base_link", "lf0_Link", "lf1_Link", "l_wheel_Link"),
        ("base_link", "rf0_Link", "rf1_Link", "r_wheel_Link"),
    ):
        ids = [name_to_id[name] for name in chain if name in name_to_id]
        if len(ids) >= 2:
            strips.append(body_pos[ids])
    if strips:
        rr.log("/world/robot_overlay/links", rr.LineStrips3D(strips))


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def _log_scalar(rr, path: str, value: object) -> None:
    rr.log(path, rr.Scalars(scalars=float(value)))


def _log_array(rr, path: str, values: object, labels: tuple[str, ...]) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    for idx, value in enumerate(arr):
        label = labels[idx] if idx < len(labels) else str(idx)
        _log_scalar(rr, f"{path}/{label}", value)


def _log_ctbc_side(rr, action_term, env_id: int) -> None:
    actual_xz = _finite_np(action_term.actual_wheel_xz[env_id]).reshape(2, 2)
    target_xz = _finite_np(action_term.target_wheel_xz[env_id]).reshape(2, 2)
    requested_delta_xz = _finite_np(action_term.ctbc_wheel_delta_xz[env_id]).reshape(2, 2)
    side_y = np.asarray([-0.16, 0.16], dtype=np.float32)
    actual = np.column_stack([actual_xz[:, 0], side_y, actual_xz[:, 1]]).astype(np.float32)
    target = np.column_stack([target_xz[:, 0], side_y, target_xz[:, 1]]).astype(np.float32)
    requested = np.column_stack(
        [requested_delta_xz[:, 0], np.zeros(2, dtype=np.float32), requested_delta_xz[:, 1]]
    ).astype(np.float32)
    rr.log(
        "/ctbc_side/wheel/actual",
        rr.Points3D(
            positions=actual,
            labels=["actual_left", "actual_right"],
            colors=[[43, 132, 255, 255], [154, 91, 216, 255]],
            radii=0.018,
        ),
    )
    rr.log(
        "/ctbc_side/wheel/target",
        rr.Points3D(
            positions=target,
            labels=["target_left", "target_right"],
            colors=[[32, 190, 110, 255], [255, 145, 40, 255]],
            radii=0.018,
        ),
    )
    rr.log(
        "/ctbc_side/wheel/action_delta",
        rr.Arrows3D(origins=actual, vectors=target - actual, colors=[255, 230, 0, 255]),
    )
    rr.log(
        "/ctbc_side/wheel/requested_ctbc_delta",
        rr.Arrows3D(origins=actual, vectors=requested, colors=[255, 80, 80, 180]),
    )
    rr.log(
        "/ctbc_side/reference/ground",
        rr.LineStrips3D([np.asarray([[-0.55, -0.26, 0.0], [0.25, 0.26, 0.0]])]),
        static=True,
    )


def _log_rollout_sample(
    rr,
    base_env,
    action_term,
    reward: torch.Tensor,
    step: int,
    env_id: int,
    initial_wheel_terrain_z: torch.Tensor,
    step_height: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    robot = base_env.scene["robot"]
    state = getattr(base_env, "stair_climb_state", None)
    projected_gravity = torch.nan_to_num(robot.data.projected_gravity_b[env_id].float())
    tilt_deg = float(
        torch.rad2deg(torch.acos(torch.clamp(-projected_gravity[2], -1.0, 1.0))).item()
    )
    pitch_deg = float(
        torch.rad2deg(torch.atan2(projected_gravity[0], -projected_gravity[2])).item()
    )
    base_vx = _scalar(torch.nan_to_num(robot.data.root_link_lin_vel_b[env_id, 0]))
    cmd_vx = 0.0
    cmd_height = 0.0
    try:
        command = base_env.command_manager.get_command("velocity_height")
        cmd_vx = _scalar(command[env_id, 0])
        if command.shape[1] > 4:
            cmd_height = _scalar(command[env_id, 4])
    except Exception:
        pass

    active = np.zeros(2, dtype=np.float32)
    stable = np.zeros(2, dtype=np.float32)
    force = np.zeros(2, dtype=np.float32)
    phase = np.full(2, -1.0, dtype=np.float32)
    kff = 0.0
    ctbc_pitch_abs_deg = abs(pitch_deg)
    if state is not None:
        kff = float(state.kff)
        active = _finite_np((state.ff_phase[env_id] >= 0).float()).reshape(2)
        stable = _finite_np(state.stable_contact[env_id].float()).reshape(2)
        force = _finite_np(state.latest_contact_force[env_id]).reshape(2)
        phase = _finite_np(state.ff_phase[env_id].float()).reshape(2)
        pitch_getter = getattr(state, "latest_pitch_abs_deg", None)
        if isinstance(pitch_getter, torch.Tensor):
            ctbc_pitch_abs_deg = float(torch.nan_to_num(pitch_getter[env_id]).item())

    ctbc_delta = _finite_np(action_term.ctbc_action_delta[env_id]).reshape(-1)
    ctbc_bias = _finite_np(action_term.ctbc_output_bias[env_id]).reshape(-1)
    raw_action = _finite_np(action_term.raw_action[env_id]).reshape(-1)
    unclipped_action = _finite_np(action_term.unclipped_action[env_id]).reshape(-1)
    delayed_action = _finite_np(action_term.delayed_action[env_id]).reshape(-1)
    wheel_body_ids = _wheel_body_ids(base_env)
    wheel_terrain_z, wheel_heights = _wheel_terrain_measurements(base_env, wheel_body_ids)
    wheel_terrain_rise = torch.nan_to_num(
        wheel_terrain_z[env_id] - initial_wheel_terrain_z,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    wheel_force = _wheel_contact_force(base_env)[env_id]
    required_support_rise = float(step_height) * float(args.pass_support_ratio)
    upper_supported = (
        (wheel_terrain_rise >= required_support_rise)
        & (wheel_force >= float(args.pass_wheel_contact_n))
        & (wheel_heights[env_id] <= float(_WHEEL_RADIUS_M) + float(args.pass_wheel_clearance_tol_m))
    )
    wheel_terrain_rise_np = _finite_np(wheel_terrain_rise).reshape(2)
    wheel_height_np = _finite_np(wheel_heights[env_id]).reshape(2)
    wheel_force_np = _finite_np(wheel_force).reshape(2)
    upper_supported_np = _finite_np(upper_supported.float()).reshape(2)

    _log_scalar(rr, "/plots/state/tilt_deg", tilt_deg)
    _log_scalar(rr, "/plots/state/pitch_deg", pitch_deg)
    _log_scalar(rr, "/plots/state/cmd_vx_mps", cmd_vx)
    _log_scalar(rr, "/plots/state/base_vx_mps", base_vx)
    _log_scalar(rr, "/plots/state/cmd_height_m", cmd_height)
    _log_scalar(rr, "/plots/state/base_height_m", robot.data.root_link_pos_w[env_id, 2].item())
    _log_scalar(rr, "/plots/state/base_x_m", robot.data.root_link_pos_w[env_id, 0].item())
    _log_scalar(rr, "/plots/state/reward", _scalar(reward[env_id]))
    _log_array(rr, "/plots/state/wheel_terrain_rise_m", wheel_terrain_rise_np, SIDE_LABELS)
    _log_array(rr, "/plots/state/wheel_height_m", wheel_height_np, SIDE_LABELS)
    _log_array(rr, "/plots/state/wheel_contact_force_n", wheel_force_np, SIDE_LABELS)
    _log_array(rr, "/plots/state/upper_supported", upper_supported_np, SIDE_LABELS)
    _log_scalar(rr, "/plots/state/upper_supported_both", float(np.all(upper_supported_np > 0.5)))
    _log_scalar(rr, "/plots/state/required_support_rise_m", required_support_rise)
    _log_scalar(rr, "/plots/ctbc/kff", kff)
    _log_array(rr, "/plots/ctbc/active", active, SIDE_LABELS)
    _log_array(rr, "/plots/ctbc/stable", stable, SIDE_LABELS)
    _log_array(rr, "/plots/ctbc/force_n", force, SIDE_LABELS)
    _log_scalar(rr, "/plots/ctbc/pitch_abs_deg", ctbc_pitch_abs_deg)
    _log_array(rr, "/plots/ctbc/phase_step", phase, SIDE_LABELS)
    _log_array(rr, "/plots/ctbc/action_delta", ctbc_delta, ACTION_LABELS)
    _log_array(rr, "/plots/ctbc/output_bias", ctbc_bias, ACTION_LABELS)
    _log_array(rr, "/plots/action/raw", raw_action, ACTION_LABELS)
    _log_array(rr, "/plots/action/unclipped", unclipped_action, ACTION_LABELS)
    _log_array(rr, "/plots/action/delayed", delayed_action, ACTION_LABELS)
    _log_ctbc_side(rr, action_term, env_id)

    return {
        "step": int(step),
        "kff": kff,
        "active_any": bool(np.any(active > 0.5)),
        "stable_any": bool(np.any(stable > 0.5)),
        "force_max_n": float(np.max(force)) if force.size else 0.0,
        "ctbc_pitch_abs_deg": ctbc_pitch_abs_deg,
        "ctbc_delta_abs_max": float(np.max(np.abs(ctbc_delta[:4]))) if ctbc_delta.size else 0.0,
        "tilt_deg": tilt_deg,
        "pitch_deg": pitch_deg,
        "cmd_vx_mps": cmd_vx,
        "base_vx_mps": base_vx,
        "cmd_height_m": cmd_height,
        "base_height_m": float(robot.data.root_link_pos_w[env_id, 2].item()),
        "base_x_m": float(robot.data.root_link_pos_w[env_id, 0].item()),
        "reward": _scalar(reward[env_id]),
        "wheel_terrain_rise_both_m": float(np.min(wheel_terrain_rise_np)),
        "upper_supported_both": bool(np.all(upper_supported_np > 0.5)),
    }


def _maybe_manual_trigger(
    base_env,
    time_s: float,
    trigger_time: float | None,
    triggered: bool,
    side: str,
) -> bool:
    if triggered or trigger_time is None or time_s < float(trigger_time):
        return triggered
    state = getattr(base_env, "stair_climb_state", None)
    if state is None:
        return triggered
    side_ids = (0, 1) if side == "both" else (0,) if side == "left" else (1,)
    # 诊断专用入口：手动置相位，观察 CTBC 前馈本身的姿态效果。
    state._ff_phase[:] = -1
    for side_id in side_ids:
        state._ff_phase[:, side_id] = 0
    state._cooldown[:] = 0
    return True


def _step_height_from_level(level: int) -> float:
    level_clamped = max(0, min(int(level), 9))
    return 0.05 + (float(level_clamped) / 9.0) * (0.20 - 0.05)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _set_watch_env(args)

    import rerun as rr
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.manual_seed(int(args.seed))

    cfg = _configure_cfg(args)
    record_env_id = int(args.record_env_id)
    if record_env_id < 0 or record_env_id >= int(args.num_envs):
        raise ValueError(f"--record-env-id 必须在 [0, {int(args.num_envs) - 1}] 内")
    base_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env)
    policy = _load_policy(env, args.checkpoint, args.task, args.device)
    env.reset()
    _maybe_set_start_pose(base_env, args)
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()
    state = getattr(base_env, "stair_climb_state", None)
    if state is not None and not args.no_fixed_iteration:
        state.set_fixed_iteration(int(args.iteration))

    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("delayed_action")
    initial_base_pos = _finite_np(robot.data.root_link_pos_w[record_env_id]).reshape(3).copy()
    wheel_body_ids = _wheel_body_ids(base_env)
    initial_wheel_terrain_z = _wheel_terrain_measurements(base_env, wheel_body_ids)[0][
        record_env_id
    ].clone()
    step_height = _step_height_from_level(int(args.terrain_level))
    mirror = _MujocoMirror(base_env.sim.mj_model, robot)
    renderer = _MujocoRgbRenderer(
        mirror.model,
        width=int(args.render_width),
        height=int(args.render_height),
        distance=float(args.camera_distance),
        azimuth=float(args.camera_azimuth),
        elevation=float(args.camera_elevation),
    )
    rr.init("se3_stair_ctbc", spawn=False)
    if bool(args.spawn):
        rr.spawn(
            connect=True,
            detach_process=True,
            memory_limit=str(args.memory_limit),
            server_memory_limit=str(args.memory_limit),
        )
    rr.save(str(args.output))
    _send_blueprint(rr)

    steps = max(1, math.ceil(float(args.seconds) / float(base_env.step_dt)))
    sample_every = max(1, int(args.sample_every))
    render_every = max(1, int(args.render_every))
    samples: list[dict[str, Any]] = []
    manual_triggered = False

    try:
        for step in range(steps):
            time_s = step * float(base_env.step_dt)
            manual_triggered = _maybe_manual_trigger(
                base_env,
                time_s,
                args.manual_trigger_time,
                manual_triggered,
                args.manual_trigger_side,
            )
            with torch.no_grad():
                obs = env.get_observations()
                action = policy(obs)
                _, reward, _, _ = env.step(action)

            if step % sample_every != 0 and step != steps - 1:
                continue

            time_s = step * float(base_env.step_dt)
            rr.set_time("time", duration=time_s)
            rr.set_time("step", sequence=int(step))
            mirror.sync(robot, time_s, env_id=record_env_id)
            if step % render_every == 0 or step == steps - 1:
                _log_rgb_frame(
                    rr,
                    renderer.render(mirror.data),
                    jpeg_quality=int(args.jpeg_quality),
                )
            samples.append(
                _log_rollout_sample(
                    rr,
                    base_env,
                    action_term,
                    reward,
                    step,
                    record_env_id,
                    initial_wheel_terrain_z,
                    step_height,
                    args,
                )
            )
    finally:
        renderer.close()
        env.close()
        disconnect = getattr(rr, "disconnect", None)
        if callable(disconnect):
            disconnect()

    active_samples = [sample for sample in samples if sample["active_any"]]
    first_trigger_step = active_samples[0]["step"] if active_samples else None
    base_heights = [sample["base_height_m"] for sample in samples]
    base_xs = [sample["base_x_m"] for sample in samples]
    support_flags = [bool(sample["upper_supported_both"]) for sample in samples]
    max_support_run = 0
    current_support_run = 0
    for supported in support_flags:
        current_support_run = current_support_run + 1 if supported else 0
        max_support_run = max(max_support_run, current_support_run)
    summary = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output.resolve()),
        "task": args.task,
        "device": args.device,
        "seconds": float(args.seconds),
        "steps": int(steps),
        "sample_every": int(sample_every),
        "render_every": int(render_every),
        "record_env_id": int(record_env_id),
        "iteration": None if args.no_fixed_iteration else int(args.iteration),
        "ctbc_kff_last": samples[-1]["kff"] if samples else 0.0,
        "ctbc_active_sample_rate": len(active_samples) / max(1, len(samples)),
        "ctbc_first_trigger_time_s": (
            None if first_trigger_step is None else first_trigger_step * float(base_env.step_dt)
        ),
        "ctbc_force_max_n": max((sample["force_max_n"] for sample in samples), default=0.0),
        "ctbc_delta_abs_max": max(
            (sample["ctbc_delta_abs_max"] for sample in samples), default=0.0
        ),
        "base_height_initial_m": float(initial_base_pos[2]),
        "base_height_max_m": max(base_heights, default=float(initial_base_pos[2])),
        "base_height_gain_max_m": max(base_heights, default=float(initial_base_pos[2]))
        - float(initial_base_pos[2]),
        "x_progress_final_m": (
            0.0 if not base_xs else float(base_xs[-1]) - float(initial_base_pos[0])
        ),
        "x_progress_max_m": (0.0 if not base_xs else max(base_xs) - float(initial_base_pos[0])),
        "tilt_max_deg": max((sample["tilt_deg"] for sample in samples), default=0.0),
        "base_vx_mean_mps": float(np.mean([sample["base_vx_mps"] for sample in samples]))
        if samples
        else 0.0,
        "wheel_terrain_rise_both_max_m": max(
            (sample["wheel_terrain_rise_both_m"] for sample in samples),
            default=0.0,
        ),
        "upper_supported_both_sample_rate": float(np.mean(support_flags)) if samples else 0.0,
        "upper_supported_both_max_duration_s": float(max_support_run * sample_every)
        * float(base_env.step_dt),
        "required_support_rise_m": float(step_height) * float(args.pass_support_ratio),
        "step_height_m": float(step_height),
        "reward_mean": float(np.mean([sample["reward"] for sample in samples])) if samples else 0.0,
        "manual_triggered": bool(manual_triggered),
        "manual_trigger_side": args.manual_trigger_side,
        "ff_x_m": None if args.ff_x_m is None else float(args.ff_x_m),
        "ff_lift_m": None if args.ff_lift_m is None else float(args.ff_lift_m),
        "ff_period_s": None if args.ff_period_s is None else float(args.ff_period_s),
        "ff_wheel_action": None if args.ff_wheel_action is None else float(args.ff_wheel_action),
        "ff_rise_ratio": None if args.ff_rise_ratio is None else float(args.ff_rise_ratio),
        "ff_hold_ratio": None if args.ff_hold_ratio is None else float(args.ff_hold_ratio),
    }
    return summary


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = _run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
