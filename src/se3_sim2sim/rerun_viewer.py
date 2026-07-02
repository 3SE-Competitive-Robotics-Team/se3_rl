"""Rerun viewer sink for MuJoCo models and rollout telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from .math_utils import quat_wxyz_to_xyzw

GeomView = Literal["visual", "collision", "both"]

SIDE_LABELS = ("left", "right")
ACTION_LABELS = (
    "left_front",
    "left_drive",
    "right_front",
    "right_drive",
    "left_wheel",
    "right_wheel",
)
CTRL_LABELS = ACTION_LABELS
XYZ_LABELS = ("x", "y", "z")
EULER_LABELS = ("roll", "pitch", "yaw")
ENTITY_COLORS = {
    "base": (72, 115, 180, 255),
    "left_thigh": (35, 156, 93, 255),
    "left_calf": (35, 181, 190, 255),
    "left_wheel": (38, 132, 255, 255),
    "right_thigh": (242, 145, 51, 255),
    "right_calf": (225, 85, 84, 255),
    "right_wheel": (154, 91, 216, 255),
}


class RerunViewer:
    def __init__(
        self,
        *,
        app_id: str,
        spawn: bool = True,
        address: str | None = None,
        record_to_rrd: Path | None = None,
        memory_limit: str = "1GB",
        follow_body: str = "base_link",
        geom_view: GeomView = "visual",
        manage_recording: bool = True,
    ) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        self.follow_body = follow_body
        self.geom_view = geom_view
        self.body_paths: list[str] = []
        self.geom_paths: dict[int, str] = {}
        self.follow_body_id = -1
        blueprint = self._make_blueprint(rrb, follow_body=follow_body)
        if manage_recording:
            rr.init(app_id, spawn=False)
            if address:
                rr.connect_grpc(address)
            elif spawn:
                rr.spawn(
                    connect=True,
                    detach_process=True,
                    memory_limit=memory_limit,
                    server_memory_limit=memory_limit,
                )
            if record_to_rrd is not None:
                record_to_rrd.parent.mkdir(parents=True, exist_ok=True)
                rr.save(str(record_to_rrd))
        rr.send_blueprint(blueprint, make_active=True, make_default=True)

    def log_model(self, model: mujoco.MjModel) -> None:
        self.body_paths = [self._body_path(model, body_id) for body_id in range(model.nbody)]
        self.follow_body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.follow_body)
        )
        for geom_id in range(model.ngeom):
            if not self._should_log_geom(model, geom_id):
                continue
            body_id = int(model.geom_bodyid[geom_id])
            path = f"{self.body_paths[body_id]}/geoms/{self._geom_name(model, geom_id)}"
            self.geom_paths[geom_id] = path
            self._log_static_geom(model, geom_id, path)

    def _should_log_geom(self, model: mujoco.MjModel, geom_id: int) -> bool:
        """按 Rerun 显示模式筛选 MJCF 几何。"""
        geom_name = self._geom_name(model, geom_id)
        if geom_name.startswith("stair_terrain_step_"):
            return True
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            return True
        if self.geom_view == "both":
            return True

        group = int(model.geom_group[geom_id])
        has_contact = (
            int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0
        )
        if self.geom_view == "visual":
            return group == 1 or not has_contact
        return has_contact

    def log_state(
        self, model: mujoco.MjModel, data: mujoco.MjData, *, step: int, telemetry: dict[str, object]
    ) -> None:
        rr = self.rr
        rr.set_time("time", duration=float(data.time))
        rr.set_time("step", sequence=int(step))
        if not self.body_paths:
            self.log_model(model)
        for body_id, path in enumerate(self.body_paths):
            rr.log(
                path,
                rr.Transform3D(
                    translation=np.asarray(data.xpos[body_id], dtype=np.float32),
                    quaternion=quat_wxyz_to_xyzw(np.asarray(data.xquat[body_id], dtype=np.float32)),
                ),
            )
        rr.log("/metrics/height", rr.Scalars(scalars=float(telemetry["height"])))
        rr.log(
            "/metrics/wheel_clearance",
            rr.Scalars(scalars=float(telemetry.get("wheel_clearance", 0.0))),
        )
        rr.log(
            "/metrics/wheel_clearance_left",
            rr.Scalars(scalars=float(telemetry.get("wheel_clearance_left", 0.0))),
        )
        rr.log(
            "/metrics/wheel_clearance_right",
            rr.Scalars(scalars=float(telemetry.get("wheel_clearance_right", 0.0))),
        )
        rr.log(
            "/metrics/leg_clearance",
            rr.Scalars(scalars=float(telemetry.get("leg_clearance", 0.0))),
        )
        rr.log(
            "/metrics/wheel_contact",
            rr.Scalars(scalars=float(telemetry.get("wheel_contact", 0.0))),
        )
        rr.log(
            "/metrics/leg_contact",
            rr.Scalars(scalars=float(telemetry.get("leg_contact", 0.0))),
        )
        rr.log("/metrics/tilt_deg", rr.Scalars(scalars=float(telemetry["tilt_deg"])))
        rr.log("/metrics/reward", rr.Scalars(scalars=float(telemetry["reward"])))
        yaw_pid = telemetry.get("yaw_pid")
        if isinstance(yaw_pid, dict):
            rr.log(
                "/metrics/yaw_pid/current_yaw", rr.Scalars(scalars=float(yaw_pid["current_yaw"]))
            )
            rr.log("/metrics/yaw_pid/target_yaw", rr.Scalars(scalars=float(yaw_pid["target_yaw"])))
            rr.log("/metrics/yaw_pid/error", rr.Scalars(scalars=float(yaw_pid["error"])))
            rr.log("/metrics/yaw_pid/command", rr.Scalars(scalars=float(yaw_pid["command"])))
        ctrl = np.asarray(telemetry["last_ctrl"], dtype=np.float64)
        for idx, value in enumerate(ctrl):
            rr.log(f"/metrics/ctrl/{idx}", rr.Scalars(scalars=float(value)))
        self._log_2d_plots(telemetry)

    def close(self) -> None:
        disconnect = getattr(self.rr, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def _log_2d_plots(self, telemetry: dict[str, object]) -> None:
        self._log_scalar("/plots/height/base_link_m", telemetry["height"])
        self._log_scalar(
            "/plots/height/left_wheel_clearance_m",
            telemetry.get("wheel_clearance_left", 0.0),
        )
        self._log_scalar(
            "/plots/height/right_wheel_clearance_m",
            telemetry.get("wheel_clearance_right", 0.0),
        )
        self._log_scalar("/plots/height/leg_clearance_m", telemetry.get("leg_clearance", 0.0))
        self._log_scalar("/plots/height/base_clearance_m", telemetry.get("base_clearance", 0.0))

        self._log_scalar("/plots/state/height_m", telemetry["height"])
        self._log_scalar("/plots/state/wheel_clearance_m", telemetry.get("wheel_clearance", 0.0))
        self._log_scalar("/plots/state/tilt_deg", telemetry["tilt_deg"])
        self._log_scalar("/plots/state/fail_tilt_deg", telemetry["fail_tilt_deg"])
        self._log_scalar("/plots/state/reward", telemetry["reward"])
        self._log_scalar("/plots/rc/rc_switch_r", telemetry.get("rc_switch_r", 1.0))
        self._log_scalar("/plots/rc/output_enabled", telemetry.get("output_enabled", 1.0))
        self._log_scalar("/plots/rc/switch_event", telemetry.get("rc_switch_event", 0.0))
        self._log_scalar("/plots/rc/policy_reset", telemetry.get("rc_policy_reset", 0.0))
        target_mode = 1.0 if telemetry.get("target_mode", "policy") == "policy" else 0.0
        self._log_scalar("/plots/rc/target_mode_policy", target_mode)
        self._log_scalar("/plots/command/lin_vel_x", telemetry.get("command_lin_vel_x", 0.0))
        self._log_scalar("/plots/command/yaw_rate", telemetry.get("command_yaw_rate", 0.0))
        self._log_scalar("/plots/command/height", telemetry.get("command_height", 0.0))
        self._log_scalar("/plots/velocity/base_lin_vel_x", telemetry.get("base_lin_vel_x", 0.0))
        self._log_scalar("/plots/velocity/wheel_lin_vel", telemetry.get("wheel_lin_vel", 0.0))
        self._log_scalar("/plots/tracking/height/command_m", telemetry.get("command_height", 0.0))
        self._log_scalar("/plots/tracking/height/base_m", telemetry["height"])
        self._log_scalar(
            "/plots/tracking/height/wheel_clearance_m", telemetry.get("wheel_clearance", 0.0)
        )
        self._log_scalar(
            "/plots/tracking/forward_velocity/command_x_mps",
            telemetry.get("command_lin_vel_x", 0.0),
        )
        self._log_scalar(
            "/plots/tracking/forward_velocity/base_x_mps", telemetry.get("base_lin_vel_x", 0.0)
        )
        self._log_scalar(
            "/plots/tracking/forward_velocity/wheel_x_mps", telemetry.get("wheel_lin_vel", 0.0)
        )
        base_ang_vel_body = np.asarray(
            telemetry.get("base_ang_vel_body", (0.0, 0.0, 0.0)), dtype=np.float64
        ).reshape(-1)
        base_yaw_rate = float(base_ang_vel_body[2]) if base_ang_vel_body.size >= 3 else 0.0
        self._log_scalar(
            "/plots/tracking/yaw_rate/command_rad_s", telemetry.get("command_yaw_rate", 0.0)
        )
        self._log_scalar("/plots/tracking/yaw_rate/base_body_z_rad_s", base_yaw_rate)
        self._log_scalar("/plots/contact/wheel_any", telemetry.get("wheel_contact", 0.0))
        self._log_scalar("/plots/contact/wheel_full", telemetry.get("wheel_full_contact", 0.0))
        self._log_scalar("/plots/contact/wheel_left", telemetry.get("wheel_contact_left", 0.0))
        self._log_scalar("/plots/contact/wheel_right", telemetry.get("wheel_contact_right", 0.0))
        self._log_scalar("/plots/contact/leg_any", telemetry.get("leg_contact", 0.0))
        self._log_scalar("/plots/contact/leg_left", telemetry.get("leg_contact_left", 0.0))
        self._log_scalar("/plots/contact/leg_right", telemetry.get("leg_contact_right", 0.0))
        self._log_scalar("/plots/contact/base", telemetry.get("base_contact", 0.0))
        self._log_scalar("/plots/contact/nonwheel", telemetry.get("nonwheel_contact", 0.0))
        self._log_scalar("/plots/ctbc/trigger", telemetry.get("ctbc_trigger", 0.0))
        self._log_scalar(
            "/plots/ctbc/trigger_mode_pitch", telemetry.get("ctbc_trigger_mode_pitch", 0.0)
        )
        self._log_pair(
            "/plots/ctbc/active",
            (telemetry.get("ctbc_left_active", 0.0), telemetry.get("ctbc_right_active", 0.0)),
        )
        self._log_pair(
            "/plots/ctbc/stable",
            (telemetry.get("ctbc_stable_left", 0.0), telemetry.get("ctbc_stable_right", 0.0)),
        )
        self._log_pair(
            "/plots/ctbc/contact_force_n",
            (telemetry.get("ctbc_contact_left", 0.0), telemetry.get("ctbc_contact_right", 0.0)),
        )
        self._log_pair(
            "/plots/ctbc/trigger_score",
            (
                telemetry.get("ctbc_trigger_score_left", 0.0),
                telemetry.get("ctbc_trigger_score_right", 0.0),
            ),
        )
        self._log_pair(
            "/plots/ctbc/phase",
            (telemetry.get("ctbc_phase_left", 0.0), telemetry.get("ctbc_phase_right", 0.0)),
        )
        self._log_scalar("/plots/ctbc/pitch_abs_deg", telemetry.get("ctbc_pitch_abs_deg", 0.0))
        self._log_scalar(
            "/plots/ctbc/pitch_threshold_deg",
            telemetry.get("ctbc_pitch_threshold_deg", 0.0),
        )
        self._log_scalar(
            "/plots/leg_alignment/wheel_lateral_distance_m",
            telemetry.get("wheel_lateral_distance", 0.0),
        )
        self._log_scalar(
            "/plots/leg_alignment/wheel_fore_aft_offset_m",
            telemetry.get("wheel_fore_aft_offset", 0.0),
        )
        self._log_scalar(
            "/plots/leg_alignment/leg_mirror_error_rad",
            telemetry.get("leg_mirror_error", 0.0),
        )
        yaw_pid = telemetry.get("yaw_pid")
        if isinstance(yaw_pid, dict):
            self._log_scalar("/plots/yaw_pid/current_yaw", yaw_pid["current_yaw"])
            self._log_scalar("/plots/yaw_pid/target_yaw", yaw_pid["target_yaw"])
            self._log_scalar("/plots/yaw_pid/error", yaw_pid["error"])
            self._log_scalar("/plots/yaw_pid/command", yaw_pid["command"])
            self._log_scalar("/plots/tracking/yaw_angle/current_rad", yaw_pid["current_yaw"])
            self._log_scalar("/plots/tracking/yaw_angle/target_rad", yaw_pid["target_yaw"])
            self._log_scalar("/plots/tracking/yaw_angle/error_rad", yaw_pid["error"])
        self._log_array(
            "/plots/imu/base_ang_vel_body_rad_s", telemetry["base_ang_vel_body"], XYZ_LABELS
        )
        self._log_array(
            "/plots/imu/base_ang_vel_world_rad_s", telemetry["base_ang_vel_world"], XYZ_LABELS
        )
        self._log_array(
            "/plots/imu/euler_deg",
            (
                telemetry.get("roll_deg", 0.0),
                telemetry.get("pitch_deg", 0.0),
                telemetry.get("yaw_deg", 0.0),
            ),
            EULER_LABELS,
        )

        self._log_array("/plots/dof/pos_rad", telemetry["dof_pos"], CTRL_LABELS)
        self._log_array("/plots/dof/vel_rad_s", telemetry["dof_vel"], CTRL_LABELS)

        self._log_array("/plots/action/raw", telemetry["policy_action_raw"], ACTION_LABELS)
        self._log_array("/plots/action/clipped", telemetry["policy_action_clipped"], ACTION_LABELS)
        self._log_array("/plots/action/obs_raw_clipped", telemetry["last_action"], ACTION_LABELS)
        self._log_array("/plots/ctrl/torque_nm", telemetry["last_ctrl"], CTRL_LABELS)

    def _log_scalar(self, path: str, value: object) -> None:
        self.rr.log(path, self.rr.Scalars(scalars=float(value)))

    def _log_pair(self, path: str, values: object) -> None:
        self._log_array(path, values, SIDE_LABELS)

    def _log_array(self, path: str, values: object, labels: tuple[str, ...]) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        for idx, value in enumerate(arr):
            label = labels[idx] if idx < len(labels) else str(idx)
            self._log_scalar(f"{path}/{label}", value)

    def _log_static_geom(self, model: mujoco.MjModel, geom_id: int, path: str) -> None:
        rr = self.rr
        geom_type = int(model.geom_type[geom_id])
        color = self._geom_color(model, geom_id)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float32)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            rr.log(path, rr.Boxes3D(half_sizes=size[:3], colors=color), static=True)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            extent_x = float(size[0]) if float(size[0]) > 0.0 else 5.0
            extent_y = float(size[1]) if float(size[1]) > 0.0 else 5.0
            rr.log(
                path,
                rr.Boxes3D(half_sizes=[extent_x, extent_y, 0.001], colors=color),
                static=True,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER) and hasattr(rr, "Cylinders3D"):
            rr.log(
                path,
                rr.Cylinders3D(radii=float(size[0]), lengths=2.0 * float(size[1]), colors=color),
                static=True,
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            rr.log(
                path, rr.Ellipsoids3D(half_sizes=[float(size[0])] * 3, colors=color), static=True
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertices, faces = self._mesh(model, mesh_id)
            vertex_colors = np.repeat(color.reshape(1, 4), vertices.shape[0], axis=0)
            rr.log(
                path,
                rr.Mesh3D(
                    vertex_positions=vertices, triangle_indices=faces, vertex_colors=vertex_colors
                ),
                static=True,
            )
        pos = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
        quat = quat_wxyz_to_xyzw(np.asarray(model.geom_quat[geom_id], dtype=np.float32))
        rr.log(path, rr.Transform3D(translation=pos, quaternion=quat), static=True)

    @staticmethod
    def _make_blueprint(rrb, *, follow_body: str = "base_link"):
        eye_controls = rrb.archetypes.EyeControls3D(
            tracking_entity=f"/world/bodies/{follow_body}",
        )
        spatial = rrb.Spatial3DView(origin="/world", name="Scene", eye_controls=eye_controls)
        time_panel = rrb.TimePanel(state="collapsed")
        time_series_view = rrb.TimeSeriesView

        overview = rrb.Grid(
            time_series_view(
                origin="/plots/tracking/forward_velocity",
                contents="/plots/tracking/forward_velocity/**",
                name="Velocity Tracking",
            ),
            time_series_view(
                origin="/plots/tracking/height",
                contents="/plots/tracking/height/**",
                name="Height Tracking",
            ),
            time_series_view(
                origin="/plots/tracking/yaw_rate",
                contents="/plots/tracking/yaw_rate/**",
                name="Yaw Rate Tracking",
            ),
            time_series_view(origin="/plots/contact", contents="/plots/contact/**", name="Contact"),
            time_series_view(origin="/plots/ctbc", contents="/plots/ctbc/**", name="CTBC"),
            time_series_view(
                origin="/plots/ctrl/torque_nm",
                contents="/plots/ctrl/torque_nm/**",
                name="Motor Torque",
            ),
            time_series_view(origin="/plots/imu", contents="/plots/imu/**", name="IMU"),
            grid_columns=2,
            name="Overview",
        )
        yaw = rrb.Grid(
            time_series_view(
                origin="/plots/tracking/yaw_rate",
                contents="/plots/tracking/yaw_rate/**",
                name="Yaw Rate Tracking",
            ),
            time_series_view(
                origin="/plots/tracking/yaw_angle",
                contents="/plots/tracking/yaw_angle/**",
                name="Yaw Angle Tracking",
            ),
            time_series_view(origin="/plots/yaw_pid", contents="/plots/yaw_pid/**", name="Yaw PID"),
            grid_columns=1,
            name="Yaw",
        )
        raw = rrb.Vertical(
            time_series_view(origin="/plots/command", contents="/plots/command/**", name="Command"),
            time_series_view(
                origin="/plots/velocity", contents="/plots/velocity/**", name="Velocity"
            ),
            time_series_view(
                origin="/plots/leg_alignment",
                contents="/plots/leg_alignment/**",
                name="Leg Alignment",
            ),
            name="Raw",
        )
        height = time_series_view(
            origin="/plots/height",
            contents="/plots/height/**",
            name="Height",
        )
        control = rrb.Grid(
            time_series_view(origin="/plots/dof", contents="/plots/dof/**", name="DOF"),
            time_series_view(origin="/plots/action", contents="/plots/action/**", name="Action"),
            time_series_view(origin="/plots/ctrl", contents="/plots/ctrl/**", name="Ctrl"),
            grid_columns=2,
            name="Control",
        )
        plots = rrb.Tabs(overview, height, yaw, raw, control, active_tab=0, name="Debug")
        layout = rrb.Horizontal(spatial, plots, column_shares=[0.52, 0.48], name="SE3 sim2sim")
        return rrb.Blueprint(
            layout, time_panel, collapse_panels=True, auto_layout=False, auto_views=False
        )

    @staticmethod
    def _mesh(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
        v0 = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        f0 = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        scale = np.asarray(model.mesh_scale[mesh_id], dtype=np.float32).reshape(1, 3)
        vertices = (
            np.asarray(model.mesh_vert[v0 : v0 + vn], dtype=np.float32).reshape(-1, 3) * scale
        )
        faces = np.asarray(model.mesh_face[f0 : f0 + fn], dtype=np.uint32).reshape(-1, 3)
        return vertices, faces

    @staticmethod
    def _body_name(model: mujoco.MjModel, body_id: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

    @staticmethod
    def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"

    @staticmethod
    def _geom_color(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
        body_id = int(model.geom_bodyid[geom_id])
        body_name = RerunViewer._body_name(model, body_id)
        geom_name = RerunViewer._geom_name(model, geom_id)
        entity_name = f"{body_name}/{geom_name}"
        if "base_link" in entity_name:
            return np.asarray(ENTITY_COLORS["base"], dtype=np.uint8)
        if "lf0" in entity_name:
            return np.asarray(ENTITY_COLORS["left_thigh"], dtype=np.uint8)
        if "lf1" in entity_name:
            return np.asarray(ENTITY_COLORS["left_calf"], dtype=np.uint8)
        if "l_wheel" in entity_name:
            return np.asarray(ENTITY_COLORS["left_wheel"], dtype=np.uint8)
        if "rf0" in entity_name:
            return np.asarray(ENTITY_COLORS["right_thigh"], dtype=np.uint8)
        if "rf1" in entity_name:
            return np.asarray(ENTITY_COLORS["right_calf"], dtype=np.uint8)
        if "r_wheel" in entity_name:
            return np.asarray(ENTITY_COLORS["right_wheel"], dtype=np.uint8)
        rgba = np.clip(np.asarray(model.geom_rgba[geom_id], dtype=np.float32), 0.0, 1.0)
        return (rgba * 255.0).astype(np.uint8)

    def _body_path(self, model: mujoco.MjModel, body_id: int) -> str:
        return f"/world/bodies/{self._body_name(model, body_id)}"
