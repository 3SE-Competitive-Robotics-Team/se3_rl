"""通过网页实时查看 STM32 CDC state 与 recovery 观测合同。"""

from __future__ import annotations

import argparse
import binascii
import json
import math
import os
import struct
import threading
import time
import zlib
from contextlib import suppress
from dataclasses import dataclass, field
from glob import glob
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import urlopen

import numpy as np

from se3_shared import RobotConfig
from se3_shared.fourbar import policy_to_output_pos_np

from .cdc import CdcSerial
from .observation import RecoveryObservationBuilder
from .protocol import (
    MSG_LATENCY,
    MSG_POLICY_STATE,
    PolicyLatencyFrame,
    PolicyStateFrame,
    StreamParser,
    decode_policy_latency,
    decode_policy_state,
)

_ROBOT_CFG = RobotConfig()
_ZERO_ACTION = np.zeros(6, dtype=np.float32)
_DEFAULT_MJCF = Path("assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml")
_DEFAULT_RENDER_WIDTH = 1280
_DEFAULT_RENDER_HEIGHT = 720
_DEFAULT_RENDER_FPS = 50.0
_DEFAULT_RENDER_JPEG_QUALITY = 95
_COLLISION_GEOM_GROUP = 0
_VISUAL_GEOM_GROUP = 1
_FLOOR_GEOM_GROUP = 2
_DEBUG_JOINTS = (
    ("lf0", "lf0_Joint", "LF0"),
    ("lf1", "lf1_Joint", "LF1"),
    ("lb", "l_drive_bar_Joint", "LB"),
    ("lc", "l_coupler_Joint", "LC"),
    ("lw", "l_wheel_Joint", "LW"),
    ("rf0", "rf0_Joint", "RF0"),
    ("rf1", "rf1_Joint", "RF1"),
    ("rb", "r_drive_bar_Joint", "RB"),
    ("rc", "r_coupler_Joint", "RC"),
    ("rw", "r_wheel_Joint", "RW"),
)
_DEBUG_BODY_FRAMES = (("base", "base_link", "base_link"),)
_DEBUG_FRAME_KEYS = tuple(key for key, _, _ in (*_DEBUG_BODY_FRAMES, *_DEBUG_JOINTS))
_DEFAULT_NX_RELAY_URL = "http://192.168.137.100:8081"
_DEFAULT_CAMERA_AZIMUTH = 135.0
_DEFAULT_CAMERA_ELEVATION = -20.0
_DEFAULT_CAMERA_DISTANCE = 1.25
_DEFAULT_USE_GRAVITY_ATTITUDE = True
_DEFAULT_WHEEL_RENDER_MODE = "position"
_MAX_GYRO_INTEGRATION_DT_S = 0.05
_MAX_WHEEL_INTEGRATION_DT_S = 0.05
_CLOSED_CHAIN_SOLVER_EPS = 1.0e-5
_CLOSED_CHAIN_SOLVER_ITERS = 10
_CLOSED_CHAIN_RETRY_ERROR_M = 1.0e-6
_CLOSED_CHAIN_STEP_LIMIT_RAD = 0.6
_JOINT_FRAME_AXIS_LENGTH = 0.13
_JOINT_FRAME_AXIS_WIDTH = 0.008
_JOINT_FRAME_COLORS = (
    np.array([1.0, 0.16, 0.14, 0.92], dtype=np.float32),
    np.array([0.1, 0.85, 0.24, 0.92], dtype=np.float32),
    np.array([0.22, 0.52, 1.0, 0.92], dtype=np.float32),
)


@dataclass(slots=True)
class SharedSnapshot:
    """跨 CDC 读线程和 HTTP 线程共享的最新状态。"""

    latest: dict[str, Any] = field(default_factory=dict)
    latest_latency: dict[str, Any] | None = None
    latest_state: PolicyStateFrame | None = None
    event_seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, snapshot: dict[str, Any], state: PolicyStateFrame | None = None) -> None:
        with self.lock:
            self.event_seq += 1
            if self.latest_latency is not None:
                _attach_latency_snapshot(snapshot, self.latest_latency)
            snapshot["_event_seq"] = self.event_seq
            self.latest = snapshot
            self.latest_state = state

    def update_latency(self, latency: PolicyLatencyFrame) -> None:
        with self.lock:
            self.event_seq += 1
            self.latest_latency = latency_to_snapshot(latency)
            if self.latest:
                snapshot = dict(self.latest)
                _attach_latency_snapshot(snapshot, self.latest_latency)
            else:
                snapshot = {
                    "source": "cdc",
                    "connected": True,
                    "host_time_s": time.time(),
                    "seq": -1,
                }
                _attach_latency_snapshot(snapshot, self.latest_latency)
            snapshot["_event_seq"] = self.event_seq
            self.latest = snapshot

    def get(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest)

    def get_state(self) -> PolicyStateFrame | None:
        with self.lock:
            return self.latest_state


class VisualizerServer(ThreadingHTTPServer):
    """带共享状态的 HTTP server。"""

    def __init__(
        self,
        server_address: tuple[str, int],
        snapshot: SharedSnapshot,
        renderer: MujocoRenderWorker | None,
    ):
        super().__init__(server_address, VisualizerHandler)
        self.snapshot = snapshot
        self.renderer = renderer

    def renderer_info(self) -> dict[str, Any]:
        if self.renderer is None:
            return {"enabled": False}
        return self.renderer.info()

    def renderer_settings(self) -> dict[str, Any]:
        if self.renderer is None:
            return {"enabled": False}
        return self.renderer.settings()

    def update_renderer_settings(
        self,
        *,
        show_visual_model: bool | None = None,
        show_collision_model: bool | None = None,
        joint_frames: dict[str, bool] | None = None,
        use_gravity_attitude: bool | None = None,
        wheel_render_mode: str | None = None,
        camera_azimuth: float | None = None,
        camera_elevation: float | None = None,
        camera_distance: float | None = None,
    ) -> dict[str, Any]:
        if self.renderer is None:
            return {"enabled": False}
        return self.renderer.update_settings(
            show_visual_model=show_visual_model,
            show_collision_model=show_collision_model,
            joint_frames=joint_frames,
            use_gravity_attitude=use_gravity_attitude,
            wheel_render_mode=wheel_render_mode,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
        )


class VisualizerHandler(BaseHTTPRequestHandler):
    """提供网页、单帧 JSON 和 SSE 状态流。"""

    server: VisualizerServer

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/snapshot":
            payload = json.dumps(self.server.snapshot.get(), separators=(",", ":")).encode()
            self._send_bytes(payload, "application/json")
            return
        if path == "/events":
            self._stream_events()
            return
        if path in ("/render.png", "/render.jpg"):
            self._send_render()
            return
        if path == "/render_stream":
            self._stream_render()
            return
        if path == "/render_info":
            payload = json.dumps(self.server.renderer_info(), separators=(",", ":")).encode()
            self._send_bytes(payload, "application/json")
            return
        if path == "/render_settings":
            self._send_render_settings(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_render(self) -> None:
        renderer = self.server.renderer
        if renderer is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            payload, content_type = renderer.current_frame()
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_render_settings(self, query: str) -> None:
        params = parse_qs(query)
        show_visual_model = _query_bool(params["visual"][-1]) if "visual" in params else None
        show_collision_model = (
            _query_bool(params["collision"][-1]) if "collision" in params else None
        )
        use_gravity_attitude = (
            _query_bool(params["gravity_attitude"][-1]) if "gravity_attitude" in params else None
        )
        wheel_render_mode = _wheel_render_mode_from_query(params)
        joint_frames = _joint_settings_from_query(params)
        camera_azimuth = _query_float(params, "camera_azimuth")
        camera_elevation = _query_float(params, "camera_elevation")
        camera_distance = _query_float(params, "camera_distance")
        if (
            show_visual_model is None
            and show_collision_model is None
            and joint_frames is None
            and use_gravity_attitude is None
            and wheel_render_mode is None
            and camera_azimuth is None
            and camera_elevation is None
            and camera_distance is None
        ):
            settings = self.server.renderer_settings()
        else:
            settings = self.server.update_renderer_settings(
                show_visual_model=show_visual_model,
                show_collision_model=show_collision_model,
                joint_frames=joint_frames,
                use_gravity_attitude=use_gravity_attitude,
                wheel_render_mode=wheel_render_mode,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                camera_distance=camera_distance,
            )
        payload = json.dumps(settings, separators=(",", ":")).encode()
        self._send_bytes(payload, "application/json")

    def _stream_render(self) -> None:
        renderer = self.server.renderer
        if renderer is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        last_frame_id = 0
        while True:
            try:
                frame = renderer.wait_frame(last_frame_id, timeout_s=1.0)
                if frame is None:
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    continue
                frame_id, payload, content_type = frame
                part_header = (
                    b"--frame\r\n"
                    + f"Content-Type: {content_type}\r\n".encode()
                    + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                )
                self.wfile.write(part_header)
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_frame_id = frame_id
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

    def _stream_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_event_seq: int | None = None
        while True:
            snapshot = self.server.snapshot.get()
            event_seq = snapshot.get("_event_seq", snapshot.get("seq"))
            if event_seq != last_event_seq:
                payload = json.dumps(snapshot, separators=(",", ":"))
                try:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_event_seq = int(event_seq) if isinstance(event_seq, int) else None
            time.sleep(0.02)


def _query_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_float(params: dict[str, list[str]], key: str) -> float | None:
    if key not in params:
        return None
    with suppress(Exception):
        return float(params[key][-1])
    return None


def _wheel_render_mode_from_query(params: dict[str, list[str]]) -> str | None:
    if "wheel_render_mode" in params:
        return _parse_wheel_render_mode(params["wheel_render_mode"][-1])
    if "wheel_velocity" in params:
        return "velocity" if _query_bool(params["wheel_velocity"][-1]) else "position"
    return None


def _parse_wheel_render_mode(value: object) -> str | None:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"position", "pos", "wheel_pos", "encoder_pos"}:
        return "position"
    if mode in {"velocity", "vel", "wheel_vel", "speed"}:
        return "velocity"
    return None


def _joint_settings_from_query(params: dict[str, list[str]]) -> dict[str, bool] | None:
    if "joints" in params:
        value = _query_bool(params["joints"][-1])
        return {key: value for key in _DEBUG_FRAME_KEYS}

    settings: dict[str, bool] = {}
    for key in _DEBUG_FRAME_KEYS:
        query_key = f"joint_{key}"
        if query_key in params:
            settings[key] = _query_bool(params[query_key][-1])
    return settings or None


def run_cdc_reader(
    *,
    shared: SharedSnapshot,
    dev: str,
    baudrate: int,
    read_timeout_s: float,
    stop_event: threading.Event,
) -> None:
    builder = RecoveryObservationBuilder()
    parser = StreamParser()
    last_wall_time: float | None = None

    while not stop_event.is_set():
        try:
            port = _resolve_serial_port(dev)
            with CdcSerial(port, baudrate=baudrate) as serial:
                while not stop_event.is_set():
                    if not serial.wait_readable(read_timeout_s):
                        continue
                    data = serial.read_available()
                    if not data:
                        continue
                    for message in parser.feed(data):
                        if message.msg_type == MSG_POLICY_STATE:
                            now = time.monotonic()
                            frame_hz = _frame_hz(last_wall_time, now)
                            last_wall_time = now
                            state = decode_policy_state(message)
                            snapshot = state_to_snapshot(
                                state,
                                builder=builder,
                                source="cdc",
                                frame_hz=frame_hz,
                            )
                            snapshot["port"] = port
                            shared.update(
                                snapshot,
                                state,
                            )
                        elif message.msg_type == MSG_LATENCY:
                            shared.update_latency(decode_policy_latency(message))
        except Exception as exc:
            shared.update(
                {
                    "source": "cdc",
                    "connected": False,
                    "error": str(exc),
                    "port": dev,
                    "host_time_s": time.time(),
                    "seq": -1,
                },
                None,
            )
            time.sleep(1.0)


def _resolve_serial_port(dev: str) -> str:
    if dev != "auto":
        return dev
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError("no USB CDC device found under /dev/ttyACM* or /dev/ttyUSB*")


def run_remote_reader(
    *,
    shared: SharedSnapshot,
    remote_url: str,
    timeout_s: float,
    stop_event: threading.Event,
) -> None:
    base_url = remote_url.rstrip("/") + "/"
    events_url = urljoin(base_url, "events")
    while not stop_event.is_set():
        try:
            with urlopen(events_url, timeout=max(float(timeout_s), 1.0)) as response:
                for raw_line in response:
                    if stop_event.is_set():
                        return
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    snapshot = json.loads(line[5:].strip())
                    state = snapshot_to_state(snapshot)
                    snapshot = dict(snapshot)
                    snapshot["source"] = "remote"
                    snapshot["remote_url"] = base_url.rstrip("/")
                    shared.update(snapshot, state)
        except Exception as exc:
            shared.update(
                {
                    "source": "remote",
                    "connected": False,
                    "remote_url": base_url.rstrip("/"),
                    "error": str(exc),
                    "host_time_s": time.time(),
                    "seq": -1,
                },
                None,
            )
            time.sleep(1.0)


def snapshot_to_state(snapshot: dict[str, Any]) -> PolicyStateFrame | None:
    if not snapshot.get("connected", False):
        return None
    return PolicyStateFrame(
        seq=int(snapshot.get("seq", 0)),
        tick_ms=int(snapshot.get("tick_ms", 0)),
        target_seq=int(snapshot.get("target_seq", 0)),
        target_age_ms=int(snapshot.get("target_age_ms", 0)),
        target_valid=int(snapshot.get("target_valid", 0)),
        rc_switch_r=int(snapshot.get("rc_switch_r", 0)),
        output_enabled=int(snapshot.get("output_enabled", 0)),
        base_ang_vel_body=_tuple_from_snapshot(snapshot, "base_ang_vel", 3),
        projected_gravity=_tuple_from_snapshot(snapshot, "projected_gravity", 3, (0.0, 0.0, -1.0)),
        joint_pos=_tuple_from_snapshot(
            snapshot,
            "joint_pos",
            4,
            tuple(float(v) for v in _ROBOT_CFG.default_dof_pos[:4]),
        ),
        joint_vel=_tuple_from_snapshot(snapshot, "joint_vel", 4),
        wheel_pos=_tuple_from_snapshot(snapshot, "wheel_pos", 2),
        wheel_vel=_tuple_from_snapshot(snapshot, "wheel_vel", 2),
        target_joint_pos=_tuple_from_snapshot(snapshot, "target_joint_pos", 4),
        hip_torque=_tuple_from_snapshot(snapshot, "hip_torque", 4),
        wheel_torque=_tuple_from_snapshot(snapshot, "wheel_torque", 2),
        wheel_motor_torque=_tuple_from_snapshot(snapshot, "wheel_motor_torque", 2),
    )


def _tuple_from_snapshot(
    snapshot: dict[str, Any],
    key: str,
    size: int,
    fallback: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    fallback_values = (0.0,) * size if fallback is None else fallback
    values = snapshot.get(key, fallback_values)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < size:
        arr = np.pad(arr, (0, size - arr.size), constant_values=0.0)
    arr = np.nan_to_num(arr[:size], nan=0.0, posinf=0.0, neginf=0.0)
    return tuple(float(v) for v in arr)


def run_synthetic_reader(
    *, shared: SharedSnapshot, rate_hz: float, stop_event: threading.Event
) -> None:
    builder = RecoveryObservationBuilder()
    period_s = 1.0 / max(float(rate_hz), 1.0)
    seq = 0
    start = time.monotonic()

    while not stop_event.is_set():
        t = time.monotonic() - start
        state = synthetic_state(seq=seq, t=t)
        shared.update(
            state_to_snapshot(state, builder=builder, source="synthetic", frame_hz=rate_hz),
            state,
        )
        seq += 1
        time.sleep(period_s)


def synthetic_state(seq: int, t: float) -> PolicyStateFrame:
    base = np.asarray(_ROBOT_CFG.default_dof_pos, dtype=np.float32)
    joint_pos = base[:4].copy()
    joint_vel = np.zeros(4, dtype=np.float32)
    wheel_pos = np.zeros(2, dtype=np.float32)
    wheel_vel = np.zeros(2, dtype=np.float32)
    gravity_x = 0.15 * math.sin(t * 0.5)
    gravity_y = -0.22 * math.sin(t * 0.4)
    projected_gravity = (
        gravity_x,
        gravity_y,
        -math.sqrt(max(1.0 - gravity_x * gravity_x - gravity_y * gravity_y, 0.0)),
    )
    return PolicyStateFrame(
        seq=seq,
        tick_ms=int(t * 1000.0),
        target_seq=0,
        target_age_ms=0,
        target_valid=0,
        rc_switch_r=0,
        output_enabled=0,
        base_ang_vel_body=(
            0.18 * math.sin(t * 0.6),
            0.14 * math.cos(t * 0.5),
            0.10 * math.sin(t * 0.4),
        ),
        projected_gravity=projected_gravity,
        joint_pos=tuple(float(v) for v in joint_pos),
        joint_vel=tuple(float(v) for v in joint_vel),
        wheel_pos=tuple(float(v) for v in wheel_pos),
        wheel_vel=tuple(float(v) for v in wheel_vel),
        target_joint_pos=tuple(float(v) for v in joint_pos),
        hip_torque=(0.0, 0.0, 0.0, 0.0),
        wheel_torque=(0.0, 0.0),
        wheel_motor_torque=(0.0, 0.0),
    )


def state_to_snapshot(
    state: PolicyStateFrame,
    *,
    builder: RecoveryObservationBuilder,
    source: str,
    frame_hz: float | None,
) -> dict[str, Any]:
    obs_result = builder.build(state, _ZERO_ACTION)
    obs = obs_result.obs
    joint_pos = np.asarray(state.joint_pos, dtype=np.float64)
    target_joint_pos = np.asarray(state.target_joint_pos, dtype=np.float64)
    joint_pos_error = _wrap_angle_np(target_joint_pos - joint_pos)
    return {
        "source": source,
        "connected": True,
        "host_time_s": time.time(),
        "seq": int(state.seq),
        "tick_ms": int(state.tick_ms),
        "frame_hz": None if frame_hz is None else round(float(frame_hz), 2),
        "target_seq": int(state.target_seq),
        "target_age_ms": int(state.target_age_ms),
        "target_valid": int(state.target_valid),
        "rc_switch_r": int(state.rc_switch_r),
        "output_enabled": int(state.output_enabled),
        "base_ang_vel": _finite_list(state.base_ang_vel_body),
        "projected_gravity": _finite_list(state.projected_gravity),
        "joint_pos": _finite_list(state.joint_pos),
        "target_joint_pos": _finite_list(state.target_joint_pos),
        "joint_pos_error": _finite_list(joint_pos_error),
        "joint_active": _finite_list(_policy_active_angles_np(joint_pos)),
        "target_active": _finite_list(_policy_active_angles_np(target_joint_pos)),
        "render_joint_pos": _finite_list(joint_pos),
        "joint_vel": _finite_list(state.joint_vel),
        "hip_torque": _finite_list(state.hip_torque),
        "wheel_torque": _finite_list(state.wheel_torque),
        "wheel_motor_torque": _finite_list(state.wheel_motor_torque),
        "wheel_pos": _finite_list(state.wheel_pos),
        "wheel_vel": _finite_list(state.wheel_vel),
        "obs": {
            "base_ang_vel_scaled": _finite_list(obs[0:3]),
            "projected_gravity": _finite_list(obs[3:6]),
            "commands_scaled": _finite_list(obs[6:11]),
            "leg_pos_rel": _finite_list(obs[11:15]),
            "leg_vel_scaled": _finite_list(obs[15:19]),
            "wheel_pos": _finite_list(obs[19:21]),
            "wheel_vel_scaled": _finite_list(obs[21:23]),
            "last_actions": _finite_list(obs[23:29]),
            "recovery_cmd": _finite_list(obs[29:32]),
        },
        "obs_had_nonfinite_input": bool(obs_result.had_nonfinite_input),
    }


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.remainder(np.asarray(angle, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi


def _wrap_angle_scalar(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _policy_active_angles_np(joint_pos: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_pos, dtype=np.float64).reshape(4)
    return _wrap_angle_np(np.asarray([q[0] - q[1], q[3] - q[2]], dtype=np.float64))


def latency_to_snapshot(latency: PolicyLatencyFrame) -> dict[str, Any]:
    rx_to_output_us = int(latency.rx_to_output_us)
    return {
        "policy_seq": int(latency.policy_seq),
        "rx_to_output_us": rx_to_output_us,
        "rx_to_output_ms": round(float(rx_to_output_us) / 1000.0, 3),
        "output_enabled": int(latency.output_enabled),
        "host_time_s": time.time(),
    }


def _attach_latency_snapshot(snapshot: dict[str, Any], latency: dict[str, Any]) -> None:
    snapshot["latency"] = dict(latency)
    snapshot["latency_policy_seq"] = int(latency.get("policy_seq", 0))
    snapshot["rx_to_output_us"] = int(latency.get("rx_to_output_us", 0))
    snapshot["rx_to_output_ms"] = float(latency.get("rx_to_output_ms", 0.0))
    snapshot["latency_output_enabled"] = int(latency.get("output_enabled", 0))


class MujocoRenderWorker:
    def __init__(
        self,
        *,
        shared: SharedSnapshot,
        mjcf: Path,
        width: int,
        height: int,
        fps: float,
        jpeg_quality: int,
        show_visual_model: bool,
        show_collision_model: bool,
        show_joint_frames: bool,
    ) -> None:
        self.shared = shared
        self.mjcf = mjcf
        self.width = int(width)
        self.height = int(height)
        self.period_s = 1.0 / max(float(fps), 1.0)
        self.jpeg_quality = int(np.clip(jpeg_quality, 30, 95))
        self.show_visual_model = bool(show_visual_model)
        self.show_collision_model = bool(show_collision_model)
        self.use_gravity_attitude = _DEFAULT_USE_GRAVITY_ATTITUDE
        self.wheel_render_mode = _DEFAULT_WHEEL_RENDER_MODE
        self.joint_frames = {key: bool(show_joint_frames) for key in _DEBUG_FRAME_KEYS}
        self.camera_azimuth = _DEFAULT_CAMERA_AZIMUTH
        self.camera_elevation = _DEFAULT_CAMERA_ELEVATION
        self.camera_distance = _DEFAULT_CAMERA_DISTANCE
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.frame_cv = threading.Condition(self.lock)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.latest_frame: bytes | None = None
        self.latest_content_type = "image/png"
        self.latest_frame_id = 0
        self.error: str | None = None
        self.latest_info: dict[str, Any] = {
            "enabled": True,
            "ready": False,
            "width": self.width,
            "height": self.height,
            "target_fps": round(1.0 / self.period_s, 3),
            "jpeg_quality": self.jpeg_quality,
            "mjcf": str(self.mjcf),
            "model_kind": "loading",
            "closure_error_m": {},
            "show_visual_model": self.show_visual_model,
            "show_collision_model": self.show_collision_model,
            "use_gravity_attitude": self.use_gravity_attitude,
            "attitude_source": self._attitude_source_locked(),
            "wheel_render_mode": self.wheel_render_mode,
            "show_joint_frames": any(self.joint_frames.values()),
            "joint_frames": dict(self.joint_frames),
            "camera": self._camera_dict_locked(),
        }

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def current_frame(self) -> tuple[bytes, str]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame, self.latest_content_type
            if self.error:
                raise RuntimeError(self.error)
        raise RuntimeError("MJCF render frame is not ready")

    def wait_frame(
        self,
        last_frame_id: int,
        *,
        timeout_s: float,
    ) -> tuple[int, bytes, str] | None:
        with self.frame_cv:
            self.frame_cv.wait_for(
                lambda: self.latest_frame_id != last_frame_id or self.error is not None,
                timeout=timeout_s,
            )
            if self.error and self.latest_frame is None:
                raise RuntimeError(self.error)
            if self.latest_frame_id == last_frame_id or self.latest_frame is None:
                return None
            return self.latest_frame_id, self.latest_frame, self.latest_content_type

    def info(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest_info)

    def settings(self) -> dict[str, Any]:
        with self.lock:
            return {
                "enabled": True,
                "show_visual_model": self.show_visual_model,
                "show_collision_model": self.show_collision_model,
                "use_gravity_attitude": self.use_gravity_attitude,
                "attitude_source": self._attitude_source_locked(),
                "wheel_render_mode": self.wheel_render_mode,
                "show_joint_frames": any(self.joint_frames.values()),
                "joint_frames": dict(self.joint_frames),
                "camera": self._camera_dict_locked(),
            }

    def update_settings(
        self,
        *,
        show_visual_model: bool | None = None,
        show_collision_model: bool | None = None,
        joint_frames: dict[str, bool] | None = None,
        use_gravity_attitude: bool | None = None,
        wheel_render_mode: str | None = None,
        camera_azimuth: float | None = None,
        camera_elevation: float | None = None,
        camera_distance: float | None = None,
    ) -> dict[str, Any]:
        with self.frame_cv:
            if show_visual_model is not None:
                self.show_visual_model = bool(show_visual_model)
            if show_collision_model is not None:
                self.show_collision_model = bool(show_collision_model)
            if joint_frames is not None:
                for key, value in joint_frames.items():
                    if key in self.joint_frames:
                        self.joint_frames[key] = bool(value)
            if use_gravity_attitude is not None:
                self.use_gravity_attitude = bool(use_gravity_attitude)
            parsed_wheel_mode = _parse_wheel_render_mode(wheel_render_mode)
            if parsed_wheel_mode is not None:
                self.wheel_render_mode = parsed_wheel_mode
            if camera_azimuth is not None:
                self.camera_azimuth = float(camera_azimuth) % 360.0
            if camera_elevation is not None:
                self.camera_elevation = float(np.clip(camera_elevation, -80.0, 35.0))
            if camera_distance is not None:
                self.camera_distance = float(np.clip(camera_distance, 0.45, 3.0))
            self.latest_info.update(
                {
                    "show_visual_model": self.show_visual_model,
                    "show_collision_model": self.show_collision_model,
                    "use_gravity_attitude": self.use_gravity_attitude,
                    "attitude_source": self._attitude_source_locked(),
                    "wheel_render_mode": self.wheel_render_mode,
                    "show_joint_frames": any(self.joint_frames.values()),
                    "joint_frames": dict(self.joint_frames),
                    "camera": self._camera_dict_locked(),
                }
            )
            self.frame_cv.notify_all()
            return {
                "enabled": True,
                "show_visual_model": self.show_visual_model,
                "show_collision_model": self.show_collision_model,
                "use_gravity_attitude": self.use_gravity_attitude,
                "attitude_source": self._attitude_source_locked(),
                "wheel_render_mode": self.wheel_render_mode,
                "show_joint_frames": any(self.joint_frames.values()),
                "joint_frames": dict(self.joint_frames),
                "camera": self._camera_dict_locked(),
            }

    def _settings_tuple(
        self,
    ) -> tuple[bool, bool, dict[str, bool], bool, str, float, float, float]:
        with self.lock:
            return (
                self.show_visual_model,
                self.show_collision_model,
                dict(self.joint_frames),
                self.use_gravity_attitude,
                self.wheel_render_mode,
                self.camera_azimuth,
                self.camera_elevation,
                self.camera_distance,
            )

    def _attitude_source_locked(self) -> str:
        return "gravity" if self.use_gravity_attitude else "gyro"

    def _camera_dict_locked(self) -> dict[str, float]:
        return {
            "azimuth": round(float(self.camera_azimuth), 3),
            "elevation": round(float(self.camera_elevation), 3),
            "distance": round(float(self.camera_distance), 3),
        }

    def _wait_until(self, deadline_s: float) -> None:
        while not self.stop_event.is_set():
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0.0:
                return
            if remaining_s > 0.006:
                self.stop_event.wait(min(remaining_s - 0.003, 0.01))
            else:
                time.sleep(0)

    def _run(self) -> None:
        renderer: MujocoStateRenderer | None = None
        try:
            renderer = MujocoStateRenderer(
                mjcf=self.mjcf,
                width=self.width,
                height=self.height,
            )
            last_render_time: float | None = None
            next_frame_at = time.perf_counter()
            while not self.stop_event.is_set():
                self._wait_until(next_frame_at)
                if self.stop_event.is_set():
                    break
                started = time.perf_counter()
                state = self.shared.get_state()
                if state is not None:
                    render_frame_hz = _frame_hz(last_render_time, started)
                    last_render_time = started
                    render_started = time.perf_counter()
                    (
                        show_visual_model,
                        show_collision_model,
                        joint_frames,
                        use_gravity_attitude,
                        wheel_render_mode,
                        camera_azimuth,
                        camera_elevation,
                        camera_distance,
                    ) = self._settings_tuple()
                    renderer.set_render_settings(
                        show_visual_model=show_visual_model,
                        show_collision_model=show_collision_model,
                        joint_frames=joint_frames,
                        use_gravity_attitude=use_gravity_attitude,
                        wheel_render_mode=wheel_render_mode,
                        camera_azimuth=camera_azimuth,
                        camera_elevation=camera_elevation,
                        camera_distance=camera_distance,
                    )
                    rgb = renderer.render_rgb(state)
                    render_ms = (time.perf_counter() - render_started) * 1000.0
                    encode_started = time.perf_counter()
                    frame, content_type = _encode_render_frame_rgb(rgb, self.jpeg_quality)
                    encode_ms = (time.perf_counter() - encode_started) * 1000.0
                    with self.frame_cv:
                        self.latest_frame = frame
                        self.latest_content_type = content_type
                        self.latest_frame_id += 1
                        self.error = None
                        self.latest_info.update(
                            {
                                "ready": True,
                                "frame_id": self.latest_frame_id,
                                "content_type": content_type,
                                "bytes": len(frame),
                                "render_fps": (
                                    round(render_frame_hz, 3)
                                    if render_frame_hz is not None
                                    else None
                                ),
                                "render_ms": round(render_ms, 3),
                                "encode_ms": round(encode_ms, 3),
                                "backend": renderer.backend_info(),
                                "model_kind": renderer.model_kind,
                                "closure_error_m": renderer.closure_error(),
                                "show_visual_model": show_visual_model,
                                "show_collision_model": show_collision_model,
                                "use_gravity_attitude": use_gravity_attitude,
                                "attitude_source": ("gravity" if use_gravity_attitude else "gyro"),
                                "wheel_render_mode": wheel_render_mode,
                                "show_joint_frames": any(joint_frames.values()),
                                "joint_frames": dict(joint_frames),
                                "camera": {
                                    "azimuth": round(float(camera_azimuth), 3),
                                    "elevation": round(float(camera_elevation), 3),
                                    "distance": round(float(camera_distance), 3),
                                },
                            }
                        )
                        self.frame_cv.notify_all()
                next_frame_at += self.period_s
                if next_frame_at < started:
                    next_frame_at = started + self.period_s
        except Exception as exc:
            with self.frame_cv:
                self.error = str(exc)
                self.latest_info.update({"ready": False, "error": str(exc)})
                self.frame_cv.notify_all()
        finally:
            if renderer is not None:
                renderer.close()


class MujocoStateRenderer:
    def __init__(self, *, mjcf: Path, width: int, height: int) -> None:
        if os.environ.get("MUJOCO_GL") is None and os.name != "nt":
            os.environ["MUJOCO_GL"] = "egl"
        import mujoco

        self.mujoco = mujoco
        self.lock = threading.Lock()
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.neutral_qpos = (
            np.asarray(self.model.key_qpos[0], dtype=np.float64).copy()
            if self.model.nkey > 0
            else np.asarray(self.model.qpos0, dtype=np.float64).copy()
        )
        self.scene_option = mujoco.MjvOption()
        self._prepare_geom_groups()
        self.renderer = mujoco.Renderer(
            self.model,
            height=max(int(height), 1),
            width=max(int(width), 1),
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 1.25
        self.camera.azimuth = 135.0
        self.camera.elevation = -20.0
        self.camera.lookat[:] = (0.0, 0.0, 0.10)
        self.joint_qpos = {
            name: self._joint_qpos_addr(name)
            for name in (
                "lf0_Joint",
                "lf1_Joint",
                "l_wheel_Joint",
                "l_drive_bar_Joint",
                "l_coupler_Joint",
                "rf0_Joint",
                "rf1_Joint",
                "r_wheel_Joint",
                "r_drive_bar_Joint",
                "r_coupler_Joint",
            )
        }
        self.closure_site_ids = {
            name: self._site_id(name)
            for name in (
                "l_coupler_end",
                "lf_coupler_closure",
                "r_coupler_end",
                "rf_coupler_closure",
            )
        }
        self.closed_chain_enabled = self._has_closed_chain_render_joints()
        self.model_kind = "closed_chain" if self.closed_chain_enabled else "fourbar_surrogate"
        self.closed_chain_passive_seed = self._initial_closed_chain_passive_seed()
        self.closed_chain_failed_signatures: dict[str, tuple[float, float]] = {}
        self.last_closure_error_m: dict[str, float] = {}
        self.debug_joint_ids = {key: self._joint_id(name) for key, name, _ in _DEBUG_JOINTS}
        self.debug_body_ids = {key: self._body_id(name) for key, name, _ in _DEBUG_BODY_FRAMES}
        self.enabled_joint_frames = {key: False for key in _DEBUG_FRAME_KEYS}
        self.use_gravity_attitude = _DEFAULT_USE_GRAVITY_ATTITUDE
        self.wheel_render_mode = _DEFAULT_WHEEL_RENDER_MODE
        self._gyro_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._gyro_last_tick_ms: int | None = None
        self._gyro_last_seq: int | None = None
        self._gyro_reset_pending = True
        self._wheel_integrated_pos = np.zeros(2, dtype=np.float64)
        self._wheel_last_tick_ms: int | None = None
        self._wheel_last_seq: int | None = None
        self._wheel_reset_pending = True
        self._backend_info: dict[str, str | None] | None = None

    def set_render_settings(
        self,
        *,
        show_visual_model: bool,
        show_collision_model: bool,
        joint_frames: dict[str, bool],
        use_gravity_attitude: bool,
        wheel_render_mode: str,
        camera_azimuth: float,
        camera_elevation: float,
        camera_distance: float,
    ) -> None:
        with self.lock:
            self.scene_option.geomgroup[_COLLISION_GEOM_GROUP] = 1 if show_collision_model else 0
            self.scene_option.geomgroup[_VISUAL_GEOM_GROUP] = 1 if show_visual_model else 0
            self.scene_option.geomgroup[_FLOOR_GEOM_GROUP] = 1
            self.scene_option.flags[self.mujoco.mjtVisFlag.mjVIS_JOINT] = 0
            self.scene_option.label = self.mujoco.mjtLabel.mjLABEL_NONE
            self.enabled_joint_frames = {
                key: bool(joint_frames.get(key, False)) for key in _DEBUG_FRAME_KEYS
            }
            next_use_gravity_attitude = bool(use_gravity_attitude)
            if next_use_gravity_attitude != self.use_gravity_attitude:
                self._gyro_reset_pending = True
            self.use_gravity_attitude = next_use_gravity_attitude
            next_wheel_render_mode = (
                _parse_wheel_render_mode(wheel_render_mode) or _DEFAULT_WHEEL_RENDER_MODE
            )
            if next_wheel_render_mode != self.wheel_render_mode:
                self._wheel_reset_pending = True
            self.wheel_render_mode = next_wheel_render_mode
            self.camera.azimuth = float(camera_azimuth)
            self.camera.elevation = float(camera_elevation)
            self.camera.distance = float(camera_distance)

    def render_rgb(self, state: PolicyStateFrame) -> np.ndarray:
        with self.lock:
            self._apply_state(state)
            self.mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(
                self.data,
                camera=self.camera,
                scene_option=self.scene_option,
            )
            self._draw_joint_frames()
            rgb = self.renderer.render()
            self._capture_backend_info()
            return rgb

    def render_png(self, state: PolicyStateFrame) -> bytes:
        return _encode_png_rgb(self.render_rgb(state))

    def backend_info(self) -> dict[str, str | None]:
        return dict(self._backend_info or {})

    def closure_error(self) -> dict[str, float]:
        return dict(self.last_closure_error_m)

    def close(self) -> None:
        with self.lock, suppress(Exception):
            self.renderer.close()

    def _capture_backend_info(self) -> None:
        if self._backend_info is not None:
            return
        info: dict[str, str | None] = {}
        with suppress(Exception):
            from OpenGL import GL

            for key, enum in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
                ("shading_language", GL.GL_SHADING_LANGUAGE_VERSION),
            ):
                value = GL.glGetString(enum)
                info[key] = value.decode() if value else None
        self._backend_info = info

    def _joint_qpos_addr(self, name: str) -> int | None:
        joint_id = self._joint_id(name)
        if joint_id is None:
            return None
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_id(self, name: str) -> int | None:
        joint_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            return None
        return int(joint_id)

    def _body_id(self, name: str) -> int | None:
        body_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id < 0:
            return None
        return int(body_id)

    def _site_id(self, name: str) -> int | None:
        site_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
        if site_id < 0:
            return None
        return int(site_id)

    def _has_closed_chain_render_joints(self) -> bool:
        required_joints = (
            "lf0_Joint",
            "lf1_Joint",
            "l_wheel_Joint",
            "l_drive_bar_Joint",
            "l_coupler_Joint",
            "rf0_Joint",
            "rf1_Joint",
            "r_wheel_Joint",
            "r_drive_bar_Joint",
            "r_coupler_Joint",
        )
        required_sites = (
            "l_coupler_end",
            "lf_coupler_closure",
            "r_coupler_end",
            "rf_coupler_closure",
        )
        return all(self.joint_qpos.get(name) is not None for name in required_joints) and all(
            self.closure_site_ids.get(name) is not None for name in required_sites
        )

    def _initial_closed_chain_passive_seed(self) -> dict[str, float]:
        seed: dict[str, float] = {}
        for name in ("lf1_Joint", "l_coupler_Joint", "rf1_Joint", "r_coupler_Joint"):
            addr = self.joint_qpos.get(name)
            if addr is not None and addr < self.neutral_qpos.size:
                seed[name] = float(self.neutral_qpos[addr])
        return seed

    def _draw_joint_frames(self) -> None:
        for key, _, _ in _DEBUG_BODY_FRAMES:
            if not self.enabled_joint_frames.get(key, False):
                continue
            body_id = self.debug_body_ids.get(key)
            if body_id is None:
                continue
            anchor = np.asarray(self.data.xpos[body_id], dtype=np.float64)
            mat = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            self._draw_frame_axes(anchor, mat)

        for key, _, _ in _DEBUG_JOINTS:
            if not self.enabled_joint_frames.get(key, False):
                continue
            joint_id = self.debug_joint_ids.get(key)
            if joint_id is None:
                continue
            anchor = np.asarray(self.data.xanchor[joint_id], dtype=np.float64)
            body_id = int(self.model.jnt_bodyid[joint_id])
            mat = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            self._draw_frame_axes(anchor, mat)

    def _draw_frame_axes(self, anchor: np.ndarray, mat: np.ndarray) -> None:
        for axis_idx, rgba in enumerate(_JOINT_FRAME_COLORS):
            axis = mat[:, axis_idx]
            end = anchor + axis * _JOINT_FRAME_AXIS_LENGTH
            self._add_user_arrow(anchor, end, rgba)

    def _add_user_arrow(
        self,
        start: np.ndarray,
        end: np.ndarray,
        rgba: np.ndarray,
    ) -> None:
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        self.mujoco.mjv_initGeom(
            geom,
            self.mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        self.mujoco.mjv_connector(
            geom,
            self.mujoco.mjtGeom.mjGEOM_ARROW,
            _JOINT_FRAME_AXIS_WIDTH,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    def _prepare_geom_groups(self) -> None:
        floor_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        if floor_id >= 0:
            self.model.geom_group[floor_id] = _FLOOR_GEOM_GROUP
        collision_ids = np.flatnonzero(self.model.geom_group == _COLLISION_GEOM_GROUP)
        if collision_ids.size:
            self.model.geom_rgba[collision_ids] = (1.0, 0.72, 0.18, 0.38)

    def _apply_state(self, state: PolicyStateFrame) -> None:
        self.data.qpos[:] = self.neutral_qpos
        if self.model.nq >= 7:
            self.data.qpos[0:3] = (
                0.0,
                0.0,
                float(getattr(_ROBOT_CFG, "default_base_height", 0.22)),
            )
            self.data.qpos[3:7] = self._base_quat_for_state(state)

        policy_pos = np.asarray(state.joint_pos, dtype=np.float64).reshape(4)
        wheel_pos = self._wheel_pos_for_state(state)
        if self.closed_chain_enabled:
            self._apply_closed_chain_state(policy_pos, wheel_pos)
            return

        self._apply_surrogate_state(policy_pos, wheel_pos)

    def _apply_closed_chain_state(self, policy_pos: np.ndarray, wheel_pos: np.ndarray) -> None:
        values = {
            "lf0_Joint": float(policy_pos[0]),
            "l_drive_bar_Joint": float(policy_pos[1]),
            "l_wheel_Joint": float(wheel_pos[0]),
            "rf0_Joint": float(policy_pos[2]),
            "r_drive_bar_Joint": float(policy_pos[3]),
            "r_wheel_Joint": float(wheel_pos[1]),
        }
        self._set_qpos_values(values)
        for name, value in self.closed_chain_passive_seed.items():
            self._set_qpos_value(name, value)
        self._project_closed_chain_passive_joints()

    def _apply_surrogate_state(self, policy_pos: np.ndarray, wheel_pos: np.ndarray) -> None:
        output_pos = _policy_to_output_pos_np(policy_pos)
        self._set_qpos_values(
            {
                "lf0_Joint": float(output_pos[0]),
                "lf1_Joint": float(output_pos[1]),
                "l_wheel_Joint": float(wheel_pos[0]),
                "rf0_Joint": float(output_pos[2]),
                "rf1_Joint": float(output_pos[3]),
                "r_wheel_Joint": float(wheel_pos[1]),
            }
        )

    def _set_qpos_values(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self._set_qpos_value(name, value)

    def _set_qpos_value(self, name: str, value: float) -> None:
        addr = self.joint_qpos.get(name)
        if addr is not None and addr < self.model.nq:
            self.data.qpos[addr] = value

    def _project_closed_chain_passive_joints(self) -> None:
        self.last_closure_error_m = {
            "left": round(
                self._solve_closed_chain_side(
                    side="left",
                    active_joints=("lf0_Joint", "l_drive_bar_Joint"),
                    passive_joints=("lf1_Joint", "l_coupler_Joint"),
                    site_a="l_coupler_end",
                    site_b="lf_coupler_closure",
                ),
                9,
            ),
            "right": round(
                self._solve_closed_chain_side(
                    side="right",
                    active_joints=("rf0_Joint", "r_drive_bar_Joint"),
                    passive_joints=("rf1_Joint", "r_coupler_Joint"),
                    site_a="r_coupler_end",
                    site_b="rf_coupler_closure",
                ),
                9,
            ),
        }

    def _solve_closed_chain_side(
        self,
        *,
        side: str,
        active_joints: tuple[str, str],
        passive_joints: tuple[str, str],
        site_a: str,
        site_b: str,
    ) -> float:
        passive_addrs = tuple(self.joint_qpos.get(name) for name in passive_joints)
        active_addrs = tuple(self.joint_qpos.get(name) for name in active_joints)
        site_ids = (self.closure_site_ids.get(site_a), self.closure_site_ids.get(site_b))
        if (
            passive_addrs[0] is None
            or passive_addrs[1] is None
            or active_addrs[0] is None
            or active_addrs[1] is None
            or site_ids[0] is None
            or site_ids[1] is None
        ):
            return 0.0

        base_qpos = self.data.qpos.copy()
        active_signature = (
            round(float(base_qpos[active_addrs[0]]), 3),
            round(float(base_qpos[active_addrs[1]]), 3),
        )
        if self.closed_chain_failed_signatures.get(side) == active_signature:
            self.mujoco.mj_forward(self.model, self.data)
            return self._closed_chain_residual_norm(site_ids)

        seeds = self._closed_chain_solver_seeds(passive_joints, passive_addrs)
        best_error = math.inf
        best_passive = (
            float(self.data.qpos[passive_addrs[0]]),
            float(self.data.qpos[passive_addrs[1]]),
        )
        for seed in seeds:
            self.data.qpos[:] = base_qpos
            self.data.qpos[passive_addrs[0]] = float(seed[0])
            self.data.qpos[passive_addrs[1]] = float(seed[1])
            error = self._refine_closed_chain_side(passive_addrs, site_ids)
            if error < best_error:
                best_error = error
                best_passive = (
                    _wrap_angle_scalar(float(self.data.qpos[passive_addrs[0]])),
                    _wrap_angle_scalar(float(self.data.qpos[passive_addrs[1]])),
                )
            if error <= _CLOSED_CHAIN_RETRY_ERROR_M:
                break

        self.data.qpos[:] = base_qpos
        self.data.qpos[passive_addrs[0]] = best_passive[0]
        self.data.qpos[passive_addrs[1]] = best_passive[1]
        self.closed_chain_passive_seed[passive_joints[0]] = best_passive[0]
        self.closed_chain_passive_seed[passive_joints[1]] = best_passive[1]
        self.mujoco.mj_forward(self.model, self.data)
        final_error = self._closed_chain_residual_norm(site_ids)
        if final_error > _CLOSED_CHAIN_RETRY_ERROR_M:
            self.closed_chain_failed_signatures[side] = active_signature
        else:
            self.closed_chain_failed_signatures.pop(side, None)
        return final_error

    def _closed_chain_solver_seeds(
        self,
        passive_joints: tuple[str, str],
        passive_addrs: tuple[int, int],
    ) -> tuple[tuple[float, float], ...]:
        current = (
            float(self.data.qpos[passive_addrs[0]]),
            float(self.data.qpos[passive_addrs[1]]),
        )
        neutral = (
            float(self.neutral_qpos[passive_addrs[0]]),
            float(self.neutral_qpos[passive_addrs[1]]),
        )
        previous = (
            self.closed_chain_passive_seed.get(passive_joints[0], neutral[0]),
            self.closed_chain_passive_seed.get(passive_joints[1], neutral[1]),
        )
        return (
            previous,
            current,
            neutral,
            (0.0, 0.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-2.0, 2.0),
            (2.0, -2.0),
            (neutral[0] + math.pi, neutral[1] + math.pi),
            (neutral[0] - math.pi, neutral[1] - math.pi),
        )

    def _refine_closed_chain_side(
        self,
        passive_addrs: tuple[int, int],
        site_ids: tuple[int, int],
    ) -> float:
        for _ in range(_CLOSED_CHAIN_SOLVER_ITERS):
            residual = self._closed_chain_residual(site_ids)
            error = float(np.linalg.norm(residual))
            if error <= _CLOSED_CHAIN_RETRY_ERROR_M:
                return error

            jacobian = np.zeros((3, 2), dtype=np.float64)
            for col, addr in enumerate(passive_addrs):
                old_value = float(self.data.qpos[addr])
                self.data.qpos[addr] = old_value + _CLOSED_CHAIN_SOLVER_EPS
                residual_plus = self._closed_chain_residual(site_ids)
                self.data.qpos[addr] = old_value
                jacobian[:, col] = (residual_plus - residual) / _CLOSED_CHAIN_SOLVER_EPS

            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            delta = np.clip(delta, -_CLOSED_CHAIN_STEP_LIMIT_RAD, _CLOSED_CHAIN_STEP_LIMIT_RAD)
            for col, addr in enumerate(passive_addrs):
                self.data.qpos[addr] += float(delta[col])

        return self._closed_chain_residual_norm(site_ids)

    def _closed_chain_residual(self, site_ids: tuple[int, int]) -> np.ndarray:
        self.mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.site_xpos[site_ids[0]] - self.data.site_xpos[site_ids[1]])

    def _closed_chain_residual_norm(self, site_ids: tuple[int, int]) -> float:
        return float(np.linalg.norm(self._closed_chain_residual(site_ids)))

    def _base_quat_for_state(self, state: PolicyStateFrame) -> np.ndarray:
        gravity_quat = _quat_from_projected_gravity(state.projected_gravity)
        if self.use_gravity_attitude:
            self._reset_gyro_integrator(gravity_quat, state)
            return gravity_quat

        if self._gyro_reset_pending:
            self._reset_gyro_integrator(gravity_quat, state)
            self._gyro_reset_pending = False
            return self._gyro_quat.copy()

        if self._gyro_last_seq == int(state.seq):
            return self._gyro_quat.copy()

        dt_s = self._gyro_dt_s(state)
        self._gyro_quat = _quat_integrate_body_omega(
            self._gyro_quat,
            np.asarray(state.base_ang_vel_body, dtype=np.float64),
            dt_s,
        )
        self._gyro_last_tick_ms = int(state.tick_ms)
        self._gyro_last_seq = int(state.seq)
        return self._gyro_quat.copy()

    def _reset_gyro_integrator(self, quat: np.ndarray, state: PolicyStateFrame) -> None:
        self._gyro_quat = _quat_normalized(quat)
        self._gyro_last_tick_ms = int(state.tick_ms)
        self._gyro_last_seq = int(state.seq)

    def _gyro_dt_s(self, state: PolicyStateFrame) -> float:
        tick_ms = int(state.tick_ms)
        seq = int(state.seq)
        dt_s: float
        if self._gyro_last_tick_ms is not None and tick_ms > self._gyro_last_tick_ms:
            dt_s = float(tick_ms - self._gyro_last_tick_ms) / 1000.0
        elif self._gyro_last_seq is not None and seq > self._gyro_last_seq:
            dt_s = float(seq - self._gyro_last_seq) * float(_ROBOT_CFG.control_dt)
        else:
            dt_s = float(_ROBOT_CFG.control_dt)
        return float(np.clip(dt_s, 0.0, _MAX_GYRO_INTEGRATION_DT_S))

    def _wheel_pos_for_state(self, state: PolicyStateFrame) -> np.ndarray:
        measured_pos = np.nan_to_num(
            np.asarray(state.wheel_pos, dtype=np.float64).reshape(2),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.wheel_render_mode == "position":
            self._reset_wheel_integrator(measured_pos, state)
            return measured_pos

        if self._wheel_reset_pending:
            self._reset_wheel_integrator(measured_pos, state)
            self._wheel_reset_pending = False
            return self._wheel_integrated_pos.copy()

        if self._wheel_last_seq == int(state.seq):
            return self._wheel_integrated_pos.copy()

        wheel_vel = np.nan_to_num(
            np.asarray(state.wheel_vel, dtype=np.float64).reshape(2),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._wheel_integrated_pos = _wrap_angle_np(
            self._wheel_integrated_pos + wheel_vel * self._wheel_dt_s(state)
        )
        self._wheel_last_tick_ms = int(state.tick_ms)
        self._wheel_last_seq = int(state.seq)
        return self._wheel_integrated_pos.copy()

    def _reset_wheel_integrator(self, wheel_pos: np.ndarray, state: PolicyStateFrame) -> None:
        self._wheel_integrated_pos = _wrap_angle_np(wheel_pos)
        self._wheel_last_tick_ms = int(state.tick_ms)
        self._wheel_last_seq = int(state.seq)

    def _wheel_dt_s(self, state: PolicyStateFrame) -> float:
        tick_ms = int(state.tick_ms)
        seq = int(state.seq)
        dt_s: float
        if self._wheel_last_tick_ms is not None and tick_ms > self._wheel_last_tick_ms:
            dt_s = float(tick_ms - self._wheel_last_tick_ms) / 1000.0
        elif self._wheel_last_seq is not None and seq > self._wheel_last_seq:
            dt_s = float(seq - self._wheel_last_seq) * float(_ROBOT_CFG.control_dt)
        else:
            dt_s = float(_ROBOT_CFG.control_dt)
        return float(np.clip(dt_s, 0.0, _MAX_WHEEL_INTEGRATION_DT_S))


def _policy_to_output_pos_np(policy_pos: np.ndarray) -> np.ndarray:
    return np.asarray(policy_to_output_pos_np(np.asarray(policy_pos, dtype=np.float64))).reshape(4)


def _quat_from_projected_gravity(projected_gravity: tuple[float, float, float]) -> np.ndarray:
    body_down = _normalized(np.asarray(projected_gravity, dtype=np.float64), (0.0, 0.0, -1.0))
    world_down = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    return _quat_between_vectors(body_down, world_down).astype(np.float64, copy=False)


def _normalized(values: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    arr = np.nan_to_num(values.reshape(3), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-6:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def _quat_between_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = _normalized(src, (0.0, 0.0, -1.0))
    dst = _normalized(dst, (0.0, 0.0, -1.0))
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross((1.0, 0.0, 0.0), src)
        if float(np.linalg.norm(axis)) < 1.0e-6:
            axis = np.cross((0.0, 1.0, 0.0), src)
        axis = _normalized(axis, (1.0, 0.0, 0.0))
        return np.asarray((0.0, axis[0], axis[1], axis[2]), dtype=np.float64)

    axis = np.cross(src, dst)
    scale = math.sqrt((1.0 + dot) * 2.0)
    quat = np.asarray(
        (
            scale * 0.5,
            axis[0] / scale,
            axis[1] / scale,
            axis[2] / scale,
        ),
        dtype=np.float64,
    )
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def _quat_normalized(quat: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(quat, dtype=np.float64).reshape(4), nan=0.0)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    if arr[0] < 0.0:
        arr = -arr
    return arr / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64).reshape(4)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64).reshape(4)
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quat_integrate_body_omega(quat: np.ndarray, omega_body: np.ndarray, dt_s: float) -> np.ndarray:
    omega = np.nan_to_num(np.asarray(omega_body, dtype=np.float64).reshape(3), nan=0.0)
    angle = float(np.linalg.norm(omega)) * max(float(dt_s), 0.0)
    if angle < 1.0e-12:
        return _quat_normalized(quat)
    axis = omega / max(float(np.linalg.norm(omega)), 1.0e-12)
    half = 0.5 * angle
    delta = np.asarray(
        (
            math.cos(half),
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
        ),
        dtype=np.float64,
    )
    return _quat_normalized(_quat_multiply(quat, delta))


def _encode_png_rgb(rgb: np.ndarray) -> bytes:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("renderer returned an unexpected image shape")
    arr = arr[:, :, :3]
    height, width, _ = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, level=1))
        + chunk(b"IEND", b"")
    )


def _encode_render_frame_rgb(rgb: np.ndarray, jpeg_quality: int) -> tuple[bytes, str]:
    with suppress(Exception):
        return _encode_jpeg_rgb(rgb, jpeg_quality), "image/jpeg"
    return _encode_png_rgb(rgb), "image/png"


def _encode_jpeg_rgb(rgb: np.ndarray, jpeg_quality: int) -> bytes:
    from PIL import Image

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("renderer returned an unexpected image shape")
    arr = arr[:, :, :3]
    stream = BytesIO()
    Image.fromarray(arr).save(
        stream,
        format="JPEG",
        quality=int(np.clip(jpeg_quality, 30, 95)),
        optimize=False,
        progressive=False,
    )
    return stream.getvalue()


def _frame_hz(last_wall_time: float | None, now: float) -> float | None:
    if last_wall_time is None:
        return None
    dt = max(now - last_wall_time, 1.0e-6)
    return 1.0 / dt


def _finite_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return [round(float(v), 6) for v in arr]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web visualizer for STM32 CDC state frames.")
    parser.add_argument("--port", default="auto", help="USB CDC device path or auto.")
    parser.add_argument("--baudrate", type=int, default=921600, help="CDC baudrate hint.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument("--viewer-port", type=int, default=8097, help="HTTP viewer port.")
    parser.add_argument("--synthetic", action="store_true", help="Use animated synthetic state.")
    parser.add_argument(
        "--local-cdc",
        action="store_true",
        help="Read USB CDC directly instead of the default NX relay.",
    )
    parser.add_argument(
        "--remote-url",
        default=_DEFAULT_NX_RELAY_URL,
        help="Remote CDC visualizer base URL.",
    )
    parser.add_argument(
        "--remote-timeout-s", type=float, default=10.0, help="Remote stream timeout."
    )
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Synthetic update rate.")
    parser.add_argument("--read-timeout-s", type=float, default=0.02, help="CDC read wait.")
    parser.add_argument("--mjcf", type=Path, default=_DEFAULT_MJCF, help="MJCF used for rendering.")
    parser.add_argument(
        "--render-width", type=int, default=_DEFAULT_RENDER_WIDTH, help="MJCF render width."
    )
    parser.add_argument(
        "--render-height", type=int, default=_DEFAULT_RENDER_HEIGHT, help="MJCF render height."
    )
    parser.add_argument(
        "--render-fps", type=float, default=_DEFAULT_RENDER_FPS, help="MJCF render rate."
    )
    parser.add_argument(
        "--render-jpeg-quality",
        type=int,
        default=_DEFAULT_RENDER_JPEG_QUALITY,
        help="MJCF JPEG quality.",
    )
    parser.add_argument(
        "--show-collision", action="store_true", help="Show MJCF collision geoms by default."
    )
    parser.add_argument(
        "--show-joints", action="store_true", help="Show MJCF joint axes by default."
    )
    parser.add_argument(
        "--hide-visual-model", action="store_true", help="Hide MJCF visual geoms by default."
    )
    parser.add_argument("--no-mjcf-render", action="store_true", help="Disable MJCF rendering.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    shared = SharedSnapshot()
    stop_event = threading.Event()

    if args.synthetic:
        reader = threading.Thread(
            target=run_synthetic_reader,
            kwargs={
                "shared": shared,
                "rate_hz": float(args.rate_hz),
                "stop_event": stop_event,
            },
            daemon=True,
        )
    elif not args.local_cdc and args.remote_url:
        reader = threading.Thread(
            target=run_remote_reader,
            kwargs={
                "shared": shared,
                "remote_url": str(args.remote_url),
                "timeout_s": float(args.remote_timeout_s),
                "stop_event": stop_event,
            },
            daemon=True,
        )
    else:
        reader = threading.Thread(
            target=run_cdc_reader,
            kwargs={
                "shared": shared,
                "dev": str(args.port),
                "baudrate": int(args.baudrate),
                "read_timeout_s": float(args.read_timeout_s),
                "stop_event": stop_event,
            },
            daemon=True,
        )
    reader.start()

    renderer = None
    if not args.no_mjcf_render:
        renderer = MujocoRenderWorker(
            shared=shared,
            mjcf=Path(args.mjcf),
            width=int(args.render_width),
            height=int(args.render_height),
            fps=float(args.render_fps),
            jpeg_quality=int(args.render_jpeg_quality),
            show_visual_model=not bool(args.hide_visual_model),
            show_collision_model=bool(args.show_collision),
            show_joint_frames=bool(args.show_joints),
        )
        renderer.start()
        print(f"MJCF render worker started from {Path(args.mjcf)}")

    server = VisualizerServer((str(args.host), int(args.viewer_port)), shared, renderer)
    url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"CDC visualizer listening on http://{url_host}:{args.viewer_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
        if renderer is not None:
            renderer.close()
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SerialLeg CDC State</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111418;
      --panel: #181d23;
      --line: #2b333d;
      --text: #e6edf3;
      --muted: #9da7b3;
      --cyan: #50d6ff;
      --amber: #ffc857;
      --red: #ff6b6b;
      --green: #69db7c;
      --blue: #7aa2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    #app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(500px, 34vw);
      height: 100vh;
    }
    #viewWrap {
      position: relative;
      min-width: 0;
      border-right: 1px solid var(--line);
      background: #07090c;
      cursor: grab;
      touch-action: none;
    }
    #viewWrap.dragging {
      cursor: grabbing;
    }
    #mjcfView {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      opacity: 0;
      pointer-events: none;
      transition: opacity 120ms ease;
    }
    #view {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
    }
    #hud {
      position: absolute;
      top: 14px;
      left: 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      max-width: calc(100% - 28px);
    }
    .pill {
      padding: 5px 8px;
      border: 1px solid var(--line);
      background: rgba(24, 29, 35, 0.78);
      border-radius: 6px;
      color: var(--muted);
      backdrop-filter: blur(6px);
    }
    .pill strong { color: var(--text); font-weight: 650; }
    #viewControls {
      position: absolute;
      bottom: 14px;
      right: 14px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      max-width: calc(100% - 28px);
      cursor: default;
    }
    .controlCluster {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      background: rgba(24, 29, 35, 0.78);
      border-radius: 6px;
      color: var(--text);
      backdrop-filter: blur(6px);
      user-select: none;
      cursor: pointer;
    }
    .toggle input {
      width: 15px;
      height: 15px;
      margin: 0;
      accent-color: var(--cyan);
    }
    .toggle span {
      line-height: 1;
      white-space: nowrap;
    }
    .toggle.small {
      min-height: 26px;
      padding: 4px 7px;
      font-size: 12px;
    }
    .toggle.small input {
      width: 13px;
      height: 13px;
    }
    #side {
      overflow: auto;
      background: var(--panel);
    }
    .sideHeader {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 16px 18px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(24, 29, 35, 0.96);
    }
    #sideContent {
      display: grid;
      gap: 16px;
      padding: 14px 18px 18px;
    }
    .section {
      min-width: 0;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .section:first-child {
      padding-top: 0;
      border-top: 0;
    }
    h1, h2 {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0;
      font-weight: 700;
    }
    h2 {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .field {
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid #27313b;
      border-radius: 6px;
      background: #11161c;
    }
    .field.wide { grid-column: 1 / -1; }
    .k {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      margin-bottom: 3px;
    }
    .v {
      font-variant-numeric: tabular-nums;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .rows {
      display: grid;
      gap: 7px;
    }
    .barrow {
      display: grid;
      grid-template-columns: 48px minmax(0, 1fr) 76px;
      align-items: center;
      gap: 8px;
    }
    .bar {
      height: 8px;
      border-radius: 999px;
      background: #0f1216;
      border: 1px solid #27303a;
      overflow: hidden;
      position: relative;
    }
    .bar::before {
      content: "";
      position: absolute;
      left: 50%;
      width: 1px;
      top: 0;
      bottom: 0;
      background: #45515e;
    }
    .fill {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      background: var(--cyan);
    }
    .fill.neg {
      left: auto;
      right: 50%;
      background: var(--amber);
    }
    .value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--text);
    }
    #err {
      color: var(--red);
      margin-top: 7px;
      min-height: 16px;
      font-size: 12px;
    }
    .obsList {
      display: grid;
      gap: 8px;
    }
    .obsRow {
      display: grid;
      grid-template-columns: 140px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid #27313b;
      border-radius: 6px;
      background: #11161c;
    }
    .obsName {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .obsValues {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      min-width: 0;
    }
    .chip {
      min-width: 54px;
      padding: 2px 5px;
      border: 1px solid #303945;
      border-radius: 5px;
      background: #0d1116;
      color: var(--text);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    @media (max-width: 1180px) {
      #app { grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }
      #viewWrap { border-right: 0; border-bottom: 1px solid var(--line); }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .obsRow { grid-template-columns: 120px minmax(0, 1fr); }
    }
  </style>
</head>
<body>
  <div id="app">
    <section id="viewWrap">
      <img id="mjcfView" src="/render_stream" alt="SerialLeg MJCF render">
      <canvas id="view"></canvas>
      <div id="hud">
        <div class="pill">seq <strong id="hudSeq">-</strong></div>
        <div class="pill">hz <strong id="hudHz">-</strong></div>
        <div class="pill">source <strong id="hudSource">-</strong></div>
        <div class="pill">output <strong id="hudOutput">-</strong></div>
        <div class="pill">att <strong id="hudAttitude">gravity</strong></div>
        <div class="pill">wheel <strong id="hudWheelMode">pos</strong></div>
        <div class="pill">render <strong id="hudRender">canvas</strong></div>
      </div>
      <div id="viewControls">
        <div class="controlCluster">
          <label class="toggle"><input id="visualToggle" type="checkbox" checked><span>Visual</span></label>
          <label class="toggle"><input id="collisionToggle" type="checkbox"><span>Collision</span></label>
          <label class="toggle"><input id="gravityAttitudeToggle" type="checkbox" checked><span>Gravity</span></label>
          <label class="toggle"><input id="wheelVelocityToggle" type="checkbox"><span>Wheel Vel</span></label>
        </div>
        <div class="controlCluster" id="jointToggleGroup">
          <label class="toggle small"><input class="jointToggle" data-joint="base" type="checkbox"><span>base_link</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="lf0" type="checkbox"><span>LF0</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="lf1" type="checkbox"><span>LF1</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="lb" type="checkbox"><span>LB</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="lc" type="checkbox"><span>LC</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="lw" type="checkbox"><span>LW</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="rf0" type="checkbox"><span>RF0</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="rf1" type="checkbox"><span>RF1</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="rb" type="checkbox"><span>RB</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="rc" type="checkbox"><span>RC</span></label>
          <label class="toggle small"><input class="jointToggle" data-joint="rw" type="checkbox"><span>RW</span></label>
        </div>
      </div>
    </section>
    <aside id="side">
      <div class="sideHeader">
        <h1>SerialLeg CDC State</h1>
        <div id="err"></div>
      </div>
      <div id="sideContent">
        <section class="section">
          <h2>Comm</h2>
          <div class="grid" id="commGrid"></div>
        </section>
        <section class="section">
          <h2>Status</h2>
          <div class="grid" id="statusGrid"></div>
        </section>
        <section class="section">
          <h2>Joint Pos</h2>
          <div class="rows" id="jointBars"></div>
        </section>
        <section class="section">
          <h2>Joint Vel</h2>
          <div class="rows" id="jointVelBars"></div>
        </section>
        <section class="section">
          <h2>Wheel</h2>
          <div class="rows" id="wheelBars"></div>
        </section>
        <section class="section">
          <h2>Observation Slices</h2>
          <div class="obsList" id="obsGrid"></div>
        </section>
      </div>
    </aside>
  </div>
<script>
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const viewWrap = document.getElementById("viewWrap");
const renderImg = document.getElementById("mjcfView");
const visualToggle = document.getElementById("visualToggle");
const collisionToggle = document.getElementById("collisionToggle");
const gravityAttitudeToggle = document.getElementById("gravityAttitudeToggle");
const wheelVelocityToggle = document.getElementById("wheelVelocityToggle");
const jointToggles = Array.from(document.querySelectorAll(".jointToggle"));
const labels = ["LF", "LB", "RF", "RB"];
const CAMERA_SEND_INTERVAL_MS = 16;
let snapshot = null;
let renderInfo = {};
let mjcfRenderOk = false;
let renderSettingsTimer = null;
let renderSettingsInFlight = false;
let renderSettingsDirty = false;
let mjcfPointerId = null;
let mjcfLastPointer = [0, 0];
let mjcfCamera = { azimuth: 135, elevation: -20, distance: 1.25 };
let mouseDown = false;
let camYaw = -0.65;
let camPitch = 0.35;
let camScale = 900;
let lastMouse = [0, 0];
let canvasWheelPos = [0, 0];
let canvasWheelLastSeq = null;
let canvasWheelLastTickMs = null;

renderImg.addEventListener("load", () => {
  mjcfRenderOk = true;
  renderImg.style.opacity = "1";
  canvas.style.opacity = "0";
  canvas.style.pointerEvents = "none";
  document.getElementById("hudRender").textContent = "mjcf";
});
renderImg.addEventListener("error", () => {
  mjcfRenderOk = false;
  renderImg.style.opacity = "0";
  canvas.style.opacity = "1";
  canvas.style.pointerEvents = "auto";
  document.getElementById("hudRender").textContent = "canvas";
});

function renderSettingsParams() {
  const params = new URLSearchParams({
    visual: visualToggle.checked ? "1" : "0",
    collision: collisionToggle.checked ? "1" : "0",
    gravity_attitude: gravityAttitudeToggle.checked ? "1" : "0",
    wheel_render_mode: wheelVelocityToggle.checked ? "velocity" : "position",
    camera_azimuth: String(mjcfCamera.azimuth),
    camera_elevation: String(mjcfCamera.elevation),
    camera_distance: String(mjcfCamera.distance),
  });
  jointToggles.forEach(input => {
    params.set(`joint_${input.dataset.joint}`, input.checked ? "1" : "0");
  });
  return params;
}
function applyRenderSettings(settings, syncCamera=true) {
  if (!settings.enabled) return;
  visualToggle.checked = !!settings.show_visual_model;
  collisionToggle.checked = !!settings.show_collision_model;
  gravityAttitudeToggle.checked = !!settings.use_gravity_attitude;
  updateAttitudeLabel();
  wheelVelocityToggle.checked = settings.wheel_render_mode === "velocity";
  updateWheelModeLabel();
  const jointFrames = settings.joint_frames || {};
  jointToggles.forEach(input => {
    input.checked = !!jointFrames[input.dataset.joint];
  });
  if (syncCamera && settings.camera) {
    mjcfCamera.azimuth = Number(settings.camera.azimuth ?? mjcfCamera.azimuth);
    mjcfCamera.elevation = Number(settings.camera.elevation ?? mjcfCamera.elevation);
    mjcfCamera.distance = Number(settings.camera.distance ?? mjcfCamera.distance);
  }
}
function updateAttitudeLabel() {
  document.getElementById("hudAttitude").textContent = gravityAttitudeToggle.checked ? "gravity" : "gyro";
}
function updateWheelModeLabel() {
  document.getElementById("hudWheelMode").textContent = wheelVelocityToggle.checked ? "vel" : "pos";
}
async function fetchRenderSettings(params=null, syncCamera=true) {
  const url = params ? `/render_settings?${params.toString()}` : "/render_settings";
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) return;
  applyRenderSettings(await response.json(), syncCamera);
}
async function updateRenderSettings() {
  try {
    await fetchRenderSettings(renderSettingsParams(), true);
  } catch (err) {
    document.getElementById("err").textContent = String(err);
  }
}
function scheduleRenderSettingsUpdate() {
  renderSettingsDirty = true;
  if (renderSettingsInFlight) return;
  if (renderSettingsTimer !== null) return;
  renderSettingsTimer = window.setTimeout(() => {
    renderSettingsTimer = null;
    flushRenderSettingsUpdate();
  }, CAMERA_SEND_INTERVAL_MS);
}
async function flushRenderSettingsUpdate() {
  if (!renderSettingsDirty || renderSettingsInFlight) return;
  renderSettingsDirty = false;
  renderSettingsInFlight = true;
  try {
    const response = await fetch(`/render_settings?${renderSettingsParams().toString()}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`render settings ${response.status}`);
  } catch (err) {
    document.getElementById("err").textContent = String(err);
  } finally {
    renderSettingsInFlight = false;
    if (renderSettingsDirty) scheduleRenderSettingsUpdate();
  }
}
visualToggle.addEventListener("change", updateRenderSettings);
collisionToggle.addEventListener("change", updateRenderSettings);
gravityAttitudeToggle.addEventListener("change", () => {
  updateAttitudeLabel();
  updateRenderSettings();
});
wheelVelocityToggle.addEventListener("change", () => {
  updateWheelModeLabel();
  resetCanvasWheelIntegrator(snapshot);
  updateRenderSettings();
});
jointToggles.forEach(input => input.addEventListener("change", updateRenderSettings));
fetchRenderSettings().catch(() => {});

async function fetchRenderInfo() {
  try {
    const response = await fetch("/render_info", { cache: "no-store" });
    if (!response.ok) return;
    renderInfo = await response.json();
  } catch (_) {
    renderInfo = {};
  }
}
setInterval(fetchRenderInfo, 1000);
fetchRenderInfo().catch(() => {});

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

viewWrap.addEventListener("pointerdown", ev => {
  if (!mjcfRenderOk) return;
  if (ev.target.closest("#viewControls")) return;
  if (ev.button !== 0) return;
  ev.preventDefault();
  mjcfPointerId = ev.pointerId;
  mjcfLastPointer = [ev.clientX, ev.clientY];
  viewWrap.classList.add("dragging");
  viewWrap.setPointerCapture(ev.pointerId);
});
viewWrap.addEventListener("pointermove", ev => {
  if (!mjcfRenderOk || mjcfPointerId !== ev.pointerId) return;
  ev.preventDefault();
  const events = ev.getCoalescedEvents ? ev.getCoalescedEvents() : [ev];
  for (const item of events) {
    const dx = item.clientX - mjcfLastPointer[0];
    const dy = item.clientY - mjcfLastPointer[1];
    mjcfCamera.azimuth = (mjcfCamera.azimuth - dx * 0.28 + 360) % 360;
    mjcfCamera.elevation = clamp(mjcfCamera.elevation + dy * 0.20, -80, 35);
    mjcfLastPointer = [item.clientX, item.clientY];
  }
  scheduleRenderSettingsUpdate();
});
function endMjcfDrag(ev) {
  if (mjcfPointerId !== ev.pointerId) return;
  ev.preventDefault();
  mjcfPointerId = null;
  viewWrap.classList.remove("dragging");
  scheduleRenderSettingsUpdate();
}
viewWrap.addEventListener("pointerup", endMjcfDrag);
viewWrap.addEventListener("pointercancel", endMjcfDrag);
viewWrap.addEventListener("wheel", ev => {
  if (!mjcfRenderOk) return;
  if (ev.target.closest("#viewControls")) return;
  ev.preventDefault();
  mjcfCamera.distance = clamp(mjcfCamera.distance * Math.exp(ev.deltaY * 0.0011), 0.45, 3.0);
  scheduleRenderSettingsUpdate();
}, { passive: false });

canvas.addEventListener("mousedown", ev => {
  if (mjcfRenderOk) return;
  mouseDown = true;
  lastMouse = [ev.clientX, ev.clientY];
});
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", ev => {
  if (mjcfRenderOk || !mouseDown) return;
  const dx = ev.clientX - lastMouse[0];
  const dy = ev.clientY - lastMouse[1];
  camYaw += dx * 0.006;
  camPitch = clamp(camPitch + dy * 0.004, -1.1, 1.1);
  lastMouse = [ev.clientX, ev.clientY];
});
canvas.addEventListener("wheel", ev => {
  if (mjcfRenderOk) return;
  ev.preventDefault();
  camScale *= Math.exp(-ev.deltaY * 0.001);
  camScale = clamp(camScale, 350, 1800);
}, { passive: false });

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function fmt(v, n=3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return Number(v).toFixed(n);
}
function arr(v, n=3) {
  if (!Array.isArray(v)) return "-";
  return "[" + v.map(x => fmt(x, n)).join(", ") + "]";
}
function ageMs(hostTimeS) {
  if (!hostTimeS) return "-";
  return fmt(Date.now() - Number(hostTimeS) * 1000.0, 1);
}

function connect() {
  const events = new EventSource("/events");
  events.onmessage = ev => {
    snapshot = JSON.parse(ev.data);
    updatePanel(snapshot);
  };
  events.onerror = () => {
    document.getElementById("err").textContent = "event stream reconnecting...";
  };
}
connect();

function updatePanel(s) {
  document.getElementById("err").textContent = s.error || "";
  document.getElementById("hudSeq").textContent = s.seq ?? "-";
  document.getElementById("hudHz").textContent = fmt(s.frame_hz, 1);
  document.getElementById("hudSource").textContent = s.source ?? "-";
  document.getElementById("hudOutput").textContent = s.output_enabled ? "on" : "off";

  const latency = s.latency || {};
  setGrid("commGrid", [
    ["connected", String(!!s.connected)],
    ["source", s.source],
    ["port", s.port],
    ["state_hz", fmt(s.frame_hz, 2)],
    ["state_seq", s.seq],
    ["state_age_ms", ageMs(s.host_time_s)],
    ["target_seq", s.target_seq],
    ["target_valid", s.target_valid],
    ["target_age_ms", s.target_age_ms],
    ["rx_to_output_ms", latency.rx_to_output_ms ?? s.rx_to_output_ms],
    ["latency_policy_seq", latency.policy_seq ?? s.latency_policy_seq],
    ["latency_age_ms", ageMs(latency.host_time_s)],
    ["latency_output", latency.output_enabled ?? s.latency_output_enabled],
    ["remote_url", s.remote_url],
  ]);

  setGrid("statusGrid", [
    ["model_kind", renderInfo.model_kind],
    ["closure_error_m", closureErrorText(renderInfo.closure_error_m)],
    ["render_fps", fmt(renderInfo.render_fps, 1)],
    ["attitude_source", gravityAttitudeToggle.checked ? "gravity" : "gyro"],
    ["wheel_render_mode", renderInfo.wheel_render_mode || (wheelVelocityToggle.checked ? "velocity" : "position")],
    ["tick_ms", s.tick_ms],
    ["rc_switch_r", s.rc_switch_r],
    ["output_enabled", s.output_enabled],
    ["base_ang_vel", arr(s.base_ang_vel)],
    ["projected_g", arr(s.projected_gravity)],
    ["target_joint_pos", arr(s.target_joint_pos)],
    ["joint_pos_error", arr(s.joint_pos_error)],
    ["joint_active", arr(s.joint_active)],
    ["target_active", arr(s.target_active)],
    ["hip_torque", arr(s.hip_torque)],
    ["wheel_torque", arr(s.wheel_torque)],
    ["wheel_motor_torque", arr(s.wheel_motor_torque)],
  ]);
  setBars("jointBars", labels, s.joint_pos || [], Math.PI);
  setBars("jointVelBars", labels, s.joint_vel || [], 8.0);
  setBars("wheelBars", ["L pos", "R pos", "L vel", "R vel"],
    [...(s.wheel_pos || []), ...(s.wheel_vel || [])], 60.0);
  const obs = s.obs || {};
  setObsGrid("obsGrid", Object.entries(obs));
}

function closureErrorText(value) {
  if (!value || typeof value !== "object") return "-";
  const left = value.left ?? null;
  const right = value.right ?? null;
  return `L ${fmt(left, 5)} / R ${fmt(right, 5)}`;
}

function setGrid(id, rows) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  for (const [k, v] of rows) {
    const field = document.createElement("div");
    field.className = Array.isArray(v) || String(v ?? "").length > 24 ? "field wide" : "field";
    const key = document.createElement("div");
    key.className = "k";
    key.textContent = k;
    const val = document.createElement("div");
    val.className = "v";
    val.textContent = Array.isArray(v) ? arr(v) : String(v ?? "-");
    field.appendChild(key);
    field.appendChild(val);
    el.appendChild(field);
  }
}

function setObsGrid(id, entries) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  for (const [name, values] of entries) {
    const row = document.createElement("div");
    row.className = "obsRow";
    const label = document.createElement("div");
    label.className = "obsName";
    label.textContent = name;
    const valueWrap = document.createElement("div");
    valueWrap.className = "obsValues";
    const items = Array.isArray(values) ? values : [];
    for (const value of items) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = fmt(value);
      valueWrap.appendChild(chip);
    }
    if (!items.length) {
      const empty = document.createElement("span");
      empty.className = "chip";
      empty.textContent = "-";
      valueWrap.appendChild(empty);
    }
    row.appendChild(label);
    row.appendChild(valueWrap);
    el.appendChild(row);
  }
}

function setBars(id, names, values, limit) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  names.forEach((name, idx) => {
    const value = Number(values[idx] || 0);
    const row = document.createElement("div");
    row.className = "barrow";
    const label = document.createElement("div");
    label.className = "k";
    label.textContent = name;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "fill" + (value < 0 ? " neg" : "");
    fill.style.width = `${clamp(Math.abs(value) / limit, 0, 1) * 50}%`;
    bar.appendChild(fill);
    const num = document.createElement("div");
    num.className = "value";
    num.textContent = fmt(value);
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(num);
    el.appendChild(row);
  });
}

function resetCanvasWheelIntegrator(s) {
  const wp = Array.isArray(s?.wheel_pos) ? s.wheel_pos : [0, 0];
  canvasWheelPos = [Number(wp[0] || 0), Number(wp[1] || 0)];
  canvasWheelLastSeq = s?.seq ?? null;
  canvasWheelLastTickMs = s?.tick_ms ?? null;
}

function wheelAnglesForCanvas(s) {
  const measured = Array.isArray(s?.wheel_pos) ? s.wheel_pos : [0, 0];
  if (!wheelVelocityToggle.checked) {
    resetCanvasWheelIntegrator(s);
    return measured;
  }

  if (canvasWheelLastSeq === null) {
    resetCanvasWheelIntegrator(s);
    return canvasWheelPos;
  }
  if (canvasWheelLastSeq === s.seq) {
    return canvasWheelPos;
  }

  let dt = 0.02;
  if (canvasWheelLastTickMs !== null && Number(s.tick_ms) > Number(canvasWheelLastTickMs)) {
    dt = (Number(s.tick_ms) - Number(canvasWheelLastTickMs)) / 1000.0;
  } else if (canvasWheelLastSeq !== null && Number(s.seq) > Number(canvasWheelLastSeq)) {
    dt = (Number(s.seq) - Number(canvasWheelLastSeq)) * 0.02;
  }
  dt = clamp(dt, 0, 0.05);
  const wv = Array.isArray(s?.wheel_vel) ? s.wheel_vel : [0, 0];
  canvasWheelPos = [
    wrapAngle(canvasWheelPos[0] + Number(wv[0] || 0) * dt),
    wrapAngle(canvasWheelPos[1] + Number(wv[1] || 0) * dt),
  ];
  canvasWheelLastSeq = s.seq ?? null;
  canvasWheelLastTickMs = s.tick_ms ?? null;
  return canvasWheelPos;
}

function wrapAngle(v) {
  return ((Number(v || 0) + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
}

function rotX(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0], c*p[1] - s*p[2], s*p[1] + c*p[2]];
}
function rotY(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c*p[0] + s*p[2], p[1], -s*p[0] + c*p[2]];
}
function rotZ(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c*p[0] - s*p[1], s*p[0] + c*p[1], p[2]];
}
function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function transformBody(p, g) {
  g = g || [0, 0, -1];
  const pitch = Math.asin(clamp(g[0] || 0, -0.9, 0.9));
  const roll = -Math.asin(clamp(g[1] || 0, -0.9, 0.9));
  return add(rotY(rotX(p, roll), pitch), [0, 0, 0.22]);
}
function camera(p) {
  let q = rotZ(p, camYaw);
  q = rotX(q, camPitch);
  const rect = canvas.getBoundingClientRect();
  const z = q[2] + 1.3;
  const f = camScale / Math.max(0.35, z + 1.8);
  return [rect.width * 0.5 + q[0] * f, rect.height * 0.62 - q[1] * f, z];
}
function line(a, b, color, width=2) {
  const pa = camera(a), pb = camera(b);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
}
function dot(p, color, r=4) {
  const pp = camera(p);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(pp[0], pp[1], r, 0, Math.PI * 2);
  ctx.fill();
}
function text3(p, text, color) {
  const pp = camera(p);
  ctx.fillStyle = color;
  ctx.font = "12px ui-sans-serif, system-ui";
  ctx.fillText(text, pp[0] + 6, pp[1] - 6);
}
function cube(corners, color) {
  const edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]];
  edges.forEach(([i, j]) => line(corners[i], corners[j], color, 1.5));
}
function drawWheel(center, angle, color) {
  const pp = camera(center);
  const r = 22;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(pp[0], pp[1], r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pp[0], pp[1]);
  ctx.lineTo(pp[0] + Math.cos(angle) * r, pp[1] + Math.sin(angle) * r);
  ctx.stroke();
}

function draw() {
  resize();
  if (mjcfRenderOk) {
    requestAnimationFrame(draw);
    return;
  }
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#111418";
  ctx.fillRect(0, 0, rect.width, rect.height);

  for (let x = -0.6; x <= 0.6; x += 0.1) line([x, -0.35, 0], [x, 0.35, 0], "#222a33", 1);
  for (let y = -0.35; y <= 0.35; y += 0.1) line([-0.6, y, 0], [0.6, y, 0], "#222a33", 1);

  if (!snapshot || !snapshot.connected) {
    ctx.fillStyle = "#9da7b3";
    ctx.font = "16px ui-sans-serif, system-ui";
    ctx.fillText("waiting for CDC state...", 24, 36);
    requestAnimationFrame(draw);
    return;
  }

  const q = snapshot.render_joint_pos || snapshot.joint_pos || [0, 0, 0, 0];
  const wp = wheelAnglesForCanvas(snapshot);
  const g = snapshot.projected_gravity || [0, 0, -1];
  const body = [];
  for (const x of [-0.16, 0.16]) for (const y of [-0.11, 0.11]) for (const z of [-0.045, 0.045]) {
    body.push(transformBody([x, y, z], g));
  }
  cube(body, "#7aa2ff");

  const sides = [
    {name: "L", y: -0.13, front: 0, back: 1, wheel: 0, color: "#50d6ff"},
    {name: "R", y:  0.13, front: 2, back: 3, wheel: 1, color: "#ffc857"},
  ];
  for (const side of sides) {
    const frontAnchor = [-0.08, side.y, -0.045];
    const backAnchor = [0.08, side.y, -0.045];
    const frontEnd = [frontAnchor[0] + Math.sin(q[side.front]) * 0.17, side.y, frontAnchor[2] - Math.cos(q[side.front]) * 0.17];
    const backEnd = [backAnchor[0] + Math.sin(q[side.back]) * 0.17, side.y, backAnchor[2] - Math.cos(q[side.back]) * 0.17];
    const wheel = [(frontEnd[0] + backEnd[0]) * 0.5, side.y, Math.min(frontEnd[2], backEnd[2]) - 0.065];
    const a = transformBody(frontAnchor, g);
    const b = transformBody(backAnchor, g);
    const fe = transformBody(frontEnd, g);
    const be = transformBody(backEnd, g);
    const wc = transformBody(wheel, g);
    line(a, fe, side.color, 4);
    line(b, be, side.color, 4);
    line(fe, be, "#56616f", 2);
    line(fe, wc, "#56616f", 2);
    line(be, wc, "#56616f", 2);
    dot(a, "#e6edf3", 3);
    dot(b, "#e6edf3", 3);
    dot(fe, side.color, 4);
    dot(be, side.color, 4);
    drawWheel(wc, wp[side.wheel] || 0, side.color);
    text3(fe, labels[side.front], side.color);
    text3(be, labels[side.back], side.color);
  }
  const gravStart = transformBody([0, 0, 0.1], g);
  const gravEnd = add(gravStart, [g[0] * 0.18, g[1] * 0.18, g[2] * 0.18]);
  line(gravStart, gravEnd, "#ff6b6b", 3);
  text3(gravEnd, "g", "#ff6b6b");
  requestAnimationFrame(draw);
}
draw();
</script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
