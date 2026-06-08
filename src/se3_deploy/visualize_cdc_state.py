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
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import numpy as np

from se3_shared import RobotConfig

from .cdc import CdcSerial
from .observation import RecoveryObservationBuilder
from .protocol import MSG_POLICY_STATE, PolicyStateFrame, StreamParser, decode_policy_state

_ROBOT_CFG = RobotConfig()
_ZERO_ACTION = np.zeros(6, dtype=np.float32)
_DEFAULT_MJCF = Path("assets/robots/serialleg/mjcf/serialleg_fourbar_surrogate_train.xml")
_KNEE_X = -0.17993464
_KNEE_Z = 0.00489576
_CALF_X = 0.05003347
_CALF_Z = 0.04149627
_DRIVE_X = 0.04009536
_DRIVE_Z = 0.04530576
_COUPLER_LEN = float(np.hypot(-0.16999653, 0.00108627))
_CALF_LEN = float(np.hypot(_CALF_X, _CALF_Z))
_CALF_ZERO_ANGLE = float(np.arctan2(_CALF_Z, _CALF_X))


@dataclass(slots=True)
class SharedSnapshot:
    """跨 CDC 读线程和 HTTP 线程共享的最新状态。"""

    latest: dict[str, Any] = field(default_factory=dict)
    latest_state: PolicyStateFrame | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, snapshot: dict[str, Any], state: PolicyStateFrame | None = None) -> None:
        with self.lock:
            self.latest = snapshot
            self.latest_state = state

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


class VisualizerHandler(BaseHTTPRequestHandler):
    """提供网页、单帧 JSON 和 SSE 状态流。"""

    server: VisualizerServer

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
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

        last_seq: int | None = None
        while True:
            snapshot = self.server.snapshot.get()
            seq = snapshot.get("seq")
            if seq != last_seq:
                payload = json.dumps(snapshot, separators=(",", ":"))
                try:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_seq = int(seq) if isinstance(seq, int) else None
            time.sleep(0.02)


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
                        if message.msg_type != MSG_POLICY_STATE:
                            continue
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
    joint_pos += np.asarray(
        [
            0.35 * math.sin(t * 1.2),
            0.45 * math.sin(t * 1.1 + 0.7),
            0.35 * math.sin(t * 1.2 + 1.2),
            0.45 * math.sin(t * 1.1 + 1.9),
        ],
        dtype=np.float32,
    )
    joint_vel = np.asarray(
        [
            0.42 * math.cos(t * 1.2),
            0.50 * math.cos(t * 1.1 + 0.7),
            0.42 * math.cos(t * 1.2 + 1.2),
            0.50 * math.cos(t * 1.1 + 1.9),
        ],
        dtype=np.float32,
    )
    wheel_pos = np.asarray([t * 2.0, -t * 1.6], dtype=np.float32)
    wheel_vel = np.asarray([2.0, -1.6], dtype=np.float32)
    projected_gravity = (
        0.15 * math.sin(t * 0.5),
        -0.22 * math.sin(t * 0.4),
        -0.96,
    )
    return PolicyStateFrame(
        seq=seq,
        tick_ms=int(t * 1000.0),
        target_seq=0,
        target_age_ms=0,
        target_valid=0,
        rc_switch_r=0,
        output_enabled=0,
        base_ang_vel_body=(0.0, 0.0, 0.0),
        projected_gravity=projected_gravity,
        joint_pos=tuple(float(v) for v in joint_pos),
        joint_vel=tuple(float(v) for v in joint_vel),
        wheel_pos=tuple(float(v) for v in wheel_pos),
        wheel_vel=tuple(float(v) for v in wheel_vel),
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
        "joint_vel": _finite_list(state.joint_vel),
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
    ) -> None:
        self.shared = shared
        self.mjcf = mjcf
        self.width = int(width)
        self.height = int(height)
        self.period_s = 1.0 / max(float(fps), 1.0)
        self.jpeg_quality = int(np.clip(jpeg_quality, 30, 95))
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

    def _run(self) -> None:
        renderer: MujocoStateRenderer | None = None
        try:
            renderer = MujocoStateRenderer(
                mjcf=self.mjcf,
                width=self.width,
                height=self.height,
            )
            while not self.stop_event.is_set():
                started = time.monotonic()
                state = self.shared.get_state()
                if state is not None:
                    render_started = time.monotonic()
                    rgb = renderer.render_rgb(state)
                    render_ms = (time.monotonic() - render_started) * 1000.0
                    encode_started = time.monotonic()
                    frame, content_type = _encode_render_frame_rgb(rgb, self.jpeg_quality)
                    encode_ms = (time.monotonic() - encode_started) * 1000.0
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
                                "render_ms": round(render_ms, 3),
                                "encode_ms": round(encode_ms, 3),
                                "backend": renderer.backend_info(),
                            }
                        )
                        self.frame_cv.notify_all()
                elapsed = time.monotonic() - started
                self.stop_event.wait(max(self.period_s - elapsed, 0.0))
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
        self.renderer = mujoco.Renderer(
            self.model,
            height=max(int(height), 1),
            width=max(int(width), 1),
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.85
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0
        self.camera.lookat[:] = (0.0, 0.0, 0.08)
        self.joint_qpos = {
            name: self._joint_qpos_addr(name)
            for name in (
                "lf0_Joint",
                "lf1_Joint",
                "l_wheel_Joint",
                "rf0_Joint",
                "rf1_Joint",
                "r_wheel_Joint",
            )
        }
        self._backend_info: dict[str, str | None] | None = None

    def render_rgb(self, state: PolicyStateFrame) -> np.ndarray:
        with self.lock:
            self._apply_state(state)
            self.mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera)
            rgb = self.renderer.render()
            self._capture_backend_info()
            return rgb

    def render_png(self, state: PolicyStateFrame) -> bytes:
        return _encode_png_rgb(self.render_rgb(state))

    def backend_info(self) -> dict[str, str | None]:
        return dict(self._backend_info or {})

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
        joint_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            return None
        return int(self.model.jnt_qposadr[joint_id])

    def _apply_state(self, state: PolicyStateFrame) -> None:
        self.data.qpos[:] = self.model.qpos0
        if self.model.nq >= 7:
            self.data.qpos[0:3] = (
                0.0,
                0.0,
                float(getattr(_ROBOT_CFG, "default_base_height", 0.22)),
            )
            self.data.qpos[3:7] = _quat_from_projected_gravity(state.projected_gravity)

        policy_pos = np.asarray(state.joint_pos, dtype=np.float64).reshape(4)
        output_pos = _policy_to_output_pos_np(policy_pos)
        wheel_pos = np.asarray(state.wheel_pos, dtype=np.float64).reshape(2)
        values = {
            "lf0_Joint": float(output_pos[0]),
            "lf1_Joint": float(output_pos[1]),
            "l_wheel_Joint": float(wheel_pos[0]),
            "rf0_Joint": float(output_pos[2]),
            "rf1_Joint": float(output_pos[3]),
            "r_wheel_Joint": float(wheel_pos[1]),
        }
        for name, value in values.items():
            addr = self.joint_qpos.get(name)
            if addr is not None and addr < self.model.nq:
                self.data.qpos[addr] = value


def _policy_to_output_pos_np(policy_pos: np.ndarray) -> np.ndarray:
    arr = np.asarray(policy_pos, dtype=np.float64).reshape(4).copy()
    lower, upper = _ROBOT_CFG.active_rod_angle_limits
    left_alpha = np.clip(arr[0] - arr[1], float(lower), float(upper))
    right_alpha = np.clip(arr[3] - arr[2], float(lower), float(upper))
    arr[1] = _output_knee_from_active_angle_np(left_alpha, float(lower), float(upper))
    arr[3] = -_output_knee_from_active_angle_np(right_alpha, float(lower), float(upper))
    return arr


def _output_knee_from_active_angle_np(active_angle: float, lower: float, upper: float) -> float:
    alpha = float(np.clip(active_angle, lower, upper))
    beta = -alpha
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    px = cos_b * _DRIVE_X + sin_b * _DRIVE_Z
    pz = -sin_b * _DRIVE_X + cos_b * _DRIVE_Z

    dx = px - _KNEE_X
    dz = pz - _KNEE_Z
    dist = math.sqrt(max(dx * dx + dz * dz, 1.0e-12))
    ex = dx / dist
    ez = dz / dist

    along = (_CALF_LEN**2 - _COUPLER_LEN**2 + dist * dist) / (2.0 * dist)
    height = math.sqrt(max(_CALF_LEN**2 - along * along, 0.0))
    cx = _KNEE_X + along * ex - height * ez
    cz = _KNEE_Z + along * ez + height * ex

    phi = math.atan2(cz - _KNEE_Z, cx - _KNEE_X)
    return _wrap_angle(_CALF_ZERO_ANGLE - phi)


def _wrap_angle(angle: float) -> float:
    return math.remainder(float(angle), 2.0 * math.pi)


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
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    parser.add_argument("--viewer-port", type=int, default=8081, help="HTTP viewer port.")
    parser.add_argument("--synthetic", action="store_true", help="Use animated synthetic state.")
    parser.add_argument("--remote-url", default="", help="Remote CDC visualizer base URL.")
    parser.add_argument(
        "--remote-timeout-s", type=float, default=10.0, help="Remote stream timeout."
    )
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Synthetic update rate.")
    parser.add_argument("--read-timeout-s", type=float, default=0.02, help="CDC read wait.")
    parser.add_argument("--mjcf", type=Path, default=_DEFAULT_MJCF, help="MJCF used for rendering.")
    parser.add_argument("--render-width", type=int, default=480, help="MJCF render width.")
    parser.add_argument("--render-height", type=int, default=270, help="MJCF render height.")
    parser.add_argument("--render-fps", type=float, default=5.0, help="MJCF render rate.")
    parser.add_argument("--render-jpeg-quality", type=int, default=70, help="MJCF JPEG quality.")
    parser.add_argument("--no-mjcf-render", action="store_true", help="Disable MJCF rendering.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    shared = SharedSnapshot()
    stop_event = threading.Event()

    if args.remote_url:
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
    elif args.synthetic:
        reader = threading.Thread(
            target=run_synthetic_reader,
            kwargs={
                "shared": shared,
                "rate_hz": float(args.rate_hz),
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
      grid-template-columns: minmax(0, 1fr) 420px;
      height: 100vh;
    }
    #viewWrap {
      position: relative;
      min-width: 0;
      border-right: 1px solid var(--line);
      background: #07090c;
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
    #side {
      overflow: auto;
      padding: 14px;
      background: var(--panel);
    }
    h1, h2 {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0;
      font-weight: 700;
    }
    h2 {
      margin-top: 18px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: 110px minmax(0, 1fr);
      gap: 4px 10px;
      align-items: baseline;
    }
    .k { color: var(--muted); }
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
      grid-template-columns: 42px minmax(0, 1fr) 70px;
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
      margin-top: 8px;
      min-height: 18px;
    }
    @media (max-width: 900px) {
      #app { grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }
      #viewWrap { border-right: 0; border-bottom: 1px solid var(--line); }
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
        <div class="pill">render <strong id="hudRender">canvas</strong></div>
      </div>
    </section>
    <aside id="side">
      <h1>SerialLeg CDC State</h1>
      <div id="err"></div>
      <h2>Status</h2>
      <div class="grid" id="statusGrid"></div>
      <h2>Joint Pos</h2>
      <div class="rows" id="jointBars"></div>
      <h2>Joint Vel</h2>
      <div class="rows" id="jointVelBars"></div>
      <h2>Wheel</h2>
      <div class="rows" id="wheelBars"></div>
      <h2>Observation Slices</h2>
      <div class="grid" id="obsGrid"></div>
    </aside>
  </div>
<script>
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const renderImg = document.getElementById("mjcfView");
const labels = ["LF", "LB", "RF", "RB"];
let snapshot = null;
let mjcfRenderOk = false;
let mouseDown = false;
let camYaw = -0.65;
let camPitch = 0.35;
let camScale = 900;
let lastMouse = [0, 0];

renderImg.addEventListener("load", () => {
  mjcfRenderOk = true;
  renderImg.style.opacity = "1";
  canvas.style.opacity = "0";
  document.getElementById("hudRender").textContent = "mjcf";
});
renderImg.addEventListener("error", () => {
  mjcfRenderOk = false;
  renderImg.style.opacity = "0";
  canvas.style.opacity = "1";
  document.getElementById("hudRender").textContent = "canvas";
});
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

canvas.addEventListener("mousedown", ev => {
  mouseDown = true;
  lastMouse = [ev.clientX, ev.clientY];
});
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", ev => {
  if (!mouseDown) return;
  const dx = ev.clientX - lastMouse[0];
  const dy = ev.clientY - lastMouse[1];
  camYaw += dx * 0.006;
  camPitch = clamp(camPitch + dy * 0.004, -1.1, 1.1);
  lastMouse = [ev.clientX, ev.clientY];
});
canvas.addEventListener("wheel", ev => {
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

  setGrid("statusGrid", [
    ["connected", String(!!s.connected)],
    ["port", s.port],
    ["tick_ms", s.tick_ms],
    ["target_seq", s.target_seq],
    ["target_age_ms", s.target_age_ms],
    ["target_valid", s.target_valid],
    ["rc_switch_r", s.rc_switch_r],
    ["base_ang_vel", arr(s.base_ang_vel)],
    ["projected_g", arr(s.projected_gravity)],
  ]);
  setBars("jointBars", labels, s.joint_pos || [], Math.PI);
  setBars("jointVelBars", labels, s.joint_vel || [], 8.0);
  setBars("wheelBars", ["L pos", "R pos", "L vel", "R vel"],
    [...(s.wheel_pos || []), ...(s.wheel_vel || [])], 60.0);
  const obs = s.obs || {};
  setGrid("obsGrid", Object.entries(obs).map(([k, v]) => [k, arr(v)]));
}

function setGrid(id, rows) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  for (const [k, v] of rows) {
    const key = document.createElement("div");
    key.className = "k";
    key.textContent = k;
    const val = document.createElement("div");
    val.className = "v";
    val.textContent = Array.isArray(v) ? arr(v) : String(v ?? "-");
    el.appendChild(key);
    el.appendChild(val);
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

  const q = snapshot.joint_pos || [0, 0, 0, 0];
  const wp = snapshot.wheel_pos || [0, 0];
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
