"""启动 CTBC profile 编辑器和本地文件 API。

前端形态参考 GMR 的网页编辑器：一个 Python 命令同时提供静态页面和同源 API，
日常调参不需要额外启动 Node.js dev server。
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import mimetypes
import os
import posixpath
import subprocess
import sys
import threading
import webbrowser
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import mujoco
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WEB_ROOT = REPO_ROOT / "web"
PROFILE_ROOT = REPO_ROOT / "assets" / "ctbc_profiles"
RERUN_ROOT = REPO_ROOT / "logs" / "rerun" / "stair_ctbc"
CLOSEDCHAIN_MJCF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_closed_chain_v3_train_obb_trim.xml"
)
DEFAULT_TRIGGER_MODE = "pitch"
DEFAULT_TRIGGER_FORCE_N = 10.0
DEFAULT_CONTACT_WINDOW = 3
DEFAULT_PITCH_THRESHOLD_DEG = 6.0
DEFAULT_PITCH_WINDOW = 3


def _checkpoint_iter(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("model_"):
        return -1
    try:
        return int(stem.removeprefix("model_"))
    except ValueError:
        return -1


def _default_checkpoint_path() -> Path:
    candidates: list[Path] = []
    remote_candidates = [
        path
        for path in (REPO_ROOT / "logs" / "remote_watch").glob("**/model_*.pt")
        if path.is_file() and path.stat().st_size > 0
    ]
    if remote_candidates:
        candidates.append(
            max(
                remote_candidates,
                key=lambda path: (
                    path.stat().st_mtime,
                    _checkpoint_iter(path),
                ),
            )
        )
    candidates.extend(
        [
            REPO_ROOT / "assets" / "base_model" / "stair_base.pt",
            REPO_ROOT / "assets" / "base_model" / "rough_base.pt",
            REPO_ROOT / "assets" / "base_model" / "recovery-flat.pt",
        ]
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "no usable checkpoint found; expected logs/remote_watch/**/model_*.pt "
        "or assets/base_model/{stair_base,rough_base,recovery-flat}.pt"
    )


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_stair_ctbc_profile import _sample_polar_duration_profile  # noqa: E402

from se3_shared import (  # noqa: E402
    output_leg_wheel_xz_np,
    output_to_policy_pos_np,
    policy_to_output_pos_np,
    wheel_xz_to_output_pos_np,
)
from se3_shared.height_default import policy_default_from_height_np  # noqa: E402
from se3_sim2sim.closed_chain import ClosedChainClosureSolver  # noqa: E402

_HIP_FEEDFORWARD_RATIO = 1.2
_KNEE_FEEDFORWARD_RATIO = 2.0
_CTBC_SOURCE_OUTPUT_LEG_SCALE = 0.25
_CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN = np.asarray((-1.0, -1.0, 1.0, 1.0), dtype=np.float64)

_render_lock = threading.Lock()
_render_model: mujoco.MjModel | None = None
_render_data: mujoco.MjData | None = None
_render_closure: ClosedChainClosureSolver | None = None
_render_scene_option: mujoco.MjvOption | None = None


def _fixed_base_mjcf_xml() -> str:
    tree = ET.parse(CLOSEDCHAIN_MJCF)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = (CLOSEDCHAIN_MJCF.parent / "../meshes").resolve()
        compiler.set("meshdir", meshdir.as_posix())

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("closed-chain MJCF missing worldbody")
    base_body = worldbody.find("body[@name='base_link']")
    if base_body is None:
        raise ValueError("closed-chain MJCF missing base_link body")
    for child in list(base_body):
        if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
            base_body.remove(child)
    base_body.set("pos", "0 0 0.22")
    base_body.set("quat", "1 0 0 0")
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    return ET.tostring(root, encoding="unicode")


def _json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    _json_response(handler, {"ok": False, "error": message}, status=status)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _safe_profile_path(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("profile path is required")
    path = Path(raw)
    if not path.is_absolute():
        parts = path.parts
        if len(parts) >= 3 and parts[0] == "assets" and parts[1] == "ctbc_profiles":
            path = REPO_ROOT / path
        else:
            path = PROFILE_ROOT / path
    resolved = path.resolve()
    root = PROFILE_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"profile path must stay under {PROFILE_ROOT}")
    if resolved.suffix.lower() != ".json":
        raise ValueError("profile path must end with .json")
    return resolved


def _relative_profile_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _profile_metadata_for_commands(path: str) -> dict[str, object]:
    try:
        profile_path = _safe_profile_path(path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _sim2sim_ctbc_flags(path: str) -> str:
    metadata = _profile_metadata_for_commands(path)
    return _sim2sim_ctbc_flags_from_metadata(metadata)


def _sim2sim_ctbc_flags_from_metadata(metadata: dict[str, object]) -> str:
    if metadata.get("type") not in {
        "training_duration_polar",
        "training_duration",
        "training_cosine",
    }:
        return ""
    coordinate_mode = str(metadata.get("coordinate_mode", "body_cartesian"))
    flag_map: tuple[tuple[str, str], ...]
    if coordinate_mode == "body_polar":
        flag_map = (
            ("ff_amplitude_rad", "--stair-ctbc-ff-amplitude-rad"),
            ("duration_s", "--stair-ctbc-duration-s"),
            ("leg_length_m", "--stair-ctbc-leg-length-m"),
            ("swing_angle_deg", "--stair-ctbc-swing-angle-deg"),
            ("ff_wheel_action", "--stair-ctbc-ff-wheel-action"),
        )
    else:
        flag_map = (
            ("ff_amplitude_rad", "--stair-ctbc-ff-amplitude-rad"),
            ("duration_s", "--stair-ctbc-duration-s"),
            ("body_x_m", "--stair-ctbc-body-x-m"),
            ("body_z_m", "--stair-ctbc-body-z-m"),
            ("ff_wheel_action", "--stair-ctbc-ff-wheel-action"),
        )
    flags: list[str] = []
    for key, flag in flag_map:
        value = metadata.get(key)
        if value is None and key == "body_x_m":
            value = metadata.get("ff_x_m")
        if value is None and key == "body_z_m":
            value = metadata.get("ff_lift_m")
        if isinstance(value, int | float):
            flags.extend((flag, f"{float(value):.6g}"))
    return "" if not flags else " " + " ".join(flags)


def _trigger_config_from_metadata(metadata: dict[str, object]) -> dict[str, object]:
    mode = str(metadata.get("trigger_mode", DEFAULT_TRIGGER_MODE))
    if mode not in {"force", "pitch"}:
        raise ValueError(f"trigger_mode must be force or pitch, got {mode!r}")
    force_raw = metadata.get(
        "force_threshold_n",
        metadata.get("force_threshold", DEFAULT_TRIGGER_FORCE_N),
    )
    window_raw = metadata.get("contact_window", DEFAULT_CONTACT_WINDOW)
    pitch_threshold_raw = metadata.get("pitch_threshold_deg")
    if pitch_threshold_raw is None and "pitch_threshold_rad" in metadata:
        pitch_threshold_raw = math.degrees(float(metadata["pitch_threshold_rad"]))
    if pitch_threshold_raw is None:
        pitch_threshold_raw = DEFAULT_PITCH_THRESHOLD_DEG
    pitch_window_raw = metadata.get(
        "pitch_window",
        metadata.get("trigger_window", metadata.get("contact_window", DEFAULT_PITCH_WINDOW)),
    )
    force_threshold_n = float(force_raw)
    contact_window = int(window_raw)
    pitch_threshold_deg = float(pitch_threshold_raw)
    pitch_window = int(pitch_window_raw)
    if force_threshold_n < 0.0:
        raise ValueError("force_threshold_n must be non-negative")
    if contact_window < 1:
        raise ValueError("contact_window must be at least 1")
    if pitch_threshold_deg < 0.0:
        raise ValueError("pitch_threshold_deg must be non-negative")
    if pitch_window < 1:
        raise ValueError("pitch_window must be at least 1")
    return {
        "trigger_mode": mode,
        "force_threshold_n": force_threshold_n,
        "contact_window": contact_window,
        "pitch_threshold_deg": pitch_threshold_deg,
        "pitch_window": pitch_window,
    }


def _sim2sim_trigger_flags(trigger: dict[str, object]) -> str:
    mode = str(trigger["trigger_mode"])
    flags = ["--stair-ctbc-trigger-mode", mode]
    if mode == "pitch":
        flags.extend(
            [
                "--stair-ctbc-pitch-threshold-deg",
                f"{float(trigger['pitch_threshold_deg']):.6g}",
                "--stair-ctbc-pitch-window",
                str(int(trigger["pitch_window"])),
            ]
        )
    else:
        flags.extend(
            [
                "--stair-ctbc-force-threshold-n",
                f"{float(trigger['force_threshold_n']):.6g}",
                "--stair-ctbc-contact-window",
                str(int(trigger["contact_window"])),
            ]
        )
    return " ".join(flags)


def _validate_profile(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("profile must be a JSON object")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("profile.points must contain at least two entries")
    metadata = payload.get("metadata")
    metadata_out = dict(metadata) if isinstance(metadata, dict) else {}
    coordinate_mode = str(metadata_out.get("coordinate_mode", "body_cartesian"))
    if metadata_out.get("type") == "training_duration_polar":
        coordinate_mode = "body_polar"
    rows: list[dict[str, float]] = []
    previous_t = -1.0
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"profile point {idx} must be an object")
        t = float(point.get("t", point.get("time_s")))
        if t <= previous_t:
            raise ValueError("profile point times must be strictly increasing")
        previous_t = t
        row = {
            "t": t,
            "amp": float(point.get("amp", point.get("amp_scale", 0.0))),
            "wheel_action": float(point.get("wheel_action", point.get("wheel_action_delta", 0.0))),
        }
        if coordinate_mode == "body_polar":
            leg_length = float(point.get("leg_length_m", point.get("length_m", 0.24)))
            swing_rad = point.get("swing_angle_rad", point.get("theta_rad"))
            if swing_rad is None:
                swing_rad = math.radians(
                    float(point.get("swing_angle_deg", point.get("theta_deg", 10.0)))
                )
            else:
                swing_rad = float(swing_rad)
            x_m = float(leg_length * math.sin(swing_rad))
            z_m = float(-leg_length * math.cos(swing_rad))
            row.update(
                {
                    "leg_length_m": leg_length,
                    "swing_angle_rad": swing_rad,
                    "swing_angle_deg": math.degrees(swing_rad),
                    "body_x_m": x_m,
                    "body_z_m": z_m,
                    "x_m": x_m,
                    "z_m": z_m,
                }
            )
        else:
            x_m = float(point.get("body_x_m", point.get("x_m", point.get("x", 0.0))))
            z_m = float(point.get("body_z_m", point.get("z_m", point.get("z", 0.0))))
            row.update(
                {
                    "body_x_m": x_m,
                    "body_z_m": z_m,
                    "x_m": x_m,
                    "z_m": z_m,
                }
            )
        rows.append(row)
    period_s = float(payload.get("period_s", rows[-1]["t"]))
    if period_s <= 0.0:
        raise ValueError("profile.period_s must be positive")
    if period_s < rows[-1]["t"]:
        period_s = rows[-1]["t"]
    out = dict(payload)
    out["version"] = int(out.get("version", 1))
    out["period_s"] = period_s
    out["metadata"] = metadata_out
    out["metadata"]["cartesian_frame"] = "body"
    out["metadata"]["coordinate_mode"] = coordinate_mode
    trigger = _trigger_config_from_metadata(out["metadata"])
    out["metadata"]["trigger_mode"] = trigger["trigger_mode"]
    out["metadata"]["force_threshold_n"] = trigger["force_threshold_n"]
    out["metadata"]["contact_window"] = trigger["contact_window"]
    out["metadata"]["pitch_threshold_deg"] = trigger["pitch_threshold_deg"]
    out["metadata"]["pitch_threshold_rad"] = math.radians(float(trigger["pitch_threshold_deg"]))
    out["metadata"]["pitch_window"] = trigger["pitch_window"]
    out["metadata"].setdefault("duration_s", period_s)
    out["points"] = rows
    return out


def _command_for_profile(path: str) -> dict[str, str]:
    quoted = path.replace("\\", "/")
    metadata = _profile_metadata_for_commands(path)
    trigger = _trigger_config_from_metadata(metadata)
    trigger_flags = _sim2sim_trigger_flags(trigger)
    sim2sim_ctbc_flags = _sim2sim_ctbc_flags_from_metadata(metadata)
    checkpoint = _relative_profile_path(_default_checkpoint_path())
    sim2sim = (
        "uv run se3-sim2sim --model-variant closedchain "
        f"--checkpoint {checkpoint} "
        "--stair-terrain --stair-step-height 0.12 --stair-half-width 6.0 "
        "--stair-ctbc --stair-ctbc-iter 0 "
        f"{trigger_flags} "
        f"--stair-ctbc-profile {quoted}{sim2sim_ctbc_flags} "
        "--command 1.0 0 0 0 0.39 0 0 0 --yaw-pid "
        "--viewer rerun --rerun-record logs/rerun/stair_ctbc/profile_web.rrd "
        "--max-steps 500 --course none"
    )
    record = (
        "uv run python scripts/record_stair_ctbc_rerun.py "
        f"{checkpoint} "
        "--command-vx 1.0 --command-height 0.39 --terrain-level 0 "
        f"--trigger-mode {trigger['trigger_mode']} "
        f"--force-threshold-n {float(trigger['force_threshold_n']):.6g} "
        f"--contact-window {int(trigger['contact_window'])} "
        f"--pitch-threshold-deg {float(trigger['pitch_threshold_deg']):.6g} "
        f"--pitch-window {int(trigger['pitch_window'])} "
        f"--ctbc-profile {quoted} "
        "--seconds 10 --output logs/rerun/stair_ctbc/profile_web_mjlab.rrd"
    )
    sweep = (
        "uv run python scripts/sweep_stair_ctbc_sim2sim.py "
        f"--checkpoint {checkpoint} "
        "--stair-step-heights 0.12 --levels 0 "
        f"--profiles {quoted} "
        "--command-vxs 0.8,1.0 --command-heights 0.37,0.39 "
        f"--trigger-mode {trigger['trigger_mode']} "
        f"--force-threshold {float(trigger['force_threshold_n']):.6g} "
        f"--contact-window {int(trigger['contact_window'])} "
        f"--pitch-threshold-deg {float(trigger['pitch_threshold_deg']):.6g} "
        f"--pitch-window {int(trigger['pitch_window'])} "
        "--max-steps 500"
    )
    return {"sim2sim": sim2sim, "record": record, "sweep": sweep}


def _start_record_rerun(
    profile: dict[str, object],
    *,
    command_vx: float,
    command_height: float,
    stair_step_height: float,
    max_steps: int,
) -> dict[str, object]:
    if not (0.0 < stair_step_height <= 0.5):
        raise ValueError("stair_step_height must be in (0, 0.5]")
    if not (0.1 <= command_height <= 0.8):
        raise ValueError("command_height must be in [0.1, 0.8]")
    if not (-3.0 <= command_vx <= 3.0):
        raise ValueError("command_vx must be in [-3, 3]")
    max_steps = max(50, min(int(max_steps), 5000))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    profile_path = RERUN_ROOT / f"web_body_polar_{timestamp}.json"
    record_path = RERUN_ROOT / f"web_stair_{timestamp}.rrd"
    log_path = RERUN_ROOT / f"web_stair_{timestamp}.log"
    profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = _default_checkpoint_path()
    rel_checkpoint = _relative_profile_path(checkpoint_path)
    rel_profile = _relative_profile_path(profile_path)
    metadata = profile.get("metadata")
    trigger = _trigger_config_from_metadata(metadata if isinstance(metadata, dict) else {})
    trigger_flags = _sim2sim_trigger_flags(trigger).split()
    ctbc_flags = (
        _sim2sim_ctbc_flags_from_metadata(metadata if isinstance(metadata, dict) else {})
        .strip()
        .split()
    )
    cmd = [
        "uv",
        "run",
        "se3-sim2sim",
        "--model-variant",
        "closedchain",
        "--checkpoint",
        rel_checkpoint,
        "--stair-terrain",
        "--stair-step-height",
        f"{float(stair_step_height):.6g}",
        "--stair-half-width",
        "6.0",
        "--stair-ctbc",
        "--stair-ctbc-iter",
        "0",
        *trigger_flags,
        "--stair-ctbc-profile",
        rel_profile,
        *ctbc_flags,
        "--command",
        f"{float(command_vx):.6g}",
        "0",
        "0",
        "0",
        f"{float(command_height):.6g}",
        "0",
        "0",
        "0",
        "--yaw-pid",
        "--viewer",
        "rerun",
        "--rerun-record",
        _relative_profile_path(record_path),
        "--max-steps",
        str(max_steps),
        "--course",
        "none",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_file = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        log_file.close()
        raise
    log_file.close()
    return {
        "pid": proc.pid,
        "command": " ".join(cmd),
        "checkpoint": rel_checkpoint,
        "profile_path": _relative_profile_path(profile_path),
        "rerun_record": _relative_profile_path(record_path),
        "log_path": _relative_profile_path(log_path),
        "trigger_mode": trigger["trigger_mode"],
        "force_threshold_n": trigger["force_threshold_n"],
        "contact_window": trigger["contact_window"],
        "pitch_threshold_deg": trigger["pitch_threshold_deg"],
        "pitch_window": trigger["pitch_window"],
    }


def _sample_profile(profile: dict[str, object], phase_s: float) -> np.ndarray:
    points = profile["points"]
    if not isinstance(points, list):
        raise ValueError("profile points missing")
    metadata = profile.get("metadata")
    coordinate_mode = (
        str(metadata.get("coordinate_mode", "body_cartesian"))
        if isinstance(metadata, dict)
        else "body_cartesian"
    )
    times = np.asarray([float(point["t"]) for point in points], dtype=np.float64)
    if coordinate_mode == "body_polar":
        values = np.asarray(
            [
                [
                    float(point["leg_length_m"]),
                    float(point["swing_angle_rad"]),
                    float(point["amp"]),
                    float(point["wheel_action"]),
                ]
                for point in points
            ],
            dtype=np.float64,
        )
    else:
        values = np.asarray(
            [
                [
                    float(point["x_m"]),
                    float(point["z_m"]),
                    float(point["amp"]),
                    float(point["wheel_action"]),
                ]
                for point in points
            ],
            dtype=np.float64,
        )
    t = float(np.clip(phase_s, 0.0, float(profile["period_s"])))
    return np.asarray([np.interp(t, times, values[:, col]) for col in range(4)])


def _render_state_from_profile(
    profile: dict[str, object],
    *,
    phase_s: float,
    command_height: float,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    default_policy = policy_default_from_height_np(float(command_height)).reshape(4)
    default_output = policy_to_output_pos_np(default_policy[None, :])[0]
    default_wheel_xz = output_leg_wheel_xz_np(default_output)
    sample = _sample_profile(profile, phase_s)
    wheel_delta = np.zeros((2, 2), dtype=np.float64)
    desired_wheel_xz = default_wheel_xz.copy()
    active_sides = (0, 1) if side == "both" else (0,) if side == "left" else (1,)
    metadata = profile.get("metadata")
    amp_rad = 0.0
    coordinate_mode = "body_cartesian"
    if isinstance(metadata, dict):
        amp_rad = float(metadata.get("ff_amplitude_rad", 0.0))
        coordinate_mode = str(metadata.get("coordinate_mode", coordinate_mode))
    output_bias = np.zeros(4, dtype=np.float64)
    for side_id in active_sides:
        envelope = float(sample[2])
        feedforward = 2.0 * amp_rad * envelope
        hip_idx = 0 if side_id == 0 else 2
        knee_idx = 1 if side_id == 0 else 3
        output_bias[hip_idx] = -feedforward * _HIP_FEEDFORWARD_RATIO
        output_bias[knee_idx] = feedforward * _KNEE_FEEDFORWARD_RATIO
        if coordinate_mode == "body_polar":
            desired_wheel_xz[side_id, 0] = float(sample[0] * math.sin(sample[1]))
            desired_wheel_xz[side_id, 1] = float(-sample[0] * math.cos(sample[1]))
        else:
            wheel_delta[side_id, 0] = float(sample[0])
            wheel_delta[side_id, 1] = float(sample[1])
    if np.any(np.abs(output_bias) > 0.0):
        output_delta = (
            output_bias * _CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN * _CTBC_SOURCE_OUTPUT_LEG_SCALE
        )
        nominal_requested_output = default_output + output_delta
        nominal_realizable_output = policy_to_output_pos_np(
            output_to_policy_pos_np(nominal_requested_output[None, :])
        )[0]
        bias_wheel_delta = output_leg_wheel_xz_np(nominal_realizable_output) - default_wheel_xz
        bias_wheel_delta[..., 0] = np.minimum(bias_wheel_delta[..., 0], 0.0)
        bias_wheel_delta[..., 1] = np.maximum(bias_wheel_delta[..., 1], 0.0)
        if coordinate_mode == "body_polar":
            desired_wheel_xz += bias_wheel_delta
        else:
            wheel_delta += bias_wheel_delta
    if coordinate_mode != "body_polar":
        desired_wheel_xz = default_wheel_xz + wheel_delta
    output = wheel_xz_to_output_pos_np(desired_wheel_xz)
    return output.reshape(4), default_wheel_xz, desired_wheel_xz


def _get_render_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    global _render_closure, _render_data, _render_model
    if _render_model is None or _render_data is None:
        _render_model = mujoco.MjModel.from_xml_string(_fixed_base_mjcf_xml())
        non_visual = _render_model.geom_group != 1
        _render_model.geom_rgba[non_visual, :] = 0.0
        _render_model.vis.global_.offwidth = max(_render_model.vis.global_.offwidth, 2048)
        _render_model.vis.global_.offheight = max(_render_model.vis.global_.offheight, 2048)
        _render_data = mujoco.MjData(_render_model)
        _render_closure = ClosedChainClosureSolver.try_create(
            model=_render_model,
            data=_render_data,
        )
    return _render_model, _render_data


def _visual_mesh_scene_option() -> mujoco.MjvOption:
    global _render_scene_option
    if _render_scene_option is None:
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        option.tendongroup[:] = 0
        option.jointgroup[:] = 0
        _render_scene_option = option
    return _render_scene_option


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"joint not found in preview model: {name}")
    return int(model.jnt_qposadr[joint_id])


def _set_preview_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    output: np.ndarray,
    command_height: float,
) -> None:
    if _render_closure is None:
        raise ValueError("closed-chain preview model did not create a closure solver")
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise ValueError("base_link body not found in preview model")
    model.body_pos[base_id] = np.asarray([0.0, 0.0, float(command_height)], dtype=np.float64)
    model.body_quat[base_id] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    policy = output_to_policy_pos_np(output[None, :])[0]
    for name, value in zip(
        ("lf0_Joint", "l_drive_bar_Joint", "rf0_Joint", "r_drive_bar_Joint"),
        policy,
        strict=True,
    ):
        data.qpos[_joint_qpos_address(model, name)] = float(value)
    _render_closure.seed_passive_position(0, float(output[1]))
    _render_closure.seed_passive_position(1, float(output[3]))
    _render_closure.solve_positions()
    for name in ("l_wheel_Joint", "r_wheel_Joint"):
        data.qpos[_joint_qpos_address(model, name)] = 0.0
    mujoco.mj_forward(model, data)


def _render_pose_png(
    profile: dict[str, object],
    *,
    phase_s: float,
    command_height: float,
    side: str,
    width: int,
    height: int,
) -> bytes:
    output, _, _ = _render_state_from_profile(
        profile,
        phase_s=phase_s,
        command_height=command_height,
        side=side,
    )
    width = max(240, min(int(width), 1600))
    height = max(180, min(int(height), 1200))
    with _render_lock:
        model, data = _get_render_model()
        _set_preview_pose(model, data, output=output, command_height=command_height)
        renderer = mujoco.Renderer(model, width=width, height=height)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = np.asarray([-0.05, 0.0, 0.16], dtype=np.float64)
        cam.distance = 1.05
        cam.azimuth = 90.0
        cam.elevation = -18.0
        try:
            renderer.update_scene(data, camera=cam, scene_option=_visual_mesh_scene_option())
            rgb = renderer.render()
        finally:
            renderer.close()
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class CtbcProfileHandler(BaseHTTPRequestHandler):
    server_version = "CtbcProfileWeb/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[ctbc-web] " + fmt % args + "\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            _json_response(self, {"ok": True})
            return
        if parsed.path == "/api/profiles":
            self._profiles()
            return
        if parsed.path == "/api/profile":
            self._profile(parsed.query)
            return
        if parsed.path == "/api/commands":
            self._commands(parsed.query)
            return
        self._static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/profile":
                self._save_profile()
                return
            if parsed.path == "/api/generate":
                self._generate_profile()
                return
            if parsed.path == "/api/render_pose":
                self._render_pose()
                return
            if parsed.path == "/api/record_rerun":
                self._record_rerun()
                return
        except (ValueError, FileNotFoundError) as exc:
            _error(self, 400, str(exc))
            return
        _error(self, 404, "unknown endpoint")

    def _profiles(self) -> None:
        PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
        profiles = [
            _relative_profile_path(path)
            for path in sorted(PROFILE_ROOT.glob("*.json"), key=lambda item: item.name)
        ]
        _json_response(self, {"ok": True, "profiles": profiles})

    def _profile(self, query: str) -> None:
        params = parse_qs(query)
        try:
            path = _safe_profile_path(params.get("path", [""])[0])
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile = _validate_profile(payload)
        except FileNotFoundError:
            _error(self, 404, "profile not found")
            return
        except (ValueError, json.JSONDecodeError) as exc:
            _error(self, 400, str(exc))
            return
        rel = _relative_profile_path(path)
        _json_response(
            self,
            {
                "ok": True,
                "path": rel,
                "profile": profile,
                "commands": _command_for_profile(rel),
            },
        )

    def _commands(self, query: str) -> None:
        params = parse_qs(query)
        try:
            path = _relative_profile_path(_safe_profile_path(params.get("path", [""])[0]))
        except ValueError as exc:
            _error(self, 400, str(exc))
            return
        _json_response(self, {"ok": True, "commands": _command_for_profile(path)})

    def _save_profile(self) -> None:
        payload = _read_json_body(self)
        path = _safe_profile_path(payload.get("path"))
        profile = _validate_profile(payload.get("profile"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rel = _relative_profile_path(path)
        _json_response(
            self,
            {"ok": True, "path": rel, "profile": profile, "commands": _command_for_profile(rel)},
        )

    def _generate_profile(self) -> None:
        payload = _read_json_body(self)
        duration_s = float(payload.get("duration_s", payload.get("period_s", 0.60)))
        control_dt = float(payload.get("control_dt", 0.02))
        ff_amplitude_rad = float(payload.get("ff_amplitude_rad", 0.0))
        leg_length_m = float(payload.get("leg_length_m", 0.24))
        swing_angle_deg = float(payload.get("swing_angle_deg", 10.0))
        swing_angle_rad = float(payload.get("swing_angle_rad", math.radians(swing_angle_deg)))
        ff_wheel_action = float(payload.get("ff_wheel_action", 0.0))
        trigger_mode = str(payload.get("trigger_mode", DEFAULT_TRIGGER_MODE))
        force_threshold_n = float(payload.get("force_threshold_n", DEFAULT_TRIGGER_FORCE_N))
        contact_window = int(
            payload.get("contact_window", payload.get("trigger_window", DEFAULT_CONTACT_WINDOW))
        )
        pitch_threshold_deg = float(payload.get("pitch_threshold_deg", DEFAULT_PITCH_THRESHOLD_DEG))
        pitch_window = int(
            payload.get(
                "pitch_window",
                payload.get("trigger_window", payload.get("contact_window", DEFAULT_PITCH_WINDOW)),
            )
        )
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if leg_length_m <= 0.0:
            raise ValueError("leg_length_m must be positive")
        if trigger_mode not in {"force", "pitch"}:
            raise ValueError("trigger_mode must be force or pitch")
        if force_threshold_n < 0.0:
            raise ValueError("force_threshold_n must be non-negative")
        if contact_window < 1:
            raise ValueError("contact_window must be at least 1")
        if pitch_threshold_deg < 0.0:
            raise ValueError("pitch_threshold_deg must be non-negative")
        if pitch_window < 1:
            raise ValueError("pitch_window must be at least 1")
        profile = {
            "version": 1,
            "generator": "scripts/ctbc_profile_web.py",
            "period_s": duration_s,
            "metadata": {
                "type": "training_duration_polar",
                "cartesian_frame": "body",
                "coordinate_mode": "body_polar",
                "duration_s": duration_s,
                "control_dt": control_dt,
                "ff_amplitude_rad": ff_amplitude_rad,
                "leg_length_m": leg_length_m,
                "swing_angle_rad": swing_angle_rad,
                "swing_angle_deg": math.degrees(swing_angle_rad),
                "ff_wheel_action": ff_wheel_action,
                "trigger_mode": trigger_mode,
                "force_threshold_n": force_threshold_n,
                "contact_window": contact_window,
                "pitch_threshold_deg": pitch_threshold_deg,
                "pitch_threshold_rad": math.radians(pitch_threshold_deg),
                "pitch_window": pitch_window,
            },
            "points": _sample_polar_duration_profile(
                duration_s=duration_s,
                control_dt=control_dt,
                leg_length_m=leg_length_m,
                swing_angle_rad=swing_angle_rad,
                ff_wheel_action=ff_wheel_action,
            ),
        }
        _json_response(self, {"ok": True, "profile": _validate_profile(profile)})

    def _render_pose(self) -> None:
        payload = _read_json_body(self)
        profile = _validate_profile(payload.get("profile"))
        phase_s = float(payload.get("phase_s", 0.0))
        command_height = float(payload.get("command_height", 0.30))
        side = str(payload.get("side", "both"))
        if side not in {"left", "right", "both"}:
            raise ValueError("side must be left, right, or both")
        png = _render_pose_png(
            profile,
            phase_s=phase_s,
            command_height=command_height,
            side=side,
            width=int(payload.get("width", 900)),
            height=int(payload.get("height", 520)),
        )
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.end_headers()
        self.wfile.write(png)

    def _record_rerun(self) -> None:
        payload = _read_json_body(self)
        profile = _validate_profile(payload.get("profile"))
        result = _start_record_rerun(
            profile,
            command_vx=float(payload.get("command_vx", 1.0)),
            command_height=float(payload.get("command_height", 0.39)),
            stair_step_height=float(payload.get("stair_step_height", 0.12)),
            max_steps=int(payload.get("max_steps", 500)),
        )
        _json_response(self, {"ok": True, **result})

    def _static(self, raw_path: str) -> None:
        clean_path = posixpath.normpath(unquote(raw_path)).lstrip("/")
        if clean_path in ("", "."):
            clean_path = "ctbc_profile_editor.html"
        path = (WEB_ROOT / clean_path).resolve()
        if not path.is_relative_to(WEB_ROOT.resolve()) or not path.is_file():
            _error(self, 404, "not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in (".html", ".css", ".js"):
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("CTBC_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CTBC_WEB_PORT", "8020")))
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    server = ThreadingHTTPServer((args.host, int(args.port)), CtbcProfileHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"CTBC profile editor: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
