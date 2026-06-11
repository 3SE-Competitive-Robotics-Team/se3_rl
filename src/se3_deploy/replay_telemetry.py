"""Replay NX recovery telemetry locally and optionally write a Rerun recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from se3_shared import ObservationConfig, PolicyActionDecoder, RobotConfig

from .numpy_policy import NumpyPolicyRuntime
from .onnx_policy import OnnxPolicyRuntime

ACTION_LABELS = (
    "left_front",
    "left_active",
    "right_front",
    "right_active",
    "left_wheel",
    "right_wheel",
)
LEG_LABELS = ("left_front", "left_back", "right_front", "right_back")
WHEEL_LABELS = ("left_wheel", "right_wheel")
XYZ_LABELS = ("x", "y", "z")
POLICY_JOINT_NAMES = (
    "lf0_Joint",
    "l_drive_bar_Joint",
    "rf0_Joint",
    "r_drive_bar_Joint",
    "l_wheel_Joint",
    "r_wheel_Joint",
)
DEFAULT_RERUN_MODEL = Path(
    "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
)


class PolicyRuntimeLike(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class ReplayStats:
    total_jsonl_rows: int = 0
    sample_rows: int = 0
    event_rows: int = 0
    policy_rows: int = 0
    hold_rows: int = 0
    reset_count: int = 0
    action_row_max_abs_errors: list[float] = field(default_factory=list)
    target_joint_row_max_abs_errors: list[float] = field(default_factory=list)
    target_wheel_row_max_abs_errors: list[float] = field(default_factory=list)
    dt_ms: list[float] = field(default_factory=list)
    policy_inference_ms: list[float] = field(default_factory=list)

    def summary(self, *, period_ms: float) -> dict[str, object]:
        action = _stats(self.action_row_max_abs_errors)
        target_joint = _stats(self.target_joint_row_max_abs_errors)
        target_wheel = _stats(self.target_wheel_row_max_abs_errors)
        dt = _stats(self.dt_ms)
        policy_ms = _stats(self.policy_inference_ms)
        missed = sum(1 for value in self.dt_ms if value > period_ms * 1.5)
        return {
            "total_jsonl_rows": self.total_jsonl_rows,
            "sample_rows": self.sample_rows,
            "event_rows": self.event_rows,
            "policy_rows": self.policy_rows,
            "hold_rows": self.hold_rows,
            "reset_count": self.reset_count,
            "period_ms": period_ms,
            "missed_50hz_deadlines": missed,
            "action_max_abs_error": action,
            "target_joint_max_abs_error": target_joint,
            "target_wheel_max_abs_error": target_wheel,
            "dt_ms": dt,
            "policy_inference_ms": policy_ms,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay NX recovery telemetry JSONL with a local ONNX/NPZ policy."
    )
    parser.add_argument("telemetry", type=Path, help="NX recovery telemetry JSONL.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local .onnx/.npz checkpoint. If omitted, try the adjacent meta file.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Telemetry meta JSON. Defaults to <telemetry>.meta.json when present.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 means replay all sample rows.")
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--fail-action-error",
        type=float,
        default=None,
        help="Exit non-zero when action max abs error exceeds this threshold.",
    )
    parser.add_argument(
        "--rerun-record", type=Path, default=None, help="Optional output .rrd path."
    )
    parser.add_argument("--rerun-spawn", action="store_true", help="Open Rerun viewer.")
    parser.add_argument(
        "--rerun-address", default=None, help="Connect to an existing Rerun server."
    )
    parser.add_argument("--rerun-app-id", default="se3_nx_telemetry_replay")
    parser.add_argument(
        "--rerun-model",
        type=Path,
        default=DEFAULT_RERUN_MODEL,
        help="MJCF model to animate from logged joint positions.",
    )
    parser.add_argument(
        "--no-rerun-robot",
        action="store_true",
        help="Disable approximate MJCF robot animation in the Rerun 3D view.",
    )
    parser.add_argument(
        "--no-rerun-3d",
        action="store_true",
        help="Only log scalar plots, not approximate IMU vectors or robot geometry.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta = _load_meta(args.telemetry, args.meta)
    checkpoint = _resolve_checkpoint(args.checkpoint, args.telemetry, meta)
    _check_checkpoint_hash(checkpoint, meta)

    policy = _load_policy(checkpoint)
    decoder = PolicyActionDecoder(
        robot_cfg=RobotConfig(),
        height_conditioned_action_default=True,
        active_rod_semantics=True,
        dtype=np.float32,
    )
    command_height = _command_height(meta)
    rerun = _RerunSink(
        enabled=args.rerun_record is not None or args.rerun_spawn or args.rerun_address is not None,
        app_id=args.rerun_app_id,
        record_to_rrd=args.rerun_record,
        spawn=args.rerun_spawn,
        address=args.rerun_address,
        log_3d=not args.no_rerun_3d,
        model_path=None if args.no_rerun_robot or args.no_rerun_3d else args.rerun_model,
    )
    try:
        stats, period_ms = replay_rows(
            telemetry=args.telemetry,
            policy=policy,
            decoder=decoder,
            command_height=command_height,
            rerun=rerun,
            max_rows=args.max_rows,
            print_every=args.print_every,
        )
    finally:
        rerun.close()

    summary = stats.summary(period_ms=period_ms)
    _print_summary(args.telemetry, checkpoint, summary)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    fail_threshold = args.fail_action_error
    if fail_threshold is not None:
        max_error = float(summary["action_max_abs_error"]["max"])
        if math.isfinite(max_error) and max_error > float(fail_threshold):
            return 2
    return 0


def replay_rows(
    *,
    telemetry: Path,
    policy: PolicyRuntimeLike,
    decoder: PolicyActionDecoder,
    command_height: float,
    rerun: _RerunSink,
    max_rows: int,
    print_every: int,
) -> tuple[ReplayStats, float]:
    obs_cfg = ObservationConfig()
    stats = ReplayStats()
    first_time_s: float | None = None
    last_time_s: float | None = None
    last_reset_marker: int | None = None
    last_mode_policy = False
    period_ms = 20.0

    for line_no, row in _iter_jsonl(telemetry):
        stats.total_jsonl_rows += 1
        record_type = row.get("record_type", "sample" if "obs" in row else "event")
        if record_type == "event":
            stats.event_rows += 1
            rerun.log_event(row)
            continue
        if record_type != "sample" or "obs" not in row:
            continue
        if 0 < max_rows <= stats.sample_rows:
            break
        stats.sample_rows += 1

        row_period_ms = _optional_float(row.get("sample_period_ms"))
        if row_period_ms is not None and row_period_ms > 0.0:
            period_ms = row_period_ms

        time_s = _row_time_s(row)
        if first_time_s is None:
            first_time_s = time_s
        if last_time_s is not None:
            stats.dt_ms.append((time_s - last_time_s) * 1000.0)
        last_time_s = time_s

        policy_ms = _optional_float(row.get("policy_inference_ms"))
        if policy_ms is not None:
            stats.policy_inference_ms.append(policy_ms)

        target_mode = str(row.get("target_mode", "policy"))
        is_policy = target_mode == "policy"
        reset_marker = _optional_int(row.get("reset_id"))
        reset_now = False
        if reset_marker is not None and reset_marker != last_reset_marker:
            reset_now = True
            last_reset_marker = reset_marker
        elif reset_marker is None and is_policy and not last_mode_policy:
            reset_now = True
        if reset_now:
            _policy_reset(policy)
            stats.reset_count += 1

        obs = _array(row, "obs", obs_cfg.num_obs)
        logged_action = _array(row, "action", obs_cfg.num_actions)
        replay_action: np.ndarray | None = None
        replay_joint_target: np.ndarray | None = None
        replay_wheel_target: np.ndarray | None = None
        if is_policy:
            stats.policy_rows += 1
            replay_action = _policy_act(policy, obs)
            action_error = replay_action - logged_action
            stats.action_row_max_abs_errors.append(float(np.max(np.abs(action_error))))
            decoded = decoder.decode(replay_action, command_height=command_height)
            replay_joint_target = np.asarray(decoded.leg_target, dtype=np.float32).reshape(4)
            replay_wheel_target = np.asarray(decoded.wheel_vel_target, dtype=np.float32).reshape(2)
            if "nx_target_joint_pos" in row:
                logged_joint = _array(row, "nx_target_joint_pos", 4)
                stats.target_joint_row_max_abs_errors.append(
                    float(np.max(np.abs(replay_joint_target - logged_joint)))
                )
            if "nx_target_wheel_vel" in row:
                logged_wheel = _array(row, "nx_target_wheel_vel", 2)
                stats.target_wheel_row_max_abs_errors.append(
                    float(np.max(np.abs(replay_wheel_target - logged_wheel)))
                )
        else:
            stats.hold_rows += 1
            _policy_reset(policy)

        rerun.log_sample(
            row=row,
            relative_time_s=time_s - first_time_s,
            replay_action=replay_action,
            replay_joint_target=replay_joint_target,
            replay_wheel_target=replay_wheel_target,
        )
        last_mode_policy = is_policy
        if print_every > 0 and stats.sample_rows % print_every == 0:
            _print_progress(stats, line_no)

    return stats, period_ms


class _RerunSink:
    def __init__(
        self,
        *,
        enabled: bool,
        app_id: str,
        record_to_rrd: Path | None,
        spawn: bool,
        address: str | None,
        log_3d: bool,
        model_path: Path | None,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_3d = bool(log_3d)
        self.rr: Any | None = None
        self.robot_logger: _RobotKinematicsLogger | None = None
        if not self.enabled:
            return
        import rerun as rr

        self.rr = rr
        rr.init(app_id, spawn=False)
        if address:
            rr.connect_grpc(address)
        elif spawn:
            rr.spawn(connect=True, detach_process=True)
        if record_to_rrd is not None:
            record_to_rrd.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(record_to_rrd))
        if self.log_3d and model_path is not None:
            self.robot_logger = _RobotKinematicsLogger.try_create(
                rr=rr,
                app_id=app_id,
                model_path=model_path,
            )

    def log_event(self, row: dict[str, object]) -> None:
        if self.rr is None:
            return
        step = _optional_int(row.get("step")) or 0
        time_s = _row_time_s(row)
        self.rr.set_time("step", sequence=step)
        self.rr.set_time("time", duration=time_s)
        event = str(row.get("event", "event"))
        self._scalar("/events/count", 1.0)
        self.rr.log(f"/events/{event}", self.rr.TextLog(json.dumps(row, sort_keys=True)))

    def log_sample(
        self,
        *,
        row: dict[str, object],
        relative_time_s: float,
        replay_action: np.ndarray | None,
        replay_joint_target: np.ndarray | None,
        replay_wheel_target: np.ndarray | None,
    ) -> None:
        if self.rr is None:
            return
        step = int(row.get("step", 0))
        self.rr.set_time("step", sequence=step)
        self.rr.set_time("time", duration=float(relative_time_s))
        self._scalar(
            "/runtime/target_mode_policy", 1.0 if row.get("target_mode") == "policy" else 0.0
        )
        self._scalar("/runtime/output_enabled", float(row.get("output_enabled", 0)))
        self._scalar("/runtime/flags", float(row.get("flags", 0)))
        self._optional_scalar("/runtime/policy_inference_ms", row.get("policy_inference_ms"))
        self._optional_scalar("/runtime/loop_dt_ms", row.get("loop_dt_ms"))
        self._optional_scalar("/runtime/loop_work_ms", row.get("loop_work_ms"))
        self._optional_scalar("/runtime/state_age_ms_nx", row.get("state_age_ms_nx"))
        self._optional_scalar("/runtime/write_ms", row.get("write_ms"))

        logged_action = np.asarray(row.get("action", []), dtype=np.float32).reshape(-1)
        for idx, value in enumerate(logged_action[: len(ACTION_LABELS)]):
            self._scalar(f"/action/logged/{ACTION_LABELS[idx]}", float(value))
        if replay_action is not None:
            error = replay_action - logged_action.reshape(replay_action.shape)
            for idx, value in enumerate(replay_action[: len(ACTION_LABELS)]):
                self._scalar(f"/action/replay/{ACTION_LABELS[idx]}", float(value))
                self._scalar(f"/action/error/{ACTION_LABELS[idx]}", float(error[idx]))
            self._scalar("/action/error/max_abs", float(np.max(np.abs(error))))

        self._array("/state/joint_pos", row.get("joint_pos"), LEG_LABELS)
        self._array("/state/joint_vel", row.get("joint_vel"), LEG_LABELS)
        self._array("/state/wheel_pos", row.get("wheel_pos"), WHEEL_LABELS)
        self._array("/state/wheel_vel", row.get("wheel_vel"), WHEEL_LABELS)
        self._array("/target/nx_joint_pos", row.get("nx_target_joint_pos"), LEG_LABELS)
        self._array("/target/nx_wheel_vel", row.get("nx_target_wheel_vel"), WHEEL_LABELS)
        if replay_joint_target is not None:
            self._array("/target/replay_joint_pos", replay_joint_target, LEG_LABELS)
        if replay_wheel_target is not None:
            self._array("/target/replay_wheel_vel", replay_wheel_target, WHEEL_LABELS)
        self._array("/imu/base_ang_vel_body", row.get("base_ang_vel_body"), XYZ_LABELS)
        self._array("/imu/projected_gravity", row.get("projected_gravity"), XYZ_LABELS)
        if self.log_3d:
            if self.robot_logger is not None:
                self.robot_logger.log_sample(row, relative_time_s=relative_time_s)
            else:
                self._log_3d_vectors(row)

    def close(self) -> None:
        if self.rr is None:
            return
        disconnect = getattr(self.rr, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def _array(self, path: str, values: object, labels: tuple[str, ...]) -> None:
        if values is None:
            return
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        for idx, value in enumerate(arr[: len(labels)]):
            self._scalar(f"{path}/{labels[idx]}", float(value))

    def _optional_scalar(self, path: str, value: object) -> None:
        parsed = _optional_float(value)
        if parsed is not None:
            self._scalar(path, parsed)

    def _scalar(self, path: str, value: float) -> None:
        if self.rr is None:
            return
        self.rr.log(path, self.rr.Scalars(scalars=float(value)))

    def _log_3d_vectors(self, row: dict[str, object]) -> None:
        if self.rr is None:
            return
        try:
            gravity = np.asarray(row.get("projected_gravity", (0.0, 0.0, -1.0)), dtype=np.float32)
            ang_vel = np.asarray(row.get("base_ang_vel_body", (0.0, 0.0, 0.0)), dtype=np.float32)
            self.rr.log(
                "/approx_body/projected_gravity",
                self.rr.Arrows3D(vectors=[gravity], origins=[[0.0, 0.0, 0.0]]),
            )
            self.rr.log(
                "/approx_body/base_ang_vel_body",
                self.rr.Arrows3D(vectors=[ang_vel], origins=[[0.0, 0.0, 0.0]]),
            )
        except Exception as exc:
            self.log_3d = False
            self.rr.log("/warnings/replay_3d_disabled", self.rr.TextLog(str(exc)))


class _RobotKinematicsLogger:
    def __init__(self, *, rr: Any, model_path: Path, app_id: str) -> None:
        import mujoco

        from se3_sim2sim.rerun_viewer import RerunViewer

        self.rr = rr
        self.mujoco = mujoco
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.default_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        self.free_qpos_adr = self._free_joint_qpos_adr()
        self.joint_qpos_by_name = self._joint_qpos_by_name()
        self.base_height = float(RobotConfig().default_base_height)
        self.viewer = RerunViewer(
            app_id=app_id,
            spawn=False,
            manage_recording=False,
        )
        self.viewer.log_model(self.model)

    @classmethod
    def try_create(cls, *, rr: Any, app_id: str, model_path: Path) -> _RobotKinematicsLogger | None:
        path = Path(model_path)
        if not path.exists():
            rr.log(
                "/warnings/replay_robot_model_missing",
                rr.TextLog(f"MJCF model not found: {path}"),
            )
            return None
        try:
            return cls(rr=rr, app_id=app_id, model_path=path)
        except Exception as exc:
            rr.log("/warnings/replay_robot_disabled", rr.TextLog(str(exc)))
            return None

    def log_sample(self, row: dict[str, object], *, relative_time_s: float) -> None:
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = 0.0
        self.data.time = float(relative_time_s)
        if self.free_qpos_adr is not None:
            qadr = self.free_qpos_adr
            self.data.qpos[qadr : qadr + 3] = np.asarray([0.0, 0.0, self.base_height])
            self.data.qpos[qadr + 3 : qadr + 7] = _quat_from_projected_gravity(
                row.get("projected_gravity")
            )

        joint_pos = np.asarray(row.get("joint_pos", []), dtype=np.float64).reshape(-1)
        wheel_pos = np.asarray(row.get("wheel_pos", []), dtype=np.float64).reshape(-1)
        policy_qpos = np.concatenate([joint_pos[:4], wheel_pos[:2]])
        for name, value in zip(POLICY_JOINT_NAMES, policy_qpos, strict=False):
            qadr = self.joint_qpos_by_name.get(name)
            if qadr is not None:
                self.data.qpos[qadr] = float(value)

        self.mujoco.mj_forward(self.model, self.data)
        self.viewer.log_state(
            self.model,
            self.data,
            step=int(row.get("step", 0)),
            telemetry=_sim2sim_telemetry_from_row(row, base_height=self.base_height),
        )

    def _free_joint_qpos_adr(self) -> int | None:
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_qposadr[joint_id])
        return None

    def _joint_qpos_by_name(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for name in POLICY_JOINT_NAMES:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                result[name] = int(self.model.jnt_qposadr[joint_id])
        return result


def _sim2sim_telemetry_from_row(row: dict[str, object], *, base_height: float) -> dict[str, object]:
    joint_pos = _row_vector(row, "joint_pos", 4)
    joint_vel = _row_vector(row, "joint_vel", 4)
    wheel_pos = _row_vector(row, "wheel_pos", 2)
    wheel_vel = _row_vector(row, "wheel_vel", 2)
    action = _row_vector(row, "action", 6)
    clipped_action = _row_vector(row, "clipped_action", 6)
    hip_torque = _row_vector(row, "hip_torque", 4)
    wheel_torque = _row_vector(row, "wheel_motor_torque", 2)
    base_ang_vel_body = _row_vector(row, "base_ang_vel_body", 3)
    projected_gravity = _row_vector(row, "projected_gravity", 3, default=(0.0, 0.0, -1.0))
    roll, pitch = _roll_pitch_from_projected_gravity(projected_gravity)
    command = _row_vector(
        row, "command", 8, default=(0.0, 0.0, 0.0, 0.0, base_height, 0.0, 0.0, 0.0)
    )
    tilt_deg = math.degrees(math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0))))
    wheel_lin_vel = float(np.mean(wheel_vel)) if wheel_vel.size else 0.0
    return {
        "height": float(base_height),
        "wheel_clearance": 0.0,
        "wheel_clearance_left": 0.0,
        "wheel_clearance_right": 0.0,
        "leg_clearance": 0.0,
        "base_clearance": 0.0,
        "wheel_contact": 0.0,
        "wheel_full_contact": 0.0,
        "wheel_contact_left": 0.0,
        "wheel_contact_right": 0.0,
        "leg_contact": 0.0,
        "leg_contact_left": 0.0,
        "leg_contact_right": 0.0,
        "base_contact": 0.0,
        "nonwheel_contact": 0.0,
        "tilt_deg": float(tilt_deg),
        "fail_tilt_deg": 80.0,
        "reward": 0.0,
        "last_ctrl": np.concatenate([hip_torque, wheel_torque]).astype(np.float64, copy=False),
        "target_mode": row.get("target_mode", "policy"),
        "rc_switch_r": float(row.get("rc_switch_r", 0.0)),
        "output_enabled": float(row.get("output_enabled", 0.0)),
        "rc_switch_event": 0.0,
        "rc_policy_reset": 0.0,
        "command_lin_vel_x": float(command[0]),
        "command_yaw_rate": float(command[1]),
        "command_height": float(command[4]),
        "base_lin_vel_x": 0.0,
        "wheel_lin_vel": wheel_lin_vel,
        "base_ang_vel_body": base_ang_vel_body,
        "base_ang_vel_world": base_ang_vel_body,
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "yaw_deg": 0.0,
        "dof_pos": np.concatenate([joint_pos, wheel_pos]).astype(np.float64, copy=False),
        "dof_vel": np.concatenate([joint_vel, wheel_vel]).astype(np.float64, copy=False),
        "policy_action_raw": action,
        "policy_action_clipped": clipped_action,
        "last_action": clipped_action,
    }


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            yield line_no, json.loads(text)


def _load_meta(telemetry: Path, explicit: Path | None) -> dict[str, object]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(telemetry.with_suffix(".meta.json"))
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _resolve_checkpoint(explicit: Path | None, telemetry: Path, meta: dict[str, object]) -> Path:
    if explicit is not None:
        return explicit
    checkpoint_meta = meta.get("checkpoint") if isinstance(meta, dict) else None
    candidates: list[Path] = []
    if isinstance(checkpoint_meta, dict):
        for key in ("resolved_path", "path"):
            value = checkpoint_meta.get(key)
            if isinstance(value, str) and value:
                candidates.append(Path(value))
                candidates.append(telemetry.parent / Path(value).name)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "checkpoint not found; pass --checkpoint or place the model next to the telemetry log"
    )


def _check_checkpoint_hash(checkpoint: Path, meta: dict[str, object]) -> None:
    checkpoint_meta = meta.get("checkpoint") if isinstance(meta, dict) else None
    expected = checkpoint_meta.get("sha256") if isinstance(checkpoint_meta, dict) else None
    if not isinstance(expected, str) or not expected:
        return
    actual = _sha256_file(checkpoint)
    if actual != expected:
        print(
            "[warning] checkpoint sha256 mismatch: "
            f"expected={expected} actual={actual} path={checkpoint}"
        )


def _load_policy(checkpoint: Path) -> PolicyRuntimeLike:
    suffix = checkpoint.suffix.lower()
    if suffix == ".onnx":
        return OnnxPolicyRuntime(checkpoint)
    if suffix == ".npz":
        return NumpyPolicyRuntime(checkpoint)
    raise ValueError(f"unsupported checkpoint suffix: {checkpoint}")


def _policy_reset(policy: PolicyRuntimeLike) -> None:
    policy.reset()


def _policy_act(policy: PolicyRuntimeLike, obs: np.ndarray) -> np.ndarray:
    action = policy.act(obs)
    return np.asarray(action, dtype=np.float32).reshape(ObservationConfig().num_actions)


def _command_height(meta: dict[str, object]) -> float:
    command = meta.get("command") if isinstance(meta, dict) else None
    if isinstance(command, list) and len(command) >= 5:
        return float(command[4])
    return float(RobotConfig().default_base_height)


def _array(row: dict[str, object], key: str, size: int) -> np.ndarray:
    arr = np.asarray(row[key], dtype=np.float32).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{key} shape mismatch: expected {(size,)}, got {arr.shape}")
    return arr


def _row_vector(
    row: dict[str, object],
    key: str,
    size: int,
    *,
    default: tuple[float, ...] | None = None,
) -> np.ndarray:
    fallback = (0.0,) * int(size) if default is None else default
    arr = np.asarray(row.get(key, fallback), dtype=np.float64).reshape(-1)
    if arr.shape != (int(size),) or not np.isfinite(arr).all():
        return np.asarray(fallback, dtype=np.float64).reshape(int(size))
    return arr


def _row_time_s(row: dict[str, object]) -> float:
    for key in ("monotonic_time_s", "wall_time_s"):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    tick_ms = _optional_float(row.get("tick_ms"))
    if tick_ms is not None:
        return tick_ms / 1000.0
    step = _optional_float(row.get("step")) or 0.0
    period_ms = _optional_float(row.get("sample_period_ms")) or 20.0
    return step * period_ms / 1000.0


def _quat_from_projected_gravity(values: object) -> np.ndarray:
    gravity = np.asarray(values if values is not None else (0.0, 0.0, -1.0), dtype=np.float64)
    gravity = gravity.reshape(-1)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if norm > 1.0e-6:
        gravity = gravity / norm
    roll = math.asin(float(np.clip(-gravity[1], -1.0, 1.0)))
    pitch = math.asin(float(np.clip(gravity[0], -1.0, 1.0)))
    return _euler_xyz_to_quat_wxyz(roll, pitch, 0.0)


def _roll_pitch_from_projected_gravity(values: object) -> tuple[float, float]:
    gravity = _row_vector(
        {"projected_gravity": values}, "projected_gravity", 3, default=(0.0, 0.0, -1.0)
    )
    norm = float(np.linalg.norm(gravity))
    if norm > 1.0e-6:
        gravity = gravity / norm
    roll = math.asin(float(np.clip(-gravity[1], -1.0, 1.0)))
    pitch = math.asin(float(np.clip(gravity[0], -1.0, 1.0)))
    return roll, pitch


def _euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _optional_int(value: object) -> int | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_progress(stats: ReplayStats, line_no: int) -> None:
    last_error = (
        stats.action_row_max_abs_errors[-1] if stats.action_row_max_abs_errors else math.nan
    )
    print(
        f"replayed samples={stats.sample_rows} line={line_no} "
        f"policy={stats.policy_rows} hold={stats.hold_rows} "
        f"last_action_err={last_error:.3e}"
    )


def _print_summary(telemetry: Path, checkpoint: Path, summary: dict[str, object]) -> None:
    action = summary["action_max_abs_error"]
    dt = summary["dt_ms"]
    policy_ms = summary["policy_inference_ms"]
    print(f"telemetry: {telemetry}")
    print(f"checkpoint: {checkpoint}")
    print(
        "rows: "
        f"samples={summary['sample_rows']} policy={summary['policy_rows']} "
        f"hold={summary['hold_rows']} events={summary['event_rows']} "
        f"resets={summary['reset_count']}"
    )
    print(
        "action max abs error: "
        f"mean={action['mean']:.3e} p95={action['p95']:.3e} max={action['max']:.3e}"
    )
    print(
        "dt ms: "
        f"mean={dt['mean']:.3f} p95={dt['p95']:.3f} p99={dt['p99']:.3f} "
        f"max={dt['max']:.3f} missed={summary['missed_50hz_deadlines']}"
    )
    print(
        "logged policy ms: "
        f"mean={policy_ms['mean']:.3f} p95={policy_ms['p95']:.3f} "
        f"p99={policy_ms['p99']:.3f} max={policy_ms['max']:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
