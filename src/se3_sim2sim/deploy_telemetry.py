"""从 NX deploy telemetry 初始化 sim2sim 的辅助函数。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from se3_shared import ObservationConfig
from se3_shared import RobotConfig as SharedRobotConfig

from .config import RcSwitchEventConfig
from .runtime_spec import RuntimeSpec

DeployTelemetryInitMode = Literal["enable-transition", "first-policy", "sample"]


@dataclass(frozen=True, slots=True)
class DeployTelemetryInitialState:
    """从单条 deploy telemetry 样本解码出的 sim2sim 初始状态。"""

    telemetry_path: Path
    mode: DeployTelemetryInitMode
    selected_line_no: int
    selected_sample_index: int
    selected_step: int
    selected_time_s: float
    selected_relative_time_s: float
    selected_target_mode: str
    selected_output_enabled: bool
    first_policy_line_no: int | None
    first_policy_sample_index: int | None
    first_policy_step: int | None
    first_policy_relative_time_s: float | None
    switch_delay_s: float | None
    checkpoint_hint: Path | None
    initial_base_height: float
    initial_roll_rad: float
    initial_pitch_rad: float
    initial_yaw_rad: float
    initial_ang_vel_rad_s: tuple[float, float, float]
    initial_leg_joint_pos: tuple[float, float, float, float]
    initial_wheel_joint_pos: tuple[float, float]
    initial_dof_vel: tuple[float, float, float, float, float, float]
    initial_last_action: tuple[float, float, float, float, float, float]
    command: tuple[float, float, float, float, float, float, float, float]
    rc_initial_output_enabled: bool
    rc_events: tuple[RcSwitchEventConfig, ...]
    reference_obs: tuple[float, ...] | None

    def summary(self) -> dict[str, object]:
        """返回可写入 run summary 的紧凑 JSON 描述。"""
        return {
            "enabled": True,
            "telemetry_path": str(self.telemetry_path),
            "mode": self.mode,
            "selected_line_no": self.selected_line_no,
            "selected_sample_index": self.selected_sample_index,
            "selected_step": self.selected_step,
            "selected_relative_time_s": self.selected_relative_time_s,
            "selected_target_mode": self.selected_target_mode,
            "selected_output_enabled": self.selected_output_enabled,
            "first_policy_line_no": self.first_policy_line_no,
            "first_policy_sample_index": self.first_policy_sample_index,
            "first_policy_step": self.first_policy_step,
            "first_policy_relative_time_s": self.first_policy_relative_time_s,
            "switch_delay_s": self.switch_delay_s,
            "checkpoint_hint": None if self.checkpoint_hint is None else str(self.checkpoint_hint),
            "initial_base_height": self.initial_base_height,
            "initial_roll_deg": math.degrees(self.initial_roll_rad),
            "initial_pitch_deg": math.degrees(self.initial_pitch_rad),
            "initial_yaw_deg": math.degrees(self.initial_yaw_rad),
            "initial_ang_vel_rad_s": list(self.initial_ang_vel_rad_s),
            "initial_leg_joint_pos": list(self.initial_leg_joint_pos),
            "initial_wheel_joint_pos": list(self.initial_wheel_joint_pos),
            "initial_dof_vel": list(self.initial_dof_vel),
            "initial_last_action": list(self.initial_last_action),
            "command": list(self.command),
            "rc_initial_output_enabled": self.rc_initial_output_enabled,
            "rc_events": [
                {
                    "trigger_time_s": float(event.trigger_time_s),
                    "output_enabled": bool(event.output_enabled),
                }
                for event in self.rc_events
            ],
        }


@dataclass(frozen=True, slots=True)
class _SampleRow:
    line_no: int
    sample_index: int
    time_s: float
    row: dict[str, object]


def load_deploy_telemetry_initial_state(
    telemetry_path: Path,
    *,
    mode: DeployTelemetryInitMode,
    sample_index: int | None = None,
    pre_policy_rows: int = 3,
    base_height_override: float | None = None,
) -> DeployTelemetryInitialState:
    """把一条 deploy telemetry 样本解码为 sim2sim reset 和 RC schedule 字段。"""
    path = Path(telemetry_path).expanduser().resolve()
    samples = list(_iter_sample_rows(path))
    if not samples:
        raise ValueError(f"deploy telemetry has no sample rows: {path}")

    first_sample_time_s = samples[0].time_s
    first_policy_idx = _first_policy_sample_index(samples)
    if first_policy_idx is None and mode in {"enable-transition", "first-policy"}:
        raise ValueError(f"deploy telemetry has no policy sample rows: {path}")

    if sample_index is not None:
        selected_idx = _check_sample_index(sample_index, len(samples))
        mode = "sample"
    elif mode == "first-policy":
        selected_idx = int(first_policy_idx)
    elif mode == "enable-transition":
        selected_idx = max(0, int(first_policy_idx) - max(0, int(pre_policy_rows)))
    else:
        raise ValueError(
            "--deploy-telemetry-init-mode sample requires --deploy-telemetry-init-sample-index"
        )

    selected = samples[selected_idx]
    first_policy = None if first_policy_idx is None else samples[int(first_policy_idx)]
    state = _decode_state_fields(
        selected.row,
        base_height_override=base_height_override,
    )

    rc_initial_output_enabled = _row_output_enabled(selected.row)
    rc_events: tuple[RcSwitchEventConfig, ...] = ()
    switch_delay_s: float | None = None
    if mode == "enable-transition":
        assert first_policy is not None
        rc_initial_output_enabled = False
        switch_delay_s = max(0.0, float(first_policy.time_s - selected.time_s))
        rc_events = (RcSwitchEventConfig(trigger_time_s=switch_delay_s, output_enabled=True),)
    elif mode == "first-policy":
        rc_initial_output_enabled = True

    checkpoint_hint = _resolve_checkpoint_hint(path, _load_meta(path))
    return DeployTelemetryInitialState(
        telemetry_path=path,
        mode=mode,
        selected_line_no=selected.line_no,
        selected_sample_index=selected.sample_index,
        selected_step=_optional_int(selected.row.get("step")) or 0,
        selected_time_s=selected.time_s,
        selected_relative_time_s=float(selected.time_s - first_sample_time_s),
        selected_target_mode=str(selected.row.get("target_mode", "")),
        selected_output_enabled=_row_output_enabled(selected.row),
        first_policy_line_no=None if first_policy is None else first_policy.line_no,
        first_policy_sample_index=None if first_policy is None else first_policy.sample_index,
        first_policy_step=(
            None if first_policy is None else (_optional_int(first_policy.row.get("step")) or 0)
        ),
        first_policy_relative_time_s=(
            None if first_policy is None else float(first_policy.time_s - first_sample_time_s)
        ),
        switch_delay_s=switch_delay_s,
        checkpoint_hint=checkpoint_hint,
        rc_initial_output_enabled=rc_initial_output_enabled,
        rc_events=rc_events,
        **state,
    )


def _decode_state_fields(
    row: dict[str, object],
    *,
    base_height_override: float | None,
) -> dict[str, object]:
    obs_cfg = ObservationConfig()
    runtime = RuntimeSpec()
    shared_robot = SharedRobotConfig()

    command = _row_vector(
        row,
        "command",
        8,
        default=(0.0, 0.0, 0.0, 0.0, shared_robot.default_base_height, 0.0, 0.0, 0.0),
    )
    base_height = (
        float(base_height_override)
        if base_height_override is not None
        else float(command[4] if command.size >= 5 else shared_robot.default_base_height)
    )
    projected_gravity = _row_vector(
        row,
        "projected_gravity",
        3,
        default=(0.0, 0.0, -1.0),
    )
    roll, pitch = _roll_pitch_from_projected_gravity(projected_gravity)
    obs = _optional_row_vector(row, "obs", obs_cfg.num_obs)
    if obs is not None:
        action_slice = runtime.observation_slices["actions"]
        initial_last_action = obs[action_slice]
    else:
        initial_last_action = _row_vector(row, "clipped_action", obs_cfg.num_actions)

    joint_pos = _row_vector(row, "joint_pos", 4)
    wheel_pos = _row_vector(row, "wheel_pos", 2)
    joint_vel = _row_vector(row, "joint_vel", 4)
    wheel_vel = _row_vector(row, "wheel_vel", 2)
    base_ang_vel = _row_vector(row, "base_ang_vel_body", 3)
    return {
        "initial_base_height": base_height,
        "initial_roll_rad": float(roll),
        "initial_pitch_rad": float(pitch),
        "initial_yaw_rad": 0.0,
        "initial_ang_vel_rad_s": _as_tuple3(base_ang_vel),
        "initial_leg_joint_pos": _as_tuple4(joint_pos),
        "initial_wheel_joint_pos": _as_tuple2(wheel_pos),
        "initial_dof_vel": _as_tuple6(np.concatenate([joint_vel, wheel_vel])),
        "initial_last_action": _as_tuple6(initial_last_action),
        "command": _as_tuple8(command),
        "reference_obs": None if obs is None else tuple(float(v) for v in obs.tolist()),
    }


def _iter_sample_rows(path: Path) -> Iterator[_SampleRow]:
    sample_index = 0
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            record_type = row.get("record_type", "sample" if "obs" in row else "event")
            if record_type != "sample" or "obs" not in row:
                continue
            yield _SampleRow(
                line_no=line_no,
                sample_index=sample_index,
                time_s=_row_time_s(row),
                row=row,
            )
            sample_index += 1


def _first_policy_sample_index(samples: list[_SampleRow]) -> int | None:
    for sample in samples:
        if str(sample.row.get("target_mode", "")) == "policy" and _row_output_enabled(sample.row):
            return int(sample.sample_index)
    return None


def _check_sample_index(value: int, sample_count: int) -> int:
    index = int(value)
    if index < 0 or index >= int(sample_count):
        raise ValueError(
            f"--deploy-telemetry-init-sample-index must be in [0, {sample_count - 1}], got {index}"
        )
    return index


def _row_vector(
    row: dict[str, object],
    key: str,
    size: int,
    *,
    default: tuple[float, ...] | None = None,
) -> np.ndarray:
    if key not in row:
        if default is None:
            raise ValueError(f"deploy telemetry sample is missing required field {key!r}")
        values = default
    else:
        values = row[key]
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (int(size),) or not np.isfinite(arr).all():
        raise ValueError(
            f"deploy telemetry field {key!r} must be finite shape {(size,)}, got {arr}"
        )
    return arr


def _optional_row_vector(row: dict[str, object], key: str, size: int) -> np.ndarray | None:
    if key not in row:
        return None
    return _row_vector(row, key, size)


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


def _row_output_enabled(row: dict[str, object]) -> bool:
    value = _optional_float(row.get("output_enabled"))
    if value is not None:
        return value > 0.5
    return str(row.get("target_mode", "")) == "policy"


def _roll_pitch_from_projected_gravity(values: np.ndarray) -> tuple[float, float]:
    gravity = np.asarray(values, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(gravity))
    if norm > 1.0e-6:
        gravity = gravity / norm
    roll = math.asin(float(np.clip(-gravity[1], -1.0, 1.0)))
    pitch = math.asin(float(np.clip(gravity[0], -1.0, 1.0)))
    return roll, pitch


def _load_meta(telemetry: Path) -> dict[str, object]:
    meta_path = telemetry.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _resolve_checkpoint_hint(telemetry: Path, meta: dict[str, object]) -> Path | None:
    checkpoint_meta = meta.get("checkpoint") if isinstance(meta, dict) else None
    if not isinstance(checkpoint_meta, dict):
        return None

    candidates: list[Path] = []
    for key in ("path", "resolved_path"):
        value = checkpoint_meta.get(key)
        if not isinstance(value, str) or not value:
            continue
        raw = Path(value)
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(Path.cwd() / raw)
        candidates.append(telemetry.parent / raw.name)

    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: object) -> int | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _as_tuple2(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(2)
    return (float(arr[0]), float(arr[1]))


def _as_tuple3(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _as_tuple4(values: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(4)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _as_tuple6(values: np.ndarray) -> tuple[float, float, float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(6)
    return tuple(float(v) for v in arr)  # type: ignore[return-value]


def _as_tuple8(
    values: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(8)
    return tuple(float(v) for v in arr)  # type: ignore[return-value]
