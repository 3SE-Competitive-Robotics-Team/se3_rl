"""Jetson Orin NX 上的 recovery-only policy runtime。"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from se3_shared import (
    RECOVERY_COMMAND_HEIGHT_M,
    ObservationConfig,
    PolicyActionDecoder,
    RobotConfig,
    policy_leg_position_error_np,
)

from .cdc import CdcSerial
from .numpy_policy import NumpyPolicyRuntime
from .observation import RecoveryObservationBuilder, synthetic_recovery_state
from .onnx_policy import OnnxPolicyRuntime
from .protocol import (
    MSG_POLICY_STATE,
    PolicyStateFrame,
    PolicyTargetFrame,
    StreamParser,
    decode_policy_state,
    pack_policy_target,
)

DEFAULT_RECOVERY_CHECKPOINT = Path("logs/deploy/model_4999_recovery_obs34_gru.npz")
DEFAULT_CDC_PORT = "auto"
ACTION_FLAG_DRY_RUN = 1 << 0
ACTION_FLAG_TIMEOUT = 1 << 1
ACTION_FLAG_NONFINITE = 1 << 2
ACTION_FLAG_OUTPUT_DISABLED_HOLD = 1 << 3
OBS_CONTRACT = "se3_recovery_actor_obs34_front_sincos_active_last_actions_v1"
_ROBOT_CFG = RobotConfig()


@dataclass(frozen=True, slots=True)
class RecoveryRuntimeConfig:
    checkpoint: Path
    port: str
    baudrate: int
    device: str
    rate_hz: float
    state_timeout_s: float
    write_timeout_s: float
    max_steps: int
    dry_run: bool
    print_every: int
    telemetry_log: Path | None
    telemetry_log_every: int
    telemetry_flush_every: int


@dataclass(slots=True)
class RuntimeStats:
    steps: int = 0
    state_frames: int = 0
    action_frames: int = 0
    timeout_frames: int = 0
    nonfinite_frames: int = 0
    last_state_seq: int = 0
    last_action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_target_joint_pos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    last_target_wheel_vel: tuple[float, ...] = (0.0, 0.0)
    last_state_joint_pos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    last_state_target_joint_pos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    last_state_hip_torque: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    last_state_wheel_torque: tuple[float, ...] = (0.0, 0.0)
    last_state_wheel_motor_torque: tuple[float, ...] = (0.0, 0.0)
    last_command: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        RECOVERY_COMMAND_HEIGHT_M,
        0.0,
        0.0,
        0.0,
    )
    policy_inference_frames: int = 0
    last_policy_inference_ms: float = 0.0
    total_policy_inference_ms: float = 0.0
    max_policy_inference_ms: float = 0.0
    last_step_policy_inference_ms: float | None = None
    last_action_flags: int = 0
    last_state_output_enabled: int = 0


@dataclass(frozen=True, slots=True)
class DecodedPolicyTarget:
    """NX 下发给 STM32 的物理控制目标。"""

    joint_pos: tuple[float, float, float, float]
    wheel_vel: tuple[float, float]


class RecoveryActionTargetDecoder:
    """把 recovery raw action 解码成 STM32 侧直接执行的目标值。"""

    def __init__(self, *, command_height: float, robot_cfg: RobotConfig | None = None) -> None:
        self.robot_cfg = RobotConfig() if robot_cfg is None else robot_cfg
        self.command_height = float(command_height)
        self.decoder = PolicyActionDecoder(
            robot_cfg=self.robot_cfg,
            height_conditioned_action_default=True,
            active_rod_semantics=True,
            dtype=np.float32,
        )

    def decode(
        self, action: np.ndarray, *, command_height: float | None = None
    ) -> DecodedPolicyTarget:
        height = self.command_height if command_height is None else float(command_height)
        decoded = self.decoder.decode(action, command_height=height)
        return DecodedPolicyTarget(
            joint_pos=tuple(float(v) for v in decoded.leg_target),
            wheel_vel=tuple(float(v) for v in decoded.wheel_vel_target),
        )

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        return self.decoder.clip_action(action)


def _fmt_values(values: object) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return " ".join(f"{float(v):+.3f}" for v in arr)


def _float_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(v) for v in arr]


def _policy_active_angles(values: object) -> list[float]:
    q = np.asarray(values, dtype=np.float32).reshape(4)
    coeffs = np.asarray(_ROBOT_CFG.active_rod_angle_coeffs, dtype=np.float32).reshape(2, 2)
    active = np.asarray(
        [
            coeffs[0, 0] * q[0] + coeffs[0, 1] * q[1],
            coeffs[1, 0] * q[2] + coeffs[1, 1] * q[3],
        ],
        dtype=np.float32,
    )
    return _float_list(active)


def _policy_leg_error(target: object, current: object) -> np.ndarray:
    """按训练端 PD 语义计算腿部误差：前杆走最短角，主动杆不 wrap。"""
    return policy_leg_position_error_np(
        np.asarray(target, dtype=np.float32).reshape(4),
        np.asarray(current, dtype=np.float32).reshape(4),
        robot_cfg=_ROBOT_CFG,
    ).astype(np.float32, copy=False)


def _action_flag_names(flags: int) -> list[str]:
    names: list[str] = []
    if flags & ACTION_FLAG_DRY_RUN:
        names.append("dry_run")
    if flags & ACTION_FLAG_TIMEOUT:
        names.append("timeout")
    if flags & ACTION_FLAG_NONFINITE:
        names.append("nonfinite")
    if flags & ACTION_FLAG_OUTPUT_DISABLED_HOLD:
        names.append("output_disabled_hold")
    return names


def _resolve_telemetry_log_path(path: Path) -> Path:
    if path.suffix.lower() == ".jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path / f"recovery_telemetry_{timestamp}.jsonl"


class TelemetryLogger:
    """按 policy step 追加写入 NX runtime 调试 JSONL。"""

    def __init__(self, path: Path | None, *, every: int, flush_every: int) -> None:
        self.path: Path | None = None
        self.every = max(int(every), 1)
        self.flush_every = max(int(flush_every), 1)
        self._file = None
        self._written = 0
        if path is None:
            return
        self.path = _resolve_telemetry_log_path(path)
        self._file = self.path.open("a", encoding="utf-8")
        print(f"telemetry log: {self.path}")

    def write(self, step: int, record: dict[str, object]) -> None:
        if self._file is None or int(step) % self.every != 0:
            return
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self._file.write("\n")
        self._written += 1
        if self._written % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None


class RecoveryRuntime:
    """只运行 recovery 网络的 50 Hz 真机 runtime。"""

    def __init__(self, cfg: RecoveryRuntimeConfig) -> None:
        self.cfg = cfg
        self.obs_cfg = ObservationConfig()
        self.policy = load_policy_runtime(cfg.checkpoint, cfg.device)
        self.policy.reset()
        self.obs_builder = RecoveryObservationBuilder()
        self.target_decoder = RecoveryActionTargetDecoder(
            command_height=float(self.obs_builder.default_command[4])
        )
        self.stats = RuntimeStats()
        self._last_action = np.zeros(self.obs_cfg.num_actions, dtype=np.float32)
        self._action_seq = 0
        self._policy_memory_clean = True
        self._start_monotonic = time.monotonic()
        self._last_print_monotonic = self._start_monotonic
        self._last_print_steps = 0
        self.telemetry = TelemetryLogger(
            cfg.telemetry_log,
            every=cfg.telemetry_log_every,
            flush_every=cfg.telemetry_flush_every,
        )

    def run(self) -> RuntimeStats:
        print(
            "NX recovery runtime: "
            f"checkpoint={self.cfg.checkpoint} iter={self.policy.iteration} "
            f"type={self.policy.policy_type} device={self.cfg.device} "
            f"obs={self.obs_cfg.num_obs} contract={OBS_CONTRACT}"
        )
        try:
            if self.cfg.dry_run:
                self._run_dry()
            else:
                self._run_cdc()
        finally:
            self.telemetry.close()
        return self.stats

    def _run_dry(self) -> None:
        max_steps = self.cfg.max_steps if self.cfg.max_steps > 0 else int(self.cfg.rate_hz * 2.0)
        for step in range(max_steps):
            state = synthetic_recovery_state(seq=step)
            self.stats.state_frames += 1
            action, flags, obs, policy_inference_ms = self._act_from_state(state)
            flags |= ACTION_FLAG_DRY_RUN
            self._decode_target(action, state)
            self._record_action(state, action, flags, policy_inference_ms)
            self._write_telemetry(state, obs, action, flags, policy_inference_ms)
            self._maybe_print()

    def _run_cdc(self) -> None:
        parser = StreamParser()
        latest_state: PolicyStateFrame | None = None
        latest_state_time = 0.0
        max_steps = self.cfg.max_steps
        period_s = 1.0 / float(self.cfg.rate_hz)
        next_tick = time.monotonic()

        with CdcSerial(self.cfg.port, baudrate=self.cfg.baudrate) as serial:
            print(f"USB CDC open: port={self.cfg.port} baudrate={self.cfg.baudrate}")
            while max_steps <= 0 or self.stats.steps < max_steps:
                now = time.monotonic()
                wait_s = max(0.0, min(period_s, next_tick - now))
                if serial.wait_readable(wait_s):
                    latest_state, latest_state_time = self._read_states(
                        serial, parser, latest_state, latest_state_time
                    )

                now = time.monotonic()
                if now < next_tick:
                    continue
                next_tick += period_s
                if latest_state is None:
                    self.stats.steps += 1
                    self._maybe_print()
                    continue

                age_s = now - latest_state_time
                flags = 0
                policy_inference_ms: float | None = None
                if age_s > self.cfg.state_timeout_s:
                    self._reset_policy_memory()
                    action = np.zeros(self.obs_cfg.num_actions, dtype=np.float32)
                    flags |= ACTION_FLAG_TIMEOUT
                    obs, obs_flags = self._build_observation(latest_state)
                    flags |= obs_flags
                    if obs_flags & ACTION_FLAG_NONFINITE:
                        self.stats.nonfinite_frames += 1
                    packet = self._make_hold_target_packet(latest_state)
                elif latest_state.output_enabled == 0:
                    self._reset_policy_memory()
                    obs, obs_flags = self._build_observation(latest_state)
                    action = np.zeros(self.obs_cfg.num_actions, dtype=np.float32)
                    flags |= obs_flags | ACTION_FLAG_OUTPUT_DISABLED_HOLD
                    if obs_flags & ACTION_FLAG_NONFINITE:
                        self.stats.nonfinite_frames += 1
                    packet = self._make_hold_target_packet(latest_state)
                else:
                    action, flags, obs, policy_inference_ms = self._act_from_state(latest_state)
                    packet = self._make_target_packet(action, latest_state)

                serial.write_all(packet, timeout_s=self.cfg.write_timeout_s)
                self._record_action(latest_state, action, flags, policy_inference_ms)
                self._write_telemetry(latest_state, obs, action, flags, policy_inference_ms)
                self._maybe_print()

    def _read_states(
        self,
        serial: CdcSerial,
        parser: StreamParser,
        latest_state: PolicyStateFrame | None,
        latest_state_time: float,
    ) -> tuple[PolicyStateFrame | None, float]:
        while True:
            data = serial.read_available()
            if not data:
                return latest_state, latest_state_time
            for message in parser.feed(data):
                if message.msg_type != MSG_POLICY_STATE:
                    continue
                latest_state = decode_policy_state(message)
                latest_state_time = time.monotonic()
                self.stats.state_frames += 1
                self.stats.last_state_seq = latest_state.seq

    def _build_observation(self, state: PolicyStateFrame) -> tuple[np.ndarray, int]:
        result = self.obs_builder.build(state, self._last_action)
        flags = ACTION_FLAG_NONFINITE if result.had_nonfinite_input else 0
        return result.obs.astype(np.float32, copy=False), flags

    def _act_from_state(self, state: PolicyStateFrame) -> tuple[np.ndarray, int, np.ndarray, float]:
        obs, flags = self._build_observation(state)
        started = time.perf_counter()
        action = self.policy.act(obs)
        policy_inference_ms = (time.perf_counter() - started) * 1000.0
        self._record_policy_inference(policy_inference_ms)
        if not np.isfinite(action).all():
            flags |= ACTION_FLAG_NONFINITE
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if flags & ACTION_FLAG_NONFINITE:
            self.stats.nonfinite_frames += 1
        self._policy_memory_clean = False
        return action.astype(np.float32, copy=False), flags, obs, policy_inference_ms

    def _record_policy_inference(self, policy_inference_ms: float) -> None:
        value = float(policy_inference_ms)
        self.stats.policy_inference_frames += 1
        self.stats.last_policy_inference_ms = value
        self.stats.total_policy_inference_ms += value
        self.stats.max_policy_inference_ms = max(self.stats.max_policy_inference_ms, value)

    def _avg_policy_inference_ms(self) -> float:
        count = int(self.stats.policy_inference_frames)
        if count <= 0:
            return 0.0
        return float(self.stats.total_policy_inference_ms) / float(count)

    def _reset_policy_memory(self) -> None:
        if self._policy_memory_clean:
            return
        self.policy.reset()
        self._last_action = np.zeros(self.obs_cfg.num_actions, dtype=np.float32)
        self._policy_memory_clean = True

    def _make_target_packet(self, action: np.ndarray, state: PolicyStateFrame) -> bytes:
        target = self._decode_target(action, state)
        frame = PolicyTargetFrame(
            seq=self._action_seq,
            joint_pos=target.joint_pos,
            wheel_vel=target.wheel_vel,
        )
        self._action_seq = (self._action_seq + 1) & 0xFFFFFFFF
        return pack_policy_target(frame)

    def _make_hold_target_packet(self, state: PolicyStateFrame) -> bytes:
        joint_pos = tuple(
            float(v) for v in np.asarray(state.joint_pos, dtype=np.float32).reshape(4)
        )
        wheel_vel = (0.0, 0.0)
        self.stats.last_target_joint_pos = joint_pos
        self.stats.last_target_wheel_vel = wheel_vel
        frame = PolicyTargetFrame(
            seq=self._action_seq,
            joint_pos=joint_pos,
            wheel_vel=wheel_vel,
        )
        self._action_seq = (self._action_seq + 1) & 0xFFFFFFFF
        return pack_policy_target(frame)

    def _decode_target(
        self, action: np.ndarray, state: PolicyStateFrame | None = None
    ) -> DecodedPolicyTarget:
        command_height = (
            None if state is None else float(self.obs_builder.command_from_state(state)[4])
        )
        target = self.target_decoder.decode(action, command_height=command_height)
        joint_pos = np.asarray(target.joint_pos, dtype=np.float32).reshape(4)
        if state is not None:
            current_joint_pos = np.asarray(state.joint_pos, dtype=np.float32).reshape(4)
            joint_pos = current_joint_pos + _policy_leg_error(joint_pos, current_joint_pos)
        self.stats.last_target_joint_pos = tuple(float(v) for v in joint_pos)
        self.stats.last_target_wheel_vel = tuple(float(v) for v in target.wheel_vel)
        return DecodedPolicyTarget(
            joint_pos=self.stats.last_target_joint_pos,
            wheel_vel=target.wheel_vel,
        )

    def _record_action(
        self,
        state: PolicyStateFrame,
        action: np.ndarray,
        flags: int,
        policy_inference_ms: float | None,
    ) -> None:
        self.stats.steps += 1
        self.stats.action_frames += 1
        self.stats.last_state_seq = state.seq
        self.stats.last_action_flags = int(flags)
        self.stats.last_state_output_enabled = int(state.output_enabled)
        self.stats.last_step_policy_inference_ms = policy_inference_ms
        self._last_action = self.target_decoder.clip_action(action)
        self.stats.last_action = tuple(float(v) for v in self._last_action)
        self.stats.last_state_joint_pos = tuple(float(v) for v in state.joint_pos)
        self.stats.last_state_target_joint_pos = tuple(float(v) for v in state.target_joint_pos)
        self.stats.last_state_hip_torque = tuple(float(v) for v in state.hip_torque)
        self.stats.last_state_wheel_torque = tuple(float(v) for v in state.wheel_torque)
        self.stats.last_state_wheel_motor_torque = tuple(float(v) for v in state.wheel_motor_torque)
        self.stats.last_command = tuple(
            float(v) for v in self.obs_builder.command_from_state(state)
        )
        if flags & ACTION_FLAG_TIMEOUT:
            self.stats.timeout_frames += 1

    def _write_telemetry(
        self,
        state: PolicyStateFrame,
        obs: np.ndarray,
        action: np.ndarray,
        flags: int,
        policy_inference_ms: float | None,
    ) -> None:
        joint_pos = np.asarray(state.joint_pos, dtype=np.float32)
        stm_target = np.asarray(state.target_joint_pos, dtype=np.float32)
        nx_target = np.asarray(self.stats.last_target_joint_pos, dtype=np.float32)
        record: dict[str, object] = {
            "schema": "se3_nx_recovery_telemetry_v1",
            "obs_contract": OBS_CONTRACT,
            "num_obs": int(self.obs_cfg.num_obs),
            "num_actions": int(self.obs_cfg.num_actions),
            "wall_time_s": time.time(),
            "monotonic_time_s": time.monotonic(),
            "step": int(self.stats.steps),
            "state_seq": int(state.seq),
            "tick_ms": int(state.tick_ms),
            "target_seq": int(state.target_seq),
            "target_age_ms": int(state.target_age_ms),
            "target_valid": int(state.target_valid),
            "rc_switch_r": int(state.rc_switch_r),
            "output_enabled": int(state.output_enabled),
            "flags": int(flags),
            "flag_names": _action_flag_names(flags),
            "policy_inference_ms": None
            if policy_inference_ms is None
            else float(policy_inference_ms),
            "policy_inference_ms_last": float(self.stats.last_policy_inference_ms),
            "policy_inference_ms_avg": float(self._avg_policy_inference_ms()),
            "policy_inference_ms_max": float(self.stats.max_policy_inference_ms),
            "policy_inference_frames": int(self.stats.policy_inference_frames),
            "target_mode": "hold_current"
            if flags & (ACTION_FLAG_OUTPUT_DISABLED_HOLD | ACTION_FLAG_TIMEOUT)
            else "policy",
            "obs": _float_list(obs),
            "action": _float_list(action),
            "clipped_action": _float_list(self._last_action),
            "command": _float_list(self.obs_builder.command_from_state(state)),
            "nx_target_joint_pos": _float_list(nx_target),
            "nx_target_active": _policy_active_angles(nx_target),
            "nx_target_wheel_vel": _float_list(self.stats.last_target_wheel_vel),
            "stm_target_joint_pos": _float_list(stm_target),
            "stm_target_active": _policy_active_angles(stm_target),
            "joint_pos": _float_list(joint_pos),
            "joint_vel": _float_list(state.joint_vel),
            "joint_active": _policy_active_angles(joint_pos),
            "joint_pos_error_nx_target": _float_list(_policy_leg_error(nx_target, joint_pos)),
            "joint_pos_error_stm_target": _float_list(_policy_leg_error(stm_target, joint_pos)),
            "active_error_nx_target": _float_list(
                np.asarray(_policy_active_angles(nx_target), dtype=np.float32)
                - np.asarray(_policy_active_angles(joint_pos), dtype=np.float32)
            ),
            "active_error_stm_target": _float_list(
                np.asarray(_policy_active_angles(stm_target), dtype=np.float32)
                - np.asarray(_policy_active_angles(joint_pos), dtype=np.float32)
            ),
            "wheel_pos": _float_list(state.wheel_pos),
            "wheel_vel": _float_list(state.wheel_vel),
            "hip_torque": _float_list(state.hip_torque),
            "wheel_torque": _float_list(state.wheel_torque),
            "wheel_motor_torque": _float_list(state.wheel_motor_torque),
            "base_ang_vel_body": _float_list(state.base_ang_vel_body),
            "projected_gravity": _float_list(state.projected_gravity),
        }
        self.telemetry.write(self.stats.steps, record)

    def _maybe_print(self) -> None:
        if self.cfg.print_every <= 0 or self.stats.steps % self.cfg.print_every != 0:
            return
        now = time.monotonic()
        interval_s = max(now - self._last_print_monotonic, 1.0e-3)
        total_s = max(now - self._start_monotonic, 1.0e-3)
        recent_fps = float(self.stats.steps - self._last_print_steps) / interval_s
        avg_fps = float(self.stats.steps) / total_s
        self._last_print_monotonic = now
        self._last_print_steps = int(self.stats.steps)
        policy_ms = (
            "--"
            if self.stats.last_step_policy_inference_ms is None
            else f"{self.stats.last_step_policy_inference_ms:.3f}"
        )
        flag_names = _action_flag_names(self.stats.last_action_flags)
        flags_text = ",".join(flag_names) if flag_names else "none"
        target_mode = (
            "hold_current"
            if self.stats.last_action_flags
            & (ACTION_FLAG_OUTPUT_DISABLED_HOLD | ACTION_FLAG_TIMEOUT)
            else "policy"
        )
        action = _fmt_values(self.stats.last_action[:4])
        command = _fmt_values(self.stats.last_command[:5])
        target = _fmt_values(self.stats.last_target_joint_pos)
        stm_target = _fmt_values(self.stats.last_state_target_joint_pos)
        joint = _fmt_values(self.stats.last_state_joint_pos)
        error = _fmt_values(
            _policy_leg_error(
                np.asarray(self.stats.last_state_target_joint_pos, dtype=np.float32),
                np.asarray(self.stats.last_state_joint_pos, dtype=np.float32),
            )
        )
        torque = _fmt_values(self.stats.last_state_hip_torque)
        wheel_torque = _fmt_values(self.stats.last_state_wheel_motor_torque)
        print(
            f"step={self.stats.steps} states={self.stats.state_frames} "
            f"actions={self.stats.action_frames} last_state={self.stats.last_state_seq} "
            f"timeouts={self.stats.timeout_frames} nonfinite={self.stats.nonfinite_frames} "
            f"mode={target_mode} output={self.stats.last_state_output_enabled} "
            f"flags={flags_text} "
            f"fps={recent_fps:.1f}/{avg_fps:.1f} "
            f"policy_ms={policy_ms}/"
            f"{self._avg_policy_inference_ms():.3f}/"
            f"{self.stats.max_policy_inference_ms:.3f} "
            f"policy_n={self.stats.policy_inference_frames} "
            f"cmd5=[{command}] action4=[{action}] "
            f"target4=[{target}] stm_target4=[{stm_target}] "
            f"joint4=[{joint}] err4=[{error}] torque4=[{torque}] "
            f"wheel_motor_torque=[{wheel_torque}]"
        )


def load_policy_runtime(checkpoint: Path, device: str) -> object:
    """按 checkpoint 类型加载 ONNX、NumPy 或 PyTorch actor。"""

    if checkpoint.suffix == ".onnx":
        return OnnxPolicyRuntime(checkpoint)
    if checkpoint.suffix == ".npz":
        return NumpyPolicyRuntime(checkpoint)
    from se3_sim2sim.policy import PolicyRuntime
    from se3_sim2sim.runtime_spec import RuntimeSpec

    return PolicyRuntime(checkpoint=checkpoint, device=device, runtime=RuntimeSpec())


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _telemetry_log_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = value.strip()
    if text.lower() in {"", "0", "false", "none", "off"}:
        return None
    return Path(text)


def _default_cdc_port() -> str:
    by_id_dir = Path("/dev/serial/by-id")
    for pattern in (
        "*STMicroelectronics*Virtual_ComPort*",
        "*STM32*Virtual_ComPort*",
        "*STMicroelectronics*",
    ):
        ports = sorted(by_id_dir.glob(pattern))
        if ports:
            return str(ports[0])
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(Path("/").glob(pattern.lstrip("/")))
        if ports:
            return str(ports[0])
    return "/dev/ttyACM0"


def _resolve_cdc_port(value: object) -> str:
    port = str(value).strip()
    if port.lower() == "auto":
        return _default_cdc_port()
    return port


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recovery-only policy on Jetson Orin NX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RECOVERY_CHECKPOINT,
        help="Recovery checkpoint path (.pt or exported .npz).",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("SE3_CDC_PORT", DEFAULT_CDC_PORT),
        help="USB CDC device path, or auto to prefer STM32 /dev/serial/by-id.",
    )
    parser.add_argument("--baudrate", type=int, default=921600, help="CDC baudrate hint.")
    parser.add_argument(
        "--device", default="cpu", help="Torch device, usually cpu on NX first pass."
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=1.0 / _ROBOT_CFG.control_dt,
        help="Policy send rate. Defaults to RobotConfig.control_dt.",
    )
    parser.add_argument(
        "--state-timeout-s", type=float, default=0.10, help="State freshness timeout."
    )
    parser.add_argument("--write-timeout-s", type=float, default=0.02, help="CDC write timeout.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run policy on synthetic stand state."
    )
    parser.add_argument(
        "--print-every", type=int, default=50, help="Print one status line per N steps."
    )
    parser.add_argument(
        "--telemetry-log",
        type=_telemetry_log_path,
        default=_telemetry_log_path(os.environ.get("SE3_TELEMETRY_LOG")),
        help="Write JSONL telemetry to this file, or create a timestamped file in this directory.",
    )
    parser.add_argument(
        "--telemetry-log-every",
        type=int,
        default=_env_int("SE3_TELEMETRY_LOG_EVERY", 1),
        help="Write one telemetry row per N policy steps.",
    )
    parser.add_argument(
        "--telemetry-flush-every",
        type=int,
        default=_env_int("SE3_TELEMETRY_FLUSH_EVERY", 25),
        help="Flush telemetry after N written rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = RecoveryRuntimeConfig(
        checkpoint=args.checkpoint,
        port=_resolve_cdc_port(args.port),
        baudrate=int(args.baudrate),
        device=str(args.device),
        rate_hz=float(args.rate_hz),
        state_timeout_s=float(args.state_timeout_s),
        write_timeout_s=float(args.write_timeout_s),
        max_steps=int(args.max_steps),
        dry_run=bool(args.dry_run),
        print_every=int(args.print_every),
        telemetry_log=args.telemetry_log,
        telemetry_log_every=int(args.telemetry_log_every),
        telemetry_flush_every=int(args.telemetry_flush_every),
    )
    RecoveryRuntime(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
