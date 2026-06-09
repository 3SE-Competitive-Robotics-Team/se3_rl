"""Jetson Orin NX 上的 recovery-only policy runtime。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from se3_shared import (
    Joint,
    JointGroup,
    ObservationConfig,
    RobotConfig,
)

from .cdc import CdcSerial
from .numpy_policy import NumpyPolicyRuntime
from .observation import RecoveryObservationBuilder, synthetic_recovery_state
from .protocol import (
    MSG_POLICY_STATE,
    PolicyStateFrame,
    PolicyTargetFrame,
    StreamParser,
    decode_policy_state,
    pack_policy_target,
)

DEFAULT_RECOVERY_CHECKPOINT = Path("logs/deploy/model_5999_recovery_gru.npz")
ACTION_FLAG_DRY_RUN = 1 << 0
ACTION_FLAG_TIMEOUT = 1 << 1
ACTION_FLAG_NONFINITE = 1 << 2
_HEIGHT_LUT_SIZE = 1024
_WHEEL_RADIUS = 0.06
_BASE_COM_X = -0.01780372
_LF1_BODY_XZ = (-0.12990117, 0.04639203)
_LF1_JOINT_XZ = (-0.05003347, -0.04149627)
_WHEEL_BODY_XZ = (-0.15699, -0.21049)
_KNEE_X = -0.17993464
_KNEE_Z = 0.00489576
_CALF_X = 0.05003347
_CALF_Z = 0.04149627
_DRIVE_X = 0.04009536
_DRIVE_Z = 0.04530576
_COUPLER_LEN = float(np.hypot(-0.16999653, 0.00108627))
_CALF_LEN = float(np.hypot(_CALF_X, _CALF_Z))
_CALF_ZERO_ANGLE = float(np.arctan2(_CALF_Z, _CALF_X))


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


@dataclass(slots=True)
class RuntimeStats:
    steps: int = 0
    state_frames: int = 0
    action_frames: int = 0
    timeout_frames: int = 0
    nonfinite_frames: int = 0
    last_state_seq: int = 0
    last_action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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
        self.leg_scales = np.asarray(self.robot_cfg.action_scale, dtype=np.float32)[
            JointGroup.LEG_ACTUATORS
        ]
        self.wheel_scale = float(self.robot_cfg.action_scale[Joint.L_WHEEL])
        self.action_clip = self.robot_cfg.action_clip
        self.active_coeffs = np.asarray(
            self.robot_cfg.active_rod_angle_coeffs,
            dtype=np.float32,
        )
        lower, upper = self.robot_cfg.active_rod_angle_limits
        self.active_lower = float(lower) - float(self.robot_cfg.active_rod_lower_target_overdrive)
        self.active_upper = float(upper)
        self.policy_default = (
            _policy_default_from_height_np(
                self.command_height,
                self.robot_cfg,
            )
            .astype(np.float32, copy=False)
            .reshape(4)
        )

    def decode(self, action: np.ndarray) -> DecodedPolicyTarget:
        raw = self.clip_action(action)

        policy_default = self.policy_default

        target = np.empty(4, dtype=np.float32)
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = self.active_coeffs[side_idx]
            front_target = policy_default[front_idx] + raw[front_idx] * self.leg_scales[front_idx]
            active_default = (
                front_coef * policy_default[front_idx] + back_coef * policy_default[back_idx]
            )
            active_raw = active_default + raw[back_idx] * self.leg_scales[back_idx]
            active_target = np.clip(active_raw, self.active_lower, self.active_upper)
            target[front_idx] = front_target
            target[back_idx] = (active_target - front_coef * front_target) / back_coef

        wheel_vel = raw[JointGroup.CTRL_WHEELS] * self.wheel_scale
        return DecodedPolicyTarget(
            joint_pos=tuple(float(v) for v in target),
            wheel_vel=tuple(float(v) for v in wheel_vel),
        )

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32).reshape(6)
        if self.action_clip is None:
            return raw.astype(np.float32, copy=True)
        clip = float(self.action_clip)
        return np.clip(raw, -clip, clip).astype(np.float32, copy=False)


def _policy_default_from_height_np(command_height: float, cfg: RobotConfig) -> np.ndarray:
    height = np.asarray(command_height, dtype=np.float64).reshape(-1)
    active_by_length, length_grid, active_grid, vec_x_grid, vec_z_grid = _height_default_lut_np(cfg)
    target_x = np.full_like(height, _BASE_COM_X)
    target_z = _WHEEL_RADIUS - height
    target_length = np.clip(
        np.hypot(target_x, target_z),
        length_grid[0],
        length_grid[-1],
    )
    active = np.interp(target_length, length_grid, active_by_length)
    vec_x = np.interp(active, active_grid, vec_x_grid)
    vec_z = np.interp(active, active_grid, vec_z_grid)
    lf = np.arctan2(vec_x, -vec_z) - np.arctan2(target_x, -target_z)
    rf = -lf
    lb = lf - active
    rb = rf + active
    return np.stack((lf, lb, rf, rb), axis=1)


def _height_default_lut_np(
    cfg: RobotConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = cfg.active_rod_angle_limits
    active = np.linspace(float(lower), float(upper), _HEIGHT_LUT_SIZE, dtype=np.float64)
    output_knee = _output_knee_from_active_angle_np_array(active, float(lower), float(upper))
    vec_x, vec_z = _leg_vector_np(output_knee)
    length = np.hypot(vec_x, vec_z)
    order = np.argsort(length)
    return active[order], length[order], active, vec_x, vec_z


def _output_knee_from_active_angle_np_array(
    active_angle: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    alpha = np.clip(np.asarray(active_angle, dtype=np.float64), lower, upper)
    beta = -alpha
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    px = cos_b * _DRIVE_X + sin_b * _DRIVE_Z
    pz = -sin_b * _DRIVE_X + cos_b * _DRIVE_Z

    dx = px - _KNEE_X
    dz = pz - _KNEE_Z
    dist = np.sqrt(np.maximum(dx * dx + dz * dz, 1.0e-12))
    ex = dx / dist
    ez = dz / dist

    along = (_CALF_LEN**2 - _COUPLER_LEN**2 + dist * dist) / (2.0 * dist)
    height = np.sqrt(np.maximum(_CALF_LEN**2 - along * along, 0.0))
    cx = _KNEE_X + along * ex - height * ez
    cz = _KNEE_Z + along * ez + height * ex

    phi = np.arctan2(cz - _KNEE_Z, cx - _KNEE_X)
    return _wrap_angle_np_array(_CALF_ZERO_ANGLE - phi)


def _leg_vector_np(output_knee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body = np.asarray(_LF1_BODY_XZ, dtype=np.float64)
    joint = np.asarray(_LF1_JOINT_XZ, dtype=np.float64)
    wheel = np.asarray(_WHEEL_BODY_XZ, dtype=np.float64)
    cos_q = np.cos(output_knee)
    sin_q = np.sin(output_knee)
    rot_joint_x = cos_q * joint[0] + sin_q * joint[1]
    rot_joint_z = -sin_q * joint[0] + cos_q * joint[1]
    rot_wheel_x = cos_q * wheel[0] + sin_q * wheel[1]
    rot_wheel_z = -sin_q * wheel[0] + cos_q * wheel[1]
    x = body[0] + joint[0] - rot_joint_x + rot_wheel_x
    z = body[1] + joint[1] - rot_joint_z + rot_wheel_z
    return x, z


def _wrap_angle_np_array(angle: np.ndarray) -> np.ndarray:
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


class RecoveryRuntime:
    """只运行 recovery 网络的 50 Hz 真机 runtime。"""

    def __init__(self, cfg: RecoveryRuntimeConfig) -> None:
        self.cfg = cfg
        self.obs_cfg = ObservationConfig()
        self.policy = load_policy_runtime(cfg.checkpoint, cfg.device)
        self.policy.reset()
        self.obs_builder = RecoveryObservationBuilder()
        self.target_decoder = RecoveryActionTargetDecoder(
            command_height=float(self.obs_builder.command[4])
        )
        self.stats = RuntimeStats()
        self._last_action = np.zeros(self.obs_cfg.num_actions, dtype=np.float32)
        self._action_seq = 0

    def run(self) -> RuntimeStats:
        print(
            "NX recovery runtime: "
            f"checkpoint={self.cfg.checkpoint} iter={self.policy.iteration} "
            f"type={self.policy.policy_type} device={self.cfg.device}"
        )
        if self.cfg.dry_run:
            self._run_dry()
        else:
            self._run_cdc()
        return self.stats

    def _run_dry(self) -> None:
        max_steps = self.cfg.max_steps if self.cfg.max_steps > 0 else int(self.cfg.rate_hz * 2.0)
        for step in range(max_steps):
            state = synthetic_recovery_state(seq=step)
            self.stats.state_frames += 1
            action, flags = self._act_from_state(state)
            flags |= ACTION_FLAG_DRY_RUN
            self._record_action(state, action, flags)
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
                if age_s > self.cfg.state_timeout_s:
                    action = np.zeros(self.obs_cfg.num_actions, dtype=np.float32)
                    flags |= ACTION_FLAG_TIMEOUT
                else:
                    action, flags = self._act_from_state(latest_state)

                packet = self._make_target_packet(action)
                serial.write_all(packet, timeout_s=self.cfg.write_timeout_s)
                self._record_action(latest_state, action, flags)
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

    def _act_from_state(self, state: PolicyStateFrame) -> tuple[np.ndarray, int]:
        result = self.obs_builder.build(state, self._last_action)
        action = self.policy.act(result.obs)
        flags = ACTION_FLAG_NONFINITE if result.had_nonfinite_input else 0
        if not np.isfinite(action).all():
            flags |= ACTION_FLAG_NONFINITE
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if flags & ACTION_FLAG_NONFINITE:
            self.stats.nonfinite_frames += 1
        return action.astype(np.float32, copy=False), flags

    def _make_target_packet(self, action: np.ndarray) -> bytes:
        target = self.target_decoder.decode(action)
        frame = PolicyTargetFrame(
            seq=self._action_seq,
            joint_pos=target.joint_pos,
            wheel_vel=target.wheel_vel,
        )
        self._action_seq = (self._action_seq + 1) & 0xFFFFFFFF
        return pack_policy_target(frame)

    def _record_action(self, state: PolicyStateFrame, action: np.ndarray, flags: int) -> None:
        self.stats.steps += 1
        self.stats.action_frames += 1
        self.stats.last_state_seq = state.seq
        self._last_action = self.target_decoder.clip_action(action)
        self.stats.last_action = tuple(float(v) for v in self._last_action)
        if flags & ACTION_FLAG_TIMEOUT:
            self.stats.timeout_frames += 1

    def _maybe_print(self) -> None:
        if self.cfg.print_every <= 0 or self.stats.steps % self.cfg.print_every != 0:
            return
        action = " ".join(f"{v:+.3f}" for v in self.stats.last_action)
        print(
            f"step={self.stats.steps} states={self.stats.state_frames} "
            f"actions={self.stats.action_frames} last_state={self.stats.last_state_seq} "
            f"timeouts={self.stats.timeout_frames} nonfinite={self.stats.nonfinite_frames} "
            f"action=[{action}]"
        )


def load_policy_runtime(checkpoint: Path, device: str) -> object:
    """按 checkpoint 类型加载 PyTorch 或 NumPy actor。"""

    if checkpoint.suffix == ".npz":
        return NumpyPolicyRuntime(checkpoint)
    from se3_sim2sim.policy import PolicyRuntime
    from se3_sim2sim.runtime_spec import RuntimeSpec

    return PolicyRuntime(checkpoint=checkpoint, device=device, runtime=RuntimeSpec())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recovery-only policy on Jetson Orin NX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RECOVERY_CHECKPOINT,
        help="Recovery checkpoint path (.pt or exported .npz).",
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB CDC device path.")
    parser.add_argument("--baudrate", type=int, default=921600, help="CDC baudrate hint.")
    parser.add_argument(
        "--device", default="cpu", help="Torch device, usually cpu on NX first pass."
    )
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Policy send rate.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = RecoveryRuntimeConfig(
        checkpoint=args.checkpoint,
        port=str(args.port),
        baudrate=int(args.baudrate),
        device=str(args.device),
        rate_hz=float(args.rate_hz),
        state_timeout_s=float(args.state_timeout_s),
        write_timeout_s=float(args.write_timeout_s),
        max_steps=int(args.max_steps),
        dry_run=bool(args.dry_run),
        print_every=int(args.print_every),
    )
    RecoveryRuntime(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
