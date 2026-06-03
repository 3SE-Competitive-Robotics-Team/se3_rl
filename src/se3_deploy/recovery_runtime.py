"""Jetson Orin NX 上的 recovery-only policy runtime。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from se3_shared import ObservationConfig

from .cdc import CdcSerial
from .numpy_policy import NumpyPolicyRuntime
from .observation import RecoveryObservationBuilder, synthetic_recovery_state
from .protocol import (
    ACTION_FLAG_DRY_RUN,
    ACTION_FLAG_NONFINITE,
    ACTION_FLAG_TIMEOUT,
    MODE_RECOVERY_ONLY,
    MSG_POLICY_STATE,
    PolicyActionFrame,
    PolicyStateFrame,
    StreamParser,
    decode_policy_state,
    pack_policy_action,
)

DEFAULT_RECOVERY_CHECKPOINT = Path("logs/rsl_rl/se3_wheel_leg/2026-05-31_14-08-23/model_2999.pt")


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


class RecoveryRuntime:
    """只运行 recovery 网络的 50 Hz 真机 runtime。"""

    def __init__(self, cfg: RecoveryRuntimeConfig) -> None:
        self.cfg = cfg
        self.obs_cfg = ObservationConfig()
        self.policy = load_policy_runtime(cfg.checkpoint, cfg.device)
        self.policy.reset()
        self.obs_builder = RecoveryObservationBuilder()
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

                packet = self._make_action_packet(latest_state, action, flags)
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

    def _make_action_packet(self, state: PolicyStateFrame, action: np.ndarray, flags: int) -> bytes:
        frame = PolicyActionFrame(
            seq=self._action_seq,
            source_state_seq=state.seq,
            timestamp_us=monotonic_us(),
            mode=MODE_RECOVERY_ONLY,
            flags=int(flags),
            action=tuple(float(v) for v in action),
        )
        self._action_seq = (self._action_seq + 1) & 0xFFFFFFFF
        return pack_policy_action(frame)

    def _record_action(self, state: PolicyStateFrame, action: np.ndarray, flags: int) -> None:
        self.stats.steps += 1
        self.stats.action_frames += 1
        self.stats.last_state_seq = state.seq
        self._last_action = action.astype(np.float32, copy=True)
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


def monotonic_us() -> int:
    return (time.monotonic_ns() // 1000) & 0xFFFFFFFF


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
        help="Recovery checkpoint path (.pt or exported .npz). Defaults to the phase-1 model_2999.pt.",
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="USB CDC device path.")
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
