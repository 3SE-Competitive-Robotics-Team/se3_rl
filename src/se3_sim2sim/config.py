"""SE3 sim2sim workflow 的运行配置。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import se3_shared
from se3_shared import ActionDelayConfig, Termination

ViewerMode = Literal["rerun", "none"]
MAX_YAW_RATE_RAD_S = 4.0 * math.pi

_shared_robot = se3_shared.RobotConfig()
_shared_obs = se3_shared.ObservationConfig()


@dataclass(slots=True)
class YawPidConfig:
    """yaw 轴闭环控制配置。"""

    enabled: bool = True
    target_yaw_rad: float = 0.0
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    max_rate: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_yaw_rad):
            raise ValueError(f"target_yaw_rad must be finite, got {self.target_yaw_rad}")
        if not math.isfinite(self.kp):
            raise ValueError(f"kp must be finite, got {self.kp}")
        if not math.isfinite(self.ki):
            raise ValueError(f"ki must be finite, got {self.ki}")
        if not math.isfinite(self.kd):
            raise ValueError(f"kd must be finite, got {self.kd}")
        if not math.isfinite(self.max_rate):
            raise ValueError(f"max_rate must be finite, got {self.max_rate}")
        if self.max_rate <= 0.0:
            raise ValueError(f"max_rate must be positive, got {self.max_rate}")
        if self.max_rate > MAX_YAW_RATE_RAD_S:
            raise ValueError(f"max_rate must be <= {MAX_YAW_RATE_RAD_S}, got {self.max_rate}")


@dataclass(slots=True)
class RobotConfig:
    model_path: Path = Path("assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml")
    task: str = "wheel_legged_joint_pos"
    seed: int = 0
    sim_dt: float = _shared_robot.sim_dt
    control_decimation: int = _shared_robot.control_decimation
    base_height: float = 0.301
    command: tuple[float, float, float, float, float] = (0.5, 0.0, 0.0, 0.0, 0.28)
    command_scale: tuple[float, ...] = _shared_obs.command_scale
    default_dof_pos: tuple[float, ...] = _shared_robot.default_dof_pos
    action_scale: tuple[float, ...] = _shared_robot.action_scale
    torque_limits: tuple[float, ...] = _shared_robot.torque_limits
    leg_kp: float = _shared_robot.leg_kp
    leg_kd: float = _shared_robot.leg_kd
    wheel_kd: float = _shared_robot.wheel_kd
    yaw_pid: YawPidConfig = field(default_factory=YawPidConfig)
    action_delay: ActionDelayConfig = field(
        default_factory=lambda: _shared_robot.action_delay.model_copy()
    )
    action_delay_steps: int | None = None
    """兼容旧 CLI 的固定步数延迟入口; 新配置使用 action_delay。"""


@dataclass(slots=True)
class PolicyConfig:
    checkpoint: Path | None = None
    device: str = "cpu"


@dataclass(slots=True)
class ViewerConfig:
    mode: ViewerMode = "rerun"
    app_id: str = "se3_sim2sim"
    spawn: bool = True
    address: str | None = None
    record_to_rrd: Path | None = None
    log_every: int = 1
    follow_body: str = "base_link"


@dataclass(slots=True)
class RunConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    max_steps: int = 0
    fixed_reset: bool = True
    randomize_root: bool = False
    print_every: int = 100
    print_debug: bool = False
    json_output: Path | None = None
    termination: Termination = field(default_factory=Termination)
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12

    def resolved(self, root: Path | None = None) -> RunConfig:
        base = Path.cwd() if root is None else Path(root)
        self.robot.model_path = _resolve_path(base, self.robot.model_path)
        self._resolve_legacy_action_delay_steps()
        self.policy.checkpoint = (
            _latest_checkpoint(base)
            if self.policy.checkpoint is None
            else _resolve_path(base, self.policy.checkpoint)
        )
        if self.viewer.record_to_rrd is not None:
            self.viewer.record_to_rrd = _resolve_path(base, self.viewer.record_to_rrd)
        if self.json_output is not None:
            self.json_output = _resolve_path(base, self.json_output)
        self.termination = Termination(
            terminate_on_fall=self.terminate_on_fall,
            fail_tilt_deg=self.fail_tilt_deg,
            fail_height_m=self.fail_height_m,
        )
        return self

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "robot": asdict(self.robot),
            "policy": asdict(self.policy),
            "viewer": asdict(self.viewer),
            "max_steps": self.max_steps,
            "fixed_reset": self.fixed_reset,
            "randomize_root": self.randomize_root,
            "print_every": self.print_every,
            "print_debug": self.print_debug,
            "json_output": self.json_output,
            "termination": self.termination.model_dump(),
        }
        return _stringify_paths(payload)

    def _resolve_legacy_action_delay_steps(self) -> None:
        if self.robot.action_delay_steps is None:
            return
        delay_steps = max(0, int(self.robot.action_delay_steps))
        delay_s = delay_steps * float(self.robot.sim_dt)
        self.robot.action_delay = ActionDelayConfig(
            enabled=delay_steps > 0,
            delay_s=delay_s,
            randomize=False,
            min_delay_s=delay_s,
            max_delay_s=delay_s,
        )


def _resolve_path(base: Path, path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _latest_checkpoint(base: Path) -> Path:
    root = base / "logs" / "rsl_rl" / "se3_wheel_leg"
    runs = (
        [run for run in root.iterdir() if run.is_dir() and any(run.glob("model_*.pt"))]
        if root.exists()
        else []
    )
    if not runs:
        raise FileNotFoundError(
            "未找到 checkpoint, 请使用 --checkpoint 指定 logs/rsl_rl/se3_wheel_leg/<timestamp>/model_*.pt"
        )
    latest_run = max(runs, key=lambda path: (path.stat().st_mtime, path.name))
    candidates = list(latest_run.glob("model_*.pt"))
    return max(candidates, key=_checkpoint_iteration).resolve()


def _checkpoint_iteration(path: Path) -> int:
    stem = path.stem
    prefix = "model_"
    if not stem.startswith(prefix):
        return -1
    try:
        return int(stem.removeprefix(prefix))
    except ValueError:
        return -1


def _stringify_paths(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _stringify_paths(value.model_dump())
    if isinstance(value, dict):
        return {k: _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(v) for v in value]
    if isinstance(value, tuple):
        return [_stringify_paths(v) for v in value]
    return value
