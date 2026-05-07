"""SE3 sim2sim workflow 的运行配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ViewerMode = Literal["rerun", "none"]


@dataclass(slots=True)
class RobotConfig:
    model_path: Path = Path("assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml")
    task: str = "wheel_legged_joint_pos"
    seed: int = 0
    sim_dt: float = 0.005
    control_decimation: int = 4
    base_height: float = 0.28
    command: tuple[float, float, float, float, float] = (0.5, 0.0, 0.0, 0.0, 0.28)
    command_scale: tuple[float, float, float, float, float] = (2.0, 0.25, 5.0, 5.0, 5.0)
    default_dof_pos: tuple[float, ...] = (0.6171, 0.2070, 0.0, 0.6171, 0.2070, 0.0)
    action_scale: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25, 20.0, 20.0)
    torque_limits: tuple[float, ...] = (30.0, 30.0, 3.3, 30.0, 30.0, 3.3)
    leg_kp: float = 40.0
    leg_kd: float = 2.0
    wheel_kd: float = 0.5
    action_delay_steps: int = 2
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12


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
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12

    def resolved(self, root: Path | None = None) -> RunConfig:
        base = Path.cwd() if root is None else Path(root)
        self.robot.model_path = _resolve_path(base, self.robot.model_path)
        self.policy.checkpoint = (
            _latest_checkpoint(base)
            if self.policy.checkpoint is None
            else _resolve_path(base, self.policy.checkpoint)
        )
        if self.viewer.record_to_rrd is not None:
            self.viewer.record_to_rrd = _resolve_path(base, self.viewer.record_to_rrd)
        if self.json_output is not None:
            self.json_output = _resolve_path(base, self.json_output)
        self.robot.terminate_on_fall = bool(self.terminate_on_fall)
        self.robot.fail_tilt_deg = float(self.fail_tilt_deg)
        self.robot.fail_height_m = float(self.fail_height_m)
        return self

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return _stringify_paths(payload)


def _resolve_path(base: Path, path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _latest_checkpoint(base: Path) -> Path:
    root = base / "logs" / "rsl_rl" / "se3_wheel_leg"
    candidates = list(root.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "未找到 checkpoint, 请使用 --checkpoint 指定 logs/rsl_rl/se3_wheel_leg/<timestamp>/model_*.pt"
        )
    return max(
        candidates, key=lambda path: (_checkpoint_iteration(path), path.parent.name)
    ).resolve()


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
    if isinstance(value, dict):
        return {k: _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(v) for v in value]
    if isinstance(value, tuple):
        return [_stringify_paths(v) for v in value]
    return value
