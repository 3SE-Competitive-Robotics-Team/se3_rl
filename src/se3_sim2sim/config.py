"""SE3 sim2sim workflow 的运行配置。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import se3_shared
from se3_shared import ActionDelayConfig, Termination

from .course import CourseConfig

ViewerMode = Literal["rerun", "none"]
MAX_YAW_RATE_RAD_S = 4.0 * math.pi

_shared_robot = se3_shared.RobotConfig()
_shared_obs = se3_shared.ObservationConfig()


class YawPidConfig(BaseModel):
    """yaw 轴闭环控制配置。"""

    enabled: bool = True
    target_yaw_rad: float = 0.0
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    max_rate: Annotated[float, Field(gt=0.0, le=MAX_YAW_RATE_RAD_S)] = 3.0

    @model_validator(mode="after")
    def _check_finite(self) -> YawPidConfig:
        for name in ("target_yaw_rad", "kp", "ki", "kd", "max_rate"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        return self


class JumpStateMachineConfig(BaseModel):
    """跳跃参考相位配置，与训练端 JumpCommandTerm 对齐。"""

    trajectory_steps: int = 125
    """一次跳跃参考轨迹的控制步数。"""


class JumpEventConfig(BaseModel):
    """按绝对时间触发的一次跳跃事件。"""

    trigger_time_s: Annotated[float, Field(ge=0.0)]
    """触发时间，单位秒。"""

    target_height: Annotated[float, Field(ge=0.1, le=0.6)]
    """目标跳跃高度，单位米。"""

    @model_validator(mode="after")
    def _check_finite(self) -> JumpEventConfig:
        for name in ("trigger_time_s", "target_height"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        return self


class JumpScheduleConfig(BaseModel):
    """定时跳跃调度配置。"""

    enabled: bool = False
    """是否启用定时跳跃。"""

    interval_s: Annotated[float, Field(gt=0.0)] = 5.0
    """两次跳跃之间的间隔秒数（从上一次参考轨迹结束开始计时）。"""

    target_height: Annotated[float, Field(ge=0.1, le=0.6)] = 0.4
    """目标跳跃高度 (m)，0.1~0.6。"""

    events: tuple[JumpEventConfig, ...] = ()
    """按绝对时间触发的跳跃事件列表。"""

    @model_validator(mode="after")
    def _check_schedule(self) -> JumpScheduleConfig:
        if self.enabled and self.events:
            raise ValueError("jump interval schedule and jump script events cannot both be enabled")
        last_time = -math.inf
        for event in self.events:
            if event.trigger_time_s <= last_time:
                raise ValueError("jump script event times must be strictly increasing")
            last_time = event.trigger_time_s
        return self


class RobotConfig(BaseModel):
    """sim2sim 机器人运行配置。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Path = Path("assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml")
    task: str = "wheel_legged_joint_pos"
    seed: int = 0
    sim_dt: Annotated[float, Field(gt=0.0)] = _shared_robot.sim_dt
    control_decimation: Annotated[int, Field(ge=1)] = _shared_robot.control_decimation
    base_height: float = _shared_robot.default_base_height
    command: tuple[float, float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        _shared_robot.default_base_height,
        0.0,
        0.2,
        0.0,
    )
    """8 维指令: [vx, ωz, pitch, roll, height, jump_flag, jump_target_height, jump_phase]。
    jump_phase 由 workflow 内部维护（sim2sim 初始为 0，跳跃期间自动推进）。
    """
    command_scale: tuple[float, ...] = _shared_obs.command_scale
    default_dof_pos: tuple[float, ...] = _shared_robot.default_dof_pos
    action_scale: tuple[float, ...] = _shared_robot.action_scale
    torque_limits: tuple[float, ...] = _shared_robot.torque_limits
    leg_kp: float = _shared_robot.leg_kp
    leg_kd: float = _shared_robot.leg_kd
    wheel_kd: float = _shared_robot.wheel_kd
    yaw_pid: YawPidConfig = Field(default_factory=YawPidConfig)
    action_delay: ActionDelayConfig = Field(
        default_factory=lambda: _shared_robot.action_delay.model_copy()
    )
    action_delay_steps: int | None = None
    """兼容旧 CLI 的固定步数延迟入口; 新配置使用 action_delay。"""
    jump_state_machine: JumpStateMachineConfig = Field(default_factory=JumpStateMachineConfig)
    """跳跃参考相位参数，与训练端 JumpCommandTerm 对齐。"""
    jump_schedule: JumpScheduleConfig = Field(default_factory=JumpScheduleConfig)
    """定时跳跃调度配置。"""


class PolicyConfig(BaseModel):
    """策略 checkpoint 配置。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoint: Path | None = None
    device: str = "cpu"


class ViewerConfig(BaseModel):
    """可视化配置。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: ViewerMode = "rerun"
    app_id: str = "se3_sim2sim"
    spawn: bool = True
    address: str | None = None
    record_to_rrd: Path | None = None
    memory_limit: str = "1GB"
    log_every: int = 1
    follow_body: str = "base_link"


class RunConfig(BaseModel):
    """sim2sim 完整运行配置。

    所有子配置均为 BaseModel，model_dump(mode='json') 可直接序列化整棵树，
    Path 自动转为字符串，不再需要手动 asdict / _stringify_paths。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    robot: RobotConfig = Field(default_factory=RobotConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    viewer: ViewerConfig = Field(default_factory=ViewerConfig)
    max_steps: int = 0
    fixed_reset: bool = True
    randomize_root: bool = False
    print_every: int = 100
    print_debug: bool = False
    json_output: Path | None = None
    course: CourseConfig = Field(default_factory=CourseConfig)
    termination: Termination = Field(default_factory=Termination)
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12

    def resolved(self, root: Path | None = None) -> RunConfig:
        """原地解析所有路径，返回 self（供链式调用）。"""
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
        """序列化为纯 JSON 兼容字典（Path → str，嵌套 BaseModel 递归展开）。"""
        return self.model_dump(mode="json")

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
