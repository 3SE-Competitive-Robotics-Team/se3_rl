"""SE3 sim2sim workflow 的运行配置。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import se3_shared
from se3_shared import ActionDelayConfig, Termination

from .course import CourseConfig

ViewerMode = Literal["rerun", "mujoco", "viser", "none"]
RerunGeomView = Literal["visual", "collision", "both"]
SimModelVariant = Literal["fourbar-surrogate", "closedchain", "openchain"]
RecoveryPose = Literal["standing", "left_side", "right_side", "prone", "supine"]
RcOffMode = Literal["no-torque", "hold-current"]
CustomTerrain = Literal["none", "gap-ramp-facility", "slope-17"]
RoughTerrainType = Literal[
    "mixed",
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave_terrain",
]
MAX_YAW_RATE_RAD_S = 4.0 * math.pi
RECOVERY_COMMAND_HEIGHT_M = se3_shared.RECOVERY_COMMAND_HEIGHT_M
RECOVERY_POSE_CHOICES: tuple[RecoveryPose, ...] = (
    "standing",
    "left_side",
    "right_side",
    "prone",
    "supine",
)
ROUGH_TERRAIN_TYPE_CHOICES: tuple[RoughTerrainType, ...] = (
    "mixed",
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave_terrain",
)
CUSTOM_TERRAIN_CHOICES: tuple[CustomTerrain, ...] = (
    "none",
    "gap-ramp-facility",
    "slope-17",
)
RECOVERY_POSE_RP_RAD: dict[RecoveryPose, tuple[float, float]] = {
    "standing": (0.0, 0.0),
    "left_side": (0.5 * math.pi, 0.0),
    "right_side": (-0.5 * math.pi, 0.0),
    "prone": (0.0, math.pi),
    "supine": (0.0, -math.pi),
}

_shared_robot = se3_shared.RobotConfig()
_shared_obs = se3_shared.ObservationConfig()
_MJCF_DIR = Path("assets/robots/serialleg/mjcf")

DEFAULT_SIM_MODEL_VARIANT: SimModelVariant = "closedchain"
SIM_MODEL_VARIANT_CHOICES: tuple[SimModelVariant, ...] = (
    "fourbar-surrogate",
    "closedchain",
    "openchain",
)
SIM_MODEL_VARIANT_PATHS: dict[SimModelVariant, Path] = {
    "fourbar-surrogate": _MJCF_DIR / "serialleg_fourbar_surrogate_train.xml",
    "closedchain": _MJCF_DIR / "serialleg_closed_chain_v3_train_obb_trim.xml",
    "openchain": _MJCF_DIR / "serialleg_fidelity_cylinder_wheels.xml",
}
_SIM_MODEL_VARIANT_ALIASES: dict[str, SimModelVariant] = {
    "default": "closedchain",
    "fourbar": "fourbar-surrogate",
    "fourbar-surrogate": "fourbar-surrogate",
    "fourbar_surrogate": "fourbar-surrogate",
    "surrogate": "fourbar-surrogate",
    "equivalent-openchain": "fourbar-surrogate",
    "closedchain": "closedchain",
    "closed-chain": "closedchain",
    "closedchain-obb": "closedchain",
    "closedchain_obb": "closedchain",
    "no-spring": "closedchain",
    "no_spring": "closedchain",
    "openchain": "openchain",
    "open-chain": "openchain",
}


def normalize_model_variant(value: str) -> SimModelVariant:
    """规范化 sim2sim 模型变体名称。"""
    key = value.strip().lower()
    try:
        return _SIM_MODEL_VARIANT_ALIASES[key]
    except KeyError as exc:
        allowed = "/".join(SIM_MODEL_VARIANT_CHOICES)
        raise ValueError(f"不支持的 sim2sim 模型变体 {value!r}；可选 {allowed}") from exc


def model_path_for_variant(value: str) -> Path:
    """返回指定 sim2sim 模型变体对应的 MJCF 路径。"""
    return SIM_MODEL_VARIANT_PATHS[normalize_model_variant(value)]


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


class RcSwitchEventConfig(BaseModel):
    """遥控器输出使能切换事件。"""

    trigger_time_s: Annotated[float, Field(ge=0.0)]
    """触发时间，单位秒。"""

    output_enabled: bool
    """是否允许 policy 输出真正接管机器人。"""

    @model_validator(mode="after")
    def _check_finite(self) -> RcSwitchEventConfig:
        if not math.isfinite(self.trigger_time_s):
            raise ValueError(f"trigger_time_s must be finite, got {self.trigger_time_s}")
        return self


class RcSwitchScheduleConfig(BaseModel):
    """sim2sim 中模拟遥控器开关 / output enable 的时间表。"""

    initial_output_enabled: bool = True
    """仿真开始时是否允许 policy 输出。"""

    off_mode: RcOffMode = "no-torque"
    """output disabled 时的物理语义；真机当前为 no-torque。"""

    events: tuple[RcSwitchEventConfig, ...] = ()
    """按绝对时间切换 output enable 的事件列表。"""

    @model_validator(mode="after")
    def _check_schedule(self) -> RcSwitchScheduleConfig:
        last_time = -math.inf
        for event in self.events:
            if event.trigger_time_s <= last_time:
                raise ValueError("rc switch event times must be strictly increasing")
            last_time = event.trigger_time_s
        return self


class StairCtbcConfig(BaseModel):
    """sim2sim 台阶 CTBC 前馈配置，与训练端默认值保持一致。"""

    enabled: bool = False
    cartesian_frame: Literal["body"] = "body"
    coordinate_mode: Literal["body_polar", "body_cartesian"] = "body_polar"
    trigger_mode: Literal["force", "pitch"] = "pitch"
    contact_window: Annotated[int, Field(ge=1)] = 3
    force_threshold_n: Annotated[float, Field(ge=0.0)] = 10.0
    pitch_threshold_rad: Annotated[float, Field(ge=0.0)] = math.radians(6.0)
    pitch_threshold_deg: Annotated[float, Field(ge=0.0)] = 6.0
    pitch_window: Annotated[int, Field(ge=1)] = 3
    ff_amplitude_rad: float = 0.0
    leg_length_m: Annotated[float, Field(gt=0.0)] = 0.18
    swing_angle_rad: float = math.radians(-35.0)
    swing_angle_deg: float = -35.0
    body_x_m: float = 0.20
    body_z_m: float = 0.15
    ff_x_m: float = 0.20
    ff_lift_m: float = 0.15
    ff_duration_s: Annotated[float, Field(gt=0.0)] = 0.60
    ff_period_s: Annotated[float, Field(gt=0.0)] = 0.60
    ff_rise_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    ff_hold_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    ff_wheel_action: float = 0.0
    profile_path: Path | None = None
    ann_start_iter: Annotated[int, Field(ge=0)] = 200
    ann_end_iter: Annotated[int, Field(ge=0)] = 500
    fixed_iter: Annotated[int, Field(ge=0)] | None = None
    obs_scale: Annotated[float, Field(gt=0.0)] = 0.01
    allow_bilateral_trigger: bool = False

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_cartesian_aliases(cls, data: object) -> object:
        """把旧 ff_x/ff_lift 字段升级成显式 body-frame 字段。"""

        if isinstance(data, dict):
            upgraded = dict(data)
            if "body_x_m" not in upgraded and "ff_x_m" in upgraded:
                upgraded["body_x_m"] = upgraded["ff_x_m"]
            if "body_z_m" not in upgraded and "ff_lift_m" in upgraded:
                upgraded["body_z_m"] = upgraded["ff_lift_m"]
            if "swing_angle_rad" not in upgraded and "swing_angle_deg" in upgraded:
                upgraded["swing_angle_rad"] = math.radians(float(upgraded["swing_angle_deg"]))
            if "pitch_threshold_rad" not in upgraded and "pitch_threshold_deg" in upgraded:
                upgraded["pitch_threshold_rad"] = math.radians(
                    float(upgraded["pitch_threshold_deg"])
                )
            has_legacy_cartesian = any(
                key in upgraded for key in ("body_x_m", "body_z_m", "ff_x_m", "ff_lift_m")
            )
            has_polar = any(key in upgraded for key in ("leg_length_m", "swing_angle_rad"))
            if "coordinate_mode" not in upgraded and has_legacy_cartesian and not has_polar:
                upgraded["coordinate_mode"] = "body_cartesian"
            if "ff_duration_s" not in upgraded and "duration_s" in upgraded:
                upgraded["ff_duration_s"] = upgraded["duration_s"]
            if "ff_duration_s" not in upgraded and "ff_period_s" in upgraded:
                upgraded["ff_duration_s"] = upgraded["ff_period_s"]
            return upgraded
        return data

    @model_validator(mode="after")
    def _sync_body_cartesian_aliases(self) -> StairCtbcConfig:
        """保持新 body 字段和旧 ff 字段一致，旧字段只作为兼容入口。"""

        self.ff_x_m = float(self.body_x_m)
        self.ff_lift_m = float(self.body_z_m)
        self.ff_period_s = float(self.ff_duration_s)
        self.swing_angle_deg = math.degrees(float(self.swing_angle_rad))
        self.pitch_threshold_deg = math.degrees(float(self.pitch_threshold_rad))
        return self


class RobotConfig(BaseModel):
    """sim2sim 机器人运行配置。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Path = model_path_for_variant(DEFAULT_SIM_MODEL_VARIANT)
    task: str = "wheel_legged_joint_pos"
    seed: int = 0
    stair_terrain: bool = False
    """是否在原生 MuJoCo sim2sim 中添加台阶碰撞地形。"""
    stair_terrain_level: Annotated[int, Field(ge=0, le=9)] = 0
    """台阶课程等级，0 对应 5cm，9 对应 20cm。"""
    stair_step_height_range: tuple[float, float] = (0.05, 0.20)
    """台阶高度范围，与训练端 stair terrain 课程保持一致。"""
    stair_step_count: Annotated[int, Field(ge=1, le=20)] = 6
    stair_step_depth_m: Annotated[float, Field(gt=0.0)] = 0.8
    stair_start_x_m: float = 1.0
    stair_half_width_m: Annotated[float, Field(gt=0.0)] = 6.0
    stair_ctbc: StairCtbcConfig = Field(default_factory=StairCtbcConfig)
    """原生 MuJoCo 台阶值守时使用的 CTBC 前馈注入器。"""
    rough_terrain: bool = False
    """是否在原生 MuJoCo sim2sim 中添加 MJLab rough terrain。"""
    rough_terrain_type: RoughTerrainType = "mixed"
    """MJLab ROUGH_TERRAINS_CFG 的完整混合地形，或指定单个 sub-terrain 类型。"""
    rough_terrain_origin_type: RoughTerrainType | None = None
    """mixed rough grid 中放到机器人脚下的 terrain column；None 时使用中间列。"""
    rough_terrain_level: Annotated[int, Field(ge=0, le=9)] = 1
    """rough terrain 等级；按 MJLab terrain row difficulty 映射到 0-1。"""
    rough_terrain_size_m: tuple[float, float] = (8.0, 8.0)
    """rough heightfield 的全尺寸，默认对齐训练端 ROUGH_TERRAINS_CFG.size。"""
    rough_stair_step_height_range: tuple[float, float] = (0.0, 0.05)
    """rough pyramid stairs 的单级高度范围；默认对齐 5cm rough discovery 泛化训练。"""
    custom_terrain: CustomTerrain = "none"
    """sim2sim 额外自定义地形；slope-17 为单个 17° 坡面。"""
    sim_dt: Annotated[float, Field(gt=0.0)] = _shared_robot.sim_dt
    control_decimation: Annotated[int, Field(ge=1)] = _shared_robot.control_decimation
    base_height: float = _shared_robot.default_base_height
    initial_roll_rad: float = 0.0
    initial_pitch_rad: float = 0.0
    initial_yaw_rad: float = 0.0
    initial_ang_vel_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """reset 初始 base 角速度，按 MuJoCo freejoint qvel 的 body-frame xyz 顺序。"""
    initial_base_height: float | None = None
    """sim2sim reset 初始 base 姿态。默认全 0，保持原站立回放行为。"""
    command: tuple[float, float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        _shared_robot.default_base_height,
        0.0,
        0.0,
        0.0,
    )
    """8 维指令: [vx, ωz, pitch, roll, height, jump_flag, jump_target_height, jump_phase]。
    jump_phase 由 workflow 内部维护（sim2sim 初始为 0，跳跃期间自动推进）。
    """
    command_scale: tuple[float, ...] = _shared_obs.command_scale
    default_dof_pos: tuple[float, ...] = _shared_robot.default_dof_pos
    initial_leg_joint_pos: tuple[float, ...] | None = None
    """reset 时覆写腿部初始关节位置；2 个值表示左右同型，4 个值表示 policy 腿部顺序。"""
    initial_wheel_joint_pos: tuple[float, float] | None = None
    """reset 时覆写轮子连续关节位置；主要用于从 deploy telemetry 起跑。"""
    initial_dof_vel: tuple[float, ...] | None = None
    """reset 时覆写 6 维 policy-order 关节速度：[4 腿, 2 轮]。"""
    initial_last_action: tuple[float, ...] | None = None
    """reset 时覆写 observation 中的 last_action；用于从 deploy obs 精确起跑。"""
    settle_base_before_policy: bool = False
    """policy 推理前先用零控制等待 base_link 接地。"""
    pre_policy_settle_max_s: Annotated[float, Field(ge=0.0)] = 0.0
    """base_link 贴地后额外等待接触稳定的最大仿真时间；默认不做被动滚动。"""
    pre_policy_settle_contact_steps: Annotated[int, Field(ge=1)] = 5
    """判定 base_link 已接地需要连续满足的 MuJoCo step 数。"""
    action_scale: tuple[float, ...] = _shared_robot.action_scale
    action_clip: float | None = _shared_robot.action_clip
    height_conditioned_action_default: bool = True
    """让腿部 action=0 对应当前 command height 下的默认腿型。"""
    active_rod_target_lower_preload_margin: Annotated[float, Field(ge=0.0)] = (
        _shared_robot.active_rod_lower_target_overdrive
    )
    active_rod_target_upper_preload_margin: Annotated[float, Field(ge=0.0)] = 0.0
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
    rc_switch: RcSwitchScheduleConfig = Field(default_factory=RcSwitchScheduleConfig)
    """sim2sim 中的遥控器输出使能脚本；关闭时按硬件 off 语义处理。"""

    @model_validator(mode="after")
    def _check_terrain_mode(self) -> RobotConfig:
        terrain_modes = [
            bool(self.stair_terrain),
            bool(self.rough_terrain),
            self.custom_terrain != "none",
        ]
        if sum(terrain_modes) > 1:
            raise ValueError(
                "--stair-terrain, --rough-terrain and --custom-terrain cannot be enabled together"
            )
        return self


class PolicyConfig(BaseModel):
    """策略 checkpoint 配置。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoint: Path | None = None
    device: str = "cpu"


class ViewerConfig(BaseModel):
    """可视化配置。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: ViewerMode = "rerun"
    port: int = 8080
    app_id: str = "se3_sim2sim"
    spawn: bool = True
    address: str | None = None
    record_to_rrd: Path | None = None
    memory_limit: str = "1GB"
    log_every: int = 1
    follow_body: str = "base_link"
    geom_view: RerunGeomView = "visual"
    """3D viewer 显示的 MJCF 几何：visual 用于复查外观，collision 用于接触诊断。"""


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
    stair_hold_on_support: bool = False
    stair_hold_vx: float = 0.10
    stair_hold_min_step: int = 1
    termination: Termination = Field(default_factory=Termination)
    terminate_on_fall: bool = False
    fail_tilt_deg: float = 80.0
    fail_height_m: float = 0.12
    deploy_telemetry_init: dict[str, object] | None = None
    deploy_telemetry_reference_obs: tuple[float, ...] | None = Field(default=None, exclude=True)

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
