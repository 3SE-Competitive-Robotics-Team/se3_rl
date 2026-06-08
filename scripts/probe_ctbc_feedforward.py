"""用 MuJoCo 回放 warm-start checkpoint，快速筛选 CTBC 前馈候选。"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from se3_shared import ActionDelayConfig
from se3_shared.ctbc_feedforward import (
    REFERENCE_CTBC_FF_AMPLITUDE,
    current_leg_action_scales_np,
    reference_ctbc_bias_to_current_action_np,
)
from se3_shared.fourbar import policy_to_output_pos_np
from se3_shared.height_default import policy_default_from_height_np
from se3_sim2sim.config import RobotConfig, YawPidConfig
from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.rerun_viewer import RerunViewer
from se3_sim2sim.robot import WheelLeggedRobot
from se3_sim2sim.runtime_spec import RuntimeSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    ROOT / "assets" / "robots" / "serialleg" / "mjcf" / "serialleg_fourbar_surrogate_train.xml"
)
DEFAULT_CHECKPOINT = ROOT / "assets" / "base_model" / "model_4999_gru.pt"
DEFAULT_OUT_DIR = ROOT / "tmp" / "ctbc_feedforward_probe"
WHEEL_RADIUS_M = 0.06


@dataclass(frozen=True)
class ProbeConfig:
    """单次测试的全局配置。"""

    checkpoint: Path
    base_xml: Path
    out_dir: Path
    height_m: float
    initial_base_height_m: float | None
    reference_default_height_m: float | None
    step_height_m: float
    step_depth_m: float
    spawn_base_offset_m: float
    vx_mps: float
    yaw_rate_rad_s: float
    seconds: float
    fps: int
    ff_period_s: float
    force_threshold_n: float
    contact_window: int
    device: str
    render: bool
    record_rerun: bool
    rerun_spawn: bool
    rerun_memory_limit: str

    @property
    def riser_x_m(self) -> float:
        """台阶立面在世界系 x 方向的位置。"""
        return -float(self.spawn_base_offset_m)

    @property
    def reset_base_height_m(self) -> float:
        return (
            float(self.height_m)
            if self.initial_base_height_m is None
            else float(self.initial_base_height_m)
        )


@dataclass(frozen=True)
class FeedforwardCase:
    """一组 CTBC 前馈候选。"""

    name: str
    trigger_mode: str
    front_amp: float
    active_amp: float
    distance_margin_m: float = 0.08
    ff_style: str = "manual"
    reference_amplitude: float = REFERENCE_CTBC_FF_AMPLITUDE


@dataclass
class RolloutTrace:
    """单次回放的逐步诊断信号。"""

    time_s: list[float]
    wheel_front_distance_m: list[float]
    riser_force_left_n: list[float]
    riser_force_right_n: list[float]
    profile_left: list[float]
    profile_right: list[float]
    base_height_m: list[float]
    base_vx_mps: list[float]
    policy_action_abs_mean: list[float]
    ff_action_abs_mean: list[float]
    trigger_times_s: list[float]


@dataclass(frozen=True)
class TriggerDiagnostics:
    """单步 CTBC 触发诊断。"""

    distance_trigger: bool
    contact_over_threshold: tuple[bool, bool]
    contact_trigger: tuple[bool, bool]
    can_trigger: tuple[bool, bool]
    newly_triggered: tuple[bool, bool]


@dataclass(frozen=True)
class RolloutSummary:
    """单个 case 的汇总指标。"""

    name: str
    trigger_mode: str
    rerun: str | None
    video: str | None
    plot: str
    climbed: bool
    trigger_time_s: float | None
    trigger_distance_m: float | None
    max_wheel_x_past_riser_m: float
    max_wheel_bottom_over_step_m: float
    max_riser_force_n: float
    max_base_height_m: float
    final_base_height_m: float
    mean_base_height_last_1s_m: float
    min_base_height_last_1s_m: float
    max_base_height_last_1s_m: float
    mean_base_vx_mps: float
    nonwheel_contact_rate: float
    fall: bool
    steps: int


def _resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _default_cases() -> tuple[FeedforwardCase, ...]:
    return (
        FeedforwardCase("policy_only", "none", 0.0, 0.0),
        FeedforwardCase(
            "reference_distance",
            "distance",
            0.0,
            0.0,
            0.015,
            ff_style="reference",
        ),
        FeedforwardCase("reference_contact", "contact", 0.0, 0.0, ff_style="reference"),
        FeedforwardCase("distance_current", "distance", 0.14, 0.07, 0.015),
        FeedforwardCase("distance_early_current", "distance", 0.14, 0.07, 0.08),
        FeedforwardCase("contact_current", "contact", 0.14, 0.07),
    )


def _parse_case(value: str) -> FeedforwardCase:
    parts = [item.strip() for item in value.split(":")]
    if len(parts) not in (3, 4, 5):
        raise argparse.ArgumentTypeError(
            "case 格式必须是 name:trigger:front_amp:active_amp[:distance_margin_m]"
        )
    name, trigger_mode = parts[:2]
    trigger_mode = trigger_mode.lower()
    if trigger_mode not in {"none", "distance", "contact"}:
        raise argparse.ArgumentTypeError("trigger 必须是 none、distance 或 contact")
    if len(parts) in (3, 4) and parts[2].lower() == "reference":
        margin = float(parts[3]) if len(parts) == 4 else 0.08
        return FeedforwardCase(
            name=name,
            trigger_mode=trigger_mode,
            front_amp=0.0,
            active_amp=0.0,
            distance_margin_m=margin,
            ff_style="reference",
        )
    if len(parts) in (3, 4) and parts[2].lower() in {
        "reference_train_raw",
        "reference_sim2sim_raw",
    }:
        margin = float(parts[3]) if len(parts) == 4 else 0.08
        return FeedforwardCase(
            name=name,
            trigger_mode=trigger_mode,
            front_amp=0.0,
            active_amp=0.0,
            distance_margin_m=margin,
            ff_style=parts[2].lower(),
        )
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            "manual case must be name:trigger:front_amp:active_amp[:distance_margin_m]"
        )
    front_amp, active_amp = parts[2:4]
    margin = float(parts[4]) if len(parts) == 5 else 0.08
    return FeedforwardCase(
        name=name,
        trigger_mode=trigger_mode,
        front_amp=float(front_amp),
        active_amp=float(active_amp),
        distance_margin_m=margin,
    )


def build_stair_xml(cfg: ProbeConfig) -> Path:
    """生成带一级台阶的临时 MJCF 场景。"""

    xml = cfg.base_xml.read_text(encoding="utf-8")
    meshdir = (cfg.base_xml.parent / "../meshes").resolve().as_posix()
    xml = xml.replace('meshdir="../meshes"', f'meshdir="{meshdir}"')
    stage = f"""
    <geom name="ctbc_step_1" type="box"
          pos="{cfg.riser_x_m + cfg.step_depth_m * 0.5:.4f} 0 {cfg.step_height_m * 0.5:.4f}"
          size="{cfg.step_depth_m * 0.5:.4f} 1.2 {cfg.step_height_m * 0.5:.4f}"
          rgba="0.62 0.36 0.24 1"
          friction="0.8 0.005 0.0001"
          contype="2" conaffinity="1" />
    <geom name="ctbc_step_face_marker" type="box"
          pos="{cfg.riser_x_m:.4f} -0.62 {cfg.step_height_m * 0.5:.4f}"
          size="0.006 0.02 {cfg.step_height_m * 0.5:.4f}"
          rgba="0.1 0.1 0.1 1"
          contype="0" conaffinity="0" group="1" />
"""
    out = cfg.out_dir / "ctbc_stage_scene.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml.replace("</worldbody>", stage + "\n  </worldbody>"), encoding="utf-8")
    return out


def set_height_default_pose(robot: WheelLeggedRobot, cfg: ProbeConfig) -> None:
    """把 reset 姿态设为当前 command height 对应的默认腿型。"""

    policy = np.asarray(policy_default_from_height_np(cfg.height_m), dtype=np.float64).reshape(4)
    output = policy_to_output_pos_np(policy)
    joint_values = {
        "lf0_Joint": output[0],
        "lf1_Joint": output[1],
        "rf0_Joint": output[2],
        "rf1_Joint": output[3],
        "l_wheel_Joint": 0.0,
        "r_wheel_Joint": 0.0,
    }
    robot.data.qpos[0:3] = (0.0, 0.0, cfg.reset_base_height_m)
    robot.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in joint_values.items():
        jid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        robot.data.qpos[robot.model.jnt_qposadr[jid]] = float(value)
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def make_robot(scene_xml: Path, cfg: ProbeConfig) -> WheelLeggedRobot:
    """构造 sim2sim MuJoCo 机器人运行时。"""

    command = (
        cfg.vx_mps,
        cfg.yaw_rate_rad_s,
        0.0,
        0.0,
        cfg.height_m,
        0.0,
        0.0,
        0.0,
    )
    robot_cfg = RobotConfig(
        model_path=scene_xml,
        base_height=cfg.reset_base_height_m,
        initial_base_height=cfg.reset_base_height_m,
        command=command,
        height_conditioned_action_default=True,
        action_delay_steps=0,
        action_delay=ActionDelayConfig(
            enabled=False,
            delay_s=0.0,
            randomize=False,
            min_delay_s=0.0,
            max_delay_s=0.0,
        ),
        yaw_pid=YawPidConfig(enabled=False),
        action_clip=None,
    )
    robot = WheelLeggedRobot(cfg=robot_cfg, runtime=RuntimeSpec())
    robot.reset()
    set_height_default_pose(robot, cfg)
    return robot


def action_for_case(
    case: FeedforwardCase,
    profiles: np.ndarray,
    *,
    cfg: ProbeConfig,
    policy_action: np.ndarray,
    robot: WheelLeggedRobot,
) -> np.ndarray:
    """生成左右轮独立触发的 CTBC raw action bias。"""

    action = np.zeros(6, dtype=np.float64)
    if case.ff_style == "reference_train_raw":
        side_pulse = 2.0 * float(case.reference_amplitude) * np.asarray(profiles, dtype=np.float64)
        action[0] = -1.5 * side_pulse[0]
        action[1] = 1.0 * side_pulse[0]
        action[2] = -1.5 * side_pulse[1]
        action[3] = 1.0 * side_pulse[1]
        return action
    if case.ff_style == "reference_sim2sim_raw":
        side_pulse = 2.0 * float(case.reference_amplitude) * np.asarray(profiles, dtype=np.float64)
        action[0] = side_pulse[0]
        action[1] = 2.0 * side_pulse[0]
        action[2] = side_pulse[1]
        action[3] = 2.0 * side_pulse[1]
        return action
    if case.ff_style == "reference":
        side_pulse = 2.0 * float(case.reference_amplitude) * np.asarray(profiles, dtype=np.float64)
        policy_default = (
            policy_default_from_height_np(float(cfg.reference_default_height_m)).reshape(4)
            if cfg.reference_default_height_m is not None
            else robot._policy_action_default()
        )
        action[:4] = reference_ctbc_bias_to_current_action_np(
            np.asarray(policy_action[:4], dtype=np.float64),
            policy_default,
            side_pulse,
            leg_scales=current_leg_action_scales_np(),
            height_conditioned_action_default=robot.cfg.height_conditioned_action_default,
        ).reshape(4)
        return action

    action[0] = -case.front_amp * profiles[0]
    action[1] = case.active_amp * profiles[0]
    action[2] = case.front_amp * profiles[1]
    action[3] = case.active_amp * profiles[1]
    return action


def riser_contact_force_xy(robot: WheelLeggedRobot, step_geom_id: int) -> np.ndarray:
    """近似训练端 wheel_riser_sensor，统计轮子与台阶立面的水平法向力。"""

    forces = np.zeros(2, dtype=np.float64)
    wheel_sets = (robot._left_wheel_geom_ids, robot._right_wheel_geom_ids)
    force6 = np.zeros(6, dtype=np.float64)
    for idx in range(robot.data.ncon):
        contact = robot.data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 != step_geom_id and geom2 != step_geom_id:
            continue
        other = geom2 if geom1 == step_geom_id else geom1
        side = None
        if other in wheel_sets[0]:
            side = 0
        elif other in wheel_sets[1]:
            side = 1
        if side is None:
            continue
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        if abs(float(normal[2])) > 0.5:
            continue
        mujoco.mj_contactForce(robot.model, robot.data, idx, force6)
        force_world = normal * abs(float(force6[0]))
        forces[side] += float(np.linalg.norm(force_world[:2]))
    return forces


def update_trigger_state(
    case: FeedforwardCase,
    cfg: ProbeConfig,
    *,
    t: float,
    wheel_front_x: float,
    riser_force: np.ndarray,
    phases: np.ndarray,
    cooldown: np.ndarray,
    contact_buf: np.ndarray,
    trigger_times: list[float],
) -> TriggerDiagnostics:
    """按指定触发器推进左右两侧 CTBC 相位。"""

    cooldown[cooldown > 0] -= 1

    contact_buf[1:] = contact_buf[:-1]
    contact_buf[0] = riser_force
    distance_hit = wheel_front_x >= cfg.riser_x_m - case.distance_margin_m
    contact_over_threshold = riser_force > cfg.force_threshold_n
    contact_trigger = np.all(contact_buf > cfg.force_threshold_n, axis=0)

    newly_triggered = np.zeros(2, dtype=bool)
    if case.trigger_mode == "distance":
        newly_triggered[:] = distance_hit
    elif case.trigger_mode == "contact":
        newly_triggered = contact_trigger.copy()

    can_trigger = (phases < 0) & (cooldown == 0)
    newly_triggered &= can_trigger
    phases[newly_triggered] = 0
    if bool(np.any(newly_triggered)):
        trigger_times.append(t)
    return TriggerDiagnostics(
        distance_trigger=bool(distance_hit),
        contact_over_threshold=(bool(contact_over_threshold[0]), bool(contact_over_threshold[1])),
        contact_trigger=(bool(contact_trigger[0]), bool(contact_trigger[1])),
        can_trigger=(bool(can_trigger[0]), bool(can_trigger[1])),
        newly_triggered=(bool(newly_triggered[0]), bool(newly_triggered[1])),
    )


def profile_from_phases(phases: np.ndarray, ff_steps: int) -> np.ndarray:
    """把 CTBC 相位转换为半余弦前馈包络。"""

    profiles = np.zeros(2, dtype=np.float64)
    active = phases >= 0
    if not bool(np.any(active)):
        return profiles
    phase = phases[active].astype(np.float64) / float(ff_steps)
    profiles[active] = 0.5 * (1.0 - np.cos(2.0 * math.pi * phase))
    return profiles


def advance_phases(phases: np.ndarray, cooldown: np.ndarray, ff_steps: int) -> None:
    """完成一个 control step 后推进 CTBC 相位和冷却。"""

    active = phases >= 0
    phases[active] += 1
    finished = phases >= ff_steps
    cooldown[finished] = max(1, round(0.3 / 0.02))
    phases[finished] = -1


def log_rerun_ctbc(
    viewer: RerunViewer,
    robot: WheelLeggedRobot,
    *,
    step: int,
    info: dict[str, object],
    trigger: TriggerDiagnostics,
    wheel_front_distance_m: float,
    riser_force: np.ndarray,
    profiles: np.ndarray,
    phases: np.ndarray,
    policy_action: np.ndarray,
    ff_action: np.ndarray,
    action: np.ndarray,
) -> None:
    """在 Rerun 中记录 CTBC 触发与前馈诊断。"""

    viewer.log_state(robot.model, robot.data, step=step, telemetry=info)
    rr = viewer.rr

    def scalar(path: str, value: object) -> None:
        rr.log(path, rr.Scalars(scalars=float(value)))

    scalar("/plots/contact/front_riser_distance_m", wheel_front_distance_m)
    scalar("/plots/contact/distance_trigger", float(trigger.distance_trigger))
    scalar("/plots/contact/contact_force_left_n", riser_force[0])
    scalar("/plots/contact/contact_force_right_n", riser_force[1])
    scalar("/plots/contact/contact_over_threshold_left", float(trigger.contact_over_threshold[0]))
    scalar("/plots/contact/contact_over_threshold_right", float(trigger.contact_over_threshold[1]))
    scalar("/plots/contact/contact_trigger_left", float(trigger.contact_trigger[0]))
    scalar("/plots/contact/contact_trigger_right", float(trigger.contact_trigger[1]))
    scalar("/plots/contact/ctbc_can_trigger_left", float(trigger.can_trigger[0]))
    scalar("/plots/contact/ctbc_can_trigger_right", float(trigger.can_trigger[1]))
    scalar("/plots/contact/ctbc_triggered_left", float(trigger.newly_triggered[0]))
    scalar("/plots/contact/ctbc_triggered_right", float(trigger.newly_triggered[1]))
    scalar("/plots/contact/ctbc_active_left", float(profiles[0] > 0.0))
    scalar("/plots/contact/ctbc_active_right", float(profiles[1] > 0.0))
    scalar("/plots/contact/ctbc_phase_left", phases[0])
    scalar("/plots/contact/ctbc_phase_right", phases[1])

    labels = ("lf0", "l_active", "rf0", "r_active", "l_wheel", "r_wheel")
    for idx, label in enumerate(labels):
        scalar(f"/plots/action/policy_raw/{label}", policy_action[idx])
        scalar(f"/plots/action/ff_raw/{label}", ff_action[idx])
        scalar(f"/plots/action/policy_plus_ff/{label}", action[idx])
    scalar("/plots/action/ff_profile_left", profiles[0])
    scalar("/plots/action/ff_profile_right", profiles[1])


def plot_trace(case: FeedforwardCase, trace: RolloutTrace, out_dir: Path) -> Path:
    """保存触发、力、前馈和速度信号图。"""

    out = out_dir / f"{case.name}_signals.png"
    time = np.asarray(trace.time_s)
    fig, axes = plt.subplots(5, 1, figsize=(9.6, 10.0), sharex=True)

    axes[0].plot(time, np.asarray(trace.wheel_front_distance_m) * 1000.0, color="#1f77b4")
    axes[0].axhline(-case.distance_margin_m * 1000.0, color="#888", linestyle="--", linewidth=1)
    axes[0].axhline(0.0, color="#222", linewidth=1)
    axes[0].set_ylabel("front-riser mm")
    axes[0].set_title(f"{case.name} ({case.trigger_mode})")

    axes[1].plot(time, trace.riser_force_left_n, label="left", color="#d62728")
    axes[1].plot(time, trace.riser_force_right_n, label="right", color="#2ca02c")
    axes[1].set_ylabel("riser force N")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, trace.profile_left, label="left", color="#d62728")
    axes[2].plot(time, trace.profile_right, label="right", color="#2ca02c")
    axes[2].set_ylabel("ff profile")
    axes[2].legend(loc="upper right")

    axes[3].plot(time, trace.base_height_m, color="#9467bd")
    axes[3].set_ylabel("base z m")

    axes[4].plot(time, trace.base_vx_mps, color="#ff7f0e", label="base vx")
    axes[4].plot(time, trace.policy_action_abs_mean, color="#555555", label="|policy| mean")
    axes[4].plot(time, trace.ff_action_abs_mean, color="#17becf", label="|ff| mean")
    axes[4].set_ylabel("vx / action")
    axes[4].set_xlabel("time s")
    axes[4].legend(loc="upper right")

    for ax in axes:
        for trigger_time in trace.trigger_times_s:
            ax.axvline(trigger_time, color="#111", linestyle=":", linewidth=1.2)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def rollout(case: FeedforwardCase, scene_xml: Path, cfg: ProbeConfig) -> RolloutSummary:
    """运行一个物理回放并保存诊断产物。"""

    robot = make_robot(scene_xml, cfg)
    policy = PolicyRuntime(checkpoint=cfg.checkpoint, device=cfg.device, runtime=RuntimeSpec())
    policy.reset()
    rerun_path: Path | None = None
    rerun_viewer: RerunViewer | None = None
    if cfg.record_rerun:
        rerun_path = cfg.out_dir / f"{case.name}.rrd"
        rerun_viewer = RerunViewer(
            app_id=f"ctbc_feedforward_probe_{case.name}",
            spawn=cfg.rerun_spawn,
            record_to_rrd=rerun_path,
            memory_limit=cfg.rerun_memory_limit,
        )
        rerun_viewer.log_model(robot.model)
    renderer = None
    frames: list[np.ndarray] = []
    if cfg.render:
        renderer = mujoco.Renderer(robot.model, height=480, width=640)
        for geom_id in range(robot.model.ngeom):
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if "collision" in name and not name.startswith("ctbc_step"):
                robot.model.geom_rgba[geom_id, 3] = 0.0

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.18, 0.0, 0.18)
    cam.distance = 1.25
    cam.elevation = -12
    cam.azimuth = 90

    left_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_Link")
    right_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_Link")
    step_geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "ctbc_step_1")

    trace = RolloutTrace([], [], [], [], [], [], [], [], [], [], [])
    phases = np.full(2, -1, dtype=np.int64)
    cooldown = np.zeros(2, dtype=np.int64)
    contact_buf = np.zeros((cfg.contact_window, 2), dtype=np.float64)
    max_x_past = -math.inf
    max_bottom_over_step = -math.inf
    max_riser_force = 0.0
    nonwheel_contacts = 0
    fall = False
    steps_run = 0
    control_steps = int(cfg.seconds / robot.control_dt)
    ff_steps = max(1, round(cfg.ff_period_s / robot.control_dt))
    render_every = max(1, round(1.0 / (cfg.fps * robot.control_dt)))

    obs = robot.observation()
    for step in range(control_steps):
        steps_run += 1
        t = step * robot.control_dt
        wheel_pos = np.stack((robot.data.xpos[left_id], robot.data.xpos[right_id]))
        wheel_front_x = float(wheel_pos[:, 0].max() + WHEEL_RADIUS_M)
        riser_force = riser_contact_force_xy(robot, step_geom_id)
        max_riser_force = max(max_riser_force, float(np.max(riser_force)))

        trigger = update_trigger_state(
            case,
            cfg,
            t=t,
            wheel_front_x=wheel_front_x,
            riser_force=riser_force,
            phases=phases,
            cooldown=cooldown,
            contact_buf=contact_buf,
            trigger_times=trace.trigger_times_s,
        )
        profiles = profile_from_phases(phases, ff_steps)

        policy_action = np.clip(np.asarray(policy.act(obs), dtype=np.float64), -1.0, 1.0)
        ff_action = action_for_case(
            case,
            profiles,
            cfg=cfg,
            policy_action=policy_action,
            robot=robot,
        )
        action = policy_action + ff_action
        obs, _, done, info = robot.step(action)
        if rerun_viewer is not None:
            log_rerun_ctbc(
                rerun_viewer,
                robot,
                step=step,
                info=info,
                trigger=trigger,
                wheel_front_distance_m=wheel_front_x - cfg.riser_x_m,
                riser_force=riser_force,
                profiles=profiles,
                phases=phases,
                policy_action=policy_action,
                ff_action=ff_action,
                action=action,
            )
        advance_phases(phases, cooldown, ff_steps)

        trace.time_s.append(t)
        trace.wheel_front_distance_m.append(wheel_front_x - cfg.riser_x_m)
        trace.riser_force_left_n.append(float(riser_force[0]))
        trace.riser_force_right_n.append(float(riser_force[1]))
        trace.profile_left.append(float(profiles[0]))
        trace.profile_right.append(float(profiles[1]))
        trace.base_height_m.append(float(info["height"]))
        trace.base_vx_mps.append(float(info["base_lin_vel_x"]))
        trace.policy_action_abs_mean.append(float(np.mean(np.abs(policy_action))))
        trace.ff_action_abs_mean.append(float(np.mean(np.abs(ff_action))))

        if bool(info.get("nonwheel_contact", False)):
            nonwheel_contacts += 1
        fall = fall or bool(info.get("fall_detected", False)) or bool(done)

        wheel_pos = np.stack((robot.data.xpos[left_id], robot.data.xpos[right_id]))
        max_x_past = max(max_x_past, float(wheel_pos[:, 0].min() - cfg.riser_x_m))
        max_bottom_over_step = max(
            max_bottom_over_step,
            float(wheel_pos[:, 2].min() - WHEEL_RADIUS_M - cfg.step_height_m),
        )

        if renderer is not None and step % render_every == 0:
            renderer.update_scene(robot.data, camera=cam, scene_option=opt)
            frames.append(renderer.render())

    climbed = max_x_past > WHEEL_RADIUS_M and max_bottom_over_step > -0.01 and not fall
    video_path: Path | None = None
    if renderer is not None:
        video_path = cfg.out_dir / f"{case.name}.mp4"
        if frames:
            imageio.mimsave(video_path, frames, fps=cfg.fps, quality=8)
        renderer.close()
    if rerun_viewer is not None:
        rerun_viewer.close()
    plot_path = plot_trace(case, trace, cfg.out_dir)

    trigger_time = trace.trigger_times_s[0] if trace.trigger_times_s else None
    trigger_distance = None
    if trigger_time is not None and trace.time_s:
        idx = int(np.argmin(np.abs(np.asarray(trace.time_s) - trigger_time)))
        trigger_distance = trace.wheel_front_distance_m[idx]

    base_height = np.asarray(trace.base_height_m, dtype=np.float64)
    last_1s_samples = max(1, round(1.0 / robot.control_dt))
    last_height = base_height[-last_1s_samples:] if base_height.size else np.asarray([0.0])

    return RolloutSummary(
        name=case.name,
        trigger_mode=case.trigger_mode,
        rerun=None if rerun_path is None else rerun_path.name,
        video=None if video_path is None else video_path.name,
        plot=plot_path.name,
        climbed=climbed,
        trigger_time_s=trigger_time,
        trigger_distance_m=trigger_distance,
        max_wheel_x_past_riser_m=max_x_past,
        max_wheel_bottom_over_step_m=max_bottom_over_step,
        max_riser_force_n=max_riser_force,
        max_base_height_m=float(np.max(base_height)) if base_height.size else 0.0,
        final_base_height_m=float(base_height[-1]) if base_height.size else 0.0,
        mean_base_height_last_1s_m=float(np.mean(last_height)),
        min_base_height_last_1s_m=float(np.min(last_height)),
        max_base_height_last_1s_m=float(np.max(last_height)),
        mean_base_vx_mps=float(np.mean(trace.base_vx_mps)) if trace.base_vx_mps else 0.0,
        nonwheel_contact_rate=nonwheel_contacts / max(1, steps_run),
        fall=fall,
        steps=steps_run,
    )


def write_html(cfg: ProbeConfig, summaries: list[RolloutSummary], out: Path) -> None:
    """写出可浏览的回放页面。"""

    rows = []
    cards = []
    for item in summaries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.name)}</td>"
            f"<td>{html.escape(item.trigger_mode)}</td>"
            f"<td>{'yes' if item.climbed else 'no'}</td>"
            f"<td>{'' if item.trigger_time_s is None else f'{item.trigger_time_s:.2f}'}</td>"
            f"<td>{'' if item.trigger_distance_m is None else f'{item.trigger_distance_m * 1000:+.1f}'}</td>"
            f"<td>{item.max_riser_force_n:.1f}</td>"
            f"<td>{item.max_wheel_x_past_riser_m * 1000:+.1f}</td>"
            f"<td>{item.max_wheel_bottom_over_step_m * 1000:+.1f}</td>"
            f"<td>{item.max_base_height_m:.3f}</td>"
            f"<td>{item.mean_base_height_last_1s_m:.3f}</td>"
            f"<td>{item.mean_base_vx_mps:+.3f}</td>"
            f"<td>{item.nonwheel_contact_rate:.3f}</td>"
            f"<td>{'yes' if item.fall else 'no'}</td>"
            "</tr>"
        )
        video_block = (
            f'<video src="{html.escape(item.video)}" controls loop muted autoplay></video>'
            if item.video is not None
            else "<p>未生成视频（--no-render）。</p>"
        )
        rerun_block = (
            f"<p>Rerun: <code>{html.escape(item.rerun)}</code></p>"
            if item.rerun is not None
            else ""
        )
        cards.append(
            f"""
            <section>
              <h2>{html.escape(item.name)}</h2>
              {rerun_block}
              {video_block}
              <img src="{html.escape(item.plot)}" alt="{html.escape(item.name)} signals">
            </section>
            """
        )
    reference_default_label = (
        "command height"
        if cfg.reference_default_height_m is None
        else f"{cfg.reference_default_height_m:.2f} m"
    )
    initial_height_label = (
        "same as command"
        if cfg.initial_base_height_m is None
        else f"{cfg.initial_base_height_m:.3f} m"
    )
    out.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>CTBC feedforward probe</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f8; color: #1f2933; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #d9dee5; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
    section {{ background: white; border: 1px solid #d9dee5; padding: 12px; }}
    video, img {{ width: 100%; display: block; }}
    video {{ margin-bottom: 10px; }}
    p {{ line-height: 1.55; }}
    code {{ background: #eef1f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>CTBC feedforward probe</h1>
  <p>
    checkpoint: <code>{html.escape(str(cfg.checkpoint))}</code><br>
    vx={cfg.vx_mps:.2f} m/s, height_command={cfg.height_m:.2f} m,
    initial_base_z={initial_height_label},
    step={cfg.step_height_m:.2f} m, ff_period={cfg.ff_period_s:.2f} s,
    reference_default={reference_default_label}
  </p>
  <table>
    <thead>
      <tr><th>case</th><th>trigger</th><th>climbed</th><th>trigger s</th><th>trigger dist mm</th><th>max riser N</th><th>min wheel x past riser mm</th><th>min wheel bottom over step mm</th><th>max base z</th><th>last 1s base z</th><th>mean vx</th><th>nonwheel contact</th><th>fall</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <div class="grid">{"".join(cards)}</div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_summary_json(
    cfg: ProbeConfig, cases: list[FeedforwardCase], summaries: list[RolloutSummary]
) -> Path:
    """写出机器可读的汇总结果。"""

    out = cfg.out_dir / "summary.json"
    payload = {
        "config": {
            **asdict(cfg),
            "checkpoint": str(cfg.checkpoint),
            "base_xml": str(cfg.base_xml),
            "out_dir": str(cfg.out_dir),
        },
        "cases": [asdict(case) for case in cases],
        "summaries": [asdict(item) for item in summaries],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 MuJoCo 加载 warm-start checkpoint，测试 CTBC 前馈候选。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--height", type=float, default=0.34)
    parser.add_argument(
        "--initial-base-height",
        type=float,
        default=None,
        help="reset qpos base z; defaults to --height.",
    )
    parser.add_argument(
        "--reference-default-height",
        type=float,
        default=None,
        help="只用于 reference FF 换算的默认腿型高度；不改变 base height command。",
    )
    parser.add_argument("--step-height", type=float, default=0.20)
    parser.add_argument("--step-depth", type=float, default=1.10)
    parser.add_argument("--spawn-base-offset", type=float, default=-0.08)
    parser.add_argument("--vx", type=float, default=0.80)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--ff-period", type=float, default=0.60)
    parser.add_argument("--force-threshold", type=float, default=30.0)
    parser.add_argument("--contact-window", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help="格式：name:trigger:front_amp:active_amp[:distance_margin_m]，可重复指定。",
    )
    parser.add_argument("--no-render", action="store_true", help="只跑指标和曲线，不保存 mp4。")
    parser.add_argument("--record-rerun", action="store_true", help="为每个 case 保存 .rrd。")
    parser.add_argument("--rerun-spawn", action="store_true", help="录制时同时打开 Rerun viewer。")
    parser.add_argument("--rerun-memory-limit", default="1GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProbeConfig(
        checkpoint=_resolve_path(args.checkpoint),
        base_xml=_resolve_path(args.base_xml),
        out_dir=_resolve_path(args.out_dir),
        height_m=float(args.height),
        initial_base_height_m=(
            None if args.initial_base_height is None else float(args.initial_base_height)
        ),
        reference_default_height_m=(
            None if args.reference_default_height is None else float(args.reference_default_height)
        ),
        step_height_m=float(args.step_height),
        step_depth_m=float(args.step_depth),
        spawn_base_offset_m=float(args.spawn_base_offset),
        vx_mps=float(args.vx),
        yaw_rate_rad_s=float(args.yaw_rate),
        seconds=float(args.seconds),
        fps=int(args.fps),
        ff_period_s=float(args.ff_period),
        force_threshold_n=float(args.force_threshold),
        contact_window=max(1, int(args.contact_window)),
        device=str(args.device),
        render=not bool(args.no_render),
        record_rerun=bool(args.record_rerun),
        rerun_spawn=bool(args.rerun_spawn),
        rerun_memory_limit=str(args.rerun_memory_limit),
    )
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{cfg.checkpoint}")
    if not cfg.base_xml.exists():
        raise FileNotFoundError(f"MJCF 不存在：{cfg.base_xml}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cases = list(args.case) if args.case else list(_default_cases())
    scene_xml = build_stair_xml(cfg)
    summaries = [rollout(case, scene_xml, cfg) for case in cases]
    html_path = cfg.out_dir / "index.html"
    write_html(cfg, summaries, html_path)
    summary_path = write_summary_json(cfg, cases, summaries)

    print(html_path)
    print(summary_path)
    for item in summaries:
        print(
            item.name,
            "mode=",
            item.trigger_mode,
            "climbed=",
            item.climbed,
            "trigger=",
            item.trigger_time_s,
            "trigger_dist_mm=",
            None if item.trigger_distance_m is None else item.trigger_distance_m * 1000.0,
            "max_riser_N=",
            item.max_riser_force_n,
            "x_past_mm=",
            item.max_wheel_x_past_riser_m * 1000.0,
            "bottom_over_step_mm=",
            item.max_wheel_bottom_over_step_m * 1000.0,
            "max_base_z=",
            item.max_base_height_m,
            "last_1s_base_z=",
            item.mean_base_height_last_1s_m,
            "mean_vx=",
            item.mean_base_vx_mps,
            "nonwheel=",
            item.nonwheel_contact_rate,
            "fall=",
            item.fall,
        )


if __name__ == "__main__":
    main()
