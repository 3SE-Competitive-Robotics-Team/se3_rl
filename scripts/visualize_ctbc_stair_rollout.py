"""用物理回放检查 CTBC 触发时机和 bias 是否匹配台阶。"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from se3_shared import RobotConfig as SharedRobotConfig
from se3_shared.fourbar import policy_to_output_pos_np
from se3_shared.height_default import policy_default_from_height_np
from se3_sim2sim.config import RobotConfig, YawPidConfig
from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.robot import WheelLeggedRobot
from se3_sim2sim.runtime_spec import RuntimeSpec

ROOT = Path(__file__).resolve().parents[1]
BASE_XML = (
    ROOT / "assets" / "robots" / "serialleg" / "mjcf" / "serialleg_fourbar_surrogate_train.xml"
)
OUT_DIR = ROOT / "tmp" / "ctbc_stair_rollout"
CHECKPOINT = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "se3_wheel_leg_stair_ctbc"
    / "recovery_4300_warmstart"
    / "model_4300.pt"
)

HEIGHT_M = 0.34
STEP_HEIGHT_M = 0.20
SPAWN_BASE_OFFSET_M = -0.08
STEP_RISER_X_M = -SPAWN_BASE_OFFSET_M
WHEEL_RADIUS_M = 0.06
FF_PERIOD_S = 0.60
SECONDS = 5.0
FPS = 40
FORCE_THRESHOLD_N = 30.0
CONTACT_WINDOW = 3


@dataclass(frozen=True)
class RolloutCase:
    """单个触发器对照样例。"""

    name: str
    front_amp: float
    active_amp: float
    trigger_mode: str
    distance_margin_m: float = 0.015


@dataclass
class RolloutTrace:
    """单次回放的诊断信号。"""

    time_s: list[float]
    wheel_front_distance_m: list[float]
    riser_force_left_n: list[float]
    riser_force_right_n: list[float]
    profile_left: list[float]
    profile_right: list[float]
    trigger_times_s: list[float]


@dataclass(frozen=True)
class RolloutSummary:
    """单个回放结果。"""

    name: str
    trigger_mode: str
    video: Path
    plot: Path
    climbed: bool
    trigger_time_s: float | None
    trigger_distance_m: float | None
    max_wheel_x_past_riser_m: float
    max_wheel_bottom_over_step_m: float
    max_riser_force_n: float
    nonwheel_contact_rate: float
    fall: bool


CASES = (
    RolloutCase("policy_only", 0.0, 0.0, "none"),
    RolloutCase("distance_current", 0.14, 0.07, "distance", 0.015),
    RolloutCase("contact_current", 0.14, 0.07, "contact"),
    RolloutCase("distance_early_current", 0.14, 0.07, "distance", 0.080),
    RolloutCase("distance_early_medium", 0.50, 0.50, "distance", 0.080),
)


def build_stair_xml(path: Path) -> Path:
    """生成带一级 200 mm 台阶的临时 MJCF。"""
    xml = path.read_text(encoding="utf-8")
    meshdir = (path.parent / "../meshes").resolve().as_posix()
    xml = xml.replace('meshdir="../meshes"', f'meshdir="{meshdir}"')
    stage = f"""
    <geom name="ctbc_step_1" type="box"
          pos="{STEP_RISER_X_M + 0.55:.4f} 0 {STEP_HEIGHT_M * 0.5:.4f}"
          size="0.55 1.2 {STEP_HEIGHT_M * 0.5:.4f}"
          rgba="0.62 0.36 0.24 1"
          friction="0.8 0.005 0.0001"
          contype="2" conaffinity="1" />
    <geom name="ctbc_step_face_marker" type="box"
          pos="{STEP_RISER_X_M:.4f} -0.62 {STEP_HEIGHT_M * 0.5:.4f}"
          size="0.006 0.02 {STEP_HEIGHT_M * 0.5:.4f}"
          rgba="0.1 0.1 0.1 1"
          contype="0" conaffinity="0" group="1" />
"""
    out = OUT_DIR / "ctbc_stage_scene.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml.replace("</worldbody>", stage + "\n  </worldbody>"), encoding="utf-8")
    return out


def set_height_default_pose(robot: WheelLeggedRobot) -> None:
    """把 reset 后姿态设为当前 command height 对应的默认站姿。"""
    policy = np.asarray(policy_default_from_height_np(HEIGHT_M), dtype=np.float64).reshape(4)
    output = policy_to_output_pos_np(policy)
    joint_values = {
        "lf0_Joint": output[0],
        "lf1_Joint": output[1],
        "rf0_Joint": output[2],
        "rf1_Joint": output[3],
        "l_wheel_Joint": 0.0,
        "r_wheel_Joint": 0.0,
    }
    robot.data.qpos[0:3] = (0.0, 0.0, HEIGHT_M)
    robot.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in joint_values.items():
        jid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        robot.data.qpos[robot.model.jnt_qposadr[jid]] = float(value)
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def make_robot(scene_xml: Path) -> WheelLeggedRobot:
    """构造 sim2sim 机器人运行时。"""
    shared = SharedRobotConfig()
    command = (0.35, 0.0, 0.0, 0.0, HEIGHT_M, 0.0, 0.0, 0.0)
    cfg = RobotConfig(
        model_path=scene_xml,
        base_height=HEIGHT_M,
        initial_base_height=HEIGHT_M,
        command=command,
        height_conditioned_action_default=True,
        action_delay_steps=0,
        action_delay=shared.action_delay.model_copy(update={"min_ms": 0.0, "max_ms": 0.0}),
        yaw_pid=YawPidConfig(enabled=False),
    )
    robot = WheelLeggedRobot(cfg=cfg, runtime=RuntimeSpec())
    robot.reset()
    set_height_default_pose(robot)
    return robot


def action_for_case(case: RolloutCase, profiles: np.ndarray) -> np.ndarray:
    """生成左右轮独立触发的 CTBC raw action bias。"""
    action = np.zeros(6, dtype=np.float64)
    action[0] = -case.front_amp * profiles[0]
    action[1] = case.active_amp * profiles[0]
    action[2] = case.front_amp * profiles[1]
    action[3] = case.active_amp * profiles[1]
    return action


def riser_contact_force_xy(robot: WheelLeggedRobot, step_geom_id: int) -> np.ndarray:
    """近似训练端 wheel_riser_sensor：只统计台阶立面上的轮子水平法向力。"""
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
    case: RolloutCase,
    *,
    t: float,
    wheel_front_x: float,
    riser_force: np.ndarray,
    phases: np.ndarray,
    cooldown: np.ndarray,
    contact_buf: np.ndarray,
    trigger_times: list[float],
) -> None:
    """按指定触发器推进左右两侧 CTBC 相位。"""
    cooldown[cooldown > 0] -= 1

    newly_triggered = np.zeros(2, dtype=bool)
    if case.trigger_mode == "distance":
        hit = wheel_front_x >= STEP_RISER_X_M - case.distance_margin_m
        newly_triggered[:] = hit
    elif case.trigger_mode == "contact":
        contact_buf[1:] = contact_buf[:-1]
        contact_buf[0] = riser_force
        newly_triggered = np.all(contact_buf > FORCE_THRESHOLD_N, axis=0)

    can_trigger = (phases < 0) & (cooldown == 0)
    newly_triggered &= can_trigger
    phases[newly_triggered] = 0
    if bool(np.any(newly_triggered)):
        trigger_times.append(t)


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


def plot_trace(case: RolloutCase, trace: RolloutTrace) -> Path:
    """保存触发信号图。"""
    out = OUT_DIR / f"{case.name}_signals.png"
    time = np.asarray(trace.time_s)
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.2), sharex=True)

    axes[0].plot(time, np.asarray(trace.wheel_front_distance_m) * 1000.0, color="#1f77b4")
    axes[0].axhline(-case.distance_margin_m * 1000.0, color="#888", linestyle="--", linewidth=1)
    axes[0].axhline(0.0, color="#222", linewidth=1)
    axes[0].set_ylabel("front-riser mm")
    axes[0].set_title(f"{case.name} ({case.trigger_mode})")

    axes[1].plot(time, trace.riser_force_left_n, label="left", color="#d62728")
    axes[1].plot(time, trace.riser_force_right_n, label="right", color="#2ca02c")
    axes[1].axhline(FORCE_THRESHOLD_N, color="#888", linestyle="--", linewidth=1)
    axes[1].set_ylabel("riser force N")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, trace.profile_left, label="left", color="#d62728")
    axes[2].plot(time, trace.profile_right, label="right", color="#2ca02c")
    axes[2].set_ylabel("bias profile")
    axes[2].set_xlabel("time s")
    axes[2].legend(loc="upper right")

    for ax in axes:
        for trigger_time in trace.trigger_times_s:
            ax.axvline(trigger_time, color="#111", linestyle=":", linewidth=1.2)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def rollout(case: RolloutCase, scene_xml: Path) -> RolloutSummary:
    """运行一个物理回放并保存视频、信号图。"""
    robot = make_robot(scene_xml)
    policy = PolicyRuntime(checkpoint=CHECKPOINT, device="cpu", runtime=RuntimeSpec())
    policy.reset()
    renderer = mujoco.Renderer(robot.model, height=544, width=960)

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

    frames: list[np.ndarray] = []
    trace = RolloutTrace([], [], [], [], [], [], [])
    phases = np.full(2, -1, dtype=np.int64)
    cooldown = np.zeros(2, dtype=np.int64)
    contact_buf = np.zeros((CONTACT_WINDOW, 2), dtype=np.float64)
    max_x_past = -math.inf
    max_bottom_over_step = -math.inf
    max_riser_force = 0.0
    nonwheel_contacts = 0
    fall = False
    control_steps = int(SECONDS / robot.control_dt)
    ff_steps = max(1, round(FF_PERIOD_S / robot.control_dt))
    render_every = max(1, round(1.0 / (FPS * robot.control_dt)))

    obs = robot.observation()
    for step in range(control_steps):
        t = step * robot.control_dt
        wheel_pos = np.stack((robot.data.xpos[left_id], robot.data.xpos[right_id]))
        wheel_front_x = float(wheel_pos[:, 0].max() + WHEEL_RADIUS_M)
        riser_force = riser_contact_force_xy(robot, step_geom_id)
        max_riser_force = max(max_riser_force, float(np.max(riser_force)))

        update_trigger_state(
            case,
            t=t,
            wheel_front_x=wheel_front_x,
            riser_force=riser_force,
            phases=phases,
            cooldown=cooldown,
            contact_buf=contact_buf,
            trigger_times=trace.trigger_times_s,
        )
        profiles = profile_from_phases(phases, ff_steps)

        trace.time_s.append(t)
        trace.wheel_front_distance_m.append(wheel_front_x - STEP_RISER_X_M)
        trace.riser_force_left_n.append(float(riser_force[0]))
        trace.riser_force_right_n.append(float(riser_force[1]))
        trace.profile_left.append(float(profiles[0]))
        trace.profile_right.append(float(profiles[1]))

        policy_action = np.asarray(policy.act(obs), dtype=np.float64)
        action = policy_action + action_for_case(case, profiles)
        obs, _, done, info = robot.step(action)
        advance_phases(phases, cooldown, ff_steps)

        if bool(info.get("nonwheel_contact", False)):
            nonwheel_contacts += 1
        fall = fall or bool(info.get("fall_detected", False)) or bool(done)

        wheel_pos = np.stack((robot.data.xpos[left_id], robot.data.xpos[right_id]))
        max_x_past = max(max_x_past, float(wheel_pos[:, 0].min() - STEP_RISER_X_M))
        max_bottom_over_step = max(
            max_bottom_over_step, float(wheel_pos[:, 2].min() - WHEEL_RADIUS_M - STEP_HEIGHT_M)
        )

        if step % render_every == 0:
            renderer.update_scene(robot.data, camera=cam, scene_option=opt)
            frames.append(renderer.render())

    climbed = max_x_past > WHEEL_RADIUS_M and max_bottom_over_step > -0.01 and not fall
    video = OUT_DIR / f"{case.name}.mp4"
    imageio.mimsave(video, frames, fps=FPS, quality=8)
    plot = plot_trace(case, trace)
    renderer.close()

    trigger_time = trace.trigger_times_s[0] if trace.trigger_times_s else None
    trigger_distance = None
    if trigger_time is not None and trace.time_s:
        idx = int(np.argmin(np.abs(np.asarray(trace.time_s) - trigger_time)))
        trigger_distance = trace.wheel_front_distance_m[idx]

    return RolloutSummary(
        name=case.name,
        trigger_mode=case.trigger_mode,
        video=video,
        plot=plot,
        climbed=climbed,
        trigger_time_s=trigger_time,
        trigger_distance_m=trigger_distance,
        max_wheel_x_past_riser_m=max_x_past,
        max_wheel_bottom_over_step_m=max_bottom_over_step,
        max_riser_force_n=max_riser_force,
        nonwheel_contact_rate=nonwheel_contacts / max(1, control_steps),
        fall=fall,
    )


def write_html(summaries: list[RolloutSummary], out: Path) -> None:
    """写出回放页面。"""
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
            f"<td>{item.nonwheel_contact_rate:.3f}</td>"
            f"<td>{'yes' if item.fall else 'no'}</td>"
            "</tr>"
        )
        cards.append(
            f"""
            <section>
              <h2>{html.escape(item.name)}</h2>
              <video src="{item.video.name}" controls loop muted autoplay></video>
              <img src="{item.plot.name}" alt="{html.escape(item.name)} signals">
            </section>
            """
        )
    out.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>CTBC stair trigger rollout</title>
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
  </style>
</head>
<body>
<main>
  <h1>CTBC stair trigger rollout</h1>
  <p>同一个 recovery checkpoint 下对比触发器：distance 是几何提前触发，contact 是训练端近似的 riser 水平接触力触发。黑色虚线为触发时刻，力阈值为 30N，连续窗口为 3 个 policy step。</p>
  <table>
    <thead>
      <tr><th>case</th><th>trigger</th><th>climbed</th><th>trigger s</th><th>trigger dist mm</th><th>max riser N</th><th>min wheel x past riser mm</th><th>min wheel bottom over step mm</th><th>nonwheel contact</th><th>fall</th></tr>
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_xml = build_stair_xml(BASE_XML)
    summaries = [rollout(case, scene_xml) for case in CASES]
    html_path = OUT_DIR / "index.html"
    write_html(summaries, html_path)
    print(html_path)
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
            "nonwheel=",
            item.nonwheel_contact_rate,
            "fall=",
            item.fall,
        )


if __name__ == "__main__":
    main()
