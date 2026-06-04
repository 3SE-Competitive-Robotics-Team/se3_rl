"""SE3 sim2sim workflow 的命令行入口。"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from se3_shared import ActionDelayConfig

from .config import (
    MAX_YAW_RATE_RAD_S,
    JumpEventConfig,
    JumpScheduleConfig,
    PolicyConfig,
    RobotConfig,
    RunConfig,
    ViewerConfig,
    YawPidConfig,
)
from .course import CourseConfig, CourseType


def _yaw_max_rate(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed > MAX_YAW_RATE_RAD_S:
        raise argparse.ArgumentTypeError(
            f"--yaw-max-rate must be in (0, {MAX_YAW_RATE_RAD_S}], got {parsed}"
        )
    return parsed


def _parse_unit_float(value: str, *, name: str, suffixes: tuple[str, ...]) -> float:
    text = value.strip().lower()
    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    if not text:
        raise argparse.ArgumentTypeError(f"{name} is empty")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"{name} must be finite, got {value!r}")
    return parsed


def _parse_jump_script(value: str) -> tuple[JumpEventConfig, ...]:
    """解析跳跃脚本 DSL，例如 `3s:0.4m, 8s:0.2m`。"""
    events: list[JumpEventConfig] = []
    for token in re.split(r"[,;]", value.strip()):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"jump script item must be '<time>:<height>', got {item!r}"
            )
        time_raw, height_raw = item.split(":", 1)
        trigger_time_s = _parse_unit_float(time_raw, name="jump time", suffixes=("sec", "s"))
        target_height = _parse_unit_float(height_raw, name="jump height", suffixes=("m",))
        try:
            events.append(
                JumpEventConfig(trigger_time_s=trigger_time_s, target_height=target_height)
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    if not events:
        raise argparse.ArgumentTypeError("jump script must contain at least one event")
    last_time = -math.inf
    for event in events:
        if event.trigger_time_s <= last_time:
            raise argparse.ArgumentTypeError("jump script event times must be strictly increasing")
        last_time = event.trigger_time_s
    return tuple(events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SE3 MuJoCo sim2sim workflow")
    robot_defaults = RobotConfig()
    delay_defaults = robot_defaults.action_delay
    yaw_defaults = robot_defaults.yaw_pid
    parser.add_argument("--model", type=Path, default=robot_defaults.model_path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Policy checkpoint. Defaults to the latest logs/rsl_rl/se3_wheel_leg/*/model_*.pt.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum policy steps to run. Use 0 for unlimited.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=robot_defaults.sim_dt,
        help="MuJoCo world timestep in seconds. Default is 0.005 for 200 Hz.",
    )
    parser.add_argument(
        "--control-decimation",
        type=int,
        default=robot_defaults.control_decimation,
        help="Number of MuJoCo steps per policy action. Default 4 gives 50 Hz control at 0.005s sim_dt.",
    )
    parser.add_argument("--viewer", choices=["rerun", "none"], default="rerun")
    parser.add_argument("--rerun-app-id", default="se3_sim2sim")
    parser.add_argument("--rerun-address", default=None)
    parser.add_argument("--rerun-record", type=Path, default=None)
    parser.add_argument(
        "--rerun-memory-limit",
        default="1GB",
        help="Rerun viewer 内存上限。默认 1GB,超过后由 Rerun 丢弃最老数据。",
    )
    parser.add_argument("--no-rerun-spawn", action="store_true")
    parser.add_argument("--viewer-log-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--print-debug", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--random-reset", action="store_true")
    parser.add_argument("--randomize-root", action="store_true")
    parser.add_argument(
        "--initial-roll-deg",
        type=float,
        default=0.0,
        help="Reset 初始 roll 角度（度），用于倒地自启 sim2sim/Rerun 回放。",
    )
    parser.add_argument(
        "--initial-pitch-deg",
        type=float,
        default=0.0,
        help="Reset 初始 pitch 角度（度），用于倒地自启 sim2sim/Rerun 回放。",
    )
    parser.add_argument(
        "--initial-yaw-deg",
        type=float,
        default=0.0,
        help="Reset 初始 yaw 角度（度）。训练端 yaw 随机化范围为 ±180°。",
    )
    parser.add_argument(
        "--initial-base-height",
        type=float,
        default=None,
        help="Reset 初始 base 高度（米）。默认使用共享站立高度。",
    )
    parser.add_argument(
        "--command",
        type=float,
        nargs=8,
        metavar=(
            "LIN_X",
            "YAW",
            "PITCH",
            "ROLL",
            "HEIGHT",
            "JUMP_FLAG",
            "JUMP_HEIGHT",
            "JUMP_PHASE",
        ),
        default=robot_defaults.command,
        help="Policy command: lin_vel_x yaw_rate pitch roll height jump_flag jump_target_height jump_phase. "
        "jump_phase is maintained automatically by the workflow; pass 0.0. "
        "Yaw slot is overwritten when --yaw-pid is enabled. "
        "Use --jump-interval-s to trigger periodic jumps.",
    )
    parser.add_argument(
        "--yaw-pid",
        dest="yaw_pid",
        action="store_true",
        default=yaw_defaults.enabled,
        help="Enable yaw PID control. Enabled by default.",
    )
    parser.add_argument(
        "--no-yaw-pid",
        dest="yaw_pid",
        action="store_false",
        help="Disable yaw PID control and keep the yaw slot from --command.",
    )
    parser.add_argument(
        "--yaw-target-deg",
        type=float,
        default=math.degrees(yaw_defaults.target_yaw_rad),
        help="Target yaw angle in degrees for the yaw PID controller.",
    )
    parser.add_argument("--yaw-kp", type=float, default=yaw_defaults.kp)
    parser.add_argument("--yaw-ki", type=float, default=yaw_defaults.ki)
    parser.add_argument("--yaw-kd", type=float, default=yaw_defaults.kd)
    parser.add_argument("--yaw-max-rate", type=_yaw_max_rate, default=yaw_defaults.max_rate)
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=None,
        help="Legacy fixed delay in MuJoCo sim steps. Overrides --action-delay-ms when set.",
    )
    parser.add_argument(
        "--action-delay-ms",
        type=float,
        default=delay_defaults.delay_s * 1000.0,
        help="Nominal action delay in milliseconds.",
    )
    parser.add_argument(
        "--action-delay-min-ms",
        type=float,
        default=delay_defaults.min_delay_s * 1000.0,
        help="Minimum randomized action delay in milliseconds.",
    )
    parser.add_argument(
        "--action-delay-max-ms",
        type=float,
        default=delay_defaults.max_delay_s * 1000.0,
        help="Maximum randomized action delay in milliseconds.",
    )
    parser.add_argument(
        "--action-delay-randomize",
        dest="action_delay_randomize",
        action="store_true",
        default=delay_defaults.randomize,
        help="Enable per-reset action delay randomization.",
    )
    parser.add_argument(
        "--no-action-delay-randomize",
        dest="action_delay_randomize",
        action="store_false",
        help="Disable action delay randomization and use --action-delay-ms.",
    )
    parser.add_argument(
        "--no-action-delay",
        action="store_true",
        help="Disable action delay entirely.",
    )
    parser.add_argument(
        "--leg-kp",
        type=float,
        default=robot_defaults.leg_kp,
        help="Leg joint position PD stiffness.",
    )
    parser.add_argument(
        "--leg-kd",
        type=float,
        default=robot_defaults.leg_kd,
        help="Leg joint position PD damping.",
    )
    parser.add_argument("--terminate-on-fall", action="store_true")
    parser.add_argument("--fail-tilt-deg", type=float, default=80.0)
    parser.add_argument("--fail-height-m", type=float, default=0.12)

    # 历程（Course）：指令扫描序列
    parser.add_argument(
        "--course",
        type=str,
        default=CourseType.NONE.value,
        choices=[t.value for t in CourseType],
        help="指令历程模式。"
        " walk-sweep: 前进速度扫描 0.1→0.6 m/s 每档 5 秒。"
        " jump-sweep: 跳跃高度扫描 0.1→0.6 m。"
        " upright-velocity-sweep: 自起后 locomotion 验收速度扫描。"
        " none: 固定指令（默认）。",
    )

    # 定时跳跃调度
    sched_defaults = JumpScheduleConfig()
    parser.add_argument(
        "--jump-interval-s",
        type=float,
        default=None,
        help="开启定时跳跃模式：每隔此秒触发一次原地垂直跳跃（上一次参考轨迹结束后开始计时）。"
        "启用后 --command 的 jump_flag 位被忽略。",
    )
    parser.add_argument(
        "--jump-target-height",
        type=float,
        default=sched_defaults.target_height,
        help="定时跳跃目标离地高度 (m)，0.1~0.6。默认 %(default)s m。",
    )
    parser.add_argument(
        "--jump-script",
        type=_parse_jump_script,
        default=(),
        metavar="TIME:HEIGHT[,TIME:HEIGHT...]",
        help="按绝对时间触发跳跃的简单 DSL，例如 '3s:0.4m,8s:0.2m'。"
        "时间单位默认秒，高度单位默认米；不能和 --jump-interval-s 同时使用。",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    action_delay = ActionDelayConfig(
        enabled=not bool(args.no_action_delay),
        delay_s=float(args.action_delay_ms) / 1000.0,
        randomize=bool(args.action_delay_randomize),
        min_delay_s=float(args.action_delay_min_ms) / 1000.0,
        max_delay_s=float(args.action_delay_max_ms) / 1000.0,
    )
    return RunConfig(
        robot=RobotConfig(
            model_path=args.model,
            seed=int(args.seed),
            sim_dt=float(args.sim_dt),
            control_decimation=int(args.control_decimation),
            initial_roll_rad=math.radians(float(args.initial_roll_deg)),
            initial_pitch_rad=math.radians(float(args.initial_pitch_deg)),
            initial_yaw_rad=math.radians(float(args.initial_yaw_deg)),
            initial_base_height=(
                None if args.initial_base_height is None else float(args.initial_base_height)
            ),
            command=tuple(float(v) for v in args.command),
            yaw_pid=YawPidConfig(
                enabled=bool(args.yaw_pid),
                target_yaw_rad=math.radians(float(args.yaw_target_deg)),
                kp=float(args.yaw_kp),
                ki=float(args.yaw_ki),
                kd=float(args.yaw_kd),
                max_rate=float(args.yaw_max_rate),
            ),
            action_delay=action_delay,
            action_delay_steps=(
                None if args.action_delay_steps is None else max(0, int(args.action_delay_steps))
            ),
            leg_kp=float(args.leg_kp),
            leg_kd=float(args.leg_kd),
            jump_schedule=JumpScheduleConfig(
                enabled=args.jump_interval_s is not None,
                interval_s=float(args.jump_interval_s) if args.jump_interval_s is not None else 5.0,
                target_height=float(args.jump_target_height),
                events=tuple(args.jump_script),
            ),
        ),
        policy=PolicyConfig(
            checkpoint=args.checkpoint,
            device=str(args.device),
        ),
        viewer=ViewerConfig(
            mode=args.viewer,
            app_id=str(args.rerun_app_id),
            spawn=not bool(args.no_rerun_spawn),
            address=args.rerun_address,
            record_to_rrd=args.rerun_record,
            memory_limit=str(args.rerun_memory_limit),
            log_every=max(1, int(args.viewer_log_every)),
        ),
        max_steps=int(args.max_steps),
        fixed_reset=not bool(args.random_reset),
        randomize_root=bool(args.randomize_root),
        print_every=int(args.print_every),
        print_debug=bool(args.print_debug),
        course=CourseConfig(mode=CourseType(args.course)),
        json_output=args.json_output,
        terminate_on_fall=bool(args.terminate_on_fall),
        fail_tilt_deg=float(args.fail_tilt_deg),
        fail_height_m=float(args.fail_height_m),
    )


def main() -> int:
    from .workflow import run_sim2sim

    parser = build_parser()
    args = parser.parse_args()
    if args.jump_interval_s is not None and args.jump_script:
        parser.error("--jump-script cannot be used together with --jump-interval-s")
    summary = run_sim2sim(config_from_args(args))
    rollout = summary["rollout"]
    final = rollout.get("final", {}) if isinstance(rollout, dict) else {}
    robot_cfg = summary["config"]["robot"]
    sim_dt = float(robot_cfg["sim_dt"])
    control_decimation = int(robot_cfg["control_decimation"])
    action_delay_cfg = robot_cfg["action_delay"]
    if isinstance(action_delay_cfg, dict):
        delay_enabled = bool(action_delay_cfg["enabled"])
        delay_randomize = bool(action_delay_cfg["randomize"])
    else:
        delay_enabled = False
        delay_randomize = False
    action_delay_steps = int(final.get("action_delay_steps", 0)) if final else 0
    action_delay_s = (
        float(final.get("action_delay_s", action_delay_steps * sim_dt)) if final else 0.0
    )
    print("Final summary:")
    print(f"  done_reason={summary['done_reason']}")
    print(f"  checkpoint={summary['policy']['checkpoint']}")
    print(f"  model_issues={len(summary['model_diagnostics']['issues'])}")
    print(
        f"  sim_dt={sim_dt:.4f}s control_dt={sim_dt * control_decimation:.4f}s "
        f"action_delay={action_delay_s * 1000.0:.1f}ms "
        f"steps={action_delay_steps} enabled={delay_enabled} randomize={delay_randomize}"
    )
    if final:
        print(
            f"  final_height={float(final['height']):.3f} "
            f"final_tilt_deg={float(final['tilt_deg']):.2f}"
        )
    jump_events = summary.get("jump_events")
    if isinstance(jump_events, list) and jump_events:
        print("Jump event diagnostics:")
        for event in jump_events:
            if not isinstance(event, dict) or int(event.get("samples", 0)) <= 0:
                continue
            print(
                f"  t={float(event['trigger_time_s']):.2f}s "
                f"h={float(event['target_height']):.2f}m "
                f"max_base_h={float(event['max_base_height']):.3f} "
                f"max_pitch={float(event['max_abs_pitch_deg']):.1f}deg "
                f"max_yaw={float(event['max_abs_yaw_deg']):.1f}deg"
            )
            phases = event.get("phases")
            if not isinstance(phases, dict):
                continue
            for name in ("takeoff", "early_air", "apex", "landing"):
                phase = phases.get(name)
                if not isinstance(phase, dict) or int(phase.get("samples", 0)) <= 0:
                    continue
                print(
                    f"    {name}: "
                    f"pitch_mean={float(phase['mean_abs_pitch_deg']):.1f}deg "
                    f"pitch_max={float(phase['max_abs_pitch_deg']):.1f}deg "
                    f"pitch_rate_max={float(phase['max_abs_pitch_rate_rad_s']):.2f}rad/s "
                    f"tilt_max={float(phase['max_tilt_deg']):.1f}deg "
                    f"action_rate_mean={float(phase['mean_action_delta_sq_sum']):.2f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
