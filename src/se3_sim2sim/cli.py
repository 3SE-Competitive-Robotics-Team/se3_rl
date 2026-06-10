"""SE3 sim2sim workflow 的命令行入口。"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from se3_shared import ActionDelayConfig

from .config import (
    DEFAULT_SIM_MODEL_VARIANT,
    MAX_YAW_RATE_RAD_S,
    SIM_MODEL_VARIANT_CHOICES,
    JumpEventConfig,
    JumpScheduleConfig,
    PolicyConfig,
    RobotConfig,
    RunConfig,
    ViewerConfig,
    YawPidConfig,
    model_path_for_variant,
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
    parser.add_argument(
        "--model-variant",
        choices=SIM_MODEL_VARIANT_CHOICES,
        default=DEFAULT_SIM_MODEL_VARIANT,
        help="选择内置 MJCF 模型变体：fourbar-surrogate 为训练默认等效开树，closedchain 为真实闭链 OBB 对照，openchain 为旧开链模型。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="直接指定 MJCF 路径；设置后覆盖 --model-variant。",
    )
    parser.add_argument(
        "--task",
        default=robot_defaults.task,
        help="sim2sim runtime task id.",
    )
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
        "--rerun-geom-view",
        choices=("visual", "collision", "both"),
        default=ViewerConfig().geom_view,
        help="Rerun 3D 场景显示的 MJCF 几何。visual 显示外观模型，collision 显示接触几何，both 同时显示。",
    )
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
        "--initial-ang-vel-deg-s",
        type=float,
        nargs=3,
        metavar=("ROLL_RATE", "PITCH_RATE", "YAW_RATE"),
        default=(0.0, 0.0, 0.0),
        help="Reset 初始 base 角速度（deg/s），按 roll/pitch/yaw 轴顺序写入 MuJoCo qvel[3:6]。",
    )
    parser.add_argument(
        "--initial-base-height",
        type=float,
        default=None,
        help="Reset 初始 base 高度（米）。默认使用共享站立高度。",
    )
    parser.add_argument(
        "--initial-leg-joint-pos",
        type=float,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "Reset 腿部初始关节位置。传 2 个值时左右镜像成 [LF, LB, RF, RB]；"
            "传 4 个值时按 policy 腿部顺序写入。"
        ),
    )
    parser.add_argument(
        "--settle-base-before-policy",
        action="store_true",
        help="reset 后先用零控制等待 base_link 接地，再开始 policy 推理和 Rerun 录制。",
    )
    parser.add_argument(
        "--pre-policy-settle-max-s",
        type=float,
        default=robot_defaults.pre_policy_settle_max_s,
        help="base_link 贴地后额外等待接触稳定的最大仿真时间（秒）；默认 0 表示不做被动滚动。",
    )
    parser.add_argument(
        "--pre-policy-settle-contact-steps",
        type=int,
        default=robot_defaults.pre_policy_settle_contact_steps,
        help="base_link 连续接地多少个 MuJoCo step 后开始 policy。",
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
        "--height-conditioned-action-default",
        dest="height_conditioned_action_default",
        action="store_true",
        default=None,
        help="让腿部 action=0 对应当前 command height 下的默认腿型；不传时 recovery deploy npz 自动启用。",
    )
    parser.add_argument(
        "--no-height-conditioned-action-default",
        dest="height_conditioned_action_default",
        action="store_false",
        help="显式关闭 height-conditioned action 默认腿型，覆盖 recovery deploy npz 自动检测。",
    )
    parser.add_argument(
        "--active-rod-target-lower-preload-margin",
        type=float,
        default=robot_defaults.active_rod_target_lower_preload_margin,
        help="Active rod lower target preload margin for recovery checkpoints.",
    )
    parser.add_argument(
        "--active-rod-target-upper-preload-margin",
        type=float,
        default=robot_defaults.active_rod_target_upper_preload_margin,
        help="Active rod upper target preload margin for recovery checkpoints.",
    )
    parser.add_argument(
        "--yaw-pid",
        dest="yaw_pid",
        action="store_true",
        default=None,
        help="Enable yaw PID control. Defaults to enabled only for upright resets.",
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


def _yaw_pid_enabled_from_args(args: argparse.Namespace) -> bool:
    """倒地自起 reset 默认不注入 yaw PID 指令，避免倒置 yaw 奇异点污染 actor 输入。"""
    if args.yaw_pid is not None:
        return bool(args.yaw_pid)
    initial_tilt_requested = (
        abs(float(args.initial_roll_deg)) > 1.0e-6 or abs(float(args.initial_pitch_deg)) > 1.0e-6
    )
    return bool(RobotConfig().yaw_pid.enabled and not initial_tilt_requested)


def _height_conditioned_action_default_from_args(args: argparse.Namespace) -> bool:
    """解析 sim2sim action 默认腿型语义，deploy recovery npz 默认对齐真机。"""
    explicit = getattr(args, "height_conditioned_action_default", None)
    if explicit is not None:
        return bool(explicit)
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None:
        return bool(RobotConfig().height_conditioned_action_default)
    path = Path(checkpoint)
    if path.suffix.lower() == ".npz" and "recovery" in path.name.lower():
        return True
    return bool(RobotConfig().height_conditioned_action_default)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    action_delay = ActionDelayConfig(
        enabled=not bool(args.no_action_delay),
        delay_s=float(args.action_delay_ms) / 1000.0,
        randomize=bool(args.action_delay_randomize),
        min_delay_s=float(args.action_delay_min_ms) / 1000.0,
        max_delay_s=float(args.action_delay_max_ms) / 1000.0,
    )
    model_path = (
        args.model if args.model is not None else model_path_for_variant(str(args.model_variant))
    )
    return RunConfig(
        robot=RobotConfig(
            model_path=model_path,
            task=str(args.task),
            seed=int(args.seed),
            sim_dt=float(args.sim_dt),
            control_decimation=int(args.control_decimation),
            initial_roll_rad=math.radians(float(args.initial_roll_deg)),
            initial_pitch_rad=math.radians(float(args.initial_pitch_deg)),
            initial_yaw_rad=math.radians(float(args.initial_yaw_deg)),
            initial_ang_vel_rad_s=tuple(math.radians(float(v)) for v in args.initial_ang_vel_deg_s),
            initial_base_height=(
                None if args.initial_base_height is None else float(args.initial_base_height)
            ),
            initial_leg_joint_pos=_initial_leg_joint_pos_from_args(args.initial_leg_joint_pos),
            settle_base_before_policy=bool(args.settle_base_before_policy),
            pre_policy_settle_max_s=float(args.pre_policy_settle_max_s),
            pre_policy_settle_contact_steps=max(1, int(args.pre_policy_settle_contact_steps)),
            command=tuple(float(v) for v in args.command),
            yaw_pid=YawPidConfig(
                enabled=_yaw_pid_enabled_from_args(args),
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
            height_conditioned_action_default=_height_conditioned_action_default_from_args(args),
            active_rod_target_lower_preload_margin=float(
                args.active_rod_target_lower_preload_margin
            ),
            active_rod_target_upper_preload_margin=float(
                args.active_rod_target_upper_preload_margin
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
            geom_view=str(args.rerun_geom_view),
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


def _initial_leg_joint_pos_from_args(values: list[float] | None) -> tuple[float, ...] | None:
    """规范化 CLI 输入的腿部初始关节位置。"""
    if values is None:
        return None
    if len(values) == 2:
        left_front, left_back = (float(v) for v in values)
        return (left_front, left_back, left_front, left_back)
    if len(values) == 4:
        return tuple(float(v) for v in values)
    raise ValueError("--initial-leg-joint-pos expects 2 mirrored values or 4 policy-order values")


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
    settle = summary.get("pre_policy_settle")
    if isinstance(settle, dict) and bool(settle.get("enabled", False)):
        print(
            "  pre_policy_settle="
            f"success={bool(settle.get('success', False))} "
            f"steps={int(settle.get('steps', 0))} "
            f"time={float(settle.get('time_s', 0.0)):.3f}s "
            f"base_touching={bool(settle.get('base_touching', False))} "
            f"base_contact={bool(settle.get('base_contact', False))}"
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
