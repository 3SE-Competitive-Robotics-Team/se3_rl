"""键盘交互式 sim2sim 验证入口。"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

from .cli import build_parser as build_sim2sim_parser
from .cli import config_from_args
from .course import CourseType
from .teleop_input import KeyboardTeleopSource
from .workflow import run_sim2sim

DEFAULT_TELEOP_CHECKPOINT = Path("assets/base_model/model_5999_gru.pt")
DEFAULT_MIN_COMMAND_HEIGHT = 0.195
DEFAULT_MAX_COMMAND_HEIGHT = 0.390
DEFAULT_COMMAND_HEIGHT_RATE = 0.12


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = build_sim2sim_parser()
    parser.prog = "se3-sim2sim-teleop"
    parser.description = "SE3 MuJoCo sim2sim keyboard teleop workflow"
    parser.set_defaults(
        course=CourseType.NONE.value,
        yaw_pid=False,
        rc_start_off=True,
        rerun_app_id="se3_sim2sim_teleop",
        print_every=25,
        checkpoint=DEFAULT_TELEOP_CHECKPOINT,
        viewer="mujoco",
    )
    _set_checkpoint_help(
        parser,
        f"Policy checkpoint. Defaults to the current repository base model: {DEFAULT_TELEOP_CHECKPOINT}.",
    )
    parser.add_argument(
        "--teleop-vx",
        type=_positive_float,
        default=0.5,
        help="W/S 按键对应的前后速度 command，单位 m/s。",
    )
    parser.add_argument(
        "--teleop-yaw-rate",
        type=_positive_float,
        default=0.8,
        help="A/D 按键对应的 yaw rate command，单位 rad/s。",
    )
    parser.add_argument(
        "--teleop-key-hold-s",
        type=_positive_float,
        default=0.25,
        help="按键停止重复后保持 command 的时间，单位秒。",
    )
    parser.add_argument(
        "--teleop-height-rate",
        "--teleop-height-step",
        dest="teleop_height_rate",
        type=_positive_float,
        default=DEFAULT_COMMAND_HEIGHT_RATE,
        help="按住站高/站低按键时 command height 的连续变化速度，单位 m/s。",
    )
    parser.add_argument(
        "--teleop-min-height",
        type=_positive_float,
        default=DEFAULT_MIN_COMMAND_HEIGHT,
        help="teleop 允许的最低 command height，单位 m。",
    )
    parser.add_argument(
        "--teleop-max-height",
        type=_positive_float,
        default=DEFAULT_MAX_COMMAND_HEIGHT,
        help="teleop 允许的最高 command height，单位 m。",
    )
    parser.add_argument(
        "--teleop-start-on",
        dest="rc_start_off",
        action="store_false",
        help="仿真开始时直接打开遥控器输出；默认需要先按 R。",
    )
    parser.add_argument(
        "--no-teleop-realtime",
        dest="teleop_realtime",
        action="store_false",
        default=True,
        help="不按真实时间节拍限速，仅用于自动化调试。",
    )
    parser.add_argument(
        "--rerun-record-dir",
        type=Path,
        default=Path("logs/rerun/sim2sim_teleop"),
        help="未显式传 --rerun-record 时自动保存 .rrd 的目录。",
    )
    parser.add_argument(
        "--no-auto-rerun-record",
        action="store_true",
        help="关闭交互入口默认 .rrd 记录。",
    )
    return parser


def _set_checkpoint_help(parser: argparse.ArgumentParser, help_text: str) -> None:
    for action in parser._actions:
        if "--checkpoint" in action.option_strings:
            action.help = help_text
            return


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.course != CourseType.NONE.value:
        parser.error("teleop 模式由键盘接管 command[0:2]，不能同时使用 --course")
    if args.jump_interval_s is not None and args.jump_script:
        parser.error("--jump-script cannot be used together with --jump-interval-s")

    cfg = config_from_args(args)
    if (
        cfg.viewer.mode != "none"
        and cfg.viewer.record_to_rrd is None
        and not bool(args.no_auto_rerun_record)
    ):
        cfg.viewer.record_to_rrd = _default_rerun_record_path(args.rerun_record_dir)

    source = KeyboardTeleopSource(
        command_height=float(cfg.robot.command[4]),
        default_command_height=float(cfg.robot.command[4]),
        command_lin_vel_x=float(args.teleop_vx),
        command_yaw_rate=float(args.teleop_yaw_rate),
        command_height_rate=float(args.teleop_height_rate),
        min_command_height=float(args.teleop_min_height),
        max_command_height=float(args.teleop_max_height),
        hold_s=float(args.teleop_key_hold_s),
        realtime=bool(args.teleop_realtime),
    )
    with source:
        print(source.help_text())
        if cfg.viewer.record_to_rrd is not None:
            print(f"[teleop] Rerun record: {cfg.viewer.record_to_rrd}")
        if not source.interactive:
            if int(cfg.max_steps) <= 0:
                parser.error("stdin 不是交互终端时必须设置 --max-steps，避免无限空跑")
            print("[teleop] stdin 不是交互终端，键盘 command 将保持 0。")
        summary = run_sim2sim(cfg, command_source=source)

    _print_summary(summary, cfg.viewer.record_to_rrd)
    return 0


def _default_rerun_record_path(record_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(record_dir) / f"sim2sim_teleop_{stamp}.rrd"


def _print_summary(summary: dict[str, object], record_to_rrd: Path | None) -> None:
    rollout = summary.get("rollout")
    final = rollout.get("final", {}) if isinstance(rollout, dict) else {}
    policy = summary.get("policy", {})
    print("Teleop summary:")
    print(f"  done_reason={summary.get('done_reason', 'unknown')}")
    if isinstance(policy, dict):
        print(f"  checkpoint={policy.get('checkpoint', '')}")
    if record_to_rrd is not None:
        print(f"  Rerun saved to: {record_to_rrd}")
    if final:
        print(
            f"  final_height={float(final['height']):.3f} "
            f"final_tilt_deg={float(final['tilt_deg']):.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
