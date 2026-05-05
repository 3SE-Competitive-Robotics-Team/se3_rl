"""SE3 sim2sim workflow 的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PolicyConfig, RobotConfig, RunConfig, ViewerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SE3 MuJoCo sim2sim workflow")
    robot_defaults = RobotConfig()
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
        help="MuJoCo world timestep in seconds. Default is 0.001 for 1000 Hz.",
    )
    parser.add_argument(
        "--control-decimation",
        type=int,
        default=robot_defaults.control_decimation,
        help="Number of MuJoCo steps per policy action. Default 10 gives 100 Hz control at 0.001s sim_dt.",
    )
    parser.add_argument("--viewer", choices=["rerun", "none"], default="rerun")
    parser.add_argument("--rerun-app-id", default="se3_sim2sim")
    parser.add_argument("--rerun-address", default=None)
    parser.add_argument("--rerun-record", type=Path, default=None)
    parser.add_argument("--no-rerun-spawn", action="store_true")
    parser.add_argument("--viewer-log-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--print-debug", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--random-reset", action="store_true")
    parser.add_argument("--randomize-root", action="store_true")
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("LIN_X", "YAW", "HEIGHT"),
        default=robot_defaults.command,
        help="Policy command as lin_vel_x yaw_rate height.",
    )
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=robot_defaults.action_delay_steps,
        help="Applied action delay in MuJoCo sim steps.",
    )
    parser.add_argument(
        "--theta-kd",
        type=float,
        default=robot_defaults.theta_kd,
        help="VMC leg-angle damping gain.",
    )
    parser.add_argument("--terminate-on-fall", action="store_true")
    parser.add_argument("--fail-tilt-deg", type=float, default=80.0)
    parser.add_argument("--fail-height-m", type=float, default=0.12)
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        robot=RobotConfig(
            model_path=args.model,
            seed=int(args.seed),
            sim_dt=float(args.sim_dt),
            control_decimation=int(args.control_decimation),
            command=tuple(float(v) for v in args.command),
            action_delay_steps=max(0, int(args.action_delay_steps)),
            theta_kd=float(args.theta_kd),
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
            log_every=max(1, int(args.viewer_log_every)),
        ),
        max_steps=int(args.max_steps),
        fixed_reset=not bool(args.random_reset),
        randomize_root=bool(args.randomize_root),
        print_every=int(args.print_every),
        print_debug=bool(args.print_debug),
        json_output=args.json_output,
        terminate_on_fall=bool(args.terminate_on_fall),
        fail_tilt_deg=float(args.fail_tilt_deg),
        fail_height_m=float(args.fail_height_m),
    )


def main() -> int:
    from .workflow import run_sim2sim

    args = build_parser().parse_args()
    summary = run_sim2sim(config_from_args(args))
    rollout = summary["rollout"]
    final = rollout.get("final", {}) if isinstance(rollout, dict) else {}
    robot_cfg = summary["config"]["robot"]
    sim_dt = float(robot_cfg["sim_dt"])
    control_decimation = int(robot_cfg["control_decimation"])
    action_delay_steps = int(robot_cfg["action_delay_steps"])
    print("Final summary:")
    print(f"  done_reason={summary['done_reason']}")
    print(f"  model_issues={len(summary['model_diagnostics']['issues'])}")
    print(
        f"  sim_dt={sim_dt:.4f}s control_dt={sim_dt * control_decimation:.4f}s "
        f"action_delay={action_delay_steps * sim_dt * 1000.0:.1f}ms"
    )
    if final:
        print(
            f"  final_height={float(final['height']):.3f} "
            f"final_tilt_deg={float(final['tilt_deg']):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
