"""评估纯倒地自起站立 checkpoint，并为每个 checkpoint 录制固定 Rerun。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from se3_sim2sim.config import PolicyConfig, RobotConfig, RunConfig, ViewerConfig, YawPidConfig
from se3_sim2sim.workflow import run_sim2sim

_STAND_COMMAND = (0.0, 0.0, 0.0, 0.0, 0.22, 0.0, 0.0, 0.0)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="评估 Recovery-Stand checkpoint。")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/recovery_stand_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-rerun", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--viewer-log-every", type=int, default=2)
    parser.add_argument("--rerun-memory-limit", default="512MB")
    parser.add_argument("--case", choices=("roll90", "pitch90", "inverted"), default="roll90")
    parser.add_argument("--max-final-tilt-deg", type=float, default=15.0)
    parser.add_argument("--max-height-error-m", type=float, default=0.05)
    parser.add_argument("--min-dual-wheel-contact-rate", type=float, default=0.5)
    parser.add_argument("--max-nonwheel-contact-rate", type=float, default=0.2)
    parser.add_argument(
        "--allow-final-single-wheel-contact",
        action="store_true",
        help="允许终态只有单轮触地，仅用于诊断历史 checkpoint。",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="未通过阈值时返回非 0；默认只表示评估流程是否成功执行。",
    )
    return parser


def main() -> int:
    """执行单 checkpoint 的纯站起评估。"""
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout = _run_case(args, checkpoint, output_dir)
    check = _check_case(args, rollout)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "case": args.case,
        "rollout": rollout,
        "checks": {"recovery_stand": check},
        "passed": bool(check["passed"]),
    }
    summary_path = output_dir / f"{checkpoint.stem}_recovery_stand_eval.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[eval] summary={summary_path}")
    print(f"[eval] passed={result['passed']}")
    return 1 if args.fail_on_threshold and not result["passed"] else 0


def _run_case(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """按指定初始姿态跑一次 sim2sim。"""
    name = f"{checkpoint.stem}_recovery_stand_{args.case}"
    json_output = output_dir / f"{name}.json"
    rrd_output = output_dir / f"{name}.rrd" if args.record_rerun else None
    cfg = RunConfig(
        robot=RobotConfig(
            yaw_pid=YawPidConfig(enabled=False),
            command=_STAND_COMMAND,
            initial_base_height=0.16,
        ),
        policy=PolicyConfig(checkpoint=checkpoint, device=str(args.device)),
        viewer=ViewerConfig(
            mode="rerun" if rrd_output is not None else "none",
            spawn=False,
            record_to_rrd=rrd_output,
            memory_limit=str(args.rerun_memory_limit),
            log_every=max(1, int(args.viewer_log_every)),
        ),
        max_steps=int(args.max_steps),
        print_every=int(args.print_every),
        json_output=json_output,
    )
    if args.case == "roll90":
        cfg.robot.initial_roll_rad = math.radians(90.0)
    elif args.case == "pitch90":
        cfg.robot.initial_pitch_rad = math.radians(90.0)
    elif args.case == "inverted":
        cfg.robot.initial_roll_rad = math.radians(180.0)
    summary = run_sim2sim(cfg)
    return {
        "json": str(json_output),
        "rrd": str(rrd_output) if rrd_output is not None else None,
        "done_reason": summary["done_reason"],
        "rollout": summary["rollout"],
    }


def _check_case(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """检查最终姿态和接触是否达到纯站立交接条件。"""
    rollout = payload["rollout"]
    final = rollout["final"]
    final_tilt = float(final["tilt_deg"])
    final_height = float(final["height"])
    height_error = abs(final_height - _STAND_COMMAND[4])
    dual_wheel_rate = min(
        float(rollout.get("wheel_contact_left_rate", 0.0)),
        float(rollout.get("wheel_contact_right_rate", 0.0)),
    )
    nonwheel_rate = float(rollout.get("nonwheel_contact_rate", 1.0))
    final_left_wheel_contact = float(final.get("wheel_contact_left", 0.0))
    final_right_wheel_contact = float(final.get("wheel_contact_right", 0.0))
    final_dual_wheel_contact = (
        float(final.get("wheel_full_contact", 0.0)) > 0.5
        and final_left_wheel_contact > 0.5
        and final_right_wheel_contact > 0.5
    )
    passed = (
        payload["done_reason"] == "max_steps"
        and final_tilt <= float(args.max_final_tilt_deg)
        and height_error <= float(args.max_height_error_m)
        and dual_wheel_rate >= float(args.min_dual_wheel_contact_rate)
        and nonwheel_rate <= float(args.max_nonwheel_contact_rate)
        and (args.allow_final_single_wheel_contact or final_dual_wheel_contact)
    )
    return {
        "passed": passed,
        "final_tilt_deg": final_tilt,
        "final_height_m": final_height,
        "final_height_error_m": height_error,
        "final_dual_wheel_contact": final_dual_wheel_contact,
        "final_left_wheel_contact": final_left_wheel_contact,
        "final_right_wheel_contact": final_right_wheel_contact,
        "dual_wheel_contact_rate": dual_wheel_rate,
        "nonwheel_contact_rate": nonwheel_rate,
        "max_tilt_deg": float(rollout["tilt_deg"]["max"]),
        "reset_floor_lift_m": float(final.get("reset_floor_lift_m", 0.0)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
