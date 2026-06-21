"""评估 recovery checkpoint 的自起能力和直立速度扫描表现。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from se3_sim2sim.config import (
    RECOVERY_COMMAND_HEIGHT_M,
    RECOVERY_POSE_CHOICES,
    RECOVERY_POSE_RP_RAD,
    SIM_MODEL_VARIANT_CHOICES,
    PolicyConfig,
    RobotConfig,
    RunConfig,
    ViewerConfig,
    YawPidConfig,
    model_path_for_variant,
)
from se3_sim2sim.course import CourseConfig, CourseType
from se3_sim2sim.workflow import run_sim2sim


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="评估倒地自起 checkpoint：标准自起 + upright velocity sweep。"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/recovery_eval"),
        help="保存 JSON/Rerun 的目录。",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-variant",
        choices=SIM_MODEL_VARIANT_CHOICES,
        default="closedchain",
        help="选择 sim2sim 内置 MJCF 模型变体；closedchain 会直接用真实闭链模型验证。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="直接指定 MJCF 路径；设置后覆盖 --model-variant。",
    )
    parser.add_argument("--record-rerun", action="store_true", help="保存 .rrd 回放。")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="策略未通过阈值时返回非 0；默认只表示评估流程是否成功执行。",
    )
    parser.add_argument("--skip-selfright", action="store_true")
    parser.add_argument("--skip-velocity", action="store_true")
    parser.add_argument("--selfright-max-steps", type=int, default=1000)
    parser.add_argument("--velocity-max-steps", type=int, default=3600)
    parser.add_argument(
        "--command-height",
        type=float,
        default=RECOVERY_COMMAND_HEIGHT_M,
        help="Recovery command height，默认与当前固定姿态评估一致为 0.26m。",
    )
    parser.add_argument(
        "--selfright-poses",
        nargs="+",
        choices=RECOVERY_POSE_CHOICES,
        default=list(RECOVERY_POSE_CHOICES),
        help="标准自起评估姿态列表，默认覆盖 standing/left_side/right_side/prone/supine。",
    )
    parser.add_argument(
        "--selfright-yaw-deg",
        type=float,
        default=0.0,
        help="标准自起姿态的初始 yaw，默认 0 度以对齐固定姿态批量评估。",
    )
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--viewer-log-every", type=int, default=2)
    parser.add_argument("--rerun-memory-limit", default="512MB")

    parser.add_argument("--selfright-max-tilt-deg", type=float, default=15.0)
    parser.add_argument("--selfright-min-height-m", type=float, default=0.16)
    parser.add_argument("--min-wheel-contact-rate", type=float, default=0.90)
    parser.add_argument("--max-leg-contact-rate", type=float, default=0.10)
    parser.add_argument("--max-nonwheel-contact-rate", type=float, default=0.10)
    parser.add_argument("--max-velocity-error-mps", type=float, default=0.30)
    parser.add_argument("--max-yaw-error-rad-s", type=float, default=0.25)
    parser.add_argument("--max-zero-wheel-speed-mps", type=float, default=0.25)
    parser.add_argument("--max-zero-base-speed-mps", type=float, default=0.10)
    return parser


def main() -> int:
    """执行 checkpoint 评估并返回进程退出码。"""
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "model_variant": str(args.model_variant) if args.model is None else "custom",
        "model": str(_model_path_from_args(args)),
        "command_height": float(args.command_height),
        "checks": {},
    }

    if not args.skip_selfright:
        selfright = _run_selfright(args, checkpoint, output_dir)
        result["selfright"] = selfright
        result["checks"]["selfright"] = _check_selfright(args, selfright)

    if not args.skip_velocity:
        velocity = _run_velocity_sweep(args, checkpoint, output_dir)
        result["velocity_sweep"] = velocity
        result["checks"]["velocity_sweep"] = _check_velocity_sweep(args, velocity)

    checks = result["checks"]
    passed = all(bool(check.get("passed", False)) for check in checks.values())
    result["passed"] = passed
    summary_path = output_dir / f"{checkpoint.stem}_recovery_eval.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[eval] summary={summary_path}")
    print(f"[eval] passed={passed}")
    return 1 if args.fail_on_threshold and not passed else 0


def _run_selfright(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """运行训练端标准姿态自起回放。"""
    cases: list[dict[str, Any]] = []
    for pose in args.selfright_poses:
        pose_name = str(pose)
        name = f"{checkpoint.stem}_standard_selfright_{pose_name}"
        json_output = output_dir / f"{name}.json"
        rrd_output = output_dir / f"{name}.rrd" if args.record_rerun else None
        cfg = _base_cfg(
            args,
            checkpoint,
            json_output=json_output,
            rrd_output=rrd_output,
            max_steps=args.selfright_max_steps,
        )
        roll, pitch = RECOVERY_POSE_RP_RAD[pose_name]
        cfg.robot.initial_roll_rad = float(roll)
        cfg.robot.initial_pitch_rad = float(pitch)
        cfg.robot.initial_yaw_rad = math.radians(float(args.selfright_yaw_deg))
        cfg.robot.initial_base_height = (
            float(args.command_height) if pose_name == "standing" else 0.16
        )
        cfg.robot.command = _recovery_command(float(args.command_height))
        summary = run_sim2sim(cfg)
        cases.append(
            {
                "pose": pose_name,
                "json": str(json_output),
                "rrd": str(rrd_output) if rrd_output is not None else None,
                "done_reason": summary["done_reason"],
                "rollout": summary["rollout"],
                "initial_policy_io": summary.get("initial_policy_io", {}),
            }
        )
    return {
        "poses": list(args.selfright_poses),
        "command_height": float(args.command_height),
        "done_reason": "standard_pose_sweep",
        "cases": cases,
    }


def _run_velocity_sweep(
    args: argparse.Namespace, checkpoint: Path, output_dir: Path
) -> dict[str, Any]:
    """运行直立速度扫描回放。"""
    name = f"{checkpoint.stem}_upright_velocity_sweep"
    json_output = output_dir / f"{name}.json"
    rrd_output = output_dir / f"{name}.rrd" if args.record_rerun else None
    cfg = _base_cfg(
        args,
        checkpoint,
        json_output=json_output,
        rrd_output=rrd_output,
        max_steps=args.velocity_max_steps,
    )
    cfg.robot.initial_base_height = float(args.command_height)
    cfg.robot.command = _recovery_command(float(args.command_height))
    cfg.course = CourseConfig(mode=CourseType.UPRIGHT_VELOCITY_SWEEP)
    summary = run_sim2sim(cfg)
    return {
        "json": str(json_output),
        "rrd": str(rrd_output) if rrd_output is not None else None,
        "done_reason": summary["done_reason"],
        "rollout": summary["rollout"],
        "cases": summary.get("upright_velocity_sweep", []),
    }


def _base_cfg(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    json_output: Path,
    rrd_output: Path | None,
    max_steps: int,
) -> RunConfig:
    """构造共享 sim2sim 配置。"""
    return RunConfig(
        robot=RobotConfig(
            model_path=_model_path_from_args(args),
            yaw_pid=YawPidConfig(enabled=False),
            height_conditioned_action_default=True,
        ),
        policy=PolicyConfig(checkpoint=checkpoint, device=str(args.device)),
        viewer=ViewerConfig(
            mode="rerun" if rrd_output is not None else "none",
            spawn=False,
            record_to_rrd=rrd_output,
            memory_limit=str(args.rerun_memory_limit),
            log_every=max(1, int(args.viewer_log_every)),
        ),
        max_steps=int(max_steps),
        print_every=int(args.print_every),
        json_output=json_output,
    )


def _model_path_from_args(args: argparse.Namespace) -> Path:
    """解析评估使用的 MJCF 路径。"""
    if args.model is not None:
        return args.model
    return model_path_for_variant(str(args.model_variant))


def _recovery_command(
    command_height: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """构造 recovery 策略使用的 8D command。"""
    return (0.0, 0.0, 0.0, 0.0, float(command_height), 0.0, 0.2, 0.0)


def _check_selfright(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """检查标准自起是否达标。"""
    cases = payload.get("cases")
    if isinstance(cases, list):
        case_checks = [_check_selfright_case(args, case) for case in cases]
        return {
            "passed": bool(case_checks) and all(case["passed"] for case in case_checks),
            "cases": case_checks,
        }
    return _check_selfright_case(args, payload)


def _check_selfright_case(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """检查单个标准自起姿态是否达标。"""
    rollout = payload["rollout"]
    final = rollout["final"]
    final_tilt = float(final["tilt_deg"])
    final_height = float(final["height"])
    passed = (
        payload["done_reason"] == "max_steps"
        and final_tilt <= float(args.selfright_max_tilt_deg)
        and final_height >= float(args.selfright_min_height_m)
    )
    return {
        "pose": str(payload.get("pose", "single")),
        "passed": passed,
        "done_reason": str(payload["done_reason"]),
        "final_tilt_deg": final_tilt,
        "final_height_m": final_height,
        "max_tilt_deg": float(rollout["tilt_deg"]["max"]),
    }


def _check_velocity_sweep(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    """检查直立速度扫描是否达标。"""
    cases = payload.get("cases", [])
    case_checks = [_check_velocity_case(args, case) for case in cases]
    return {
        "passed": bool(case_checks) and all(case["passed"] for case in case_checks),
        "cases": case_checks,
    }


def _check_velocity_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    """检查单个速度档是否满足轮式 locomotion 约束。"""
    samples = int(case.get("steady_samples", 0))
    vx_cmd = float(case.get("command_lin_vel_x", 0.0))
    yaw_cmd = float(case.get("command_yaw_rate", 0.0))
    wheel_contact_rate = float(case.get("wheel_contact_rate", 0.0))
    leg_contact_rate = float(case.get("leg_contact_rate", 1.0))
    nonwheel_contact_rate = float(case.get("nonwheel_contact_rate", 1.0))
    speed_error = float(case.get("mean_abs_velocity_error", math.inf))
    yaw_error = float(case.get("mean_abs_yaw_error", math.inf))
    zero_wheel_speed = float(case.get("mean_abs_wheel_lin_vel", math.inf))

    checks = {
        "has_samples": samples > 0,
        "wheel_contact": wheel_contact_rate >= float(args.min_wheel_contact_rate),
        "leg_contact": leg_contact_rate <= float(args.max_leg_contact_rate),
        "nonwheel_contact": nonwheel_contact_rate <= float(args.max_nonwheel_contact_rate),
    }
    if abs(vx_cmd) > 1.0e-6:
        checks["velocity_tracking"] = speed_error <= float(args.max_velocity_error_mps)
    elif abs(yaw_cmd) > 1.0e-6:
        checks["yaw_tracking"] = yaw_error <= float(args.max_yaw_error_rad_s)
    else:
        checks["zero_base_speed"] = speed_error <= float(args.max_zero_base_speed_mps)
        checks["zero_wheel_speed"] = zero_wheel_speed <= float(args.max_zero_wheel_speed_mps)

    return {
        "command_lin_vel_x": vx_cmd,
        "command_yaw_rate": yaw_cmd,
        "passed": all(checks.values()),
        "checks": checks,
        "steady_samples": samples,
        "mean_base_lin_vel_x": float(case.get("mean_base_lin_vel_x", 0.0)),
        "mean_abs_velocity_error": speed_error,
        "mean_yaw_rate": float(case.get("mean_yaw_rate", 0.0)),
        "mean_abs_yaw_error": yaw_error,
        "mean_abs_wheel_lin_vel": zero_wheel_speed,
        "wheel_contact_rate": wheel_contact_rate,
        "wheel_full_contact_rate": float(case.get("wheel_full_contact_rate", 0.0)),
        "leg_contact_rate": leg_contact_rate,
        "nonwheel_contact_rate": nonwheel_contact_rate,
        "min_leg_clearance": float(case.get("min_leg_clearance", 0.0)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
