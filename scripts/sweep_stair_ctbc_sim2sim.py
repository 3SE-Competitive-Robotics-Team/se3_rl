"""批量扫描 stair CTBC 参数，并按上台阶稳定性排序。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_path_list(value: str) -> list[Path]:
    return [Path(item) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("assets/base_model/model_4999.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/sweeps/stair_ctbc"))
    parser.add_argument("--levels", type=_parse_int_list, default=_parse_int_list("0,1,2,3,4,5"))
    parser.add_argument("--stair-step-height", type=float, default=None)
    parser.add_argument("--stair-step-heights", type=_parse_float_list, default=None)
    parser.add_argument("--stair-half-width", type=float, default=6.0)
    parser.add_argument(
        "--body-x",
        "--ff-x",
        dest="body_x",
        type=_parse_float_list,
        default=None,
        help="机体系腿部平面内的轮心后撤幅值；--ff-x 保留为兼容 alias。",
    )
    parser.add_argument(
        "--body-z",
        "--ff-lift",
        dest="body_z",
        type=_parse_float_list,
        default=None,
        help="机体系腿部平面内的轮心抬升幅值；--ff-lift 保留为兼容 alias。",
    )
    parser.add_argument("--leg-lengths", type=_parse_float_list, default=_parse_float_list("0.18"))
    parser.add_argument(
        "--swing-angles-deg",
        type=_parse_float_list,
        default=_parse_float_list("-35.0"),
    )
    parser.add_argument("--ff-rise", type=_parse_float_list, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--duration",
        "--ff-duration",
        "--ff-period",
        dest="duration",
        type=_parse_float_list,
        default=_parse_float_list("0.60"),
        help="CTBC 触发后保持前馈目标的持续时间，单位秒；旧 ff-period 名称保留为 alias。",
    )
    parser.add_argument("--ff-wheel", type=_parse_float_list, default=_parse_float_list("0.0,0.04"))
    parser.add_argument("--ff-amp", type=_parse_float_list, default=_parse_float_list("0.0"))
    parser.add_argument(
        "--profiles",
        type=_parse_path_list,
        default=None,
        help="Comma-separated CTBC profile JSON files. When set, sim2sim uses --stair-ctbc-profile.",
    )
    parser.add_argument("--trigger-mode", choices=("force", "pitch"), default="pitch")
    parser.add_argument("--force-threshold", type=float, default=10.0)
    parser.add_argument("--contact-window", type=int, default=3)
    parser.add_argument("--pitch-threshold-deg", type=float, default=6.0)
    parser.add_argument("--pitch-window", type=int, default=3)
    parser.add_argument("--command-vx", type=float, default=1.0)
    parser.add_argument("--command-vxs", type=_parse_float_list, default=None)
    parser.add_argument("--command-height", type=float, default=0.39)
    parser.add_argument("--command-heights", type=_parse_float_list, default=None)
    parser.add_argument("--hold-on-support", action="store_true")
    parser.add_argument("--hold-vx", type=float, default=0.10)
    parser.add_argument("--hold-vxs", type=_parse_float_list, default=None)
    parser.add_argument("--allow-bilateral-trigger", action="store_true")
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _score(stair: dict[str, Any], rollout: dict[str, Any]) -> float:
    max_step = float(stair.get("max_completed_step", 0.0))
    tail_step = float(stair.get("tail_completed_step", 0.0))
    final_step = float(stair.get("final_completed_step", 0.0))
    support = float(stair.get("support_duration_s", 0.0))
    stable = float(stair.get("stable_duration_s", 0.0))
    tail_stable = float(stair.get("tail_stable_rate", 0.0))
    tail_same_step = float(stair.get("tail_same_step_rate", 0.0))
    nonwheel = float(stair.get("nonwheel_contact_rate_after_support", 0.0))
    leg = float(rollout.get("leg_contact_rate", 0.0))
    base = float(rollout.get("base_contact_rate", rollout.get("nonwheel_contact_rate", 0.0)))
    return (
        250.0 * max_step
        + 800.0 * tail_step
        + 1200.0 * final_step
        + 120.0 * support
        + 2500.0 * stable
        + 5000.0 * tail_stable
        + 1000.0 * tail_same_step
        - 200.0 * nonwheel
        - 250.0 * leg
        - 120.0 * base
    )


def _case_name(case: dict[str, Any]) -> str:
    profile = case.get("profile")
    profile_part = "" if profile is None else f"_profile{Path(profile).stem}"
    return (
        f"lvl{case['level']:02d}_len{case['leg_length']:.3f}"
        f"_swing{case['swing_angle_deg']:.1f}_dur{case['duration']:.2f}"
        f"_wheel{case['ff_wheel']:.2f}_amp{case['ff_amp']:.2f}"
        f"_vx{case['command_vx']:.2f}_h{case['command_height']:.2f}"
        f"_hold{case['hold_vx']:.2f}"
        f"_step{case['stair_step_height']:.3f}"
        f"{profile_part}"
    ).replace(".", "p")


def _run_case(args: argparse.Namespace, case: dict[str, Any], json_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "se3_sim2sim.cli",
        "--model-variant",
        "closedchain",
        "--checkpoint",
        str(args.checkpoint),
        "--stair-terrain",
        "--stair-terrain-level",
        str(case["level"]),
        "--stair-half-width",
        str(args.stair_half_width),
        "--stair-ctbc",
        "--stair-ctbc-iter",
        "0",
        "--stair-ctbc-trigger-mode",
        str(args.trigger_mode),
        "--stair-ctbc-ff-amplitude-rad",
        str(case["ff_amp"]),
        "--stair-ctbc-leg-length-m",
        str(case["leg_length"]),
        "--stair-ctbc-swing-angle-deg",
        str(case["swing_angle_deg"]),
        "--stair-ctbc-duration-s",
        str(case["duration"]),
        "--stair-ctbc-ff-wheel-action",
        str(case["ff_wheel"]),
        "--command",
        str(case["command_vx"]),
        "0.0",
        "0.0",
        "0.0",
        str(case["command_height"]),
        "0.0",
        "0.0",
        "0.0",
        "--yaw-pid",
        "--viewer",
        "none",
        "--json-output",
        str(json_path),
        "--max-steps",
        str(args.max_steps),
        "--print-every",
        "0",
        "--course",
        "none",
    ]
    if args.trigger_mode == "pitch":
        cmd.extend(
            [
                "--stair-ctbc-pitch-threshold-deg",
                str(args.pitch_threshold_deg),
                "--stair-ctbc-pitch-window",
                str(args.pitch_window),
            ]
        )
    else:
        cmd.extend(
            [
                "--stair-ctbc-contact-window",
                str(args.contact_window),
                "--stair-ctbc-force-threshold-n",
                str(args.force_threshold),
            ]
        )
    if case["stair_step_height"] > 0.0:
        cmd.extend(["--stair-step-height", str(case["stair_step_height"])])
    if case.get("profile") is not None:
        cmd.extend(["--stair-ctbc-profile", str(case["profile"])])
    if args.allow_bilateral_trigger:
        cmd.append("--stair-ctbc-allow-bilateral-trigger")
    if args.hold_on_support:
        cmd.extend(["--stair-hold-on-support", "--stair-hold-vx", str(case["hold_vx"])])
    subprocess.run(cmd, check=True)


def _summarize(case: dict[str, Any], payload: dict[str, Any], json_path: Path) -> dict[str, Any]:
    rollout = payload.get("rollout", {})
    stair = rollout.get("stair_climb", {})
    row = {
        **case,
        "json": str(json_path),
        "profile": "" if case.get("profile") is None else str(case["profile"]),
        "score": _score(stair, rollout),
        "command_vx": case["command_vx"],
        "command_height": case["command_height"],
        "hold_vx": case["hold_vx"],
        "step_height_m": stair.get("step_height_m", 0.0),
        "max_completed_step": stair.get("max_completed_step", 0),
        "tail_completed_step": stair.get("tail_completed_step", 0),
        "final_completed_step": stair.get("final_completed_step", 0),
        "support_duration_s": stair.get("support_duration_s", 0.0),
        "stable_duration_s": stair.get("stable_duration_s", 0.0),
        "tail_stable_rate": stair.get("tail_stable_rate", 0.0),
        "tail_on_step_rate": stair.get("tail_on_step_rate", 0.0),
        "tail_same_step_rate": stair.get("tail_same_step_rate", 0.0),
        "nonwheel_contact_rate_after_support": stair.get(
            "nonwheel_contact_rate_after_support", 0.0
        ),
        "final_left_step": stair.get("final_left_step", 0),
        "final_right_step": stair.get("final_right_step", 0),
        "final_left_height_error_m": stair.get("final_left_height_error_m", 0.0),
        "final_right_height_error_m": stair.get("final_right_height_error_m", 0.0),
        "task_success": (
            int(stair.get("final_completed_step", 0)) >= 1
            and float(stair.get("tail_stable_rate", 0.0)) >= 0.5
        ),
        "leg_contact_rate": rollout.get("leg_contact_rate", 0.0),
        "base_contact_rate": rollout.get("base_contact_rate", 0.0),
        "final_height": rollout.get("final", {}).get("height", 0.0),
    }
    return row


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_vxs = args.command_vxs if args.command_vxs is not None else [float(args.command_vx)]
    command_heights = (
        args.command_heights if args.command_heights is not None else [float(args.command_height)]
    )
    hold_vxs = args.hold_vxs if args.hold_vxs is not None else [float(args.hold_vx)]
    profiles = args.profiles if args.profiles is not None else [None]
    if args.stair_step_heights is not None:
        stair_step_heights = args.stair_step_heights
    elif args.stair_step_height is not None:
        stair_step_heights = [float(args.stair_step_height)]
    else:
        stair_step_heights = [0.0]
    cases = [
        {
            "level": level,
            "leg_length": leg_length,
            "swing_angle_deg": swing_angle_deg,
            "duration": duration,
            "ff_wheel": ff_wheel,
            "ff_amp": ff_amp,
            "command_vx": command_vx,
            "command_height": command_height,
            "hold_vx": hold_vx,
            "stair_step_height": stair_step_height,
            "profile": profile,
        }
        for (
            level,
            stair_step_height,
            profile,
            leg_length,
            swing_angle_deg,
            duration,
            ff_wheel,
            ff_amp,
            command_vx,
            command_height,
            hold_vx,
        ) in itertools.product(
            args.levels,
            stair_step_heights,
            profiles,
            args.leg_lengths,
            args.swing_angles_deg,
            args.duration,
            args.ff_wheel,
            args.ff_amp,
            command_vxs,
            command_heights,
            hold_vxs,
        )
    ]
    if args.limit is not None:
        cases = cases[: max(0, int(args.limit))]

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        json_path = args.output_dir / f"{_case_name(case)}.json"
        print(f"[{idx}/{len(cases)}] {_case_name(case)}", flush=True)
        if not (args.resume and json_path.exists()):
            _run_case(args, case, json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append(_summarize(case, payload, json_path))

    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sweep] csv={csv_path}")
    print(f"[sweep] json={json_path}")
    for row in rows[: args.top_k]:
        print(
            "score={score:.1f} level={level} h={step_height_m:.3f} "
            "len={leg_length:.3f} swing={swing_angle_deg:.1f} duration={duration:.2f} wheel={ff_wheel:.2f} "
            "cmd_h={command_height:.2f} hold_vx={hold_vx:.2f} "
            "max_step={max_completed_step} tail_step={tail_completed_step} "
            "support={support_duration_s:.2f}s stable={stable_duration_s:.2f}s "
            "tail_stable={tail_stable_rate:.2f} final_lr={final_left_step}/{final_right_step} "
            "profile={profile}".format(**row)
        )


if __name__ == "__main__":
    main()
