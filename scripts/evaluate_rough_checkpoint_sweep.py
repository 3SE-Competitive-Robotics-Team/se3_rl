"""Run a broad headless sim2sim sweep for rough-discovery checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    terrain: str
    terrain_type: str | None
    origin_type: str | None
    level: int | None
    height: float
    vx: float
    yaw: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        futures = [
            pool.submit(run_case, args, checkpoint, output_dir, index, case)
            for index, case in enumerate(cases)
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{done:03d}/{len(cases):03d}] {row['case']} "
                f"ok={row['ok']} score={row['score']:.3f} "
                f"vx_err={row['vx_err_mean']:.3f} yaw_err={row['yaw_err_mean']:.3f} "
                f"h_err={row['height_err_mean']:.3f} nonwheel={row['nonwheel_contact_rate']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda r: int(r["index"]))
    csv_path = output_dir / "rough_sweep_summary.csv"
    json_path = output_dir / "rough_sweep_summary.json"
    write_csv(csv_path, rows)
    payload = {
        "checkpoint": str(checkpoint),
        "max_steps": int(args.max_steps),
        "cases": rows,
        "elapsed_s": time.monotonic() - start,
        "aggregate": aggregate(rows),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sweep] csv={csv_path}")
    print(f"[sweep] json={json_path}")
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    return 0


def build_cases() -> list[Case]:
    cases: list[Case] = []
    commands = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (0.0, 2.0), (0.0, 5.0)]
    for height in (0.26, 0.35, 0.39):
        for vx, yaw in commands:
            cases.append(
                Case(
                    f"plane_h{height:.2f}_vx{vx:.1f}_yaw{yaw:.1f}",
                    "plane",
                    None,
                    None,
                    None,
                    height,
                    vx,
                    yaw,
                )
            )

    rough_types = (
        "flat",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
        "random_rough",
        "wave_terrain",
    )
    for terrain_type in rough_types:
        for level in (0, 3, 5, 7):
            for vx, yaw in ((0.5, 0.0), (1.0, 0.0), (0.0, 2.0)):
                cases.append(
                    Case(
                        f"rough_{terrain_type}_L{level}_h0.35_vx{vx:.1f}_yaw{yaw:.1f}",
                        "rough",
                        "mixed",
                        terrain_type,
                        level,
                        0.35,
                        vx,
                        yaw,
                    )
                )

    for level in (0, 1, 3, 5, 7):
        for height in (0.26, 0.35, 0.39):
            for vx in (0.5, 1.0):
                cases.append(
                    Case(
                        f"stair_focus_L{level}_h{height:.2f}_vx{vx:.1f}",
                        "rough",
                        "mixed",
                        "pyramid_stairs",
                        level,
                        height,
                        vx,
                        0.0,
                    )
                )
    return cases


def run_case(
    args: argparse.Namespace,
    checkpoint: Path,
    output_dir: Path,
    index: int,
    case: Case,
) -> dict[str, Any]:
    case_dir = output_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    json_output = case_dir / f"{index:03d}_{case.name}.json"
    stdout_path = case_dir / f"{index:03d}_{case.name}.stdout.log"
    stderr_path = case_dir / f"{index:03d}_{case.name}.stderr.log"
    command = [
        "uv",
        "run",
        "--no-sync",
        "se3-sim2sim",
        "--checkpoint",
        str(checkpoint),
        "--model-variant",
        "closedchain",
        "--recovery-action-contract",
        "--viewer",
        "none",
        "--device",
        str(args.device),
        "--max-steps",
        str(int(args.max_steps)),
        "--print-every",
        "0",
        "--json-output",
        str(json_output),
        "--command",
        f"{case.vx:g}",
        f"{case.yaw:g}",
        "0",
        "0",
        f"{case.height:g}",
        "0",
        "0",
        "0",
    ]
    if case.terrain == "rough":
        command.extend(
            [
                "--rough-terrain",
                "--rough-terrain-type",
                str(case.terrain_type),
                "--rough-terrain-level",
                str(case.level),
                "--rough-stair-step-height-range",
                "0",
                "0.05",
            ]
        )
        if case.origin_type is not None:
            command.extend(["--rough-terrain-origin-type", str(case.origin_type)])

    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=float(args.timeout_s),
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if not json_output.exists():
        return {
            **case_fields(index, case),
            "ok": False,
            "returncode": int(proc.returncode),
            "error": (proc.stderr or proc.stdout)[-1000:],
            "elapsed_s": time.monotonic() - started,
            **empty_metrics(),
        }
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    row = summarize_payload(index, case, payload)
    row["returncode"] = int(proc.returncode)
    row["elapsed_s"] = time.monotonic() - started
    row["json"] = str(json_output)
    row["stdout"] = str(stdout_path)
    row["stderr"] = str(stderr_path)
    return row


def summarize_payload(index: int, case: Case, payload: dict[str, Any]) -> dict[str, Any]:
    rollout = payload.get("rollout", {})
    final = rollout.get("final", {}) if isinstance(rollout, dict) else {}

    def stat(key: str, field: str = "mean", default: float = math.nan) -> float:
        value = rollout.get(key, {}) if isinstance(rollout, dict) else {}
        if isinstance(value, dict):
            try:
                return float(value.get(field, default))
            except (TypeError, ValueError):
                return default
        return default

    vx_mean = stat("base_lin_vel_x")
    yaw_mean = stat("yaw_rate_rad_s")
    height_mean = stat("height")
    vx_err = abs(vx_mean - case.vx)
    yaw_err = abs(yaw_mean - case.yaw)
    height_err = abs(height_mean - case.height)
    nonwheel_rate = float(rollout.get("nonwheel_contact_rate", math.nan))
    leg_rate = float(rollout.get("leg_contact_rate", math.nan))
    base_rate = float(rollout.get("base_contact_rate", math.nan))
    wheel_rate = float(rollout.get("wheel_contact_rate", math.nan))
    tilt_max = stat("tilt_deg", "max")
    score = (
        min(vx_err / 1.0, 3.0)
        + min(yaw_err / 3.0, 3.0)
        + min(height_err / 0.08, 3.0)
        + min(max(nonwheel_rate, 0.0) * 2.0, 3.0)
        + min(max(tilt_max - 20.0, 0.0) / 30.0, 3.0)
    )
    ok = (
        vx_err <= (0.35 if abs(case.vx) > 1e-6 else 0.15)
        and yaw_err <= (0.8 if abs(case.yaw) > 1e-6 else 0.5)
        and height_err <= 0.06
        and nonwheel_rate <= 0.20
        and tilt_max <= 45.0
    )
    return {
        **case_fields(index, case),
        "ok": bool(ok),
        "score": float(score),
        "done_reason": str(payload.get("done_reason", "")),
        "samples": int(rollout.get("samples", 0) if isinstance(rollout, dict) else 0),
        "vx_mean": vx_mean,
        "vx_err_mean": vx_err,
        "yaw_rate_mean": yaw_mean,
        "yaw_err_mean": yaw_err,
        "height_mean": height_mean,
        "height_err_mean": height_err,
        "tilt_mean": stat("tilt_deg"),
        "tilt_max": tilt_max,
        "wheel_contact_rate": wheel_rate,
        "wheel_full_contact_rate": float(rollout.get("wheel_full_contact_rate", math.nan)),
        "leg_contact_rate": leg_rate,
        "base_contact_rate": base_rate,
        "nonwheel_contact_rate": nonwheel_rate,
        "wheel_clearance_min": stat("wheel_clearance", "min"),
        "leg_clearance_min": stat("leg_clearance", "min"),
        "base_clearance_min": stat("base_clearance", "min"),
        "action_delta_l2_mean": stat("action_delta_l2"),
        "action_delta_max_abs_max": stat("action_delta_max_abs", "max"),
        "final_height": float(final.get("height", math.nan))
        if isinstance(final, dict)
        else math.nan,
        "final_tilt": float(final.get("tilt_deg", math.nan))
        if isinstance(final, dict)
        else math.nan,
        "final_vx": float(final.get("base_lin_vel_x", math.nan))
        if isinstance(final, dict)
        else math.nan,
        "final_yaw_rate": (
            float(final.get("base_ang_vel_body", [math.nan, math.nan, math.nan])[2])
            if isinstance(final, dict)
            else math.nan
        ),
        "error": "",
    }


def case_fields(index: int, case: Case) -> dict[str, Any]:
    return {
        "index": int(index),
        "case": case.name,
        "terrain": case.terrain,
        "terrain_type": "" if case.terrain_type is None else case.terrain_type,
        "origin_type": "" if case.origin_type is None else case.origin_type,
        "level": "" if case.level is None else int(case.level),
        "height_cmd": float(case.height),
        "vx_cmd": float(case.vx),
        "yaw_cmd": float(case.yaw),
    }


def empty_metrics() -> dict[str, float | int | str]:
    keys = (
        "score",
        "samples",
        "vx_mean",
        "vx_err_mean",
        "yaw_rate_mean",
        "yaw_err_mean",
        "height_mean",
        "height_err_mean",
        "tilt_mean",
        "tilt_max",
        "wheel_contact_rate",
        "wheel_full_contact_rate",
        "leg_contact_rate",
        "base_contact_rate",
        "nonwheel_contact_rate",
        "wheel_clearance_min",
        "leg_clearance_min",
        "base_clearance_min",
        "action_delta_l2_mean",
        "action_delta_max_abs_max",
        "final_height",
        "final_tilt",
        "final_vx",
        "final_yaw_rate",
    )
    return {key: math.nan for key in keys} | {"done_reason": "", "samples": 0}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("samples", 0)]
    failed = [r for r in valid if not r.get("ok")]
    by_group: dict[str, dict[str, Any]] = {}
    for group_name in ("terrain", "origin_type", "level", "height_cmd", "vx_cmd", "yaw_cmd"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in valid:
            groups.setdefault(str(row[group_name]), []).append(row)
        by_group[group_name] = {
            key: {
                "count": len(items),
                "ok_rate": sum(1 for item in items if item.get("ok")) / max(1, len(items)),
                "mean_score": mean(item["score"] for item in items),
                "mean_vx_err": mean(item["vx_err_mean"] for item in items),
                "mean_yaw_err": mean(item["yaw_err_mean"] for item in items),
                "mean_height_err": mean(item["height_err_mean"] for item in items),
                "mean_nonwheel_contact": mean(item["nonwheel_contact_rate"] for item in items),
            }
            for key, items in sorted(groups.items())
        }
    worst = sorted(valid, key=lambda row: float(row["score"]), reverse=True)[:15]
    return {
        "case_count": len(rows),
        "valid_count": len(valid),
        "ok_count": sum(1 for row in valid if row.get("ok")),
        "failed_count": len(failed),
        "ok_rate": sum(1 for row in valid if row.get("ok")) / max(1, len(valid)),
        "mean_score": mean(row["score"] for row in valid),
        "mean_vx_err": mean(row["vx_err_mean"] for row in valid),
        "mean_yaw_err": mean(row["yaw_err_mean"] for row in valid),
        "mean_height_err": mean(row["height_err_mean"] for row in valid),
        "mean_nonwheel_contact": mean(row["nonwheel_contact_rate"] for row in valid),
        "by_group": by_group,
        "worst": worst,
    }


def mean(values: Any) -> float:
    data = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(data) / len(data)) if data else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
