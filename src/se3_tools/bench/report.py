"""评测结果落盘与上报：checkpoint 旁的 JSON，以及同 group 的 W&B 评测 run（不往训练 run 并发写）。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from se3_tools.bench.spec import BENCH_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_state() -> dict[str, str | bool]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--", "src")),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8"
    )
    tmp.replace(path)


def log_wandb(
    payload: dict,
    *,
    project: str,
    run_name: str,
    group: str,
    id_file: Path,
) -> str | None:
    """记到 `<run_name>-bench`（同 group）：以 iteration 为横轴写头条指标、逐条结果表与视频。"""
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    import wandb

    run_id = (
        id_file.read_text(encoding="utf-8").strip()
        if id_file.exists()
        else wandb.util.generate_id()
    )
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run_id, encoding="utf-8")
    run = wandb.init(
        project=project,
        name=f"{run_name}-bench",
        group=group,
        id=run_id,
        resume="allow",
        job_type="bench",
        config={"bench_version": BENCH_VERSION},
        reinit=True,
    )
    try:
        wandb.define_metric("iteration")
        wandb.define_metric("bench/*", step_metric="iteration")
        record: dict = {"iteration": payload["iteration"]}
        for key, value in payload["headline"].items():
            record[f"bench/{payload['suite']}/{key}"] = value
        columns = ["suite", "name", "plant", "fell", "summary"]
        table = wandb.Table(columns=columns)
        for r in payload["results"]:
            summary = {
                k: v
                for k, v in r.items()
                if k not in ("suite", "name", "plant", "kind", "riser_events")
            }
            table.add_data(
                r["suite"], r["name"], r["plant"], r.get("fell"), json.dumps(summary, default=float)
            )
        record[f"bench/{payload['suite']}/cases"] = table
        for r in payload["results"]:
            if r.get("video"):
                record[f"bench/video/{r['name']}"] = wandb.Video(r["video"], format="mp4")
        wandb.log(record)
        return run.url
    finally:
        wandb.finish()


__all__ = ["git_state", "log_wandb", "write_json"]
