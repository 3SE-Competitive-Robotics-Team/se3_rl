"""Flow Matching 蒸馏监控工具。"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_COLLECT_RE = re.compile(r"\[flow-collect\] task=(?P<task>\w+) batch=(?P<batch>\d+)/(?P<total>\d+)")
_TRAIN_RE = re.compile(r"\[flow-train\] step=(?P<step>\d+) loss=(?P<loss>[-+0-9.eE]+)")
_SAVED_RE = re.compile(r"\[flow-train\] saved (?P<path>\S+) final_loss=(?P<loss>[-+0-9.eE]+)")


@dataclass(frozen=True)
class CollectProgress:
    """采集进度摘要。"""

    per_task: dict[str, tuple[int, int]]
    saved_line: str | None


@dataclass(frozen=True)
class TrainProgress:
    """训练进度摘要。"""

    last_step: int | None
    last_loss: float | None
    saved_loss: float | None
    recent: list[tuple[int, float]]


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(description="Monitor Flow Matching collect/train/play state")
    parser.add_argument("--log", type=Path, default=Path("logs/flow_match/wheel_gait/latest.log"))
    parser.add_argument("--dataset", type=Path, default=Path("data/flow_match/wheel_gait.pt"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("logs/flow_match/wheel_gait/flow.pt")
    )
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--tail-lines", type=int, default=2000)
    parser.add_argument("--watch-s", type=float, default=0.0)
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    while True:
        _print_report(
            log_path=args.log,
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            max_steps=int(args.max_steps),
            tail_lines=int(args.tail_lines),
        )
        if float(args.watch_s) <= 0.0:
            break
        time.sleep(float(args.watch_s))


def _print_report(
    *,
    log_path: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    max_steps: int,
    tail_lines: int,
) -> None:
    """打印一次监控报告。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = _read_tail(log_path, tail_lines=tail_lines)
    collect = _parse_collect(lines)
    train = _parse_train(lines)
    dataset = _dataset_summary(dataset_path)
    checkpoint = _checkpoint_summary(checkpoint_path)
    processes = _process_summary()
    gpu = _gpu_summary()

    print(f"\n[flow-monitor] {now}")
    print(f"log={log_path} exists={log_path.exists()}")
    print(f"gpu={gpu}")
    print(f"process={processes}")
    print(_format_collect(collect))
    print(_format_train(train, max_steps=max_steps))
    print(f"dataset={dataset}")
    print(f"checkpoint={checkpoint}", flush=True)


def _read_tail(path: Path, *, tail_lines: int) -> list[str]:
    """读取日志尾部，避免大日志反复全量解析。"""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if tail_lines <= 0:
        return raw
    return raw[-tail_lines:]


def _parse_collect(lines: list[str]) -> CollectProgress:
    """解析采集批次进度。"""
    per_task: dict[str, tuple[int, int]] = {}
    saved_line: str | None = None
    for line in lines:
        match = _COLLECT_RE.search(line)
        if match is not None:
            per_task[match.group("task")] = (
                int(match.group("batch")),
                int(match.group("total")),
            )
        if "[flow-collect] saved" in line:
            saved_line = line.strip()
    return CollectProgress(per_task=per_task, saved_line=saved_line)


def _parse_train(lines: list[str]) -> TrainProgress:
    """解析训练 loss 进度。"""
    recent: list[tuple[int, float]] = []
    saved_loss: float | None = None
    for line in lines:
        match = _TRAIN_RE.search(line)
        if match is not None:
            recent.append((int(match.group("step")), float(match.group("loss"))))
            continue
        saved = _SAVED_RE.search(line)
        if saved is not None:
            saved_loss = float(saved.group("loss"))
    recent = recent[-8:]
    if recent:
        last_step, last_loss = recent[-1]
    else:
        last_step, last_loss = None, None
    return TrainProgress(
        last_step=last_step,
        last_loss=last_loss,
        saved_loss=saved_loss,
        recent=recent,
    )


def _format_collect(progress: CollectProgress) -> str:
    """格式化采集进度。"""
    if not progress.per_task and progress.saved_line is None:
        return "collect=无日志"
    parts = [f"{task}:{done}/{total}" for task, (done, total) in sorted(progress.per_task.items())]
    if progress.saved_line is not None:
        parts.append("saved")
    return "collect=" + ", ".join(parts)


def _format_train(progress: TrainProgress, *, max_steps: int) -> str:
    """格式化训练进度。"""
    if progress.last_step is None:
        return "train=无 step 日志"
    pct = 100.0 * float(progress.last_step) / max(float(max_steps), 1.0)
    recent = ", ".join(f"{step}:{loss:.4f}" for step, loss in progress.recent)
    saved = "未保存" if progress.saved_loss is None else f"saved_loss={progress.saved_loss:.6f}"
    return (
        f"train=step {progress.last_step}/{max_steps} ({pct:.1f}%) "
        f"loss={progress.last_loss:.6f} {saved} recent=[{recent}]"
    )


def _dataset_summary(path: Path) -> dict[str, Any]:
    """读取数据集形状和任务分布。"""
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    summary: dict[str, Any] = {
        "exists": True,
        "size_mb": round(stat.st_size / 1024 / 1024, 1),
    }
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        summary["error"] = str(exc)
        return summary
    if not isinstance(raw, dict):
        summary["error"] = "payload 不是 dict"
        return summary
    for key in ("obs", "actions", "commands"):
        value = raw.get(key)
        if isinstance(value, torch.Tensor):
            summary[f"{key}_shape"] = tuple(value.shape)
    modes = raw.get("modes")
    if isinstance(modes, torch.Tensor):
        unique, counts = torch.unique(modes[:, 0], return_counts=True)
        summary["modes"] = {
            str(int(mode)): int(count) for mode, count in zip(unique, counts, strict=False)
        }
    return summary


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    """读取 checkpoint 元数据。"""
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    summary: dict[str, Any] = {
        "exists": True,
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        summary["error"] = str(exc)
        return summary
    if isinstance(raw, dict) and isinstance(raw.get("metadata"), dict):
        metadata = raw["metadata"]
        for key in (
            "final_loss",
            "train_seconds",
            "reset_window_ratio",
            "burn_in_steps",
            "loss_steps",
        ):
            if key in metadata:
                summary[key] = metadata[key]
    return summary


def _process_summary() -> str:
    """读取 Flow 采集/训练进程摘要。"""
    result = _run_command(("pgrep", "-af", "se3-flow-(collect|train|play)"))
    if result.returncode != 0 or not result.stdout.strip():
        return "none"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return " | ".join(lines[-4:])


def _gpu_summary() -> str:
    """读取 GPU 摘要；无 nvidia-smi 时返回 none。"""
    result = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader",
        )
    )
    if result.returncode != 0:
        return "none"
    return result.stdout.strip().replace("\n", " | ")


def _run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """运行只读状态命令。"""
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr=str(exc))


if __name__ == "__main__":
    main()
