"""监控 recovery 训练目录，为每个 checkpoint 运行统一验收。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="轮询 recovery checkpoint，并运行 evaluate_recovery_checkpoint.py。"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="训练 run 目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="验收输出目录。")
    parser.add_argument("--poll-s", type=float, default=60.0, help="轮询间隔秒数。")
    parser.add_argument("--once", action="store_true", help="只扫描一次后退出。")
    parser.add_argument("--record-rerun", action="store_true", help="保存每个 checkpoint 的 .rrd。")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--velocity-max-steps", type=int, default=3600)
    parser.add_argument("--selfright-max-steps", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--viewer-log-every", type=int, default=2)
    parser.add_argument(
        "--stop-when-training-session-gone",
        default=None,
        help="指定 tmux session 名；session 消失且当前 checkpoint 都处理完后退出。",
    )
    return parser


def main() -> int:
    """持续监控 checkpoint 并执行验收。"""
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    state_dir = output_dir / ".processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = output_dir / "monitor.log"

    _log(monitor_log, f"run_dir={run_dir}")
    _log(monitor_log, f"output_dir={output_dir}")

    while True:
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iteration)
        for checkpoint in checkpoints:
            _process_checkpoint(args, checkpoint, output_dir, state_dir, monitor_log)

        if args.once:
            break
        if (
            args.stop_when_training_session_gone
            and _all_done(checkpoints, state_dir)
            and not _tmux_session_exists(str(args.stop_when_training_session_gone))
        ):
            _log(monitor_log, "training session ended and all checkpoints processed")
            break
        time.sleep(max(1.0, float(args.poll_s)))
    return 0


def _process_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    output_dir: Path,
    state_dir: Path,
    monitor_log: Path,
) -> None:
    """处理单个 checkpoint，使用 done/busy 文件保证可恢复。"""
    stem = checkpoint.stem
    passed_file = state_dir / f"{stem}.done"
    busy_file = state_dir / f"{stem}.busy"
    failed_file = state_dir / f"{stem}.failed"
    error_file = state_dir / f"{stem}.error"
    if passed_file.exists() or failed_file.exists() or busy_file.exists():
        return

    busy_file.write_text(str(time.time()), encoding="utf-8")
    ckpt_out = output_dir / stem
    cmd = [
        sys.executable,
        "scripts/evaluate_recovery_checkpoint.py",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(ckpt_out),
        "--device",
        str(args.device),
        "--velocity-max-steps",
        str(int(args.velocity_max_steps)),
        "--selfright-max-steps",
        str(int(args.selfright_max_steps)),
        "--print-every",
        str(int(args.print_every)),
        "--viewer-log-every",
        str(int(args.viewer_log_every)),
    ]
    if args.record_rerun:
        cmd.append("--record-rerun")

    _log(monitor_log, f"evaluating {checkpoint}")
    log_path = output_dir / f"{stem}_eval.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode == 0:
        summary_path = ckpt_out / f"{stem}_recovery_eval.json"
        passed = _summary_passed(summary_path)
        marker = passed_file if passed else failed_file
        marker.write_text(str(time.time()), encoding="utf-8")
        error_file.unlink(missing_ok=True)
        _log(monitor_log, f"{'passed' if passed else 'threshold_failed'} {stem}")
    else:
        error_file.write_text(str(time.time()), encoding="utf-8")
        _log(monitor_log, f"eval_error {stem}, returncode={result.returncode}; will retry later")
    busy_file.unlink(missing_ok=True)


def _checkpoint_iteration(path: Path) -> int:
    """从 model_<iter>.pt 解析迭代号。"""
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _all_done(checkpoints: list[Path], state_dir: Path) -> bool:
    """判断当前所有 checkpoint 是否都有 done 文件。"""
    return all(
        (state_dir / f"{checkpoint.stem}.done").exists()
        or (state_dir / f"{checkpoint.stem}.failed").exists()
        for checkpoint in checkpoints
    )


def _summary_passed(path: Path) -> bool:
    """读取 evaluate 脚本输出的总判定。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("passed", False))


def _tmux_session_exists(name: str) -> bool:
    """检查 tmux session 是否存在。"""
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _log(path: Path, message: str) -> None:
    """写入监控日志并同步打印到 stdout。"""
    line = f"[{time.strftime('%F %T')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
