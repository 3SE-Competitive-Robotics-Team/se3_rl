"""训练过程快照监控脚本。

独立运行在训练机器上，每隔 --interval-iters 个 iteration 检测最新 checkpoint，
执行一次 headless sim2sim（强制发跳跃指令），将关键指标追加写入
/tmp/sim2sim_snapshot.jsonl，供训练监控 agent 轮询读取。

使用方式：
    # 在训练启动后，在同一台机器上另开一个 tmux 窗口：
    uv run python scripts/watch_train.py --log-dir logs/rsl_rl/se3_wheel_leg --interval-iters 500

设计原则：
- 完全不入侵训练代码，随时可以 Ctrl-C 停止
- 用文件锁防止 sim2sim 进程与训练进程争抢 GPU（可选，默认跳过）
- 结果追加到 JSONL，每行一个快照，便于后续分析
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def find_latest_checkpoint(log_dir: Path) -> Path | None:
    """找最新的 model_*.pt，按数字排序取最大。"""
    checkpoints = sorted(
        log_dir.glob("*/model_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def run_sim2sim_snapshot(checkpoint: Path, jump_height: float = 0.4) -> dict | None:
    """运行 headless sim2sim，强制发跳跃指令，返回关键指标字典。

    指令格式：lin_x yaw pitch roll height jump_flag jump_target_height
    """
    cmd = [
        "uv",
        "run",
        "se3-sim2sim",
        "--checkpoint",
        str(checkpoint),
        "--viewer",
        "none",
        "--max-steps",
        "300",
        "--print-every",
        "300",
        "--command",
        "0",
        "0",
        "0",
        "0",
        "0.45",
        "1",
        str(jump_height),
        "--json-output",
        "/tmp/sim2sim_latest.json",
        "--no-action-delay-randomize",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"[watch] sim2sim 失败: {result.stderr[-300:]}", file=sys.stderr)
            return None

        output_path = Path("/tmp/sim2sim_latest.json")
        if output_path.exists():
            with open(output_path) as f:
                return json.load(f)
        # 没有 json 输出时从 stdout 解析最后一行 summary
        for line in reversed(result.stdout.splitlines()):
            if "final_height" in line:
                # 解析 "final_height=0.328 final_tilt_deg=4.08" 格式
                metrics: dict = {}
                for token in line.split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            metrics[k] = float(v)
                        except ValueError:
                            metrics[k] = v
                return metrics
        return None
    except subprocess.TimeoutExpired:
        print("[watch] sim2sim 超时", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[watch] 异常: {e}", file=sys.stderr)
        return None


def extract_iter_from_checkpoint(ckpt: Path) -> int:
    """从文件名 model_1234.pt 解析迭代数。"""
    try:
        return int(ckpt.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="训练过程跳跃能力快照监控")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs/rsl_rl/se3_wheel_leg"), help="checkpoint 根目录"
    )
    parser.add_argument(
        "--interval-iters", type=int, default=500, help="每隔多少 iteration 触发一次 sim2sim"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/sim2sim_snapshot.jsonl"),
        help="快照输出文件(JSONL 格式，追加)",
    )
    parser.add_argument(
        "--jump-height",
        type=float,
        default=0.4,
        help="强制发送的跳跃目标高度(m)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="检查新 checkpoint 的轮询间隔(秒)",
    )
    args = parser.parse_args()

    print(f"[watch] 开始监控 {args.log_dir}, 每 {args.interval_iters} iter 触发一次 sim2sim")
    print(f"[watch] 快照写入 {args.output}")

    last_snapshotted_iter = -1

    while True:
        ckpt = find_latest_checkpoint(args.log_dir)
        if ckpt is None:
            print("[watch] 未找到 checkpoint, 等待训练开始...")
            time.sleep(args.poll_seconds)
            continue

        current_iter = extract_iter_from_checkpoint(ckpt)

        # 判断是否需要触发快照
        if current_iter - last_snapshotted_iter >= args.interval_iters:
            print(f"[watch] iter={current_iter}, 触发 sim2sim 快照(checkpoint: {ckpt.name})")
            metrics = run_sim2sim_snapshot(ckpt, jump_height=args.jump_height)

            if metrics is not None:
                snapshot = {
                    "iter": current_iter,
                    "checkpoint": str(ckpt),
                    "timestamp": time.time(),
                    **metrics,
                }
                with open(args.output, "a") as f:
                    f.write(json.dumps(snapshot) + "\n")

                # 控制台简报
                h = metrics.get("final_height", metrics.get("height", "?"))
                tilt = metrics.get("final_tilt_deg", metrics.get("tilt", "?"))
                print(f"[watch] iter={current_iter} height={h:.3f} tilt={tilt:.1f}°")
            else:
                print(f"[watch] iter={current_iter} sim2sim 无输出, 跳过")

            last_snapshotted_iter = current_iter

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
