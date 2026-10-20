"""训练中旁路触发 benchmark：runner 每次导出 ONNX 后调用 `maybe_launch_bench`。

- `bench_interval`（runner 配置，0 = 关闭）：每 N 轮与训练结束的最后一次保存都跑 full 档（2026-10-19 用户定；
  Pod 128 核 32 进程约 15 s，quick 档只留给手动快速检查）。
- 以独立进程（新 session、降优先级、单线程 BLAS）运行 `python -m se3_tools.bench.cli run`，训练不等它；
  评测只吃 CPU，worker 数默认取 CPU 核数的 1/4（上限 16），给 Rough 训练（本身 CPU 瓶颈）留余量。
- `--lock`：上一次评测还没跑完时本次排队等待。
- SE3_SMOKE=1 时默认不触发（设 SE3_BENCH_IN_SMOKE=1 可在 smoke 里验证链路）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _workers(cfg: dict) -> int:
    configured = int(cfg.get("bench_workers", 0) or 0)
    if configured > 0:
        return configured
    return max(1, min(16, (os.cpu_count() or 4) // 4))


def maybe_launch_bench(
    *,
    onnx_path: Path,
    iteration: int,
    final: bool,
    cfg: dict,
    log_dir: str | os.PathLike,
) -> subprocess.Popen | None:
    """满足触发条件时启动评测进程并立即返回；任何失败只打印，不影响训练。"""
    interval = int(cfg.get("bench_interval", 0) or 0)
    if interval <= 0:
        return None
    if os.environ.get("SE3_SMOKE", "0") == "1" and os.environ.get("SE3_BENCH_IN_SMOKE") != "1":
        return None
    if not final and (iteration <= 0 or iteration % interval != 0):
        return None
    suite = "full"
    run_name = Path(log_dir).name
    command = [
        sys.executable,
        "-m",
        "se3_tools.bench.cli",
        "run",
        "--onnx",
        str(onnx_path),
        "--suite",
        suite,
        "--lock",
        "--iteration",
        str(iteration),
        "--workers",
        str(_workers(cfg)),
    ]
    if cfg.get("logger") == "wandb":
        command += [
            "--wandb-project",
            str(cfg.get("wandb_project")),
            "--wandb-run-name",
            run_name,
            "--wandb-group",
            run_name,
        ]
    if os.name == "posix" and shutil.which("nice"):
        command = ["nice", "-n", "10", *command]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    log_path = onnx_path.parent / "bench" / f"{onnx_path.stem}_{suite}.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")  # noqa: SIM115 - 交给子进程持有
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=os.name == "posix",
        )
    except Exception as error:
        print(f"[SE3 Bench] 启动失败（训练继续）：{error}", flush=True)
        return None
    print(f"[SE3 Bench] {suite} 已在后台启动（pid {process.pid}）→ {log_path}", flush=True)
    return process


__all__ = ["maybe_launch_bench"]
