"""se3-train 的 CLI 入口。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROUGH_DISCOVERY_TASK_ID = "SE3-WheelLegged-Rough-Discovery-GRU"


def _has_cli_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _apply_rough_discovery_defaults() -> None:
    """在交给 MJLab parser 前注入 SE3 专属启动默认值。"""
    if len(sys.argv) < 2 or sys.argv[1] != ROUGH_DISCOVERY_TASK_ID:
        return

    fast_logs = _env_flag(
        "SE3_ROUGH_DISCOVERY_FAST_LOGS",
        _env_flag("SE3_FAST_TRAIN_LOGS", True),
    )
    if fast_logs:
        os.environ.setdefault("SE3_LOGGER", "tensorboard")
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("SE3_ASYNC_HOST_LOGGER", "1")
        os.environ.setdefault("SE3_CONSOLE_LOG_MODE", "summary")

    if "TORCHRUNX_LOG_DIR" in os.environ or _has_cli_option(sys.argv[2:], "--torchrunx-log-dir"):
        return

    torchrunx_log_dir = os.environ.get("SE3_ROUGH_DISCOVERY_TORCHRUNX_LOG_DIR")
    if torchrunx_log_dir is None:
        torchrunx_log_dir = os.environ.get(
            "SE3_TORCHRUNX_LOG_DIR",
            str(Path("/tmp") / "se3_rl_torchrunx" / os.environ.get("USER", "user")),
        )
    sys.argv.extend(["--torchrunx-log-dir", torchrunx_log_dir])


def main() -> None:
    """训练入口,捕获 KeyboardInterrupt 实现优雅退出。"""
    sys.argv[0] = "se3-train"
    _apply_rough_discovery_defaults()

    try:
        from mjlab.scripts.train import main as mjlab_train

        mjlab_train()
    except KeyboardInterrupt:
        print("\n✓ 训练已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
