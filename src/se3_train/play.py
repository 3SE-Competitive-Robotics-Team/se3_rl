"""se3-play 的 CLI 入口。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import tyro
from mjlab import TYRO_FLAGS
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import list_tasks, load_rl_cfg


def _resolve_play_config(task_id: str, cfg: PlayConfig) -> PlayConfig:
    """为 trained agent 自动补全本地最新 checkpoint。"""
    if cfg.agent != "trained":
        return cfg
    if cfg.checkpoint_file is not None or cfg.wandb_run_path is not None:
        return cfg

    experiment_name = load_rl_cfg(task_id).experiment_name
    checkpoint = _latest_checkpoint(Path.cwd(), experiment_name)
    print(f"[INFO]: 自动选择本地最新 checkpoint: {checkpoint}")
    return replace(cfg, checkpoint_file=str(checkpoint))


def _latest_checkpoint(base: Path, experiment_name: str) -> Path:
    """按 run 修改时间和模型迭代号解析最新 checkpoint。"""
    root = base / "logs" / "rsl_rl" / experiment_name
    runs = (
        [run for run in root.iterdir() if run.is_dir() and any(run.glob("model_*.pt"))]
        if root.exists()
        else []
    )
    if not runs:
        raise FileNotFoundError(
            "未找到本地 checkpoint，请传 --checkpoint-file 或 --wandb-run-path。"
        )

    latest_run = max(runs, key=lambda path: (path.stat().st_mtime, path.name))
    candidates = list(latest_run.glob("model_*.pt"))
    return max(candidates, key=_checkpoint_iteration).resolve()


def _checkpoint_iteration(path: Path) -> int:
    """从 model_<iter>.pt 提取迭代号，避免字典序误判。"""
    stem = path.stem
    prefix = "model_"
    if not stem.startswith(prefix):
        return -1
    try:
        return int(stem.removeprefix(prefix))
    except ValueError:
        return -1


def main() -> None:
    """play 入口，先注册 se3 任务，再委托给 MJLab play。"""
    maybe_print_top_level_help("se3-play")

    __import__("mjlab.tasks")

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=TYRO_FLAGS,
    )
    del remaining_args

    run_play(chosen_task, _resolve_play_config(chosen_task, args))


if __name__ == "__main__":
    main()
