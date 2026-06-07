"""se3-play 的 CLI 入口。"""

from __future__ import annotations

import html
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import tyro
from mjlab import TYRO_FLAGS
from mjlab.scripts import play as mjlab_play
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.viewer.viser.viewer import ViserPlayViewer

_TRAINING_ITER_PATTERN = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_TRAINING_ITER_UPDATE_INTERVAL_S = 2.0
_LOG_TAIL_BYTES = 256 * 1024


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


class _Se3ViserPlayViewer(ViserPlayViewer):
    """在 MJLab Viser 面板中显示正在值守的训练进度。"""

    def setup(self) -> None:
        """初始化 viewer，并按需添加训练状态面板。"""
        super().setup()
        self._se3_train_run_dir = _training_run_dir_from_env()
        self._se3_train_iter_html = None
        self._se3_train_iter_last_update = 0.0
        if self._se3_train_run_dir is None:
            return

        with self._server.gui.add_folder("Training"):
            self._se3_train_iter_html = self._server.gui.add_html("")
        self._update_training_iter_display(force=True)

    def sync_env_to_viewer(self) -> None:
        """同步仿真画面，并刷新外部训练进度。"""
        super().sync_env_to_viewer()
        self._update_training_iter_display()

    def _update_training_iter_display(self, force: bool = False) -> None:
        """低频读取训练日志，把当前 PPO iter 写入 Viser GUI。"""
        if self._se3_train_iter_html is None or self._se3_train_run_dir is None:
            return

        now = time.monotonic()
        if not force and now - self._se3_train_iter_last_update < _TRAINING_ITER_UPDATE_INTERVAL_S:
            return
        self._se3_train_iter_last_update = now

        progress = _read_training_progress(self._se3_train_run_dir)
        self._se3_train_iter_html.content = _format_training_progress_html(
            self._se3_train_run_dir, progress
        )


def _training_run_dir_from_env() -> Path | None:
    """从环境变量读取需要显示在 Viser 里的训练 run 目录。"""
    raw = os.environ.get("SE3_VISER_TRAIN_RUN_DIR")
    if not raw:
        return None
    return Path(raw).expanduser()


def _read_training_progress(run_dir: Path) -> dict[str, int | str | None]:
    """读取训练日志和 checkpoint，返回 Viser 面板需要的进度字段。"""
    iteration, total = _latest_logged_iteration(run_dir)
    checkpoint_iter = _latest_checkpoint_iteration(run_dir)
    return {
        "iteration": iteration,
        "total": total,
        "checkpoint_iter": checkpoint_iter,
        "updated_at": time.strftime("%H:%M:%S"),
    }


def _latest_logged_iteration(run_dir: Path) -> tuple[int | None, int | None]:
    """从 rank0 日志尾部提取最新的 Learning iteration。"""
    if not run_dir.exists():
        return None, None

    log_files = sorted(
        run_dir.glob("torchrunx/*/localhost[0].log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        log_files = sorted(
            run_dir.glob("torchrunx/**/*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    for log_file in log_files:
        text = _read_file_tail(log_file, _LOG_TAIL_BYTES)
        matches = list(_TRAINING_ITER_PATTERN.finditer(text))
        if matches:
            match = matches[-1]
            return int(match.group(1)), int(match.group(2))
    return None, None


def _latest_checkpoint_iteration(run_dir: Path) -> int | None:
    """读取 run 目录下最新 checkpoint 的迭代号。"""
    if not run_dir.exists():
        return None
    candidates = list(run_dir.glob("model_*.pt"))
    if not candidates:
        return None
    latest = max((_checkpoint_iteration(path) for path in candidates), default=-1)
    return latest if latest >= 0 else None


def _read_file_tail(path: Path, max_bytes: int) -> str:
    """只读取日志尾部，避免 Viser 高频刷新时扫描完整大文件。"""
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes), os.SEEK_SET)
            return file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _format_training_progress_html(run_dir: Path, progress: dict[str, int | str | None]) -> str:
    """格式化训练进度面板 HTML。"""
    iteration = progress["iteration"]
    total = progress["total"]
    checkpoint_iter = progress["checkpoint_iter"]
    updated_at = progress["updated_at"]

    if isinstance(iteration, int) and isinstance(total, int):
        iter_text = f"{iteration} / {total}"
    else:
        iter_text = "waiting for log"
    ckpt_text = str(checkpoint_iter) if isinstance(checkpoint_iter, int) else "none"
    run_name = html.escape(run_dir.name)
    iter_text = html.escape(iter_text)
    ckpt_text = html.escape(ckpt_text)
    updated_text = html.escape(str(updated_at))

    return f"""
      <div style="font-size:0.85em; line-height:1.35; padding:0 1em 0.5em 1em;">
        <strong>Training iter:</strong> {iter_text}<br/>
        <strong>Latest checkpoint:</strong> model_{ckpt_text}.pt<br/>
        <strong>Run:</strong> {run_name}<br/>
        <strong>Updated:</strong> {updated_text}
      </div>
    """


def _install_se3_viser_viewer() -> None:
    """替换 MJLab 默认 Viser viewer，增加 SE3 训练值守面板。"""
    mjlab_play.ViserPlayViewer = _Se3ViserPlayViewer


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

    _install_se3_viser_viewer()
    mjlab_play.run_play(chosen_task, _resolve_play_config(chosen_task, args))


if __name__ == "__main__":
    main()
