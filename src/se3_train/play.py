"""se3-play 的 CLI 入口。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import tyro
from mjlab import TYRO_FLAGS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts import play as mjlab_play
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import list_tasks, load_rl_cfg


@dataclass(frozen=True)
class Se3PlayConfig(PlayConfig):
    """SE3 play 扩展参数。"""

    play_terrain_difficulty: float | None = None
    """play 模式固定课程地形难度，范围 0.0 到 1.0。"""


def _resolve_play_config(task_id: str, cfg: Se3PlayConfig) -> Se3PlayConfig:
    """为 trained agent 自动补全本地最新 checkpoint。"""
    if cfg.agent != "trained":
        return cfg
    if cfg.checkpoint_file is not None or cfg.wandb_run_path is not None:
        return cfg

    experiment_name = load_rl_cfg(task_id).experiment_name
    checkpoint = _latest_checkpoint(Path.cwd(), experiment_name)
    print(f"[INFO]: 自动选择本地最新 checkpoint: {checkpoint}")
    return replace(cfg, checkpoint_file=str(checkpoint))


def _apply_play_terrain_difficulty(env_cfg: ManagerBasedRlEnvCfg, difficulty: float | None) -> None:
    """把 play 地形固定到指定课程难度。"""
    if difficulty is None:
        return
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("--play-terrain-difficulty 必须在 0.0 到 1.0 之间")

    terrain_cfg = getattr(env_cfg.scene, "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_cfg is None or terrain_generator is None:
        raise ValueError("--play-terrain-difficulty 只支持 generator terrain 任务")

    terrain_generator.num_rows = 1
    terrain_generator.difficulty_range = (float(difficulty), float(difficulty))
    terrain_cfg.max_init_terrain_level = 0
    print(f"[INFO]: play terrain difficulty fixed at {difficulty:.3f}")


def _run_play_with_se3_overrides(task_id: str, cfg: Se3PlayConfig) -> None:
    """在 MJLab play 前注入 SE3 专属 play 配置。"""
    if cfg.play_terrain_difficulty is None:
        mjlab_play.run_play(task_id, cfg)
        return

    original_load_env_cfg = mjlab_play.load_env_cfg

    def load_env_cfg_with_play_difficulty(
        task_name: str, play: bool = False
    ) -> ManagerBasedRlEnvCfg:
        env_cfg = original_load_env_cfg(task_name, play=play)
        if play:
            _apply_play_terrain_difficulty(env_cfg, cfg.play_terrain_difficulty)
        return env_cfg

    mjlab_play.load_env_cfg = load_env_cfg_with_play_difficulty
    try:
        mjlab_play.run_play(task_id, cfg)
    finally:
        mjlab_play.load_env_cfg = original_load_env_cfg


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
        Se3PlayConfig,
        args=remaining_args,
        default=Se3PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=TYRO_FLAGS,
    )
    del remaining_args

    _run_play_with_se3_overrides(chosen_task, _resolve_play_config(chosen_task, args))


if __name__ == "__main__":
    main()
