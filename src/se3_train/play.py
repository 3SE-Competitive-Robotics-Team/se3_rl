"""se3-play 的 CLI 入口。"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import tyro
from mjlab import TYRO_FLAGS
from mjlab.scripts import play as mjlab_play
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.viewer.viser import viewer as mjlab_viser_viewer
from mjlab.viewer.viser.scene import MjlabViserScene
from mjlab.viewer.viser.viewer import ViserPlayViewer

from se3_train.training_runtime import TRAINING_STATUS_FILENAME

_TRAINING_ITER_PATTERN = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_TRAINING_COLLECT_PATTERN = re.compile(r"Collection time:\s+([0-9.]+)s")
_TRAINING_LEARNING_PATTERN = re.compile(r"Learning time:\s+([0-9.]+)s")
_TRAINING_ITER_TIME_PATTERN = re.compile(r"Iteration time:\s+([0-9.]+)s")
_TRAINING_ITER_UPDATE_INTERVAL_S = 2.0
_CTBC_UPDATE_INTERVAL_S = 0.25
_CTBC_LOG_INTERVAL_S = 2.0
_LOG_TAIL_BYTES = 256 * 1024
_VISER_FRAME_RATE_ENV = "SE3_VISER_FRAME_RATE"
_VISER_INITIAL_SPEED_ENV = "SE3_VISER_INITIAL_SPEED"
_VISER_FOLLOW_CTBC_ENV = "SE3_VISER_FOLLOW_CTBC"


def _hide_collision_group(scene: Any) -> None:
    """隐藏 MJCF group 0 碰撞体，保留视觉模型默认显示。"""
    if not hasattr(scene, "geom_groups_visible"):
        return
    if len(scene.geom_groups_visible) > 0:
        scene.geom_groups_visible[0] = False
    sync_visibilities = getattr(scene, "_sync_visibilities", None)
    if callable(sync_visibilities):
        sync_visibilities()


class _Se3MjlabViserScene(MjlabViserScene):
    """SE3 play 专用 scene，默认不显示碰撞体。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _hide_collision_group(self)


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

    def __init__(self, *args, **kwargs) -> None:
        """按环境变量降低常驻值守的刷新和仿真速率。"""
        kwargs.setdefault("frame_rate", _viser_frame_rate_from_env(default=60.0))
        super().__init__(*args, **kwargs)
        initial_speed = _viser_initial_speed_from_env()
        if initial_speed is not None:
            self._set_initial_speed(initial_speed)

    def setup(self) -> None:
        """初始化 viewer，并按需添加训练状态面板。"""
        super().setup()
        self._se3_train_run_dir = _training_run_dir_from_env()
        self._se3_train_iter_html = None
        self._se3_train_iter_last_update = 0.0
        self._se3_ctbc_html = None
        self._se3_ctbc_last_update = 0.0
        self._se3_ctbc_last_log = 0.0
        self._se3_ctbc_checkpoint_iter = None
        self._se3_follow_ctbc = None

        if self._se3_train_run_dir is not None:
            with self._server.gui.add_folder("Training"):
                self._se3_train_iter_html = self._server.gui.add_html("")
        if getattr(self.env.unwrapped, "stair_climb_state", None) is not None:
            with self._server.gui.add_folder("CTBC Feedforward"):
                self._se3_follow_ctbc = self._server.gui.add_checkbox(
                    "Follow active CTBC",
                    initial_value=_bool_env(_VISER_FOLLOW_CTBC_ENV, default=False),
                )
                self._se3_ctbc_html = self._server.gui.add_html("")

        self._hide_collision_geoms_by_default()
        self._sync_ctbc_checkpoint_iteration(force=True)
        self._update_training_iter_display(force=True)
        self._update_ctbc_display(force=True)

    def _hide_collision_geoms_by_default(self) -> None:
        """默认隐藏 MJCF group 0 碰撞体，只显示视觉模型。"""
        if hasattr(self, "_scene"):
            _hide_collision_group(self._scene)

    def sync_env_to_viewer(self) -> None:
        """同步仿真画面，并刷新外部训练进度。"""
        super().sync_env_to_viewer()
        self._sync_ctbc_checkpoint_iteration()
        self._update_training_iter_display()
        self._update_ctbc_display()

    def _set_initial_speed(self, speed: float) -> None:
        """把初始播放速度吸附到 MJLab 支持的最近档位。"""
        speeds = list(self.SPEED_MULTIPLIERS)
        speed_index = min(range(len(speeds)), key=lambda index: abs(speeds[index] - speed))
        self._speed_index = speed_index
        self._time_multiplier = speeds[speed_index]

    def _handle_custom_action(self, action: Any, payload: Any | None) -> bool:
        """处理 SE3 自定义 Viser GUI action。"""
        if isinstance(payload, dict) and payload.get("type") == "gui_push_robot":
            self._handle_gui_push_robot(payload)
            return True
        return super()._handle_custom_action(action, payload)

    def _handle_gui_push_robot(self, payload: dict[str, Any]) -> None:
        """给选中环境一次性叠加 root velocity 扰动。"""
        env = self.env.unwrapped
        if bool(payload.get("all_envs", False)):
            env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        else:
            env_idx = max(0, min(int(payload.get("env_idx", self._scene.env_idx)), env.num_envs - 1))
            env_ids = torch.tensor([env_idx], dtype=torch.int64, device=env.device)

        raw_delta = payload.get("delta_velocity", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta = torch.as_tensor(raw_delta, dtype=torch.float32, device=env.device)
        if delta.numel() != 6:
            print(f"[WARN]: 忽略非法 Viser push delta: {raw_delta!r}")
            return

        asset = env.scene["robot"]
        with self._sim_lock:
            vel_w = asset.data.root_link_vel_w[env_ids].clone()
            vel_w += delta.reshape(1, 6)
            asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.sim.sense()

        if hasattr(self, "_scene"):
            self._scene.request_update()
        print(
            "[VISER] push_robot "
            f"env_ids={env_ids.detach().cpu().tolist()} "
            f"delta={[round(float(value), 3) for value in delta.detach().cpu().tolist()]}"
        )

    def _update_training_iter_display(self, force: bool = False) -> None:
        """低频读取训练日志，把当前 PPO iter 写入 Viser GUI。"""
        if self._se3_train_iter_html is None or self._se3_train_run_dir is None:
            return

        now = time.monotonic()
        if not force and now - self._se3_train_iter_last_update < _TRAINING_ITER_UPDATE_INTERVAL_S:
            return
        self._se3_train_iter_last_update = now

        progress = _read_training_progress(self._se3_train_run_dir)
        if self._ckpt_mgr is not None:
            progress["selected_checkpoint"] = self._ckpt_mgr.current_name
        self._se3_train_iter_html.content = _format_training_progress_html(
            self._se3_train_run_dir, progress
        )

    def _sync_ctbc_checkpoint_iteration(self, force: bool = False) -> None:
        """让 watch 的 CTBC 课程始终跟随当前所选 checkpoint。"""
        state = getattr(self.env.unwrapped, "stair_climb_state", None)
        if state is None:
            return
        checkpoint_name = self._ckpt_mgr.current_name if self._ckpt_mgr is not None else ""
        iteration = _checkpoint_iteration(Path(checkpoint_name))
        if iteration < 0:
            raw_iteration = os.environ.get("SE3_WATCH_ITER", "")
            iteration = int(raw_iteration) if raw_iteration.isdigit() else -1
        if iteration < 0 or (not force and iteration == self._se3_ctbc_checkpoint_iter):
            return
        state.set_fixed_iteration(iteration)
        self._se3_ctbc_checkpoint_iter = iteration

    def _update_ctbc_display(self, force: bool = False) -> None:
        """显示所选环境从碰撞判据到动作注入的完整 CTBC 链路。"""
        if self._se3_ctbc_html is None:
            return
        now = time.monotonic()
        if not force and now - self._se3_ctbc_last_update < _CTBC_UPDATE_INTERVAL_S:
            return
        self._se3_ctbc_last_update = now

        env = self.env.unwrapped
        state = getattr(env, "stair_climb_state", None)
        if state is None:
            return
        active_mask = state.contact_triggered()
        active_ids = active_mask.nonzero().flatten().detach().cpu().tolist()
        env_idx = int(self._scene.env_idx)
        if (
            self._se3_follow_ctbc is not None
            and self._se3_follow_ctbc.value
            and active_ids
            and not bool(active_mask[env_idx].item())
        ):
            env_idx = int(active_ids[0])
            self._scene.env_idx = env_idx
        contact = state.latest_contact_force[env_idx].detach().cpu().tolist()
        stable = state.stable_contact[env_idx].detach().cpu().tolist()
        phase = state.ff_phase[env_idx].detach().cpu().tolist()
        cooldown = state.cooldown[env_idx].detach().cpu().tolist()
        active_count = len(active_ids)
        stable_count = int(state.stable_contact.any(dim=-1).sum().item())
        cycles = int(state.complete_ff_cycle_count[env_idx].item())

        output_bias = [0.0] * 4
        action_delta = [0.0] * 4
        requested_wheel_delta = [[0.0, 0.0], [0.0, 0.0]]
        actual_wheel_xz = [[0.0, 0.0], [0.0, 0.0]]
        target_wheel_xz = [[0.0, 0.0], [0.0, 0.0]]
        action_term = env.action_manager.get_term("delayed_action")
        if hasattr(action_term, "ctbc_output_bias"):
            output_bias = action_term.ctbc_output_bias[env_idx, :4].detach().cpu().tolist()
        if hasattr(action_term, "ctbc_action_delta"):
            action_delta = action_term.ctbc_action_delta[env_idx, :4].detach().cpu().tolist()
        if hasattr(action_term, "ctbc_wheel_delta_xz"):
            requested_wheel_delta = action_term.ctbc_wheel_delta_xz[env_idx].detach().cpu().tolist()
        if hasattr(action_term, "actual_wheel_xz"):
            actual_wheel_xz = action_term.actual_wheel_xz[env_idx].detach().cpu().tolist()
        if hasattr(action_term, "target_wheel_xz"):
            target_wheel_xz = action_term.target_wheel_xz[env_idx].detach().cpu().tolist()

        self._se3_ctbc_html.content = _format_ctbc_html(
            env_idx=env_idx,
            num_envs=int(env.num_envs),
            local_iter=int(state.local_iteration),
            kff=float(state.kff),
            force_threshold=float(state.force_threshold),
            contact_window=int(state.contact_window),
            contact=contact,
            stable=stable,
            phase=phase,
            period_steps=int(state.ff_period_steps),
            cooldown=cooldown,
            active_count=active_count,
            active_ids=active_ids,
            stable_count=stable_count,
            cycles=cycles,
            output_bias=output_bias,
            action_delta=action_delta,
            requested_wheel_delta=requested_wheel_delta,
            actual_wheel_xz=actual_wheel_xz,
            target_wheel_xz=target_wheel_xz,
        )
        if now - self._se3_ctbc_last_log >= _CTBC_LOG_INTERVAL_S:
            self._se3_ctbc_last_log = now
            print(
                "[CTBC] "
                f"iter={state.local_iteration} kff={state.kff:.3f} "
                f"active={active_count}/{env.num_envs} stable={stable_count}/{env.num_envs} "
                f"active_ids={active_ids} env={env_idx} force={contact} phase={phase} "
                f"bias={[round(value, 3) for value in output_bias]} "
                f"delta={[round(value, 3) for value in action_delta]} "
                f"wheel_delta_cm={_round_nested_cm(requested_wheel_delta)} "
                f"actual_xz_cm={_round_nested_cm(actual_wheel_xz)} "
                f"target_xz_cm={_round_nested_cm(target_wheel_xz)}"
            )


def _format_ctbc_html(
    *,
    env_idx: int,
    num_envs: int,
    local_iter: int,
    kff: float,
    force_threshold: float,
    contact_window: int,
    contact: list[float],
    stable: list[bool],
    phase: list[int],
    period_steps: int,
    cooldown: list[int],
    active_count: int,
    active_ids: list[int],
    stable_count: int,
    cycles: int,
    output_bias: list[float],
    action_delta: list[float],
    requested_wheel_delta: list[list[float]],
    actual_wheel_xz: list[list[float]],
    target_wheel_xz: list[list[float]],
) -> str:
    """格式化 CTBC 传感、状态机和动作注入遥测。"""
    active = [value >= 0 for value in phase]
    return f"""
      <div style="font-size:0.85em; line-height:1.4; padding:0 1em 0.5em 1em;">
        <strong>Checkpoint iteration:</strong> {local_iter}<br/>
        <strong>kff:</strong> {kff:.3f}<br/>
        <strong>All envs:</strong> active {active_count}/{num_envs},
        stable-contact {stable_count}/{num_envs}<br/>
        <strong>Active env IDs:</strong> {active_ids}<br/>
        <strong>Selected env:</strong> {env_idx}, completed cycles {cycles}<br/>
        <strong>Trigger:</strong> force &gt; {force_threshold:.1f} N for
        {contact_window} consecutive frames<br/>
        <strong>Contact L/R:</strong> {contact[0]:.1f} / {contact[1]:.1f} N<br/>
        <strong>Stable L/R:</strong> {stable[0]} / {stable[1]}<br/>
        <strong>Active L/R:</strong> {active[0]} / {active[1]}<br/>
        <strong>Phase L/R:</strong> {phase[0]} / {phase[1]} of {period_steps}<br/>
        <strong>Cooldown L/R:</strong> {cooldown[0]} / {cooldown[1]}<br/>
        <strong>Legacy bias:</strong>
        [{", ".join(f"{value:+.3f}" for value in output_bias)}]<br/>
        <strong>Injected action delta:</strong>
        [{", ".join(f"{value:+.3f}" for value in action_delta)}]<br/>
        <strong>Requested wheel dX/dZ (cm) L/R:</strong>
        {_format_xz_cm(requested_wheel_delta)}<br/>
        <strong>Actual wheel X/Z (cm) L/R:</strong>
        {_format_xz_cm(actual_wheel_xz)}<br/>
        <strong>Target wheel X/Z (cm) L/R:</strong>
        {_format_xz_cm(target_wheel_xz)}
      </div>
    """


def _format_xz_cm(values: list[list[float]]) -> str:
    return " / ".join(f"({side[0] * 100.0:+.1f}, {side[1] * 100.0:+.1f})" for side in values)


def _round_nested_cm(values: list[list[float]]) -> list[list[float]]:
    return [[round(component * 100.0, 2) for component in side] for side in values]


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _viser_frame_rate_from_env(*, default: float) -> float:
    """读取 Viser 目标刷新率；无配置时保持 MJLab 默认行为。"""
    raw = os.environ.get(_VISER_FRAME_RATE_ENV)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN]: 忽略非法 {_VISER_FRAME_RATE_ENV}={raw!r}")
        return default
    if value <= 0.0:
        print(f"[WARN]: 忽略非正 {_VISER_FRAME_RATE_ENV}={raw!r}")
        return default
    return value


def _viser_initial_speed_from_env() -> float | None:
    """读取 Viser 初始播放速度；支持小数或 1/32 这类分数写法。"""
    raw = os.environ.get(_VISER_INITIAL_SPEED_ENV)
    if not raw:
        return None
    try:
        value = _parse_positive_float(raw)
    except ValueError:
        print(f"[WARN]: 忽略非法 {_VISER_INITIAL_SPEED_ENV}={raw!r}")
        return None
    if value <= 0.0:
        print(f"[WARN]: 忽略非正 {_VISER_INITIAL_SPEED_ENV}={raw!r}")
        return None
    return value


def _parse_positive_float(raw: str) -> float:
    """解析正浮点数或分数字符串。"""
    text = raw.strip()
    if "/" not in text:
        return float(text)
    numerator, denominator = text.split("/", 1)
    return float(numerator) / float(denominator)


def _training_run_dir_from_env() -> Path | None:
    """从环境变量读取需要显示在 Viser 里的训练 run 目录。"""
    raw = os.environ.get("SE3_VISER_TRAIN_RUN_DIR")
    if not raw:
        return None
    return Path(raw).expanduser()


def _read_training_progress(run_dir: Path) -> dict[str, int | float | str | None]:
    """读取训练日志和 checkpoint，返回 Viser 面板需要的进度字段。"""
    logged_progress = _latest_logged_progress(run_dir)
    status_progress = _read_training_status(run_dir)
    progress = logged_progress
    if status_progress is not None:
        progress = {
            key: value if value is not None else logged_progress[key]
            for key, value in status_progress.items()
        }
    checkpoint_iter = _latest_checkpoint_iteration(run_dir)
    selected_checkpoint = _selected_checkpoint_from_env()
    return {
        "iteration": progress["iteration"],
        "total": progress["total"],
        "collect_time_s": progress["collect_time_s"],
        "learning_time_s": progress["learning_time_s"],
        "iter_time_s": progress["iter_time_s"],
        "checkpoint_iter": checkpoint_iter,
        "selected_checkpoint": selected_checkpoint,
        "updated_at": time.strftime("%H:%M:%S"),
    }


def _selected_checkpoint_from_env() -> str | None:
    """读取当前 se3-play 实际加载的 checkpoint 文件名。"""
    raw = os.environ.get("SE3_VISER_SELECTED_CHECKPOINT")
    if not raw:
        return None
    return Path(raw).name


def _read_training_status(run_dir: Path) -> dict[str, int | float | None] | None:
    """读取训练 runner 写出的实时状态文件。"""
    status_path = run_dir / TRAINING_STATUS_FILENAME
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    return {
        "iteration": _coerce_int(payload.get("iteration")),
        "total": _coerce_int(payload.get("total_iterations")),
        "collect_time_s": _coerce_float(payload.get("collect_time_s")),
        "learning_time_s": _coerce_float(payload.get("learning_time_s")),
        "iter_time_s": _coerce_float(payload.get("iter_time_s")),
    }


def _latest_logged_progress(run_dir: Path) -> dict[str, int | float | None]:
    """从 rank0 日志尾部提取最新的训练进度和耗时。"""
    empty = {
        "iteration": None,
        "total": None,
        "collect_time_s": None,
        "learning_time_s": None,
        "iter_time_s": None,
    }
    if not run_dir.exists():
        return empty

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
            latest_block = text[match.end() :]
            return {
                "iteration": int(match.group(1)),
                "total": int(match.group(2)),
                "collect_time_s": _latest_pattern_float(_TRAINING_COLLECT_PATTERN, latest_block),
                "learning_time_s": _latest_pattern_float(_TRAINING_LEARNING_PATTERN, latest_block),
                "iter_time_s": _latest_pattern_float(_TRAINING_ITER_TIME_PATTERN, latest_block),
            }
    return empty


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


def _latest_pattern_float(pattern: re.Pattern[str], text: str) -> float | None:
    """返回文本中最后一次匹配到的浮点数。"""
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group(1))


def _coerce_int(value: object) -> int | None:
    """把 JSON 字段收窄为 int。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_float(value: object) -> float | None:
    """把 JSON 字段收窄为 float。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _format_seconds(value: object) -> str:
    """格式化秒级耗时字段。"""
    if isinstance(value, int | float):
        return f"{float(value):.3f}s"
    return "waiting"


def _format_training_progress_html(
    run_dir: Path, progress: dict[str, int | float | str | None]
) -> str:
    """格式化训练进度面板 HTML。"""
    iteration = progress["iteration"]
    total = progress["total"]
    collect_time_s = progress["collect_time_s"]
    learning_time_s = progress["learning_time_s"]
    iter_time_s = progress["iter_time_s"]
    checkpoint_iter = progress["checkpoint_iter"]
    selected_checkpoint = progress["selected_checkpoint"]
    updated_at = progress["updated_at"]

    if isinstance(iteration, int) and isinstance(total, int):
        iter_text = f"{iteration} / {total}"
    else:
        iter_text = "waiting for status"
    ckpt_text = f"model_{checkpoint_iter}.pt" if isinstance(checkpoint_iter, int) else "none"
    selected_text = (
        str(selected_checkpoint) if isinstance(selected_checkpoint, str) else "unspecified"
    )
    run_name = html.escape(run_dir.name)
    iter_text = html.escape(iter_text)
    ckpt_text = html.escape(ckpt_text)
    selected_text = html.escape(selected_text)
    iter_time_text = html.escape(_format_seconds(iter_time_s))
    collect_time_text = html.escape(_format_seconds(collect_time_s))
    learning_time_text = html.escape(_format_seconds(learning_time_s))
    updated_text = html.escape(str(updated_at))

    return f"""
      <div style="font-size:0.85em; line-height:1.35; padding:0 1em 0.5em 1em;">
        <strong>Training progress:</strong> {iter_text}<br/>
        <strong>iter_time:</strong> {iter_time_text}<br/>
        <strong>collect_time:</strong> {collect_time_text}<br/>
        <strong>learning_time:</strong> {learning_time_text}<br/>
        <strong>Selected checkpoint:</strong> {selected_text}<br/>
        <strong>Latest checkpoint:</strong> {ckpt_text}<br/>
        <strong>Run:</strong> {run_name}<br/>
        <strong>Updated:</strong> {updated_text}
      </div>
    """


def _configure_viser_training_env(task_id: str, cfg: PlayConfig) -> None:
    """为 se3-play 自动选择要显示的训练 run。"""
    if cfg.checkpoint_file is not None and not os.environ.get("SE3_VISER_SELECTED_CHECKPOINT"):
        os.environ["SE3_VISER_SELECTED_CHECKPOINT"] = Path(cfg.checkpoint_file).name

    if os.environ.get("SE3_VISER_TRAIN_RUN_DIR"):
        return

    run_dir = _training_run_dir_from_checkpoint(cfg.checkpoint_file)
    if run_dir is None and cfg.wandb_run_path is None:
        run_dir = _latest_training_run_dir(task_id, Path(cfg.log_root))
    if run_dir is not None:
        os.environ["SE3_VISER_TRAIN_RUN_DIR"] = str(run_dir)


def _training_run_dir_from_checkpoint(checkpoint_file: str | None) -> Path | None:
    """从本地 checkpoint 路径推断训练 run 目录。"""
    if checkpoint_file is None:
        return None
    checkpoint = Path(checkpoint_file).expanduser()
    return checkpoint.parent if checkpoint.exists() else None


def _latest_training_run_dir(task_id: str, log_root: Path) -> Path | None:
    """按任务配置自动选择最近更新的本地训练 run。"""
    root = log_root.expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    experiment_name = load_rl_cfg(task_id).experiment_name
    experiment_root = root / experiment_name
    if not experiment_root.exists():
        return None
    runs = [path for path in experiment_root.iterdir() if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: (path.stat().st_mtime, path.name))


def _install_se3_viser_viewer() -> None:
    """替换 MJLab 默认 Viser viewer，增加 SE3 训练值守面板。"""
    mjlab_viser_viewer.MjlabViserScene = _Se3MjlabViserScene
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
    resolved_args = _resolve_play_config(chosen_task, args)
    _configure_viser_training_env(chosen_task, resolved_args)
    mjlab_play.run_play(chosen_task, resolved_args)


if __name__ == "__main__":
    main()
