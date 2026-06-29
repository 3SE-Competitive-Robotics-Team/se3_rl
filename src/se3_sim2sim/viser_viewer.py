"""sim2sim 原生 MuJoCo 的 Viser 浏览器 viewer。"""

from __future__ import annotations

import html
import json
import re
import threading
import time
from pathlib import Path

import mujoco

from .teleop_input import CommandInputUpdate

_SPEEDS = (1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0)
_TRAINING_STATUS_FILENAME = "se3_training_status.json"
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")
_NO_CHECKPOINT_OPTION = "(no model_*.pt)"
_NORMAL_RESET_MODE = "normal"
_RECOVERY_RESET_OPTIONS = ("left_side", "right_side", "prone", "supine")
_COMMAND_VX_LIMIT = 1.89
_COMMAND_YAW_LIMIT = 9.41
_COMMAND_HEIGHT_MIN = 0.195
_COMMAND_HEIGHT_MAX = 0.390
_COMMAND_DEFAULT_HEIGHT = 0.260
_PUSH_VEL_LIMIT = 2.0
_PUSH_YAW_LIMIT = 8.0


class ViserViewer:
    """把 sim2sim 的单环境 MuJoCo 状态推送到 Viser。"""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        control_dt: float,
        frame_rate: float = 30.0,
        port: int = 8080,
        label: str = "se3_sim2sim",
        geom_view: str = "visual",
        checkpoint_path: Path | None = None,
        policy_iteration: object = None,
        initial_command: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        import viser
        from mjviser import ViserMujocoScene

        self._viser = viser
        self._server = viser.ViserServer(host="0.0.0.0", port=int(port), label=label)
        self._scene = ViserMujocoScene(self._server, model, num_envs=1)
        self._control_dt = float(control_dt)
        self._frame_time = 1.0 / max(float(frame_rate), 1.0)
        self._speed_index = _SPEEDS.index(1.0)
        self._speed = 1.0
        self._paused = False
        self._closed = False
        self._reset_request_lock = threading.Lock()
        self._reset_request_mode: str | None = None
        self._reset_request_last_time = 0.0
        self._last_reset_mode = "initial"
        self._reset_count = 0
        self._last_render_time = 0.0
        self._start_wall_time: float | None = None
        self._wall_start_time = time.perf_counter()
        self._last_step = 0
        self._stats_steps = 0
        self._stats_frames = 0
        self._stats_last_time = time.perf_counter()
        self._actual_rt = 0.0
        self._fps = 0.0
        self._step_wall_ms = 0.0
        self._checkpoint_path = checkpoint_path
        self._policy_iteration = policy_iteration
        self._checkpoint_dropdown = None
        self._checkpoint_dropdown_updating = False
        self._checkpoint_options_last_update = 0.0
        self._checkpoint_request_lock = threading.Lock()
        self._checkpoint_switch_requested: Path | None = None
        self._checkpoint_switch_status = "ready"
        self._training_progress_last_update = 0.0
        self._training_progress: dict[str, object] = {}
        self._last_telemetry: dict[str, object] = {}
        self._command_lock = threading.Lock()
        self._command_vx = _command_value(initial_command, 0, 0.0)
        self._command_yaw_rate = _command_value(initial_command, 1, 0.0)
        self._command_height = _clamp(
            _command_value(initial_command, 4, _COMMAND_DEFAULT_HEIGHT),
            _COMMAND_HEIGHT_MIN,
            _COMMAND_HEIGHT_MAX,
        )
        self._command_gui_status = "ready"
        self._push_lock = threading.Lock()
        self._pending_push_delta: list[float] | None = None
        self._push_count = 0
        self._push_status = "none"
        self._configure_geom_groups(geom_view)
        self._setup_gui()

    def log_model(self, model: mujoco.MjModel) -> None:
        del model

    def log_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        step: int,
        telemetry: dict[str, object],
    ) -> None:
        del model
        if self._closed:
            return
        self._last_telemetry = telemetry
        self._wait_if_paused()
        self._pace(step)
        now = time.perf_counter()
        delta_steps = max(0, int(step) - int(self._last_step))
        self._stats_steps += delta_steps
        self._last_step = int(step)
        if now - self._last_render_time >= self._frame_time or step == 0:
            self._scene.update_from_mjdata(data)
            self._stats_frames += 1
            self._last_render_time = now
        self._update_stats(now)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.stop()

    @property
    def closed(self) -> bool:
        """viewer 是否已关闭。"""

        return bool(self._closed)

    def consume_reset_request(self) -> str | None:
        """消费 GUI 触发的 reset 请求，并返回 reset 模式。"""

        with self._reset_request_lock:
            mode = self._reset_request_mode
            self._reset_request_mode = None
        return mode

    def consume_reset_requested(self) -> bool:
        """消费 GUI 触发的 reset 请求。"""

        return self.consume_reset_request() is not None

    def notify_reset(self, mode: str = _NORMAL_RESET_MODE) -> None:
        """主仿真线程完成 reset 后同步 viewer 计数和计时。"""

        self._last_reset_mode = str(mode)
        self._reset_count += 1
        self._last_step = 0
        self._stats_steps = 0
        self._stats_frames = 0
        self._actual_rt = 0.0
        self._fps = 0.0
        self._step_wall_ms = 0.0
        now = time.perf_counter()
        self._stats_last_time = now
        self._last_render_time = 0.0
        self._start_wall_time = None
        self._wall_start_time = now
        self._update_status_display()

    def consume_checkpoint_request(self) -> Path | None:
        """消费 GUI 触发的 checkpoint 切换请求。"""

        with self._checkpoint_request_lock:
            requested = self._checkpoint_switch_requested
            self._checkpoint_switch_requested = None
        return requested

    def consume_push_request(self) -> tuple[float, float, float, float, float, float] | None:
        """消费 GUI 触发的一次性 root velocity 扰动。"""

        with self._push_lock:
            delta = self._pending_push_delta
            self._pending_push_delta = None
            if delta is None:
                return None
            self._push_count += 1
            self._push_status = (
                f"applied #{self._push_count}: "
                f"vx={delta[0]:+.2f}, vy={delta[1]:+.2f}, yaw={delta[5]:+.2f}"
            )
        self._update_status_display()
        return tuple(float(value) for value in delta)

    def notify_checkpoint_loaded(self, checkpoint_path: Path, policy_iteration: object) -> None:
        """主仿真线程完成 policy 切换后同步 GUI 状态。"""

        self._checkpoint_path = Path(checkpoint_path)
        self._policy_iteration = policy_iteration
        self._checkpoint_switch_status = f"loaded {self._checkpoint_path.name}"
        self._refresh_checkpoint_dropdown(force=True)
        self._update_status_display()

    def notify_checkpoint_failed(self, checkpoint_path: Path, message: str) -> None:
        """主仿真线程加载 checkpoint 失败后同步 GUI 状态。"""

        self._checkpoint_switch_status = f"failed {Path(checkpoint_path).name}: {message}"
        self._update_status_display()

    def pace(self, sim_time_s: float) -> None:
        """CommandInputSource 兼容接口；Viser 自己在 log_state 中做播放节拍。"""

        del sim_time_s

    def poll(self, sim_time_s: float) -> CommandInputUpdate:
        """返回 GUI 当前设置的速度、角速度和高度指令。"""

        del sim_time_s
        with self._command_lock:
            return CommandInputUpdate(
                lin_vel_x=float(self._command_vx),
                yaw_rate=float(self._command_yaw_rate),
                command_height=float(self._command_height),
            )

    def _setup_gui(self) -> None:
        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Controls", icon=self._viser.Icon.SETTINGS):
            with self._server.gui.add_folder("Info"):
                self._status_html = self._server.gui.add_html("")
            with self._server.gui.add_folder("Policy"):
                options = self._checkpoint_options()
                initial_value = self._initial_checkpoint_option(options)
                self._checkpoint_dropdown = self._server.gui.add_dropdown(
                    "Checkpoint",
                    options=options,
                    initial_value=initial_value,
                    disabled=self._checkpoint_path is None,
                    hint="Select a model_*.pt from the current run directory.",
                )

                @self._checkpoint_dropdown.on_update
                def _(event) -> None:
                    if self._checkpoint_dropdown_updating:
                        return
                    selected = str(event.target.value)
                    self._request_checkpoint_switch_by_name(selected)

                refresh_button = self._server.gui.add_button("Refresh Checkpoints")

                @refresh_button.on_click
                def _(_) -> None:
                    self._refresh_checkpoint_dropdown(force=True)
                    self._update_status_display()

                latest_button = self._server.gui.add_button("Load Latest Checkpoint")

                @latest_button.on_click
                def _(_) -> None:
                    latest = self._latest_checkpoint_path()
                    if latest is None:
                        self._checkpoint_switch_status = "no checkpoint found"
                        self._update_status_display()
                        return
                    self._set_checkpoint_dropdown_value(latest.name)
                    self._request_checkpoint_switch(latest)

            with self._server.gui.add_folder("Simulation"):
                pause_button = self._server.gui.add_button(
                    "Pause",
                    icon=self._viser.Icon.PLAYER_PAUSE,
                )

                @pause_button.on_click
                def _(_) -> None:
                    self._paused = not self._paused
                    pause_button.label = "Play" if self._paused else "Pause"
                    pause_button.icon = (
                        self._viser.Icon.PLAYER_PLAY
                        if self._paused
                        else self._viser.Icon.PLAYER_PAUSE
                    )
                    self._stats_last_time = time.perf_counter()
                    self._update_status_display()

                reset_button = self._server.gui.add_button("Reset Environment")

                @reset_button.on_click
                def _(_) -> None:
                    self._request_reset(_NORMAL_RESET_MODE)

                recovery_pose = self._server.gui.add_dropdown(
                    "Recovery Pose",
                    options=_RECOVERY_RESET_OPTIONS,
                    initial_value=_RECOVERY_RESET_OPTIONS[0],
                    hint="Reset to a fixed recovery pose and clear policy state.",
                )
                recovery_reset_button = self._server.gui.add_button("Recovery Reset")

                @recovery_reset_button.on_click
                def _(_) -> None:
                    self._request_reset(str(recovery_pose.value))

                speed_buttons = self._server.gui.add_button_group(
                    "Speed",
                    options=["Slower", "1x", "Faster"],
                )

                @speed_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Slower":
                        self._speed_index = max(0, self._speed_index - 1)
                    elif event.target.value == "Faster":
                        self._speed_index = min(len(_SPEEDS) - 1, self._speed_index + 1)
                    else:
                        self._speed_index = _SPEEDS.index(1.0)
                    self._speed = float(_SPEEDS[self._speed_index])
                    self._start_wall_time = None
                    self._update_status_display()

            with self._server.gui.add_folder("Command"):
                vx_slider = self._server.gui.add_slider(
                    "vx (m/s)",
                    min=-_COMMAND_VX_LIMIT,
                    max=_COMMAND_VX_LIMIT,
                    step=0.05,
                    initial_value=self._command_vx,
                )
                yaw_slider = self._server.gui.add_slider(
                    "yaw_rate (rad/s)",
                    min=-_COMMAND_YAW_LIMIT,
                    max=_COMMAND_YAW_LIMIT,
                    step=0.05,
                    initial_value=self._command_yaw_rate,
                )
                height_slider = self._server.gui.add_slider(
                    "height (m)",
                    min=_COMMAND_HEIGHT_MIN,
                    max=_COMMAND_HEIGHT_MAX,
                    step=0.005,
                    initial_value=self._command_height,
                )
                command_buttons = self._server.gui.add_button_group(
                    "Command",
                    options=["Zero Motion", "Default Height"],
                )

                def _sync_command_from_gui(status: str = "updated") -> None:
                    with self._command_lock:
                        self._command_vx = float(vx_slider.value)
                        self._command_yaw_rate = float(yaw_slider.value)
                        self._command_height = _clamp(
                            float(height_slider.value),
                            _COMMAND_HEIGHT_MIN,
                            _COMMAND_HEIGHT_MAX,
                        )
                    self._command_gui_status = status
                    self._update_status_display()

                @vx_slider.on_update
                def _(_) -> None:
                    _sync_command_from_gui()

                @yaw_slider.on_update
                def _(_) -> None:
                    _sync_command_from_gui()

                @height_slider.on_update
                def _(_) -> None:
                    _sync_command_from_gui()

                @command_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Zero Motion":
                        vx_slider.value = 0.0
                        yaw_slider.value = 0.0
                        _sync_command_from_gui("zero motion")
                    elif event.target.value == "Default Height":
                        height_slider.value = _COMMAND_DEFAULT_HEIGHT
                        _sync_command_from_gui("default height")

            with self._server.gui.add_folder("Push Robot"):
                push_vx_slider = self._server.gui.add_slider(
                    "delta vx (m/s)",
                    min=-_PUSH_VEL_LIMIT,
                    max=_PUSH_VEL_LIMIT,
                    step=0.05,
                    initial_value=0.8,
                )
                push_vy_slider = self._server.gui.add_slider(
                    "delta vy (m/s)",
                    min=-_PUSH_VEL_LIMIT,
                    max=_PUSH_VEL_LIMIT,
                    step=0.05,
                    initial_value=0.0,
                )
                push_yaw_slider = self._server.gui.add_slider(
                    "delta yaw_rate (rad/s)",
                    min=-_PUSH_YAW_LIMIT,
                    max=_PUSH_YAW_LIMIT,
                    step=0.1,
                    initial_value=0.0,
                )
                push_button = self._server.gui.add_button("Apply Push")

                @push_button.on_click
                def _(_) -> None:
                    delta = [
                        float(push_vx_slider.value),
                        float(push_vy_slider.value),
                        0.0,
                        0.0,
                        0.0,
                        float(push_yaw_slider.value),
                    ]
                    with self._push_lock:
                        self._pending_push_delta = delta
                        self._push_status = (
                            f"queued: vx={delta[0]:+.2f}, vy={delta[1]:+.2f}, yaw={delta[5]:+.2f}"
                        )
                    self._update_status_display()

            with self._server.gui.add_folder("Scene"):
                self._scene.create_scene_gui(
                    camera_distance=2.2,
                    camera_azimuth=135.0,
                    camera_elevation=-18.0,
                )
        with tabs.add_tab("Visualization", icon=self._viser.Icon.EYE):
            self._scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=self._viser.Icon.LAYERS_INTERSECT):
            self._scene.create_groups_gui()
        self._update_status_display()

    def _checkpoint_info(self) -> dict[str, str]:
        if self._checkpoint_path is None:
            return {
                "run": "unspecified",
                "selected": "unspecified",
                "latest": "none",
                "checkpoint_iter": "unknown",
                "path": "",
            }
        checkpoint = Path(self._checkpoint_path)
        latest = _latest_checkpoint_name(checkpoint.parent)
        return {
            "run": checkpoint.parent.name,
            "selected": checkpoint.name,
            "latest": latest,
            "checkpoint_iter": _checkpoint_iteration_text(checkpoint),
            "path": str(checkpoint),
        }

    def _training_progress_info(self) -> dict[str, object]:
        now = time.monotonic()
        if now - self._training_progress_last_update < 2.0:
            return self._training_progress
        self._training_progress_last_update = now
        if self._checkpoint_path is None:
            self._training_progress = {}
            return self._training_progress
        status_path = Path(self._checkpoint_path).parent / _TRAINING_STATUS_FILENAME
        if not status_path.exists():
            self._training_progress = {}
            return self._training_progress
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._training_progress = {}
            return self._training_progress
        self._training_progress = payload if isinstance(payload, dict) else {}
        return self._training_progress

    def _checkpoint_options(self) -> tuple[str, ...]:
        if self._checkpoint_path is None:
            return (_NO_CHECKPOINT_OPTION,)
        candidates = _checkpoint_paths(Path(self._checkpoint_path).parent)
        if not candidates:
            return (_NO_CHECKPOINT_OPTION,)
        return tuple(path.name for path in candidates)

    def _initial_checkpoint_option(self, options: tuple[str, ...]) -> str:
        if self._checkpoint_path is not None and Path(self._checkpoint_path).name in options:
            return Path(self._checkpoint_path).name
        return options[-1]

    def _refresh_checkpoint_dropdown(self, *, force: bool = False) -> None:
        dropdown = self._checkpoint_dropdown
        if dropdown is None:
            return
        now = time.monotonic()
        if not force and now - self._checkpoint_options_last_update < 2.0:
            return
        self._checkpoint_options_last_update = now
        options = self._checkpoint_options()
        current_value = str(dropdown.value)
        preferred = (
            current_value if current_value in options else self._initial_checkpoint_option(options)
        )
        self._checkpoint_dropdown_updating = True
        try:
            dropdown.options = options
            dropdown.value = preferred
            dropdown.disabled = self._checkpoint_path is None or options == (_NO_CHECKPOINT_OPTION,)
        finally:
            self._checkpoint_dropdown_updating = False

    def _set_checkpoint_dropdown_value(self, value: str) -> None:
        dropdown = self._checkpoint_dropdown
        if dropdown is None:
            return
        self._checkpoint_dropdown_updating = True
        try:
            if value in dropdown.options:
                dropdown.value = value
        finally:
            self._checkpoint_dropdown_updating = False

    def _request_checkpoint_switch_by_name(self, selected: str) -> None:
        if self._checkpoint_path is None or selected == _NO_CHECKPOINT_OPTION:
            self._checkpoint_switch_status = "no checkpoint selected"
            self._update_status_display()
            return
        target = Path(self._checkpoint_path).parent / selected
        self._request_checkpoint_switch(target)

    def _request_checkpoint_switch(self, target: Path) -> None:
        target = Path(target)
        if not target.exists():
            self._checkpoint_switch_status = f"missing {target.name}"
            self._update_status_display()
            return
        if (
            self._checkpoint_path is not None
            and target.resolve() == Path(self._checkpoint_path).resolve()
        ):
            self._checkpoint_switch_status = f"already loaded {target.name}"
            self._update_status_display()
            return
        with self._checkpoint_request_lock:
            self._checkpoint_switch_requested = target.resolve()
        self._checkpoint_switch_status = f"loading {target.name}"
        self._update_status_display()

    def _request_reset(self, mode: str) -> None:
        now = time.monotonic()
        if now - self._reset_request_last_time < 0.5:
            return
        self._reset_request_last_time = now
        with self._reset_request_lock:
            self._reset_request_mode = str(mode)
        self._update_status_display()

    def _latest_checkpoint_path(self) -> Path | None:
        if self._checkpoint_path is None:
            return None
        candidates = _checkpoint_paths(Path(self._checkpoint_path).parent)
        if not candidates:
            return None
        return candidates[-1]

    def _configure_geom_groups(self, geom_view: str) -> None:
        groups = [False, False, True, False, False, False]
        if geom_view == "visual":
            groups[1] = True
        elif geom_view == "collision":
            groups[0] = True
        elif geom_view == "both":
            groups[0] = True
            groups[1] = True
        else:
            raise ValueError(f"未知几何显示模式: {geom_view}")
        self._scene.geom_groups_visible = groups
        sync = getattr(self._scene, "_sync_visibilities", None)
        if callable(sync):
            sync()

    def _wait_if_paused(self) -> None:
        while self._paused and not self._closed:
            self._update_status_display()
            time.sleep(0.05)
            self._stats_last_time = time.perf_counter()

    def _pace(self, step: int) -> None:
        if self._speed <= 0.0:
            return
        now = time.perf_counter()
        if self._start_wall_time is None:
            self._start_wall_time = now - int(step) * self._control_dt / self._speed
            return
        target_elapsed = int(step) * self._control_dt / self._speed
        ahead_s = target_elapsed - (now - self._start_wall_time)
        if ahead_s > 0.0:
            time.sleep(min(ahead_s, 0.05))

    def _update_stats(self, now: float) -> None:
        dt = now - self._stats_last_time
        if dt < 0.5:
            return
        self._actual_rt = self._stats_steps * self._control_dt / dt
        self._fps = self._stats_frames / dt
        self._step_wall_ms = dt / max(1, self._stats_steps) * 1000.0
        self._stats_steps = 0
        self._stats_frames = 0
        self._stats_last_time = now
        self._update_status_display()

    def _update_status_display(self) -> None:
        self._refresh_checkpoint_dropdown()
        progress = self._training_progress_info()
        checkpoint = self._checkpoint_info()
        rt_display = f"{self._actual_rt:.2f}x" if self._actual_rt > 0.0 else "-"
        fps_display = f"{self._fps:.0f}"
        status = "Paused" if self._paused else "Running"
        capped = (
            ' <span style="color:#e74c3c;">[CAPPED]</span>'
            if (not self._paused and self._actual_rt < 0.98 * self._speed)
            else ""
        )
        height = _fmt_float(self._last_telemetry.get("height"))
        tilt = _fmt_float(self._last_telemetry.get("tilt_deg"))
        vx = _fmt_float(self._last_telemetry.get("base_lin_vel_x"))
        yaw_rate = _fmt_float(_sequence_item(self._last_telemetry.get("base_ang_vel_body"), 2))
        sim_time = _fmt_seconds(self._last_telemetry.get("time"))
        wall_time = _fmt_seconds(time.perf_counter() - self._wall_start_time)
        progress_text = _format_training_progress(progress)
        policy_iteration = html.escape(str(self._policy_iteration))
        with self._command_lock:
            command_vx = self._command_vx
            command_yaw_rate = self._command_yaw_rate
            command_height = self._command_height
        with self._push_lock:
            push_status = self._push_status
        self._status_html.content = f"""
          <div style="font-size: 0.85em; line-height: 1.35;
                      padding: 0 1em 0.5em 1em;">
            <strong>Status:</strong> {status}{capped}<br/>
            <strong>Steps:</strong> {self._last_step}<br/>
            <strong>Sim time:</strong> {sim_time}<br/>
            <strong>Wall time:</strong> {wall_time}<br/>
            <strong>Reset count:</strong> {self._reset_count}<br/>
            <strong>Last reset:</strong> {html.escape(self._last_reset_mode)}<br/>
            <strong>Speed:</strong> {_format_speed(self._speed)}<br/>
            <strong>Target RT:</strong> {self._speed:.2f}x<br/>
            <strong>Actual RT:</strong> {rt_display} ({fps_display} FPS)<br/>
            <strong>Policy-step wall:</strong> {self._step_wall_ms:.2f} ms<br/>
            <strong>Height:</strong> {height} m<br/>
            <strong>Tilt:</strong> {tilt} deg<br/>
            <strong>Base vx:</strong> {vx} m/s<br/>
            <strong>Base yaw rate:</strong> {yaw_rate} rad/s<br/>
            <strong>Command:</strong>
              vx={command_vx:+.2f} m/s,
              yaw={command_yaw_rate:+.2f} rad/s,
              h={command_height:.3f} m
              ({html.escape(self._command_gui_status)})<br/>
            <strong>Push:</strong> {html.escape(push_status)}<br/>
            <hr style="border:0; border-top:1px solid #ddd; margin:0.45em 0;"/>
            <strong>Training progress:</strong> {progress_text}<br/>
            <strong>iter_time:</strong> {_fmt_seconds(progress.get("iter_time_s"))}<br/>
            <strong>collect_time:</strong> {_fmt_seconds(progress.get("collect_time_s"))}<br/>
            <strong>learning_time:</strong> {_fmt_seconds(progress.get("learning_time_s"))}<br/>
            <strong>Checkpoint iteration:</strong> {html.escape(checkpoint["checkpoint_iter"])}<br/>
            <strong>Policy iteration:</strong> {policy_iteration}<br/>
            <strong>Checkpoint switch:</strong> {html.escape(self._checkpoint_switch_status)}<br/>
            <strong>Selected checkpoint:</strong> {html.escape(checkpoint["selected"])}<br/>
            <strong>Latest checkpoint:</strong> {html.escape(checkpoint["latest"])}<br/>
            <strong>Run:</strong> {html.escape(checkpoint["run"])}
          </div>
          """


def _format_speed(value: float) -> str:
    if value >= 1.0:
        return f"{value:.0f}x"
    return f"1/{round(1.0 / value)}x"


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _command_value(
    command: tuple[float, ...] | list[float] | None, index: int, default: float
) -> float:
    if command is None or len(command) <= index:
        return float(default)
    try:
        return float(command[index])
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _sequence_item(value: object, index: int) -> object:
    try:
        return value[index]  # type: ignore[index]
    except (IndexError, TypeError):
        return None


def _fmt_seconds(value: object) -> str:
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "waiting"


def _format_training_progress(progress: dict[str, object]) -> str:
    iteration = _coerce_int(progress.get("iteration"))
    total = _coerce_int(progress.get("total_iterations"))
    if iteration is None or total is None:
        return "waiting for status"
    return html.escape(f"{iteration} / {total}")


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _checkpoint_iteration_text(path: Path) -> str:
    iteration = _checkpoint_iteration(path)
    return str(iteration) if iteration >= 0 else "unknown"


def _checkpoint_iteration(path: Path) -> int:
    match = _CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _latest_checkpoint_name(run_dir: Path) -> str:
    candidates = _checkpoint_paths(run_dir)
    if not candidates:
        return "none"
    return candidates[-1].name


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    candidates = list(run_dir.glob("model_*.pt")) if run_dir.exists() else []
    return sorted(candidates, key=_checkpoint_iteration)
