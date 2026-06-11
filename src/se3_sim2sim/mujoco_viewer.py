"""无 MuJoCo 默认键盘快捷键的原生渲染窗口。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import glfw
import mujoco
import numpy as np

from .math_utils import euler_xyz_to_quat_wxyz, quat_wxyz_to_xyzw

_PANEL_MIN_WIDTH = 280
_PANEL_MAX_WIDTH = 360
_JOINT_FALLBACK_LIMIT_RAD = 2.2
_BASE_ATTITUDE_LIMIT_RAD = math.radians(90.0)
_BASE_YAW_LIMIT_RAD = math.pi


@dataclass(frozen=True)
class _PoseControl:
    """右侧面板中的一个姿态滑条。"""

    key: str
    label: str
    lower: float
    upper: float
    kind: str
    index: int


@dataclass(frozen=True)
class _SliderRect:
    """滑条热区，坐标为 framebuffer 顶部原点。"""

    key: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class _ViewerLayout:
    """MuJoCo 画面和右侧面板布局。"""

    width: int
    height: int
    scene_width: int
    panel_left: int
    panel_width: int


class MujocoViewer:
    """轻量 MuJoCo 渲染窗口：保留鼠标相机，键盘只交给上层 teleop。"""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        key_callback: Callable[[int], None] | None = None,
        pose_joint_names: tuple[str, ...] = (),
        follow_body: str = "base_link",
        title: str = "SE3 sim2sim teleop",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        if not glfw.init():
            raise RuntimeError("无法初始化 GLFW，不能打开 MuJoCo viewer")
        self._model = model
        self._data = data
        self._key_callback = key_callback
        self._pose_editor = _PoseEditor(model, data, joint_names=pose_joint_names[:4])
        self._window = glfw.create_window(int(width), int(height), title, None, None)
        if self._window is None:
            glfw.terminate()
            raise RuntimeError("无法创建 MuJoCo viewer 窗口")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_pos)
        glfw.set_scroll_callback(self._window, self._on_scroll)

        self._camera = mujoco.MjvCamera()
        self._option = mujoco.MjvOption()
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self._configure_tracking_camera(model, follow_body)

        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._layout: _ViewerLayout | None = None
        self._closed = False

    def _configure_tracking_camera(self, model: mujoco.MjModel, follow_body: str) -> None:
        mujoco.mjv_defaultFreeCamera(model, self._camera)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, follow_body)
        if body_id < 0:
            return
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._camera.trackbodyid = int(body_id)
        self._camera.distance = 2.2
        self._camera.azimuth = 135.0
        self._camera.elevation = -18.0

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
        del step
        if self._closed or glfw.window_should_close(self._window):
            self._closed = True
            return

        glfw.make_context_current(self._window)
        viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
        layout = _make_layout(int(viewport_width), int(viewport_height))
        self._layout = layout
        self._pose_editor.set_output_enabled(float(telemetry.get("output_enabled", 1.0)) > 0.5)
        glfw.poll_events()
        if self._closed or glfw.window_should_close(self._window):
            self._closed = True
            return

        scene_viewport = mujoco.MjrRect(0, 0, layout.scene_width, layout.height)
        mujoco.mjv_updateScene(
            model,
            data,
            self._option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )
        mujoco.mjr_render(scene_viewport, self._scene, self._context)
        self._pose_editor.draw(self._context, layout)
        glfw.swap_buffers(self._window)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        glfw.destroy_window(self._window)
        glfw.terminate()

    def _on_key(self, window: object, key: int, scancode: int, action: int, mods: int) -> None:
        del window, scancode, mods
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if self._key_callback is not None:
            self._key_callback(int(key))

    def _on_mouse_button(self, window: object, button: int, action: int, mods: int) -> None:
        del window, mods
        cursor_x, cursor_y = glfw.get_cursor_pos(self._window)
        x_fb, y_fb_top = self._cursor_to_framebuffer(cursor_x, cursor_y)
        self._last_x = float(cursor_x)
        self._last_y = float(cursor_y)
        if (
            button == glfw.MOUSE_BUTTON_LEFT
            and action == glfw.PRESS
            and self._pose_editor.begin_drag(x_fb, y_fb_top, self._current_layout())
        ):
            self._button_left = False
            self._button_middle = False
            self._button_right = False
            return
        if (
            button == glfw.MOUSE_BUTTON_LEFT
            and action == glfw.RELEASE
            and self._pose_editor.end_drag()
        ):
            self._button_left = False
            self._button_middle = False
            self._button_right = False
            return
        if self._pose_editor.contains_point(x_fb, y_fb_top, self._current_layout()):
            self._button_left = False
            self._button_middle = False
            self._button_right = False
            return
        self._button_left = (
            glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_middle = (
            glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self._button_right = (
            glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

    def _on_cursor_pos(self, window: object, xpos: float, ypos: float) -> None:
        del window
        x_fb, y_fb_top = self._cursor_to_framebuffer(xpos, ypos)
        if self._pose_editor.drag(x_fb, y_fb_top, self._current_layout()):
            self._last_x = float(xpos)
            self._last_y = float(ypos)
            return
        if self._pose_editor.contains_point(x_fb, y_fb_top, self._current_layout()):
            self._last_x = float(xpos)
            self._last_y = float(ypos)
            return
        if not (self._button_left or self._button_middle or self._button_right):
            self._last_x = float(xpos)
            self._last_y = float(ypos)
            return

        width, height = glfw.get_window_size(self._window)
        height = max(1, int(height))
        dx = float(xpos) - self._last_x
        dy = float(ypos) - self._last_y
        self._last_x = float(xpos)
        self._last_y = float(ypos)

        shift = (
            glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(self._window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(
            self._model,
            action,
            dx / max(1, int(width)),
            dy / height,
            self._scene,
            self._camera,
        )

    def _on_scroll(self, window: object, x_offset: float, y_offset: float) -> None:
        del window, x_offset
        cursor_x, cursor_y = glfw.get_cursor_pos(self._window)
        x_fb, y_fb_top = self._cursor_to_framebuffer(cursor_x, cursor_y)
        if self._pose_editor.contains_point(x_fb, y_fb_top, self._current_layout()):
            return
        mujoco.mjv_moveCamera(
            self._model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * float(y_offset),
            self._scene,
            self._camera,
        )

    def _current_layout(self) -> _ViewerLayout:
        if self._layout is not None:
            return self._layout
        viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
        self._layout = _make_layout(int(viewport_width), int(viewport_height))
        return self._layout

    def _cursor_to_framebuffer(self, xpos: float, ypos: float) -> tuple[float, float]:
        win_width, win_height = glfw.get_window_size(self._window)
        fb_width, fb_height = glfw.get_framebuffer_size(self._window)
        scale_x = fb_width / max(1, int(win_width))
        scale_y = fb_height / max(1, int(win_height))
        return float(xpos) * scale_x, float(ypos) * scale_y


class _PoseEditor:
    """在 MuJoCo 窗口右侧提供仅 RC off 时可用的姿态编辑面板。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        joint_names: tuple[str, ...],
    ) -> None:
        self._model = model
        self._data = data
        self._controls = self._build_controls(tuple(joint_names))
        self._slider_rects: list[_SliderRect] = []
        self._dragging_key: str | None = None
        self._output_enabled = True

    def set_output_enabled(self, enabled: bool) -> None:
        self._output_enabled = bool(enabled)

    def draw(self, context: mujoco.MjrContext, layout: _ViewerLayout) -> None:
        if layout.panel_width < 80:
            return
        panel = mujoco.MjrRect(layout.panel_left, 0, layout.panel_width, layout.height)
        mujoco.mjr_rectangle(panel, 0.08, 0.085, 0.09, 0.96)
        mujoco.mjr_rectangle(
            mujoco.MjrRect(layout.panel_left, 0, 2, layout.height),
            0.28,
            0.31,
            0.34,
            1.0,
        )
        self._draw_text(
            context, layout, "Pose editor", layout.panel_left + 18, 24, (0.95, 0.95, 0.95)
        )
        status = "RC ON - locked" if self._output_enabled else "RC OFF - drag sliders"
        status_color = (0.95, 0.66, 0.32) if self._output_enabled else (0.42, 0.88, 0.62)
        self._draw_text(context, layout, status, layout.panel_left + 18, 52, status_color)

        self._slider_rects = []
        row_top = 94
        row_gap = 58
        for index, control in enumerate(self._controls):
            y_top = row_top + index * row_gap
            self._draw_control(context, layout, control, y_top)

    def contains_point(self, x_fb: float, y_fb_top: float, layout: _ViewerLayout) -> bool:
        del y_fb_top
        if layout.panel_width < 80:
            return False
        return float(x_fb) >= float(layout.panel_left)

    def begin_drag(self, x_fb: float, y_fb_top: float, layout: _ViewerLayout) -> bool:
        if not self.contains_point(x_fb, y_fb_top, layout):
            return False
        if self._output_enabled:
            return True
        control_key = self._control_key_at(x_fb, y_fb_top)
        if control_key is None:
            return True
        self._dragging_key = control_key
        self._set_control_from_x(control_key, x_fb)
        return True

    def drag(self, x_fb: float, y_fb_top: float, layout: _ViewerLayout) -> bool:
        del y_fb_top
        if self._dragging_key is None:
            return False
        if self._output_enabled:
            return True
        if not self.contains_point(x_fb, 0.0, layout):
            return True
        self._set_control_from_x(self._dragging_key, x_fb)
        return True

    def end_drag(self) -> bool:
        was_dragging = self._dragging_key is not None
        self._dragging_key = None
        return was_dragging

    def _build_controls(self, joint_names: tuple[str, ...]) -> list[_PoseControl]:
        controls: list[_PoseControl] = []
        for index, joint_name in enumerate(joint_names[:4]):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            lower, upper = self._joint_limits(jid)
            label = joint_name.removesuffix("_Joint")
            controls.append(
                _PoseControl(
                    key=f"joint:{index}",
                    label=label,
                    lower=lower,
                    upper=upper,
                    kind="joint",
                    index=int(jid),
                )
            )
        controls.extend(
            (
                _PoseControl(
                    key="base:roll",
                    label="base roll",
                    lower=-_BASE_ATTITUDE_LIMIT_RAD,
                    upper=_BASE_ATTITUDE_LIMIT_RAD,
                    kind="base",
                    index=0,
                ),
                _PoseControl(
                    key="base:pitch",
                    label="base pitch",
                    lower=-_BASE_ATTITUDE_LIMIT_RAD,
                    upper=_BASE_ATTITUDE_LIMIT_RAD,
                    kind="base",
                    index=1,
                ),
                _PoseControl(
                    key="base:yaw",
                    label="base yaw",
                    lower=-_BASE_YAW_LIMIT_RAD,
                    upper=_BASE_YAW_LIMIT_RAD,
                    kind="base",
                    index=2,
                ),
            )
        )
        return controls

    def _joint_limits(self, joint_id: int) -> tuple[float, float]:
        if bool(self._model.jnt_limited[joint_id]):
            lower, upper = self._model.jnt_range[joint_id]
            if math.isfinite(float(lower)) and math.isfinite(float(upper)) and lower < upper:
                return float(lower), float(upper)
        current = float(self._data.qpos[self._model.jnt_qposadr[joint_id]])
        return (
            min(-_JOINT_FALLBACK_LIMIT_RAD, current - 1.0),
            max(_JOINT_FALLBACK_LIMIT_RAD, current + 1.0),
        )

    def _draw_control(
        self,
        context: mujoco.MjrContext,
        layout: _ViewerLayout,
        control: _PoseControl,
        y_top: int,
    ) -> None:
        value = self._control_value(control)
        value = _clip(value, control.lower, control.upper)
        slider_x = layout.panel_left + 20
        slider_y = y_top + 28
        slider_width = layout.panel_width - 40
        slider_height = 10
        label = f"{control.label}: {math.degrees(value):+.1f} deg"
        self._draw_text(context, layout, label, slider_x, y_top, (0.82, 0.86, 0.90))

        track_color = (0.24, 0.26, 0.28, 1.0)
        fill_color = (0.26, 0.55, 0.86, 0.95)
        knob_color = (0.90, 0.92, 0.94, 1.0)
        if self._output_enabled:
            fill_color = (0.42, 0.42, 0.42, 0.86)
            knob_color = (0.62, 0.62, 0.62, 0.92)

        progress = (value - control.lower) / max(1.0e-9, control.upper - control.lower)
        fill_width = max(0, int(slider_width * progress))
        knob_x = int(slider_x + slider_width * progress)
        self._rect(layout, slider_x, slider_y, slider_width, slider_height, track_color)
        self._rect(layout, slider_x, slider_y, fill_width, slider_height, fill_color)
        self._rect(layout, knob_x - 5, slider_y - 5, 10, slider_height + 10, knob_color)
        self._slider_rects.append(
            _SliderRect(
                key=control.key,
                x=float(slider_x),
                y=float(slider_y - 12),
                width=float(slider_width),
                height=float(slider_height + 24),
            )
        )

    def _control_key_at(self, x_fb: float, y_fb_top: float) -> str | None:
        for rect in self._slider_rects:
            if (
                rect.x <= float(x_fb) <= rect.x + rect.width
                and rect.y <= float(y_fb_top) <= rect.y + rect.height
            ):
                return rect.key
        return None

    def _set_control_from_x(self, key: str, x_fb: float) -> None:
        control = self._control_by_key(key)
        rect = next((item for item in self._slider_rects if item.key == key), None)
        if control is None or rect is None:
            return
        alpha = _clip((float(x_fb) - rect.x) / max(1.0, rect.width), 0.0, 1.0)
        value = control.lower + alpha * (control.upper - control.lower)
        self._set_control_value(control, value)

    def _control_by_key(self, key: str) -> _PoseControl | None:
        return next((control for control in self._controls if control.key == key), None)

    def _control_value(self, control: _PoseControl) -> float:
        if control.kind == "joint":
            qpos_id = int(self._model.jnt_qposadr[control.index])
            return float(self._data.qpos[qpos_id])
        euler = self._base_euler_xyz()
        return float(euler[control.index])

    def _set_control_value(self, control: _PoseControl, value: float) -> None:
        value = _clip(value, control.lower, control.upper)
        if control.kind == "joint":
            qpos_id = int(self._model.jnt_qposadr[control.index])
            qvel_id = int(self._model.jnt_dofadr[control.index])
            self._data.qpos[qpos_id] = value
            self._data.qvel[qvel_id] = 0.0
        else:
            euler = self._base_euler_xyz()
            euler[control.index] = value
            self._data.qpos[3:7] = euler_xyz_to_quat_wxyz(
                float(euler[0]), float(euler[1]), float(euler[2])
            )
            self._data.qvel[3:6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def _base_euler_xyz(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        quat_xyzw = quat_wxyz_to_xyzw(np.asarray(self._data.qpos[3:7], dtype=np.float64))
        return Rotation.from_quat(quat_xyzw).as_euler("xyz")

    def _draw_text(
        self,
        context: mujoco.MjrContext,
        layout: _ViewerLayout,
        text: str,
        x_px: float,
        y_top_px: float,
        color: tuple[float, float, float],
    ) -> None:
        x_rel = _clip(float(x_px) / max(1.0, float(layout.width)), 0.0, 1.0)
        y_rel = _clip(1.0 - float(y_top_px) / max(1.0, float(layout.height)), 0.0, 1.0)
        mujoco.mjr_text(
            mujoco.mjtFont.mjFONT_NORMAL,
            text,
            context,
            x_rel,
            y_rel,
            float(color[0]),
            float(color[1]),
            float(color[2]),
        )

    def _rect(
        self,
        layout: _ViewerLayout,
        x_top: float,
        y_top: float,
        width: float,
        height: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        rect = mujoco.MjrRect(
            round(x_top),
            round(layout.height - y_top - height),
            max(0, round(width)),
            max(0, round(height)),
        )
        mujoco.mjr_rectangle(rect, float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


def _make_layout(width: int, height: int) -> _ViewerLayout:
    panel_width = min(_PANEL_MAX_WIDTH, max(_PANEL_MIN_WIDTH, int(width * 0.28)))
    panel_width = min(panel_width, max(0, int(width) - 320))
    scene_width = max(1, int(width) - panel_width)
    return _ViewerLayout(
        width=int(width),
        height=int(height),
        scene_width=scene_width,
        panel_left=scene_width,
        panel_width=panel_width,
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


class CompositeViewer:
    """同时驱动多个 viewer sink。"""

    def __init__(self, viewers: list[object]) -> None:
        self._viewers = viewers

    def log_model(self, model: mujoco.MjModel) -> None:
        for viewer in self._viewers:
            viewer.log_model(model)

    def log_state(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        step: int,
        telemetry: dict[str, object],
    ) -> None:
        for viewer in self._viewers:
            viewer.log_state(model, data, step=step, telemetry=telemetry)

    def close(self) -> None:
        for viewer in reversed(self._viewers):
            viewer.close()
