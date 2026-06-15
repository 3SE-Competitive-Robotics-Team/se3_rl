"""无 MuJoCo 默认键盘快捷键的原生渲染窗口。"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import glfw
import mujoco
import numpy as np

from se3_shared import JointGroup, RobotConfig
from se3_shared.fourbar import output_knee_from_active_angle_np

from .closed_chain import ClosedChainClosureSolver
from .math_utils import euler_xyz_to_quat_wxyz, quat_wxyz_to_xyzw

_PANEL_MIN_WIDTH = 390
_PANEL_MAX_WIDTH = 460
_GROUND_GEOM_GROUP = 2
_JOINT_FALLBACK_LIMIT_RAD = 2.2
_BASE_ATTITUDE_LIMIT_RAD = math.pi
_BASE_YAW_LIMIT_RAD = math.pi
_ROBOT_CFG = RobotConfig()


@dataclass(frozen=True)
class _PoseControl:
    """右侧面板中的一个姿态滑条。"""

    key: str
    label: str
    lower: float
    upper: float
    kind: str
    index: int
    other_index: int | None = None
    side_index: int | None = None
    front_coef: float = 0.0
    back_coef: float = 0.0


@dataclass(frozen=True)
class _SliderRect:
    """滑条热区，坐标为 framebuffer 顶部原点。"""

    key: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class _TendonJointLimit:
    """由 limited fixed tendon 推导出的单关节动态限位。"""

    tendon_id: int
    joint_id: int
    joint_coef: float
    other_joint_id: int
    other_coef: float
    lower: float
    upper: float


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
        key_callback: Callable[..., None] | None = None,
        pose_joint_names: tuple[str, ...] = (),
        follow_body: str = "base_link",
        geom_view: str = "visual",
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
        self._geom_view = str(geom_view)
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
        self._configure_geom_groups()
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

    def _configure_geom_groups(self) -> None:
        if self._geom_view == "visual":
            self._option.geomgroup[:] = 0
            self._option.geomgroup[1] = 1
        elif self._geom_view == "collision":
            self._option.geomgroup[:] = 0
            self._option.geomgroup[0] = 1
        elif self._geom_view == "both":
            self._option.geomgroup[:] = 1
        else:
            raise ValueError(f"未知几何显示模式: {self._geom_view}")
        self._option.geomgroup[_GROUND_GEOM_GROUP] = 1

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
        self._configure_geom_groups()
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

    @property
    def closed(self) -> bool:
        """窗口是否已经关闭。"""

        return bool(self._closed)

    def _on_key(self, window: object, key: int, scancode: int, action: int, mods: int) -> None:
        del window, scancode, mods
        if action not in (glfw.PRESS, glfw.REPEAT, glfw.RELEASE):
            return
        if self._key_callback is not None:
            self._key_callback(int(key), int(action))

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
        self._closure_solver = ClosedChainClosureSolver.try_create(model=model, data=data)
        self._tendon_joint_limits = self._build_tendon_joint_limits()
        self._reference_values = self._build_reference_values()
        self._slider_rects: list[_SliderRect] = []
        self._dragging_key: str | None = None
        self._base_drag_euler_xyz: np.ndarray | None = None
        self._output_enabled = True

    def set_output_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._output_enabled:
            self._base_drag_euler_xyz = None
        self._output_enabled = enabled

    def draw(self, context: mujoco.MjrContext, layout: _ViewerLayout) -> None:
        if layout.panel_width < 80:
            return
        panel = mujoco.MjrRect(layout.panel_left, 0, layout.panel_width, layout.height)
        mujoco.mjr_rectangle(panel, 0.048, 0.054, 0.060, 0.98)
        mujoco.mjr_rectangle(
            mujoco.MjrRect(layout.panel_left, 0, 2, layout.height),
            0.20,
            0.23,
            0.25,
            1.0,
        )

        pad_x = 24
        left = layout.panel_left + pad_x
        right = layout.panel_left + layout.panel_width - pad_x
        self._draw_header(context, layout, left, right)

        self._slider_rects = []
        control_count = max(1, len(self._controls))
        header_y = 96
        section_gap = 24
        group_gap = 10
        bottom_pad = 18
        available_rows = layout.height - header_y - section_gap * 2 - group_gap - bottom_pad
        row_gap = int(_clip(available_rows / control_count, 44, 54))

        y_top = header_y
        y_top = self._draw_section(context, layout, "LEGS", y_top, left, right)
        for control in self._controls:
            if control.kind == "base":
                continue
            self._draw_control(context, layout, control, y_top, left, right)
            y_top += row_gap

        y_top += group_gap
        y_top = self._draw_section(context, layout, "BASE", y_top, left, right)
        for control in self._controls:
            if control.kind != "base":
                continue
            self._draw_control(context, layout, control, y_top, left, right)
            y_top += row_gap

    def _draw_header(
        self,
        context: mujoco.MjrContext,
        layout: _ViewerLayout,
        left: int,
        right: int,
    ) -> None:
        """绘制面板标题和 RC 状态。"""

        status_text = "RC ON" if self._output_enabled else "RC OFF"
        status_bg = (0.30, 0.25, 0.14, 1.0) if self._output_enabled else (0.08, 0.28, 0.20, 1.0)
        status_fg = (0.96, 0.70, 0.36) if self._output_enabled else (0.42, 0.92, 0.58)
        hint = "LOCKED" if self._output_enabled else "EDIT"

        self._rect(layout, left, 14, right - left, 62, (0.070, 0.078, 0.086, 1.0))
        self._rect(layout, left, 75, right - left, 1, (0.18, 0.21, 0.23, 1.0))
        self._draw_text(context, layout, "Pose Editor", left + 12, 28, (0.88, 0.93, 0.96))
        self._draw_text(context, layout, hint, left + 12, 52, (0.52, 0.60, 0.66))
        self._rect(layout, right - 92, 31, 76, 24, status_bg)
        self._draw_text(context, layout, status_text, right - 82, 36, status_fg)

    def _draw_section(
        self,
        context: mujoco.MjrContext,
        layout: _ViewerLayout,
        label: str,
        y_top: int,
        left: int,
        right: int,
    ) -> int:
        """绘制右侧面板分组标题。"""

        self._rect(layout, left, y_top, right - left, 19, (0.082, 0.095, 0.106, 1.0))
        self._draw_text(context, layout, label, left + 8, y_top + 3, (0.50, 0.58, 0.64))
        self._rect(
            layout,
            left,
            y_top + 20,
            right - left,
            1,
            (0.17, 0.20, 0.22, 1.0),
        )
        return y_top + 24

    def contains_point(self, x_fb: float, y_fb_top: float, layout: _ViewerLayout) -> bool:
        if layout.panel_width < 80:
            return False
        return self._control_key_at(x_fb, y_fb_top) is not None

    def begin_drag(self, x_fb: float, y_fb_top: float, layout: _ViewerLayout) -> bool:
        if not self.contains_point(x_fb, y_fb_top, layout):
            return False
        if self._output_enabled:
            return True
        control_key = self._control_key_at(x_fb, y_fb_top)
        if control_key is None:
            return False
        self._dragging_key = control_key
        control = self._control_by_key(control_key)
        if control is not None and control.kind == "base" and self._base_drag_euler_xyz is None:
            # RC off 编辑期间固定其它 Euler 轴，避免 quaternion 反解在奇异点附近跳变。
            self._base_drag_euler_xyz = self._base_euler_xyz()
        self._set_control_from_x(control_key, x_fb)
        return True

    def drag(self, x_fb: float, y_fb_top: float, layout: _ViewerLayout) -> bool:
        del y_fb_top, layout
        if self._dragging_key is None:
            return False
        if self._output_enabled:
            return True
        self._set_control_from_x(self._dragging_key, x_fb)
        return True

    def end_drag(self) -> bool:
        was_dragging = self._dragging_key is not None
        self._dragging_key = None
        return was_dragging

    def _build_controls(self, joint_names: tuple[str, ...]) -> list[_PoseControl]:
        controls: list[_PoseControl] = []
        policy_leg_names = JointGroup.POLICY_LEG_NAMES
        joint_ids = {
            name: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names[:4]
        }
        if all(joint_ids.get(name, -1) >= 0 for name in policy_leg_names):
            active_lower, active_upper = _ROBOT_CFG.active_rod_angle_limits
            side_specs = (
                (
                    "left",
                    "lf0",
                    "l_active_rod_angle",
                    policy_leg_names[0],
                    policy_leg_names[1],
                    _ROBOT_CFG.active_rod_angle_coeffs[0],
                ),
                (
                    "right",
                    "rf0",
                    "r_active_rod_angle",
                    policy_leg_names[2],
                    policy_leg_names[3],
                    _ROBOT_CFG.active_rod_angle_coeffs[1],
                ),
            )
            for side_index, (
                _side,
                front_label,
                active_label,
                front_name,
                back_name,
                coeffs,
            ) in enumerate(side_specs):
                front_id = int(joint_ids[front_name])
                back_id = int(joint_ids[back_name])
                front_coef, back_coef = (float(coeffs[0]), float(coeffs[1]))
                front_lower, front_upper = self._front_joint_limits(front_id)
                controls.append(
                    _PoseControl(
                        key=f"joint:{front_name}",
                        label=front_label,
                        lower=front_lower,
                        upper=front_upper,
                        kind="joint",
                        index=front_id,
                        other_index=back_id,
                        side_index=side_index,
                        front_coef=front_coef,
                        back_coef=back_coef,
                    )
                )
                controls.append(
                    _PoseControl(
                        key=f"active:{side_index}",
                        label=active_label,
                        lower=float(active_lower),
                        upper=float(active_upper),
                        kind="active_rod",
                        index=front_id,
                        other_index=back_id,
                        side_index=side_index,
                        front_coef=front_coef,
                        back_coef=back_coef,
                    )
                )
        else:
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

    def _front_joint_limits(self, joint_id: int) -> tuple[float, float]:
        if bool(self._model.jnt_limited[joint_id]):
            return self._joint_limits(joint_id)
        return -math.pi, math.pi

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

    def _build_tendon_joint_limits(self) -> dict[int, tuple[_TendonJointLimit, ...]]:
        limits: dict[int, list[_TendonJointLimit]] = {}
        joint_wrap_type = int(mujoco.mjtWrap.mjWRAP_JOINT)
        for tendon_id in range(self._model.ntendon):
            if not bool(self._model.tendon_limited[tendon_id]):
                continue
            adr = int(self._model.tendon_adr[tendon_id])
            num = int(self._model.tendon_num[tendon_id])
            entries: list[tuple[int, float]] = []
            for wrap_id in range(adr, adr + num):
                if int(self._model.wrap_type[wrap_id]) != joint_wrap_type:
                    continue
                entries.append(
                    (
                        int(self._model.wrap_objid[wrap_id]),
                        float(self._model.wrap_prm[wrap_id]),
                    )
                )
            if len(entries) != 2:
                continue
            (joint_a, coef_a), (joint_b, coef_b) = entries
            lower, upper = (float(v) for v in self._model.tendon_range[tendon_id])
            if not (
                math.isfinite(lower)
                and math.isfinite(upper)
                and lower < upper
                and abs(coef_a) > 1.0e-9
                and abs(coef_b) > 1.0e-9
            ):
                continue
            limits.setdefault(joint_a, []).append(
                _TendonJointLimit(tendon_id, joint_a, coef_a, joint_b, coef_b, lower, upper)
            )
            limits.setdefault(joint_b, []).append(
                _TendonJointLimit(tendon_id, joint_b, coef_b, joint_a, coef_a, lower, upper)
            )
        return {joint_id: tuple(items) for joint_id, items in limits.items()}

    def _control_limits(self, control: _PoseControl) -> tuple[float, float]:
        lower = float(control.lower)
        upper = float(control.upper)
        if control.kind != "joint" or control.other_index is not None:
            return lower, upper
        for tendon_limit in self._tendon_joint_limits.get(control.index, ()):
            other_qpos = int(self._model.jnt_qposadr[tendon_limit.other_joint_id])
            other_value = float(self._data.qpos[other_qpos])
            offset = tendon_limit.other_coef * other_value
            if tendon_limit.joint_coef > 0.0:
                t_lower = (tendon_limit.lower - offset) / tendon_limit.joint_coef
                t_upper = (tendon_limit.upper - offset) / tendon_limit.joint_coef
            else:
                t_lower = (tendon_limit.upper - offset) / tendon_limit.joint_coef
                t_upper = (tendon_limit.lower - offset) / tendon_limit.joint_coef
            lower = max(lower, float(t_lower))
            upper = min(upper, float(t_upper))
        if lower < upper:
            return lower, upper
        current = float(self._data.qpos[self._model.jnt_qposadr[control.index]])
        return current, current

    def _build_reference_values(self) -> dict[str, float]:
        """记录滑条参考刻度：关节用初始角度，机身姿态用零位。"""

        values: dict[str, float] = {}
        for control in self._controls:
            if control.kind == "joint":
                qpos_id = int(self._model.jnt_qposadr[control.index])
                values[control.key] = float(self._data.qpos[qpos_id])
            elif control.kind == "active_rod":
                values[control.key] = self._active_rod_value(control)
            else:
                values[control.key] = 0.0
        return values

    def _draw_control(
        self,
        context: mujoco.MjrContext,
        layout: _ViewerLayout,
        control: _PoseControl,
        y_top: int,
        left: int,
        right: int,
    ) -> None:
        value = self._control_value(control)
        lower, upper = self._control_limits(control)
        value = _clip(value, lower, upper)
        value_width = 100
        slider_x = left
        slider_y = y_top + 26
        slider_width = right - left - value_width - 16
        slider_height = 6
        label = self._control_display_label(control)
        value_label = f"{math.degrees(value):+6.1f} deg"
        span = max(1.0e-9, upper - lower)
        progress = (value - lower) / span
        near_limit = min(value - lower, upper - value) <= 0.025 * span

        row_width = right - left
        row_bg = (0.060, 0.068, 0.075, 0.84)
        if self._dragging_key == control.key:
            row_bg = (0.076, 0.095, 0.108, 0.96)
        self._rect(layout, left, y_top - 3, row_width, 40, row_bg)
        self._draw_text(context, layout, label, left + 10, y_top + 1, (0.82, 0.87, 0.90))
        self._rect(
            layout, right - value_width, y_top + 4, value_width, 24, (0.050, 0.057, 0.063, 1.0)
        )
        self._draw_text(
            context, layout, value_label, right - value_width + 10, y_top + 9, (0.72, 0.78, 0.82)
        )

        track_color = (0.15, 0.17, 0.18, 1.0)
        fill_color = (0.28, 0.48, 0.62, 0.98)
        knob_color = (0.84, 0.91, 0.94, 1.0)
        tick_color = (0.36, 0.40, 0.43, 0.90)
        if self._output_enabled:
            fill_color = (0.27, 0.30, 0.32, 0.96)
            knob_color = (0.58, 0.61, 0.63, 0.92)
            tick_color = (0.27, 0.30, 0.32, 0.90)
        elif near_limit:
            knob_color = (0.98, 0.68, 0.32, 1.0)
            fill_color = (0.76, 0.44, 0.18, 0.98)

        knob_x = int(slider_x + slider_width * progress)
        default_x = self._reference_x(control, lower, upper, slider_x, slider_width)
        self._rect(layout, slider_x, slider_y, slider_width, slider_height, track_color)
        self._rect(layout, slider_x, slider_y + 2, slider_width, 1, tick_color)
        if default_x is not None:
            self._rect(
                layout, default_x - 1, slider_y - 4, 2, slider_height + 8, (0.45, 0.51, 0.55, 0.95)
            )
        self._rect(layout, max(slider_x, knob_x - 1), slider_y, 2, slider_height, fill_color)
        self._rect(layout, knob_x - 5, slider_y - 7, 10, slider_height + 14, knob_color)
        self._slider_rects.append(
            _SliderRect(
                key=control.key,
                x=float(slider_x),
                y=float(slider_y - 16),
                width=float(slider_width),
                height=float(slider_height + 32),
            )
        )

    def _reference_x(
        self,
        control: _PoseControl,
        lower: float,
        upper: float,
        slider_x: int,
        slider_width: int,
    ) -> int | None:
        """返回参考刻度在滑条上的像素位置。"""

        reference = self._reference_values.get(control.key)
        if reference is None:
            return None
        if not (lower <= reference <= upper):
            return None
        alpha = (reference - lower) / max(1.0e-9, upper - lower)
        return int(slider_x + slider_width * alpha)

    def _control_display_label(self, control: _PoseControl) -> str:
        """把 MJCF 关节名压缩成面板可读标签。"""

        labels = {
            "lf0": "LF swing",
            "l_active_rod_angle": "LF bend",
            "rf0": "RF swing",
            "r_active_rod_angle": "RF bend",
            "l_drive_bar": "LF drive",
            "r_drive_bar": "RF drive",
            "base roll": "Roll",
            "base pitch": "Pitch",
            "base yaw": "Yaw",
        }
        return labels.get(control.label, control.label)

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
        lower, upper = self._control_limits(control)
        alpha = _clip((float(x_fb) - rect.x) / max(1.0, rect.width), 0.0, 1.0)
        value = lower + alpha * (upper - lower)
        self._set_control_value(control, value)

    def _control_by_key(self, key: str) -> _PoseControl | None:
        return next((control for control in self._controls if control.key == key), None)

    def _control_value(self, control: _PoseControl) -> float:
        if control.kind == "joint":
            qpos_id = int(self._model.jnt_qposadr[control.index])
            return float(self._data.qpos[qpos_id])
        if control.kind == "active_rod":
            return self._active_rod_value(control)
        if self._base_drag_euler_xyz is not None:
            return float(self._base_drag_euler_xyz[control.index])
        euler = self._base_euler_xyz()
        return float(euler[control.index])

    def _set_control_value(self, control: _PoseControl, value: float) -> None:
        lower, upper = self._control_limits(control)
        value = _clip(value, lower, upper)
        if control.kind == "joint":
            qpos_id = int(self._model.jnt_qposadr[control.index])
            qvel_id = int(self._model.jnt_dofadr[control.index])
            if control.other_index is None:
                self._data.qpos[qpos_id] = value
                self._data.qvel[qvel_id] = 0.0
            else:
                active_angle = self._active_rod_value(control)
                self._write_active_rod_pair(control, front_value=value, active_angle=active_angle)
            self._solve_closed_chain(control)
        elif control.kind == "active_rod":
            front_value = float(self._data.qpos[self._model.jnt_qposadr[control.index]])
            self._write_active_rod_pair(control, front_value=front_value, active_angle=value)
            self._solve_closed_chain(control)
        else:
            if self._dragging_key == control.key and self._base_drag_euler_xyz is not None:
                euler = np.array(self._base_drag_euler_xyz, dtype=np.float64, copy=True)
            else:
                euler = self._base_euler_xyz()
            euler[control.index] = value
            if self._dragging_key == control.key:
                self._base_drag_euler_xyz = euler
            self._data.qpos[3:7] = euler_xyz_to_quat_wxyz(
                float(euler[0]), float(euler[1]), float(euler[2])
            )
            self._data.qvel[3:6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def _active_rod_value(self, control: _PoseControl) -> float:
        """读取同侧主动杆夹角。"""

        if control.other_index is None:
            return 0.0
        front_qpos = int(self._model.jnt_qposadr[control.index])
        back_qpos = int(self._model.jnt_qposadr[control.other_index])
        return float(
            control.front_coef * self._data.qpos[front_qpos]
            + control.back_coef * self._data.qpos[back_qpos]
        )

    def _write_active_rod_pair(
        self,
        control: _PoseControl,
        *,
        front_value: float,
        active_angle: float,
    ) -> None:
        """按 policy 语义写入同侧 front angle 和 active angle。"""

        if control.other_index is None or abs(control.back_coef) <= 1.0e-9:
            return
        active_lower, active_upper = _ROBOT_CFG.active_rod_angle_limits
        active_angle = _clip(active_angle, float(active_lower), float(active_upper))
        back_value = (active_angle - control.front_coef * float(front_value)) / control.back_coef

        front_qpos = int(self._model.jnt_qposadr[control.index])
        front_qvel = int(self._model.jnt_dofadr[control.index])
        back_qpos = int(self._model.jnt_qposadr[control.other_index])
        back_qvel = int(self._model.jnt_dofadr[control.other_index])
        self._data.qpos[front_qpos] = float(front_value)
        self._data.qpos[back_qpos] = float(back_value)
        self._data.qvel[front_qvel] = 0.0
        self._data.qvel[back_qvel] = 0.0
        self._seed_closed_chain_branch(control, active_angle)

    def _seed_closed_chain_branch(self, control: _PoseControl, active_angle: float) -> None:
        """用解析四连杆膝角把闭链保持在当前装配分支。"""

        if self._closure_solver is None or control.side_index is None:
            return
        knee_angle = output_knee_from_active_angle_np(float(active_angle))
        if int(control.side_index) == 1:
            knee_angle = -knee_angle
        self._closure_solver.seed_passive_position(int(control.side_index), float(knee_angle))

    def _solve_closed_chain(self, control: _PoseControl) -> None:
        """写入腿部姿态后投影闭链被动关节。"""

        if control.side_index is None or self._closure_solver is None:
            return
        self._closure_solver.solve_positions()
        self._closure_solver.solve_velocities()

    def _base_euler_xyz(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        quat_xyzw = quat_wxyz_to_xyzw(np.asarray(self._data.qpos[3:7], dtype=np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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

    @property
    def closed(self) -> bool:
        """任一子 viewer 关闭时，组合 viewer 视为关闭。"""

        return any(bool(getattr(viewer, "closed", False)) for viewer in self._viewers)
