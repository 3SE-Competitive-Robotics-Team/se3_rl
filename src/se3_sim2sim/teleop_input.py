"""键盘 teleop 输入源，用于交互式 sim2sim command 验证。"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from threading import Lock
from types import TracebackType
from typing import Protocol, Self


@dataclass(frozen=True)
class CommandInputUpdate:
    """单个控制步读取到的人工 command 更新。"""

    lin_vel_x: float
    yaw_rate: float
    command_height: float
    toggle_output: bool = False
    quit_requested: bool = False
    key_events: tuple[str, ...] = ()


class CommandInputSource(Protocol):
    """workflow 可消费的 command 输入源协议。"""

    def pace(self, sim_time_s: float) -> None:
        """按仿真时间节拍阻塞，保证人能实时输入。"""

    def poll(self, sim_time_s: float) -> CommandInputUpdate:
        """读取当前控制步的 command。"""


class KeyboardTeleopSource:
    """无第三方依赖的非阻塞键盘遥控输入。"""

    def __init__(
        self,
        *,
        lin_vel_x: float = 0.0,
        yaw_rate: float = 0.0,
        command_height: float = 0.22,
        default_command_height: float = 0.22,
        command_lin_vel_x: float = 0.5,
        command_yaw_rate: float = 0.8,
        command_height_rate: float = 0.12,
        min_command_height: float = 0.195,
        max_command_height: float = 0.390,
        hold_s: float = 0.25,
        realtime: bool = True,
    ) -> None:
        self.command_lin_vel_x = _finite_positive(command_lin_vel_x, "command_lin_vel_x")
        self.command_yaw_rate = _finite_positive(command_yaw_rate, "command_yaw_rate")
        self.command_height_rate = _finite_positive(command_height_rate, "command_height_rate")
        self.min_command_height = _finite(min_command_height, "min_command_height")
        self.max_command_height = _finite(max_command_height, "max_command_height")
        if self.min_command_height >= self.max_command_height:
            raise ValueError("min_command_height must be lower than max_command_height")
        self.default_command_height = self._clamp_height(
            _finite(default_command_height, "default_command_height")
        )
        self.hold_s = _finite_positive(hold_s, "hold_s")
        self.realtime = bool(realtime)
        self._lin_vel_x = _finite(lin_vel_x, "lin_vel_x")
        self._yaw_rate = _finite(yaw_rate, "yaw_rate")
        self._command_height = self._clamp_height(_finite(command_height, "command_height"))
        self._last_vx_key_at = -math.inf
        self._last_yaw_key_at = -math.inf
        self._last_height_key_at = -math.inf
        self._last_height_update_at = time.monotonic()
        self._height_direction = 0.0
        self._interactive = False
        self._wall_start_s: float | None = None
        self._old_terminal_attrs: list[object] | None = None
        self._termios: object | None = None
        self._msvcrt: object | None = None
        self._queued_keys: list[str] = []
        self._queued_keys_lock = Lock()

    def __enter__(self) -> Self:
        self._wall_start_s = time.monotonic()
        self._interactive = bool(sys.stdin.isatty())
        if not self._interactive:
            return self
        if os.name == "nt":
            import msvcrt

            self._msvcrt = msvcrt
            return self

        import termios
        import tty

        self._termios = termios
        fd = sys.stdin.fileno()
        self._old_terminal_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._old_terminal_attrs is None or self._termios is None:
            return
        self._termios.tcsetattr(
            sys.stdin.fileno(),
            self._termios.TCSADRAIN,
            self._old_terminal_attrs,
        )
        self._old_terminal_attrs = None

    @property
    def interactive(self) -> bool:
        """当前 stdin 是否支持直接读键。"""

        return self._interactive

    def help_text(self) -> str:
        """返回终端提示文本。"""

        return (
            "[teleop] R: 切换遥控器输出 | W/S: 前进/后退 | A/D: 左/右旋转 | "
            "按住 ↑/E: 连续站高 | 按住 ↓/C: 连续站低 | H: 默认站高 | "
            "右侧面板: RC off 时可拖动四关节和 base 姿态 | Space/X: 清零速度 | Q/Esc: 退出"
        )

    def key_callback(self, keycode: int) -> None:
        """接收 MuJoCo viewer 的按键回调。"""

        if keycode == 256:
            self.queue_key("\x1b")
            return
        if keycode == 265:
            self.queue_key("up")
            return
        if keycode == 264:
            self.queue_key("down")
            return
        if 0 <= int(keycode) < 256:
            self.queue_key(chr(int(keycode)))

    def queue_key(self, key: str) -> None:
        """把外部 viewer 收到的按键加入下一控制步处理队列。"""

        if not key:
            return
        with self._queued_keys_lock:
            self._queued_keys.append(key.lower())

    def pace(self, sim_time_s: float) -> None:
        if not self.realtime or self._wall_start_s is None:
            return
        target_s = self._wall_start_s + max(0.0, float(sim_time_s))
        delay_s = target_s - time.monotonic()
        if delay_s > 0.0:
            time.sleep(min(delay_s, 0.05))

    def poll(self, sim_time_s: float) -> CommandInputUpdate:
        del sim_time_s
        now_s = time.monotonic()
        keys = tuple(self._read_pending_keys())
        toggle_output = False
        quit_requested = False

        for key in keys:
            if key == "r":
                toggle_output = True
            elif key == "w":
                self._lin_vel_x = self.command_lin_vel_x
                self._last_vx_key_at = now_s
            elif key == "s":
                self._lin_vel_x = -self.command_lin_vel_x
                self._last_vx_key_at = now_s
            elif key == "a":
                self._yaw_rate = self.command_yaw_rate
                self._last_yaw_key_at = now_s
            elif key == "d":
                self._yaw_rate = -self.command_yaw_rate
                self._last_yaw_key_at = now_s
            elif key in {"up", "e"}:
                self._height_direction = 1.0
                self._last_height_key_at = now_s
            elif key in {"down", "c"}:
                self._height_direction = -1.0
                self._last_height_key_at = now_s
            elif key == "h":
                self._command_height = self.default_command_height
                self._height_direction = 0.0
                self._last_height_key_at = -math.inf
            elif key in {" ", "x"}:
                self._lin_vel_x = 0.0
                self._yaw_rate = 0.0
                self._last_vx_key_at = -math.inf
                self._last_yaw_key_at = -math.inf
            elif key in {"q", "\x1b", "\x03"}:
                quit_requested = True

        if now_s - self._last_vx_key_at > self.hold_s:
            self._lin_vel_x = 0.0
        if now_s - self._last_yaw_key_at > self.hold_s:
            self._yaw_rate = 0.0
        if now_s - self._last_height_key_at > self.hold_s:
            self._height_direction = 0.0
        if self._height_direction != 0.0:
            dt = max(0.0, now_s - self._last_height_update_at)
            self._command_height = self._clamp_height(
                self._command_height + self._height_direction * self.command_height_rate * dt
            )
        self._last_height_update_at = now_s

        return CommandInputUpdate(
            lin_vel_x=self._lin_vel_x,
            yaw_rate=self._yaw_rate,
            command_height=self._command_height,
            toggle_output=toggle_output,
            quit_requested=quit_requested,
            key_events=keys,
        )

    def _read_pending_keys(self) -> list[str]:
        keys = self._drain_queued_keys()
        if not self._interactive:
            return keys
        if os.name == "nt":
            keys.extend(self._read_windows_keys())
        else:
            keys.extend(self._read_posix_keys())
        return keys

    def _drain_queued_keys(self) -> list[str]:
        with self._queued_keys_lock:
            keys = list(self._queued_keys)
            self._queued_keys.clear()
        return keys

    def _read_windows_keys(self) -> list[str]:
        if self._msvcrt is None:
            return []
        keys: list[str] = []
        while self._msvcrt.kbhit():
            key = self._msvcrt.getwch()
            if key in {"\x00", "\xe0"}:
                if self._msvcrt.kbhit():
                    ext_key = self._msvcrt.getwch()
                    if ext_key == "H":
                        keys.append("up")
                    elif ext_key == "P":
                        keys.append("down")
                continue
            keys.append(key.lower())
        return keys

    @staticmethod
    def _read_posix_keys() -> list[str]:
        import select

        keys: list[str] = []
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return keys
            key = sys.stdin.read(1)
            if not key:
                return keys
            if key == "\x1b":
                readable, _, _ = select.select([sys.stdin], [], [], 0.0)
                if readable and sys.stdin.read(1) == "[":
                    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if readable:
                        arrow = sys.stdin.read(1)
                        if arrow == "A":
                            keys.append("up")
                            continue
                        if arrow == "B":
                            keys.append("down")
                            continue
            keys.append(key.lower())

    def _clamp_height(self, value: float) -> float:
        return float(np_clip(value, self.min_command_height, self.max_command_height))


def _finite(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def _finite_positive(value: float, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def np_clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))
