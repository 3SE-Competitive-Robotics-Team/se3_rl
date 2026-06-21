"""Linux USB CDC 串口读写封装。"""

from __future__ import annotations

import errno
import os
import select
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class CdcSerial:
    """面向 Jetson Linux 的非阻塞 USB CDC 端口。"""

    path: str
    baudrate: int = 921600
    read_chunk_size: int = 4096
    _fd: int | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> CdcSerial:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        if self._fd is not None:
            return
        fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            configure_raw_serial(fd, self.baudrate)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def close(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None

    def read_available(self) -> bytes:
        fd = self._require_fd()
        try:
            return os.read(fd, int(self.read_chunk_size))
        except BlockingIOError:
            return b""
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return b""
            raise

    def wait_readable(self, timeout_s: float) -> bool:
        fd = self._require_fd()
        readable, _, _ = select.select([fd], [], [], max(0.0, float(timeout_s)))
        return bool(readable)

    def write_all(self, data: bytes, *, timeout_s: float = 0.02) -> None:
        fd = self._require_fd()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            try:
                written = os.write(fd, view[offset:])
            except BlockingIOError:
                written = 0
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                written = 0
            if written > 0:
                offset += int(written)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"USB CDC write timeout after {timeout_s:.3f}s")
            _, writable, _ = select.select([], [fd], [], min(0.001, deadline - time.monotonic()))
            if not writable and time.monotonic() >= deadline:
                raise TimeoutError(f"USB CDC write timeout after {timeout_s:.3f}s")

    def _require_fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("USB CDC port is not open")
        return self._fd


def configure_raw_serial(fd: int, baudrate: int) -> None:
    """把 CDC 端口设为 raw 模式；USB CDC 通常忽略实际 baudrate。"""

    import termios

    attrs = termios.tcgetattr(fd)
    iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = attrs

    iflag &= ~(
        termios.IGNBRK
        | termios.BRKINT
        | termios.PARMRK
        | termios.ISTRIP
        | termios.INLCR
        | termios.IGNCR
        | termios.ICRNL
        | termios.IXON
    )
    oflag &= ~termios.OPOST
    lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
    cflag &= ~(termios.CSIZE | termios.PARENB)
    cflag |= termios.CS8 | termios.CLOCAL | termios.CREAD

    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0

    speed = baud_to_termios(baudrate)
    termios.tcsetattr(fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, speed, speed, cc])
    termios.tcflush(fd, termios.TCIOFLUSH)


def baud_to_termios(baudrate: int) -> int:
    import termios

    table = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
        230400: getattr(termios, "B230400", termios.B115200),
        460800: getattr(termios, "B460800", termios.B115200),
        921600: getattr(termios, "B921600", termios.B115200),
    }
    return table.get(int(baudrate), table[921600])
