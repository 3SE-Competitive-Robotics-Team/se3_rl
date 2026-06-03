"""NX 与 STM32 之间的 recovery policy USB CDC 协议。

协议目标是把真机部署边界写死：
- STM32 上行 policy 顺序的物理状态。
- NX 下行网络输出的 6 维归一化 raw action。
- STM32 负责把 raw action 转换为物理目标并做限幅和底层安全。
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

MAGIC = b"S3"
VERSION = 1
MAX_PAYLOAD_SIZE = 256

MSG_POLICY_STATE = 1
MSG_POLICY_ACTION = 2

HEADER_STRUCT = struct.Struct("<2sBBH")
CRC_STRUCT = struct.Struct("<I")

POLICY_STATE_STRUCT = struct.Struct("<III3f3f6f6f6H")
POLICY_ACTION_STRUCT = struct.Struct("<IIIHH6f")

ACTION_FLAG_TIMEOUT = 1 << 0
ACTION_FLAG_NONFINITE = 1 << 1
ACTION_FLAG_DRY_RUN = 1 << 2

MODE_RECOVERY_ONLY = 1


@dataclass(frozen=True, slots=True)
class PolicyStateFrame:
    """STM32 上行给 NX 的 policy 状态帧。"""

    seq: int
    timestamp_us: int
    status_bits: int
    base_ang_vel_body: tuple[float, float, float]
    projected_gravity: tuple[float, float, float]
    dof_pos: tuple[float, float, float, float, float, float]
    dof_vel: tuple[float, float, float, float, float, float]
    motor_status: tuple[int, int, int, int, int, int]

    payload_struct: ClassVar[struct.Struct] = POLICY_STATE_STRUCT

    def pack_payload(self) -> bytes:
        values = (
            int(self.seq),
            int(self.timestamp_us),
            int(self.status_bits),
            *finite_floats(self.base_ang_vel_body, 3, "base_ang_vel_body"),
            *finite_floats(self.projected_gravity, 3, "projected_gravity"),
            *finite_floats(self.dof_pos, 6, "dof_pos"),
            *finite_floats(self.dof_vel, 6, "dof_vel"),
            *uints(self.motor_status, 6, "motor_status", max_value=0xFFFF),
        )
        return self.payload_struct.pack(*values)

    @classmethod
    def from_payload(cls, payload: bytes) -> PolicyStateFrame:
        if len(payload) != cls.payload_struct.size:
            raise ValueError(f"state payload size mismatch: {len(payload)}")
        values = cls.payload_struct.unpack(payload)
        return cls(
            seq=int(values[0]),
            timestamp_us=int(values[1]),
            status_bits=int(values[2]),
            base_ang_vel_body=tuple(float(v) for v in values[3:6]),
            projected_gravity=tuple(float(v) for v in values[6:9]),
            dof_pos=tuple(float(v) for v in values[9:15]),
            dof_vel=tuple(float(v) for v in values[15:21]),
            motor_status=tuple(int(v) for v in values[21:27]),
        )


@dataclass(frozen=True, slots=True)
class PolicyActionFrame:
    """NX 下行给 STM32 的 raw policy action 帧。"""

    seq: int
    source_state_seq: int
    timestamp_us: int
    mode: int
    flags: int
    action: tuple[float, float, float, float, float, float]

    payload_struct: ClassVar[struct.Struct] = POLICY_ACTION_STRUCT

    def pack_payload(self) -> bytes:
        values = (
            int(self.seq),
            int(self.source_state_seq),
            int(self.timestamp_us),
            int(self.mode),
            int(self.flags),
            *finite_floats(self.action, 6, "action"),
        )
        return self.payload_struct.pack(*values)

    @classmethod
    def from_payload(cls, payload: bytes) -> PolicyActionFrame:
        if len(payload) != cls.payload_struct.size:
            raise ValueError(f"action payload size mismatch: {len(payload)}")
        values = cls.payload_struct.unpack(payload)
        return cls(
            seq=int(values[0]),
            source_state_seq=int(values[1]),
            timestamp_us=int(values[2]),
            mode=int(values[3]),
            flags=int(values[4]),
            action=tuple(float(v) for v in values[5:11]),
        )


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    """已通过 magic、版本、长度和 CRC 校验的消息。"""

    msg_type: int
    payload: bytes


class StreamParser:
    """面向 USB CDC 字节流的增量解析器。"""

    def __init__(self, *, max_payload_size: int = MAX_PAYLOAD_SIZE) -> None:
        self.max_payload_size = int(max_payload_size)
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[ParsedMessage]:
        if data:
            self._buffer.extend(data)
        messages: list[ParsedMessage] = []

        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                self._drop_noise()
                break
            if start > 0:
                del self._buffer[:start]
            if len(self._buffer) < HEADER_STRUCT.size:
                break

            magic, version, msg_type, payload_len = HEADER_STRUCT.unpack_from(self._buffer)
            if magic != MAGIC:
                del self._buffer[0]
                continue
            if version != VERSION or payload_len > self.max_payload_size:
                del self._buffer[0]
                continue

            frame_len = HEADER_STRUCT.size + int(payload_len) + CRC_STRUCT.size
            if len(self._buffer) < frame_len:
                break

            frame = bytes(self._buffer[:frame_len])
            expected_crc = CRC_STRUCT.unpack_from(frame, HEADER_STRUCT.size + payload_len)[0]
            actual_crc = zlib.crc32(frame[: HEADER_STRUCT.size + payload_len]) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                del self._buffer[0]
                continue

            payload = frame[HEADER_STRUCT.size : HEADER_STRUCT.size + payload_len]
            messages.append(ParsedMessage(msg_type=int(msg_type), payload=payload))
            del self._buffer[:frame_len]

        return messages

    def _drop_noise(self) -> None:
        if len(self._buffer) > len(MAGIC) - 1:
            del self._buffer[: -(len(MAGIC) - 1)]


def pack_message(msg_type: int, payload: bytes) -> bytes:
    """打包带 CRC32 的协议消息。"""

    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD_SIZE}")
    header = HEADER_STRUCT.pack(MAGIC, VERSION, int(msg_type), len(payload))
    crc = zlib.crc32(header + payload) & 0xFFFFFFFF
    return header + payload + CRC_STRUCT.pack(crc)


def pack_policy_state(frame: PolicyStateFrame) -> bytes:
    return pack_message(MSG_POLICY_STATE, frame.pack_payload())


def pack_policy_action(frame: PolicyActionFrame) -> bytes:
    return pack_message(MSG_POLICY_ACTION, frame.pack_payload())


def decode_policy_state(message: ParsedMessage) -> PolicyStateFrame:
    if message.msg_type != MSG_POLICY_STATE:
        raise ValueError(f"unexpected message type for state: {message.msg_type}")
    return PolicyStateFrame.from_payload(message.payload)


def decode_policy_action(message: ParsedMessage) -> PolicyActionFrame:
    if message.msg_type != MSG_POLICY_ACTION:
        raise ValueError(f"unexpected message type for action: {message.msg_type}")
    return PolicyActionFrame.from_payload(message.payload)


def finite_floats(values: object, size: int, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"{name} shape mismatch: expected {(size,)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return tuple(float(v) for v in arr)


def uints(values: object, size: int, name: str, *, max_value: int) -> tuple[int, ...]:
    items = tuple(int(v) for v in values)  # type: ignore[arg-type]
    if len(items) != int(size):
        raise ValueError(f"{name} size mismatch: expected {size}, got {len(items)}")
    for value in items:
        if value < 0 or value > int(max_value):
            raise ValueError(f"{name} value out of range: {value}")
    return items
