"""NX 与 STM32 之间的 recovery policy USB CDC 协议。"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from se3_shared import RECOVERY_DEFAULT_STM_COMMAND_5D

SOF = b"\xa5\x5a"
VERSION = 1
MAX_PAYLOAD_SIZE = 192

MSG_STATE = 0x01
MSG_TARGET = 0x02
MSG_LATENCY = 0x03

MSG_POLICY_STATE = MSG_STATE
MSG_POLICY_ACTION = MSG_TARGET

HEADER_STRUCT = struct.Struct("<BBBBHH")
CRC_STRUCT = struct.Struct("<H")

POLICY_STATE_STRUCT_V1 = struct.Struct("<IIIHBBB3x3f3f4f4f2f2f")
POLICY_STATE_STRUCT_V2 = struct.Struct("<IIIHBBB3x3f3f4f4f2f2f4f4f")
POLICY_STATE_STRUCT_V3 = struct.Struct("<IIIHBBB3x3f3f4f4f2f2f4f4f2f2f")
POLICY_STATE_STRUCT = struct.Struct("<IIIHBBB3x3f3f4f4f2f2f4f4f2f2f5f")
POLICY_TARGET_STRUCT = struct.Struct("<I4f2f")
POLICY_LATENCY_STRUCT = struct.Struct("<IIB3x")
DEFAULT_POLICY_COMMAND = RECOVERY_DEFAULT_STM_COMMAND_5D


@dataclass(frozen=True, slots=True)
class PolicyStateFrame:
    """STM32 上行给 NX 的 actor 状态帧。"""

    seq: int
    tick_ms: int
    target_seq: int
    target_age_ms: int
    target_valid: int
    rc_switch_r: int
    output_enabled: int
    base_ang_vel_body: tuple[float, float, float]
    projected_gravity: tuple[float, float, float]
    joint_pos: tuple[float, float, float, float]
    joint_vel: tuple[float, float, float, float]
    wheel_pos: tuple[float, float]
    wheel_vel: tuple[float, float]
    target_joint_pos: tuple[float, float, float, float]
    hip_torque: tuple[float, float, float, float]
    wheel_torque: tuple[float, float]
    wheel_motor_torque: tuple[float, float]
    command: tuple[float, float, float, float, float] = DEFAULT_POLICY_COMMAND

    payload_struct: ClassVar[struct.Struct] = POLICY_STATE_STRUCT

    @property
    def timestamp_us(self) -> int:
        return (int(self.tick_ms) * 1000) & 0xFFFFFFFF

    @property
    def dof_pos(self) -> tuple[float, float, float, float, float, float]:
        return (*self.joint_pos, *self.wheel_pos)

    @property
    def dof_vel(self) -> tuple[float, float, float, float, float, float]:
        return (*self.joint_vel, *self.wheel_vel)

    @property
    def policy_command(self) -> tuple[float, ...]:
        return (*self.command, 0.0, 0.0, 0.0)

    def pack_payload(self) -> bytes:
        values = (
            int(self.tick_ms),
            int(self.seq),
            int(self.target_seq),
            int(self.target_age_ms),
            int(self.target_valid),
            int(self.rc_switch_r),
            int(self.output_enabled),
            *finite_floats(self.base_ang_vel_body, 3, "base_ang_vel_body"),
            *finite_floats(self.projected_gravity, 3, "projected_gravity"),
            *finite_floats(self.joint_pos, 4, "joint_pos"),
            *finite_floats(self.joint_vel, 4, "joint_vel"),
            *finite_floats(self.wheel_pos, 2, "wheel_pos"),
            *finite_floats(self.wheel_vel, 2, "wheel_vel"),
            *finite_floats(self.target_joint_pos, 4, "target_joint_pos"),
            *finite_floats(self.hip_torque, 4, "hip_torque"),
            *finite_floats(self.wheel_torque, 2, "wheel_torque"),
            *finite_floats(self.wheel_motor_torque, 2, "wheel_motor_torque"),
            *finite_floats(self.command, 5, "command"),
        )
        return self.payload_struct.pack(*values)

    @classmethod
    def from_payload(cls, payload: bytes) -> PolicyStateFrame:
        if len(payload) == POLICY_STATE_STRUCT_V1.size:
            values = POLICY_STATE_STRUCT_V1.unpack(payload)
            return cls(
                tick_ms=int(values[0]),
                seq=int(values[1]),
                target_seq=int(values[2]),
                target_age_ms=int(values[3]),
                target_valid=int(values[4]),
                rc_switch_r=int(values[5]),
                output_enabled=int(values[6]),
                base_ang_vel_body=tuple(float(v) for v in values[7:10]),
                projected_gravity=tuple(float(v) for v in values[10:13]),
                joint_pos=tuple(float(v) for v in values[13:17]),
                joint_vel=tuple(float(v) for v in values[17:21]),
                wheel_pos=tuple(float(v) for v in values[21:23]),
                wheel_vel=tuple(float(v) for v in values[23:25]),
                target_joint_pos=(0.0, 0.0, 0.0, 0.0),
                hip_torque=(0.0, 0.0, 0.0, 0.0),
                wheel_torque=(0.0, 0.0),
                wheel_motor_torque=(0.0, 0.0),
            )
        if len(payload) == POLICY_STATE_STRUCT_V2.size:
            values = POLICY_STATE_STRUCT_V2.unpack(payload)
            return cls(
                tick_ms=int(values[0]),
                seq=int(values[1]),
                target_seq=int(values[2]),
                target_age_ms=int(values[3]),
                target_valid=int(values[4]),
                rc_switch_r=int(values[5]),
                output_enabled=int(values[6]),
                base_ang_vel_body=tuple(float(v) for v in values[7:10]),
                projected_gravity=tuple(float(v) for v in values[10:13]),
                joint_pos=tuple(float(v) for v in values[13:17]),
                joint_vel=tuple(float(v) for v in values[17:21]),
                wheel_pos=tuple(float(v) for v in values[21:23]),
                wheel_vel=tuple(float(v) for v in values[23:25]),
                target_joint_pos=tuple(float(v) for v in values[25:29]),
                hip_torque=tuple(float(v) for v in values[29:33]),
                wheel_torque=(0.0, 0.0),
                wheel_motor_torque=(0.0, 0.0),
            )
        if len(payload) == POLICY_STATE_STRUCT_V3.size:
            values = POLICY_STATE_STRUCT_V3.unpack(payload)
            return cls(
                tick_ms=int(values[0]),
                seq=int(values[1]),
                target_seq=int(values[2]),
                target_age_ms=int(values[3]),
                target_valid=int(values[4]),
                rc_switch_r=int(values[5]),
                output_enabled=int(values[6]),
                base_ang_vel_body=tuple(float(v) for v in values[7:10]),
                projected_gravity=tuple(float(v) for v in values[10:13]),
                joint_pos=tuple(float(v) for v in values[13:17]),
                joint_vel=tuple(float(v) for v in values[17:21]),
                wheel_pos=tuple(float(v) for v in values[21:23]),
                wheel_vel=tuple(float(v) for v in values[23:25]),
                target_joint_pos=tuple(float(v) for v in values[25:29]),
                hip_torque=tuple(float(v) for v in values[29:33]),
                wheel_torque=tuple(float(v) for v in values[33:35]),
                wheel_motor_torque=tuple(float(v) for v in values[35:37]),
            )
        if len(payload) != cls.payload_struct.size:
            raise ValueError(f"state payload size mismatch: {len(payload)}")
        values = cls.payload_struct.unpack(payload)
        return cls(
            tick_ms=int(values[0]),
            seq=int(values[1]),
            target_seq=int(values[2]),
            target_age_ms=int(values[3]),
            target_valid=int(values[4]),
            rc_switch_r=int(values[5]),
            output_enabled=int(values[6]),
            base_ang_vel_body=tuple(float(v) for v in values[7:10]),
            projected_gravity=tuple(float(v) for v in values[10:13]),
            joint_pos=tuple(float(v) for v in values[13:17]),
            joint_vel=tuple(float(v) for v in values[17:21]),
            wheel_pos=tuple(float(v) for v in values[21:23]),
            wheel_vel=tuple(float(v) for v in values[23:25]),
            target_joint_pos=tuple(float(v) for v in values[25:29]),
            hip_torque=tuple(float(v) for v in values[29:33]),
            wheel_torque=tuple(float(v) for v in values[33:35]),
            wheel_motor_torque=tuple(float(v) for v in values[35:37]),
            command=tuple(float(v) for v in values[37:42]),
        )


@dataclass(frozen=True, slots=True)
class PolicyTargetFrame:
    """NX 下行给 STM32 的物理目标帧。"""

    seq: int
    joint_pos: tuple[float, float, float, float]
    wheel_vel: tuple[float, float]

    payload_struct: ClassVar[struct.Struct] = POLICY_TARGET_STRUCT

    def pack_payload(self) -> bytes:
        values = (
            int(self.seq),
            *finite_floats(self.joint_pos, 4, "joint_pos"),
            *finite_floats(self.wheel_vel, 2, "wheel_vel"),
        )
        return self.payload_struct.pack(*values)

    @classmethod
    def from_payload(cls, payload: bytes) -> PolicyTargetFrame:
        if len(payload) != cls.payload_struct.size:
            raise ValueError(f"target payload size mismatch: {len(payload)}")
        values = cls.payload_struct.unpack(payload)
        return cls(
            seq=int(values[0]),
            joint_pos=tuple(float(v) for v in values[1:5]),
            wheel_vel=tuple(float(v) for v in values[5:7]),
        )


@dataclass(frozen=True, slots=True)
class PolicyLatencyFrame:
    """STM32 输出链路延迟诊断帧。"""

    policy_seq: int
    rx_to_output_us: int
    output_enabled: int

    payload_struct: ClassVar[struct.Struct] = POLICY_LATENCY_STRUCT

    @classmethod
    def from_payload(cls, payload: bytes) -> PolicyLatencyFrame:
        if len(payload) != cls.payload_struct.size:
            raise ValueError(f"latency payload size mismatch: {len(payload)}")
        values = cls.payload_struct.unpack(payload)
        return cls(
            policy_seq=int(values[0]),
            rx_to_output_us=int(values[1]),
            output_enabled=int(values[2]),
        )


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    """已通过帧头、版本、长度和 CRC 校验的消息。"""

    msg_type: int
    frame_seq: int
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
            start = self._buffer.find(SOF)
            if start < 0:
                self._drop_noise()
                break
            if start > 0:
                del self._buffer[:start]
            if len(self._buffer) < HEADER_STRUCT.size:
                break

            sof0, sof1, msg_type, version, payload_len, frame_seq = HEADER_STRUCT.unpack_from(
                self._buffer
            )
            if bytes((sof0, sof1)) != SOF:
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
            actual_crc = crc16(frame[: HEADER_STRUCT.size + payload_len])
            if actual_crc != expected_crc:
                del self._buffer[0]
                continue

            payload = frame[HEADER_STRUCT.size : HEADER_STRUCT.size + payload_len]
            messages.append(
                ParsedMessage(msg_type=int(msg_type), frame_seq=int(frame_seq), payload=payload)
            )
            del self._buffer[:frame_len]

        return messages

    def _drop_noise(self) -> None:
        if len(self._buffer) > len(SOF) - 1:
            del self._buffer[: -(len(SOF) - 1)]


def pack_message(msg_type: int, payload: bytes, *, frame_seq: int = 0) -> bytes:
    """打包带 CRC16 的协议消息。"""

    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD_SIZE}")
    header = HEADER_STRUCT.pack(
        SOF[0],
        SOF[1],
        int(msg_type),
        VERSION,
        len(payload),
        int(frame_seq) & 0xFFFF,
    )
    body = header + payload
    return body + CRC_STRUCT.pack(crc16(body))


def pack_policy_state(frame: PolicyStateFrame) -> bytes:
    return pack_message(MSG_STATE, frame.pack_payload(), frame_seq=frame.seq)


def pack_policy_target(frame: PolicyTargetFrame) -> bytes:
    return pack_message(MSG_TARGET, frame.pack_payload(), frame_seq=frame.seq)


def pack_policy_action(frame: PolicyTargetFrame) -> bytes:
    return pack_policy_target(frame)


def decode_policy_state(message: ParsedMessage) -> PolicyStateFrame:
    if message.msg_type != MSG_STATE:
        raise ValueError(f"unexpected message type for state: {message.msg_type}")
    return PolicyStateFrame.from_payload(message.payload)


def decode_policy_target(message: ParsedMessage) -> PolicyTargetFrame:
    if message.msg_type != MSG_TARGET:
        raise ValueError(f"unexpected message type for target: {message.msg_type}")
    return PolicyTargetFrame.from_payload(message.payload)


def decode_policy_action(message: ParsedMessage) -> PolicyTargetFrame:
    return decode_policy_target(message)


def decode_policy_latency(message: ParsedMessage) -> PolicyLatencyFrame:
    if message.msg_type != MSG_LATENCY:
        raise ValueError(f"unexpected message type for latency: {message.msg_type}")
    return PolicyLatencyFrame.from_payload(message.payload)


def finite_floats(values: object, size: int, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"{name} shape mismatch: expected {(size,)}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return tuple(float(v) for v in arr)


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = ((crc >> 8) ^ CRC_TABLE[(crc ^ byte) & 0xFF]) & 0xFFFF
    return crc


def make_crc_table() -> tuple[int, ...]:
    table: list[int] = []
    for byte in range(256):
        crc = 0
        value = byte
        for _ in range(8):
            if (crc ^ value) & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
            value >>= 1
        table.append(crc & 0xFFFF)
    return tuple(table)


CRC_TABLE = make_crc_table()
