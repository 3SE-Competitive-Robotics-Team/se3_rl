"""AMP 运动输入：19 维物理状态与间隔 20 ms 的相邻帧，不包含 actor 专用缩放。"""

from __future__ import annotations

import torch

AMP_FRAME_DIM = 19
AMP_TRANSITION_DIM = 38
AMP_CONTROL_DT_S = 0.02
AMP_FEATURE_NAMES = (
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "base_omega_x",
    "base_omega_y",
    "base_omega_z",
    "base_velocity_x",
    "base_velocity_y",
    "base_velocity_z",
    "left_wheel_x",
    "left_wheel_z",
    "right_wheel_x",
    "right_wheel_z",
    "left_wheel_vx",
    "left_wheel_vz",
    "right_wheel_vx",
    "right_wheel_vz",
    "left_wheel_spin",
    "right_wheel_spin",
)


def amp_frame_from_world(
    *,
    rotation_world_from_body: torch.Tensor,
    base_lin_vel_world: torch.Tensor,
    base_ang_vel_world: torch.Tensor,
    wheel_pos_world: torch.Tensor,
    hip_pos_world: torch.Tensor,
    wheel_lin_vel_world: torch.Tensor,
    hip_lin_vel_world: torch.Tensor,
    wheel_spin_forward: torch.Tensor,
) -> torch.Tensor:
    """从 link 原点运动学构造 AMP 单帧，支持任意批量维度。

    body 为右手「前、左、上」坐标，矩阵将 body 向量转到 world；两轮按物理左、右排列。
    位置单位 m，线速度 m/s，角速度 rad/s，轮子自转正值统一为向前滚动。
    轮心相对速度是随体坐标下相对位置的时间导数，必须扣除 omega × p。
    """
    for value in (wheel_pos_world, hip_pos_world, wheel_lin_vel_world, hip_lin_vel_world):
        if value.shape[-2:] != (2, 3):
            raise ValueError("轮心/髋轴运动学末两维必须为 (2, 3)，顺序为左、右")
    if rotation_world_from_body.shape[-2:] != (3, 3) or wheel_spin_forward.shape[-1] != 2:
        raise ValueError("机身旋转矩阵或轮速形状错误")
    rotation = rotation_world_from_body
    gravity = -rotation[..., 2, :]
    omega = torch.einsum("...ji,...j->...i", rotation, base_ang_vel_world)
    velocity = torch.einsum("...ji,...j->...i", rotation, base_lin_vel_world)
    relative_position = torch.einsum(
        "...ji,...kj->...ki", rotation, wheel_pos_world - hip_pos_world
    )
    relative_velocity = torch.einsum(
        "...ji,...kj->...ki", rotation, wheel_lin_vel_world - hip_lin_vel_world
    )
    relative_velocity -= torch.linalg.cross(
        omega.unsqueeze(-2).expand_as(relative_position), relative_position
    )
    batch_shape = relative_position.shape[:-2]
    return torch.cat(
        (
            gravity,
            omega,
            velocity,
            relative_position[..., [0, 2]].reshape(*batch_shape, 4),
            relative_velocity[..., [0, 2]].reshape(*batch_shape, 4),
            wheel_spin_forward,
        ),
        dim=-1,
    )


def normalize_amp_frame(frame: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """应用统一统计量；示范与策略、前后两帧必须使用同一份 mean/std。"""
    if (
        frame.shape[-1] != AMP_FRAME_DIM
        or mean.shape != (AMP_FRAME_DIM,)
        or std.shape != (AMP_FRAME_DIM,)
    ):
        raise ValueError("AMP 归一化只接受 19 维单帧及同维度统计量")
    return (frame - mean.to(frame)) / std.to(frame).clamp_min(1e-6)


def amp_transition(
    previous: torch.Tensor, current: torch.Tensor, *, reset_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """拼成 [上一帧, 当前帧]；排除跨 episode 和非有限状态，不将它们送入判别损失。"""
    if previous.shape != current.shape or current.shape[-1] != AMP_FRAME_DIM:
        raise ValueError("AMP 相邻帧必须是同形状的 19 维状态")
    if reset_mask.shape != current.shape[:-1] or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask 必须是与批量维度一致的布尔张量")
    valid = ~reset_mask & torch.isfinite(previous).all(dim=-1) & torch.isfinite(current).all(dim=-1)
    pair = torch.cat((previous, current), dim=-1)
    return torch.where(valid.unsqueeze(-1), pair, torch.zeros_like(pair)), valid


class AmpTransitionHistory:
    """每个 20 ms 控制步调用一次；reset_mask 表示当前帧属于新 episode。"""

    def __init__(self, *, control_dt_s: float = AMP_CONTROL_DT_S) -> None:
        if abs(control_dt_s - AMP_CONTROL_DT_S) > 1e-8:
            raise ValueError("当前 AMP 契约要求相邻帧间隔为 20 ms")
        self.previous: torch.Tensor | None = None

    def update(
        self, frame: torch.Tensor, *, reset_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 38 维 transition 与有效掩码；首帧只初始化历史。"""
        if self.previous is None:
            pair, valid = amp_transition(frame, frame, reset_mask=torch.ones_like(reset_mask))
        else:
            pair, valid = amp_transition(self.previous, frame, reset_mask=reset_mask)
        self.previous = frame.detach().clone()
        return pair, valid
