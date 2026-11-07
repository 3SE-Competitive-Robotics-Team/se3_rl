"""纯 torch 四元数工具（wxyz），与 mjlab.utils.lab_api.math 同名函数数学相同。

不直接用 mjlab 的版本：它们都带 `@torch.jit.script`，torch.compile（dynamo）追不进 TorchScript 函数，
每调用一次就断一次图。2026-11-07 nulltask1 剖析：编译后一次 actor–critic 更新仍有 24 个以上编译区域、
748 次编译函数调用、2.9 万次 kernel launch，CPU 786 ms 对 GPU 329 ms。
"""

from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(min=eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az = a.unbind(-1)
    bx, by, bz = b.unbind(-1)
    return torch.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q 把 v 从机身系转到世界系。"""
    xyz = q[..., 1:4]
    t = 2.0 * _cross(xyz, v)
    return v + q[..., 0:1] * t + _cross(xyz, t)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q 的逆旋转作用在 v 上（世界系 → 机身系）。"""
    xyz = q[..., 1:4]
    t = 2.0 * _cross(xyz, v)
    return v - q[..., 0:1] * t + _cross(xyz, t)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    return m.reshape(*q.shape[:-1], 3, 3)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat([torch.cos(half), normalize(axis) * torch.sin(half)], dim=-1)


__all__ = [
    "matrix_from_quat",
    "normalize",
    "quat_apply",
    "quat_apply_inverse",
    "quat_conjugate",
    "quat_from_angle_axis",
    "quat_mul",
]
