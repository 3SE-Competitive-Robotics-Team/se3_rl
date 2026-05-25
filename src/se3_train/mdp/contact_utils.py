"""接触传感器张量工具。"""

from __future__ import annotations

import torch

CONTACT_FORCE_MAX_N = 5000.0


def contact_force_nonfinite_env_mask(force: torch.Tensor) -> torch.Tensor:
    """返回每个 env 的接触力是否出现 NaN/Inf。"""
    if force.ndim <= 1:
        return ~torch.isfinite(force)
    raw_nonfinite = ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    force_mag = torch.norm(force, dim=-1)
    norm_nonfinite = ~torch.isfinite(force_mag).flatten(start_dim=1).all(dim=1)
    return raw_nonfinite | norm_nonfinite


def finite_contact_force_norm(
    force: torch.Tensor,
    max_force: float = CONTACT_FORCE_MAX_N,
) -> torch.Tensor:
    """计算接触力模长，并把物理求解器偶发的非有限值限制在有限范围内。"""
    force_mag = torch.norm(force, dim=-1)
    return torch.nan_to_num(
        force_mag,
        nan=float(max_force),
        posinf=float(max_force),
        neginf=0.0,
    ).clamp(min=0.0, max=float(max_force))
