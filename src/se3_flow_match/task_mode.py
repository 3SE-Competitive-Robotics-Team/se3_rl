"""TaskMode 条件观测工具。"""

from __future__ import annotations

import torch

from se3_shared import (
    TASK_MODE_LOCOMOTION_CONTRACT,
    TASK_MODE_LOCOMOTION_TASK_MODE_SLICE,
    TASK_MODE_SEMANTICS,
    TaskMode,
)


def task_mode_tail(
    current: TaskMode,
    *,
    prev: TaskMode | None = None,
    blend: float = 1.0,
    batch: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构造 13D task_mode 尾部。"""
    prev = current if prev is None else prev
    current_semantic = torch.tensor(
        TASK_MODE_SEMANTICS[int(current)], device=device, dtype=dtype
    ).expand(batch, -1)
    prev_semantic = torch.tensor(TASK_MODE_SEMANTICS[int(prev)], device=device, dtype=dtype).expand(
        batch, -1
    )
    blend_col = torch.full((batch, 1), float(blend), device=device, dtype=dtype)
    jump_tail = torch.zeros(batch, 4, device=device, dtype=dtype)
    return torch.cat((current_semantic, prev_semantic, blend_col, jump_tail), dim=1)


def overwrite_task_mode_obs(
    obs: torch.Tensor,
    current: TaskMode,
    *,
    prev: TaskMode | None = None,
    blend: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """覆盖 42D obs 里的 task_mode 尾部。"""
    expected = TASK_MODE_LOCOMOTION_CONTRACT.num_obs
    if obs.shape[-1] != expected:
        raise ValueError(f"obs 末维必须为 {expected}，实际为 {obs.shape[-1]}")
    out = obs.clone()
    flat = out.reshape(-1, out.shape[-1])
    if isinstance(blend, torch.Tensor):
        blend_values = blend.to(device=flat.device, dtype=flat.dtype).reshape(-1)
        if blend_values.numel() == 1:
            blend_values = blend_values.expand(flat.shape[0])
        if blend_values.numel() != flat.shape[0]:
            raise ValueError(f"blend 数量必须为 {flat.shape[0]}，实际为 {blend_values.numel()}")
        tail = task_mode_tail(
            current,
            prev=prev,
            blend=1.0,
            batch=flat.shape[0],
            device=flat.device,
            dtype=flat.dtype,
        )
        tail[:, 8] = blend_values
    else:
        tail = task_mode_tail(
            current,
            prev=prev,
            blend=float(blend),
            batch=flat.shape[0],
            device=flat.device,
            dtype=flat.dtype,
        )
    flat[:, TASK_MODE_LOCOMOTION_TASK_MODE_SLICE] = tail
    return out


def mode_transition_obs(
    obs: torch.Tensor,
    current: TaskMode,
    prev: TaskMode,
) -> torch.Tensor:
    """为序列样本生成 blend 0→1 的过渡观测。"""
    if obs.ndim != 3:
        raise ValueError(
            f"obs 必须是 [N, T, {TASK_MODE_LOCOMOTION_CONTRACT.num_obs}]，实际为 {tuple(obs.shape)}"
        )
    blend = torch.linspace(0.0, 1.0, obs.shape[1], device=obs.device, dtype=obs.dtype)
    blend = blend.view(1, obs.shape[1]).expand(obs.shape[0], -1)
    return overwrite_task_mode_obs(obs, current, prev=prev, blend=blend)


__all__ = ["mode_transition_obs", "overwrite_task_mode_obs", "task_mode_tail"]
