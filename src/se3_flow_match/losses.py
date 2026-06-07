"""Flow Matching 损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import FlowVelocityField


@dataclass(frozen=True)
class FlowMatchingLoss:
    """Flow Matching 损失明细。"""

    loss: torch.Tensor
    mse: torch.Tensor
    valid_ratio: torch.Tensor


def flow_matching_loss(
    model: FlowVelocityField,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    dones: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    times: torch.Tensor | None = None,
    action_weights: torch.Tensor | None = None,
) -> FlowMatchingLoss:
    """计算 action-level Flow Matching 损失。

    参数化为 x_t = (1 - t) * action + t * noise，监督目标为 noise - action。
    dones 只用于 GRU hidden reset，终止步动作仍参与 teacher 监督。
    """
    if obs.ndim != 3:
        raise ValueError(f"obs 必须是 [B, T, obs_dim]，实际为 {tuple(obs.shape)}")
    if actions.ndim != 3:
        raise ValueError(f"actions 必须是 [B, T, action_dim]，实际为 {tuple(actions.shape)}")
    if obs.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "obs/actions 的 [B, T] 必须一致："
            f"obs={tuple(obs.shape)}, actions={tuple(actions.shape)}"
        )

    if noise is None:
        noise = torch.randn_like(actions)
    if times is None:
        times = torch.rand(*actions.shape[:2], 1, device=actions.device, dtype=actions.dtype)
    if noise.shape != actions.shape:
        raise ValueError(f"noise 必须是 {tuple(actions.shape)}，实际为 {tuple(noise.shape)}")
    if times.shape != (*actions.shape[:2], 1):
        raise ValueError(f"times 必须是 {(*actions.shape[:2], 1)}，实际为 {tuple(times.shape)}")

    noisy_action = (1.0 - times) * actions + times * noise
    target = noise - actions
    pred = model(obs, noisy_action, times, dones=dones)
    mse = (pred - target).square()
    if action_weights is None:
        mse_per_action = mse.mean(dim=-1)
        loss = mse_per_action.mean()
    else:
        weights = action_weights.to(device=actions.device, dtype=actions.dtype)
        if weights.ndim == 1:
            weights = weights.view(1, 1, -1)
        elif weights.ndim == 2:
            weights = weights.unsqueeze(1)
        if weights.shape != mse.shape:
            raise ValueError(
                f"action_weights 必须可变为 {tuple(mse.shape)}，实际为 {tuple(weights.shape)}"
            )
        denom = weights.sum().clamp_min(1.0e-6)
        loss = (mse * weights).sum() / denom
    valid_ratio = torch.ones((), device=actions.device, dtype=actions.dtype)
    return FlowMatchingLoss(loss=loss, mse=loss, valid_ratio=valid_ratio)
