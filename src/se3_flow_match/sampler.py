"""Flow policy 采样器。"""

from __future__ import annotations

import torch

from .model import FlowVelocityField


@torch.no_grad()
def sample_actions(
    model: FlowVelocityField,
    obs: torch.Tensor,
    *,
    steps: int = 5,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """用当前 obs 推进 GRU，并从噪声采样动作。"""
    if obs.ndim != 2:
        raise ValueError(f"obs 必须是 [B, obs_dim]，实际为 {tuple(obs.shape)}")
    config = model.config
    if obs.shape[-1] != config.obs_dim:
        raise ValueError(f"obs 末维应为 {config.obs_dim}，实际为 {obs.shape[-1]}")
    context = model.step_context(obs)
    return sample_actions_from_context(model, context, steps=steps, noise=noise)


@torch.no_grad()
def sample_actions_from_context(
    model: FlowVelocityField,
    context: torch.Tensor,
    *,
    steps: int = 5,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """用反向 Euler 从噪声采样动作，不推进 GRU hidden。"""
    if steps <= 0:
        raise ValueError(f"steps 必须为正数，实际为 {steps}")
    if context.ndim != 2:
        raise ValueError(f"context 必须是 [B, hidden_dim]，实际为 {tuple(context.shape)}")
    config = model.config
    if context.shape[-1] != config.rnn_hidden_dim:
        raise ValueError(f"context 末维应为 {config.rnn_hidden_dim}，实际为 {context.shape[-1]}")

    if noise is None:
        action = torch.randn(
            context.shape[0], config.action_dim, dtype=context.dtype, device=context.device
        )
    else:
        action = noise.to(device=context.device, dtype=context.dtype)
    if action.shape != (context.shape[0], config.action_dim):
        raise ValueError(
            f"noise 必须是 {(context.shape[0], config.action_dim)}，实际为 {tuple(action.shape)}"
        )

    dt = 1.0 / float(steps)
    for step in range(steps):
        t_value = 1.0 - float(step) * dt
        t = torch.full((context.shape[0], 1), t_value, dtype=context.dtype, device=context.device)
        velocity = model.velocity_from_context(context, action, t)
        action = action - velocity * dt
    return action
