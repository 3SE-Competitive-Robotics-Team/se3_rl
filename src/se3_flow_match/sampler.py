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
    """用反向 Euler 从噪声采样动作。

    obs 必须为 [B, obs_dim]，返回 [B, action_dim]。
    """
    if steps <= 0:
        raise ValueError(f"steps 必须为正数，实际为 {steps}")
    if obs.ndim != 2:
        raise ValueError(f"obs 必须是 [B, obs_dim]，实际为 {tuple(obs.shape)}")
    config = model.config
    if obs.shape[-1] != config.obs_dim:
        raise ValueError(f"obs 末维应为 {config.obs_dim}，实际为 {obs.shape[-1]}")

    if noise is None:
        action = torch.randn(obs.shape[0], config.action_dim, dtype=obs.dtype, device=obs.device)
    else:
        action = noise.to(device=obs.device, dtype=obs.dtype)
    if action.shape != (obs.shape[0], config.action_dim):
        raise ValueError(
            f"noise 必须是 {(obs.shape[0], config.action_dim)}，实际为 {tuple(action.shape)}"
        )

    dt = 1.0 / float(steps)
    for step in range(steps):
        t_value = 1.0 - float(step) * dt
        t = torch.full((obs.shape[0], 1), t_value, dtype=obs.dtype, device=obs.device)
        velocity = model(obs, action, t)
        action = action - velocity * dt
    return action
