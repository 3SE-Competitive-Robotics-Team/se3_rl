"""GRU velocity field 模型。"""

from __future__ import annotations

import torch
from torch import nn

from .config import FlowPolicyConfig


class FlowVelocityField(nn.Module):
    """条件 GRU velocity field。

    输入为当前观测、带噪动作和时间 t，输出 action 空间中的 flow velocity。
    """

    def __init__(self, config: FlowPolicyConfig) -> None:
        """初始化 GRU velocity field。"""
        super().__init__()
        self.config = config
        input_dim = config.obs_dim + config.action_dim + 1
        self.input_proj = nn.Linear(input_dim, config.rnn_hidden_dim)
        self.activation = _activation(config.activation)
        self.rnn = nn.GRU(
            input_size=config.rnn_hidden_dim,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_num_layers,
            batch_first=True,
        )
        self.output_head = nn.Linear(config.rnn_hidden_dim, config.action_dim)

    def forward(
        self,
        obs: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        *,
        dones: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测 flow velocity。

        obs/noisy_action 支持 [B, T, D] 或 [B, D]；返回形状与 noisy_action 一致。
        dones 为 [B, T] 时，会在终止步后重置 GRU hidden。
        """
        obs, noisy_action, t, squeeze_time = self._normalize_inputs(obs, noisy_action, t)
        if obs.shape[-1] != self.config.obs_dim:
            raise ValueError(f"obs 末维应为 {self.config.obs_dim}，实际为 {obs.shape[-1]}")
        if noisy_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"noisy_action 末维应为 {self.config.action_dim}，实际为 {noisy_action.shape[-1]}"
            )
        if obs.shape[:2] != noisy_action.shape[:2] or obs.shape[:2] != t.shape[:2]:
            raise ValueError(
                "obs/noisy_action/t 的 [B, T] 必须一致："
                f"obs={tuple(obs.shape)}, noisy_action={tuple(noisy_action.shape)}, t={tuple(t.shape)}"
            )

        model_input = torch.cat((obs, noisy_action, t), dim=-1)
        projected = self.activation(self.input_proj(model_input))
        rnn_out = self._run_gru(projected, dones=dones, hidden=hidden)
        velocity = self.output_head(rnn_out)
        return velocity.squeeze(1) if squeeze_time else velocity

    def _run_gru(
        self,
        inputs: torch.Tensor,
        *,
        dones: torch.Tensor | None,
        hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        """运行 GRU，并按 dones 在下一步前清零 hidden。"""
        if dones is None:
            out, _ = self.rnn(inputs, hidden)
            return out

        if dones.shape != inputs.shape[:2]:
            raise ValueError(f"dones 必须是 {tuple(inputs.shape[:2])}，实际为 {tuple(dones.shape)}")
        state = hidden
        outputs: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            out, state = self.rnn(inputs[:, step : step + 1], state)
            outputs.append(out)
            done = dones[:, step].to(device=inputs.device, dtype=torch.bool)
            if state is not None and done.any():
                keep = (~done).to(dtype=state.dtype).view(1, -1, 1)
                state = state * keep
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _normalize_inputs(
        obs: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """统一输入形状到 [B, T, D]。"""
        squeeze_time = obs.ndim == 2
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if noisy_action.ndim == 2:
            noisy_action = noisy_action.unsqueeze(1)
        if t.ndim == 0:
            t = t.reshape(1, 1, 1).expand(obs.shape[0], obs.shape[1], 1)
        elif t.ndim == 1:
            t = t.view(-1, 1, 1).expand(-1, obs.shape[1], -1)
        elif t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.ndim != 3:
            raise ValueError(f"t 必须可广播为 [B, T, 1]，实际形状为 {tuple(t.shape)}")
        if obs.ndim != 3 or noisy_action.ndim != 3:
            raise ValueError(
                "obs/noisy_action 必须是 [B, D] 或 [B, T, D]："
                f"obs={tuple(obs.shape)}, noisy_action={tuple(noisy_action.shape)}"
            )
        return obs, noisy_action, t.to(device=obs.device, dtype=obs.dtype), squeeze_time


def _activation(name: str) -> nn.Module:
    """解析激活函数。"""
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu" or normalized == "swish":
        return nn.SiLU()
    if normalized == "mish":
        return nn.Mish()
    raise ValueError(f"不支持的激活函数：{name}")
