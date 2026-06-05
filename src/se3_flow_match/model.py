"""GRU 条件 velocity field 模型。"""

from __future__ import annotations

import torch
from torch import nn

from .config import FlowPolicyConfig


class FlowVelocityField(nn.Module):
    """用观测历史条件化的 action velocity field。

    GRU 只编码 obs 历史。训练时 velocity head 接收每个时间步的 obs context、
    noisy action 和 t；部署采样时先用当前 obs 推进一步 hidden，再在 Euler
    子步里复用同一个 context。
    """

    def __init__(self, config: FlowPolicyConfig) -> None:
        """初始化 Flow Matching 学生模型。"""
        super().__init__()
        self.config = config
        self.obs_normalizer = _EmpiricalNormalizer(config.obs_dim)
        self.obs_proj = nn.Linear(config.obs_dim, config.rnn_hidden_dim)
        self.activation = _activation(config.activation)
        self.rnn = nn.GRU(
            input_size=config.rnn_hidden_dim,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_num_layers,
            batch_first=True,
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(config.rnn_hidden_dim + config.action_dim + 1, config.rnn_hidden_dim),
            _activation(config.activation),
            nn.Linear(config.rnn_hidden_dim, config.action_dim),
        )
        self._hidden: torch.Tensor | None = None

    def encode_obs(
        self,
        obs: torch.Tensor,
        *,
        dones: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把 obs 编码成 flow 条件 context。"""
        obs, squeeze_time = self._normalize_obs(obs)
        if obs.shape[-1] != self.config.obs_dim:
            raise ValueError(f"obs 末维应为 {self.config.obs_dim}，实际为 {obs.shape[-1]}")
        projected = self.activation(self.obs_proj(self.obs_normalizer(obs)))
        context = self._run_gru(projected, dones=dones, hidden=hidden)
        return context.squeeze(1) if squeeze_time else context

    def forward(
        self,
        obs: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        *,
        dones: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """从 obs/noisy_action/t 预测 action 空间 velocity。"""
        obs, noisy_action, t, squeeze_time = self._normalize_flow_inputs(obs, noisy_action, t)
        context = self.encode_obs(obs, dones=dones, hidden=hidden)
        if context.ndim == 2:
            context = context.unsqueeze(1)
        velocity = self.velocity_from_context(context, noisy_action, t)
        return velocity.squeeze(1) if squeeze_time else velocity

    def velocity_from_context(
        self,
        context: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """用已编码 context 预测 velocity，不推进 GRU hidden。"""
        context, noisy_action, t, squeeze_time = self._normalize_context_inputs(
            context, noisy_action, t
        )
        if context.shape[-1] != self.config.rnn_hidden_dim:
            raise ValueError(
                f"context 末维应为 {self.config.rnn_hidden_dim}，实际为 {context.shape[-1]}"
            )
        if noisy_action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"noisy_action 末维应为 {self.config.action_dim}，实际为 {noisy_action.shape[-1]}"
            )
        if context.shape[:2] != noisy_action.shape[:2] or context.shape[:2] != t.shape[:2]:
            raise ValueError(
                "context/noisy_action/t 的 [B, T] 必须一致："
                f"context={tuple(context.shape)}, noisy_action={tuple(noisy_action.shape)}, "
                f"t={tuple(t.shape)}"
            )
        model_input = torch.cat((context, noisy_action, t), dim=-1)
        velocity = self.velocity_head(model_input)
        return velocity.squeeze(1) if squeeze_time else velocity

    def step_context(self, obs: torch.Tensor) -> torch.Tensor:
        """部署时用单步 obs 推进内部 hidden，并返回当前 context。"""
        if obs.ndim != 2:
            raise ValueError(f"obs 必须是 [B, obs_dim]，实际为 {tuple(obs.shape)}")
        projected = self.activation(self.obs_proj(self.obs_normalizer(obs)))
        context, self._hidden = self.rnn(projected.unsqueeze(1), self._hidden)
        return context.squeeze(1)

    def set_obs_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """写入离线数据集观测均值和标准差。"""
        self.obs_normalizer.set_statistics(mean, std)

    def reset_hidden(
        self, batch_size: int | None = None, device: torch.device | None = None
    ) -> None:
        """清空或初始化部署 hidden。"""
        if batch_size is None:
            self._hidden = None
            return
        if device is None:
            device = next(self.parameters()).device
        self._hidden = torch.zeros(
            self.config.rnn_num_layers,
            int(batch_size),
            self.config.rnn_hidden_dim,
            device=device,
        )

    def reset_done(self, dones: torch.Tensor) -> None:
        """按 done mask 清空部署 hidden 的对应 batch。"""
        if self._hidden is None:
            return
        done = dones.to(device=self._hidden.device, dtype=torch.bool)
        if done.ndim != 1 or done.shape[0] != self._hidden.shape[1]:
            raise ValueError(f"dones 必须是 {(self._hidden.shape[1],)}，实际为 {tuple(done.shape)}")
        if done.any():
            keep = (~done).to(dtype=self._hidden.dtype).view(1, -1, 1)
            self._hidden = self._hidden * keep

    def _run_gru(
        self,
        inputs: torch.Tensor,
        *,
        dones: torch.Tensor | None,
        hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        """运行 GRU，并在 done 后清零下一步 hidden。"""
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
    def _normalize_obs(obs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """统一 obs 到 [B, T, D]。"""
        squeeze_time = obs.ndim == 2
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if obs.ndim != 3:
            raise ValueError(f"obs 必须是 [B, D] 或 [B, T, D]，实际为 {tuple(obs.shape)}")
        return obs, squeeze_time

    @classmethod
    def _normalize_flow_inputs(
        cls, obs: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """统一 flow 输入形状到 [B, T, D]。"""
        obs, squeeze_time = cls._normalize_obs(obs)
        noisy_action = cls._ensure_time_dim(noisy_action, "noisy_action")
        t = cls._normalize_time(
            t, batch=obs.shape[0], steps=obs.shape[1], device=obs.device, dtype=obs.dtype
        )
        return obs, noisy_action.to(device=obs.device, dtype=obs.dtype), t, squeeze_time

    @classmethod
    def _normalize_context_inputs(
        cls, context: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """统一 context/head 输入形状到 [B, T, D]。"""
        squeeze_time = context.ndim == 2
        context = cls._ensure_time_dim(context, "context")
        noisy_action = cls._ensure_time_dim(noisy_action, "noisy_action")
        t = cls._normalize_time(
            t,
            batch=context.shape[0],
            steps=context.shape[1],
            device=context.device,
            dtype=context.dtype,
        )
        return context, noisy_action.to(device=context.device, dtype=context.dtype), t, squeeze_time

    @staticmethod
    def _ensure_time_dim(tensor: torch.Tensor, name: str) -> torch.Tensor:
        """把 [B, D] 张量补成 [B, 1, D]。"""
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 3:
            raise ValueError(f"{name} 必须是 [B, D] 或 [B, T, D]，实际为 {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _normalize_time(
        t: torch.Tensor,
        *,
        batch: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """把 t 统一成 [B, T, 1]。"""
        if t.ndim == 0:
            t = t.reshape(1, 1, 1).expand(batch, steps, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1, 1).expand(-1, steps, -1)
        elif t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.ndim != 3:
            raise ValueError(f"t 必须可广播为 [B, T, 1]，实际为 {tuple(t.shape)}")
        if t.shape != (batch, steps, 1):
            raise ValueError(f"t 必须是 {(batch, steps, 1)}，实际为 {tuple(t.shape)}")
        return t.to(device=device, dtype=dtype)


class _EmpiricalNormalizer(nn.Module):
    """和 RSL 风格一致的观测归一化层。"""

    def __init__(self, num_obs: int, eps: float = 1.0e-2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(1, int(num_obs)))
        self.register_buffer("_std", torch.ones(1, int(num_obs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行经验均值方差归一化。"""
        return (x - self._mean) / (self._std + self.eps)

    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """写入固定观测统计量。"""
        mean = mean.detach().reshape(1, -1).to(device=self._mean.device, dtype=self._mean.dtype)
        std = std.detach().reshape(1, -1).to(device=self._std.device, dtype=self._std.dtype)
        if mean.shape != self._mean.shape:
            raise ValueError(
                f"mean 形状必须是 {tuple(self._mean.shape)}，实际为 {tuple(mean.shape)}"
            )
        if std.shape != self._std.shape:
            raise ValueError(f"std 形状必须是 {tuple(self._std.shape)}，实际为 {tuple(std.shape)}")
        self._mean.copy_(mean)
        self._std.copy_(std.clamp_min(1.0e-4))


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
