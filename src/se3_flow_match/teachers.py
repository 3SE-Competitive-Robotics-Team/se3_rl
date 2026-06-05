"""RSL teacher checkpoint 推理工具。"""

from __future__ import annotations

from pathlib import Path

import torch

from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.runtime_spec import RuntimeSpec


class TeacherPolicy:
    """冻结的 RSL GRU teacher。"""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        """加载 teacher checkpoint 并校验 42D->6D。"""
        self.checkpoint_path = Path(checkpoint)
        num_obs = PolicyRuntime.probe_num_obs(self.checkpoint_path, device=device)
        runtime = RuntimeSpec().with_num_obs(num_obs)
        self._runtime = PolicyRuntime(
            checkpoint=self.checkpoint_path,
            device=device,
            runtime=runtime,
        )
        if int(self._runtime.spec.num_obs) != 42:
            raise ValueError(
                f"teacher 必须是 42D actor obs，实际为 {self._runtime.spec.num_obs}: "
                f"{self.checkpoint_path}"
            )
        if int(self._runtime.spec.num_actions) != 6:
            raise ValueError(
                f"teacher 必须输出 6D action，实际为 {self._runtime.spec.num_actions}: "
                f"{self.checkpoint_path}"
            )
        if not self._runtime.spec.is_recurrent:
            raise ValueError(f"teacher 必须是 GRU checkpoint: {self.checkpoint_path}")
        self.model = self._runtime.model
        self.device = self._runtime.device
        self.model.eval()
        self.reset()

    @property
    def num_obs(self) -> int:
        """actor 观测维度。"""
        return int(self._runtime.spec.num_obs)

    @property
    def num_actions(self) -> int:
        """动作维度。"""
        return int(self._runtime.spec.num_actions)

    def reset(self, batch_size: int | None = None) -> None:
        """按 batch size 初始化 hidden。"""
        hidden_dim = int(self._runtime.spec.rnn_hidden_dim)
        layers = int(self._runtime.spec.rnn_num_layers)
        if batch_size is None:
            self.model._hidden = None  # pyright: ignore[reportAttributeAccessIssue]
            return
        self.model._hidden = torch.zeros(  # pyright: ignore[reportAttributeAccessIssue]
            layers,
            int(batch_size),
            hidden_dim,
            device=self.device,
        )

    def reset_done(self, dones: torch.Tensor) -> None:
        """按 done mask 清空对应 env hidden。"""
        hidden = getattr(self.model, "_hidden", None)
        if hidden is None:
            return
        done = dones.to(device=self.device, dtype=torch.bool)
        if done.ndim != 1 or done.shape[0] != hidden.shape[1]:
            raise ValueError(f"dones 必须是 {(hidden.shape[1],)}，实际为 {tuple(done.shape)}")
        if done.any():
            keep = (~done).to(dtype=hidden.dtype).view(1, -1, 1)
            self.model._hidden = hidden * keep  # pyright: ignore[reportAttributeAccessIssue]

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """批量 teacher 推理。"""
        if obs.ndim != 2 or obs.shape[1] != self.num_obs:
            raise ValueError(f"obs 必须是 [B, {self.num_obs}]，实际为 {tuple(obs.shape)}")
        action = self.model(obs.to(device=self.device, dtype=torch.float32))
        if action.shape != (obs.shape[0], self.num_actions):
            raise RuntimeError(
                f"teacher action 形状错误，期望 {(obs.shape[0], self.num_actions)}，"
                f"实际为 {tuple(action.shape)}"
            )
        return action


__all__ = ["TeacherPolicy"]
