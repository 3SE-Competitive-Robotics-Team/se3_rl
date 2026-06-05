"""Flow policy 运行时封装。"""

from __future__ import annotations

from pathlib import Path

import torch

from se3_shared import TaskMode

from .checkpoint import load_flow_checkpoint
from .sampler import sample_actions
from .task_mode import overwrite_task_mode_obs


class FlowPolicyRuntime:
    """和 MJLab viewer 兼容的 Flow policy callable。"""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        sample_steps: int | None = None,
        task_mode: TaskMode | None = None,
    ) -> None:
        """加载 Flow checkpoint。"""
        self.checkpoint_path = Path(checkpoint)
        self.model, self.config, self.metadata = load_flow_checkpoint(
            self.checkpoint_path, device=device
        )
        self.device = torch.device(device)
        self.sample_steps = int(sample_steps or self.config.sample_steps)
        self.task_mode = task_mode
        self.prev_mode = task_mode
        self.blend = 1.0
        self.model.eval()

    def reset(self) -> None:
        """重置 GRU hidden。"""
        self.model.reset_hidden()

    def set_task_mode(
        self,
        mode: TaskMode,
        *,
        prev_mode: TaskMode | None = None,
        blend: float = 1.0,
    ) -> None:
        """设置本步 task_mode 条件。"""
        self.task_mode = mode
        self.prev_mode = prev_mode if prev_mode is not None else mode
        self.blend = float(blend)

    @torch.no_grad()
    def __call__(self, obs_dict: object) -> torch.Tensor:
        """MJLab viewer policy 入口。"""
        obs = _extract_actor_obs(obs_dict).to(device=self.device, dtype=torch.float32)
        if self.task_mode is not None:
            obs = overwrite_task_mode_obs(
                obs,
                self.task_mode,
                prev=self.prev_mode,
                blend=self.blend,
            )
        if self.model._hidden is None:  # pyright: ignore[reportPrivateUsage]
            self.model.reset_hidden(batch_size=obs.shape[0], device=self.device)
        noise = torch.zeros(obs.shape[0], self.config.action_dim, device=self.device)
        action = sample_actions(self.model, obs, steps=self.sample_steps, noise=noise)
        if self.task_mode == TaskMode.GAIT:
            action = action.clone()
            action[:, 4:6] = 0.0
        return action


def _extract_actor_obs(obs_dict: object) -> torch.Tensor:
    """从 TensorDict 或 dict 中取 actor obs。"""
    obs = obs_dict["actor"]  # type: ignore[index]
    if not isinstance(obs, torch.Tensor):
        raise TypeError("actor obs 必须是 torch.Tensor")
    if obs.ndim != 2 or obs.shape[1] != 42:
        raise ValueError(f"actor obs 必须是 [B, 42]，实际为 {tuple(obs.shape)}")
    return obs


__all__ = ["FlowPolicyRuntime"]
