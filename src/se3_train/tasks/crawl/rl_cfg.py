"""趴地移动任务的 PPO 配置：直接沿用冻结的 Flat-MLP 基线（单帧 MLP、rollout 24 步、3500 轮）。"""

from __future__ import annotations

from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

__all__ = ["mlp_rl_cfg"]
