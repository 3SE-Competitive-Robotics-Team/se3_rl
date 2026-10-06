"""显式 vx 观测器 + MLP 策略（concurrent state estimator，Ji et al. 2022 RA-L / HIMLoco 写法）。

背景（2026-10-05，用户定）：DR1 下平地跟踪变粗主要来自 actor 不知道机身前向速度（Vx 特权 `qnd9o87r`
拿回 3 维速度特权的大部分收益，见 docs/plan/rough_dr1_oracle_20261005.md）。部署端没有 vx，用 actor 历史估。

结构（actor 观测组按 term-major、oldest→newest 展平的 16 帧历史，30 × 16 = 480 维）：

    h(480) → obs_normalizer → ĥ ──┬─ estimator MLP 480→128→64→1 ─→ v̂x ─ detach ─┐
                                    └─ 取每个 term 的最新一帧 ô_t(30) ────────────┴─ concat(31) → policy MLP → 动作分布

梯度：PPO 损失只更新 policy MLP 与分布参数（v̂x 先 detach）；估计器只由 vx 的 MSE 更新（Se3PPO 里的独立 Adam），
两路互不干扰。估计器输入与 policy 共用同一个经验归一化（统计量在 rollout 中更新，不受梯度影响）。
v̂x 以 m/s 原值进入 policy，不再归一化。部署时 ONNX 输入仍是 480 维历史、输出仍是动作，
runtime 的 History-MLP 契约不变。

隐向量（2026-10-06，用户定，latent_dim > 0）：估计器在 v̂x 之外再输出 latent_dim 维隐向量 z，**不 detach** 直接进 policy，
由 PPO 梯度端到端训练（没有额外监督）；v̂x 一列仍只由 MSE 监督、detach 后进 policy。因为 z 与 v̂x 共用估计器主干，
PPO 梯度会进入主干，与 vx 的 MSE 一起塑造主干特征。policy 输入为 [ô_t, sg(v̂x), z]。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict


def _latest_frame_index(frame_term_dims: tuple[int, ...], history_length: int) -> torch.Tensor:
    """term-major 历史展平后，每个 term 最新一帧在 480 维向量中的下标（按 term 顺序拼成单帧观测）。"""
    index: list[int] = []
    offset = 0
    for dim in frame_term_dims:
        block = dim * history_length
        index.extend(range(offset + block - dim, offset + block))
        offset += block
    return torch.tensor(index, dtype=torch.long)


class VxObserverMLPModel(MLPModel):
    """actor：估计器从历史估 vx，policy 只看最新一帧 + detach 的 v̂x。"""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64, 32),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        history_length: int,
        frame_term_dims: tuple[int, ...] | list[int],
        estimator_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        latent_dim: int = 0,
    ) -> None:
        # 父类 __init__ 里会调用 _get_latent_dim() 建 policy MLP，所以单帧宽度必须先于 super().__init__ 设好
        # （与 rsl_rl CNNModel 先设 cnn_latent_dim 的写法一致）。
        self.history_length = int(history_length)
        self.frame_term_dims = tuple(int(d) for d in frame_term_dims)
        self.frame_dim = sum(self.frame_term_dims)
        self.latent_dim = int(latent_dim)
        if self.latent_dim < 0:
            raise ValueError(f"latent_dim 不得为负，收到 {self.latent_dim}")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        if self.history_length < 2:
            raise ValueError(f"history_length 至少为 2，收到 {self.history_length}")
        if self.frame_dim * self.history_length != self.obs_dim:
            raise ValueError(
                f"观测宽度 {self.obs_dim} 与 sum(frame_term_dims)={self.frame_dim} × "
                f"history_length={self.history_length} 不一致"
            )
        self.register_buffer(
            "latest_frame_index",
            _latest_frame_index(self.frame_term_dims, self.history_length),
            persistent=False,
        )
        self.estimator = MLP(self.obs_dim, 1 + self.latent_dim, estimator_hidden_dims, activation)

    def _get_latent_dim(self) -> int:
        """policy MLP 输入：最新一帧 + v̂x + 隐向量。"""
        return self.frame_dim + 1 + self.latent_dim

    def _normalized_history(self, obs: TensorDict) -> torch.Tensor:
        return self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))

    def estimate_vx(self, obs: TensorDict) -> torch.Tensor:
        """估计器 v̂x 一列（带梯度），供 Se3PPO 的 MSE 更新用；形状 (N, 1)。"""
        return self.estimator(self._normalized_history(obs))[:, :1]

    @torch.no_grad()
    def latent_std(self, obs: TensorDict) -> float:
        """隐向量各维在 batch 上的标准差均值，用来看 z 是否塌缩成常数；latent_dim = 0 时返回 0。"""
        if self.latent_dim == 0:
            return 0.0
        return float(self.estimator(self._normalized_history(obs))[:, 1:].std(dim=0).mean())

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """policy 输入 [ô_t, sg(v̂x), z]：PPO 梯度不经 v̂x 回传，只经隐向量 z 回传到估计器。"""
        history = self._normalized_history(obs)
        estimate = self.estimator(history)
        return torch.cat(
            [
                history.index_select(-1, self.latest_frame_index),
                estimate[:, :1].detach(),
                estimate[:, 1:],
            ],
            dim=-1,
        )

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("VxObserverMLPModel 只支持 ONNX 导出")

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxVxObserverMLPModel(self, verbose)


class _OnnxVxObserverMLPModel(nn.Module):
    """ONNX 导出：输入 480 维历史，输出确定性动作；与训练前向逐项相同（无 detach，推理时无意义）。"""

    is_recurrent: bool = False

    def __init__(self, model: VxObserverMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.estimator = copy.deepcopy(model.estimator)
        self.mlp = copy.deepcopy(model.mlp)
        self.register_buffer("latest_frame_index", model.latest_frame_index.clone())
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history = self.obs_normalizer(x)
        estimate = self.estimator(history)
        latent = torch.cat([history.index_select(-1, self.latest_frame_index), estimate], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


__all__ = ["VxObserverMLPModel"]
