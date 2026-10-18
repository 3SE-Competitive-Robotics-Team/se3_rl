"""CTS（Concurrent Teacher-Student，Wang et al. RA-L 2024）+ 显式 vx 的 actor（2026-10-07，用户定）。

参照两份开源实现（Apache-2.0）：clearlab-sustech/multi_loco_isaacgym 的 rl_alg/agents/actor_critic_cts/ppo_cts.py
（教师隐向量后拼真实状态、学生回归整条向量）与 Realsbt/RL-RobotLab 的 rsl_rl MoE-CTS（32 维 L2 归一化隐向量、
教师 / 学生按 env 编号交错分组、学生编码器只由隐向量 MSE 训练）。与论文描述不同处以代码为准：学生编码器不接收 PPO 梯度。

    教师 env：critic 特权观测 → teacher_obs_normalizer → 教师编码器 MLP → L2 归一化 z_T(32) ─┐
                                                                         真实 vx（estimator_target）─┴→ [z_T, vx]
    学生 env：actor 16 帧历史 → obs_normalizer → 学生编码器 MLP → [L2(z_S)(32), v̂x] ──── detach ──→ [z_S, v̂x]
    两组共用 policy MLP：[最新一帧 ô_t(30), 隐向量(33)] → 动作分布

梯度：PPO 损失只经教师 env 的隐向量更新教师编码器，学生隐向量在 policy 前 detach；学生编码器（沿用 estimator 这个名字，
复用 Se3PPO 的独立 Adam 与多卡梯度同步）只由重建损失训练：在学生 env 样本上回归 [sg(z_T), vx_true]。
每个样本属于哪组由观测组 cts_role（教师 1 / 学生 0，env 固定）给出，随 rollout 存进 storage。
部署只走学生路径：ONNX 输入仍是 480 维历史、输出动作，与 VxObserver 的 History-MLP 契约相同。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict

from se3_train.vx_observer import _latest_frame_index


class CTSVxObserverModel(MLPModel):
    """actor：教师 env 用特权编码器的隐向量 + 真实 vx，学生 env 用历史编码器估计的隐向量 + v̂x，共用 policy。"""

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
        estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        teacher_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        latent_dim: int = 32,
        teacher_obs_group: str = "critic",
        vx_target_group: str = "estimator_target",
        role_group: str = "cts_role",
    ) -> None:
        # 父类 __init__ 会调用 _get_latent_dim() 建 policy MLP，相关宽度须先设好（同 rsl_rl CNNModel 的写法）。
        self.history_length = int(history_length)
        self.frame_term_dims = tuple(int(d) for d in frame_term_dims)
        self.frame_dim = sum(self.frame_term_dims)
        self.cts_latent_dim = int(latent_dim)
        self.teacher_obs_group = teacher_obs_group
        self.vx_target_group = vx_target_group
        self.role_group = role_group
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
        if self.frame_dim * self.history_length != self.obs_dim:
            raise ValueError(
                f"观测宽度 {self.obs_dim} 与 sum(frame_term_dims)={self.frame_dim} × "
                f"history_length={self.history_length} 不一致"
            )
        for group in (teacher_obs_group, vx_target_group, role_group):
            if group not in set(obs.keys()):
                raise ValueError(f"CTS 需要观测组 {group!r}，实际只有 {list(obs.keys())}")
        self.register_buffer(
            "latest_frame_index",
            _latest_frame_index(self.frame_term_dims, self.history_length),
            persistent=False,
        )
        teacher_dim = int(obs[teacher_obs_group].shape[-1])
        self.teacher_obs_normalizer = (
            EmpiricalNormalization(teacher_dim) if obs_normalization else nn.Identity()
        )
        self.teacher_encoder = MLP(
            teacher_dim, self.cts_latent_dim, teacher_hidden_dims, activation
        )
        # 学生编码器：输出 [z_S 原始值(latent_dim), v̂x]；沿用 estimator 名字以复用 Se3PPO 的独立优化器。
        self.estimator = MLP(
            self.obs_dim, self.cts_latent_dim + 1, estimator_hidden_dims, activation
        )

    def _get_latent_dim(self) -> int:
        """policy MLP 输入：最新一帧 + 隐向量 z + vx。"""
        return self.frame_dim + self.cts_latent_dim + 1

    def _normalized_history(self, obs: TensorDict) -> torch.Tensor:
        return self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))

    def teacher_latent(self, obs: TensorDict) -> torch.Tensor:
        """[L2(z_T), vx_true]，形状 (N, latent_dim + 1)；带梯度（PPO 经教师 env 训练教师编码器）。"""
        z = F.normalize(
            self.teacher_encoder(self.teacher_obs_normalizer(obs[self.teacher_obs_group])), dim=-1
        )
        return torch.cat([z, obs[self.vx_target_group]], dim=-1)

    def student_latent(self, obs: TensorDict, history: torch.Tensor | None = None) -> torch.Tensor:
        """[L2(z_S), v̂x]，形状 (N, latent_dim + 1)；带梯度（只用于重建损失）。"""
        out = self.estimator(self._normalized_history(obs) if history is None else history)
        return torch.cat(
            [F.normalize(out[:, : self.cts_latent_dim], dim=-1), out[:, self.cts_latent_dim :]],
            dim=-1,
        )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """policy 输入 [ô_t, 隐向量]：教师样本用教师隐向量（带梯度），学生样本用学生隐向量（detach）。"""
        history = self._normalized_history(obs)
        teacher = obs[self.role_group] > 0.5
        latent = torch.where(
            teacher, self.teacher_latent(obs), self.student_latent(obs, history).detach()
        )
        return torch.cat([history.index_select(-1, self.latest_frame_index), latent], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        super().update_normalization(obs)
        if self.obs_normalization:
            self.teacher_obs_normalizer.update(obs[self.teacher_obs_group])  # type: ignore[operator]

    def estimator_loss(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """重建损失：学生样本上 [z_S, v̂x] 回归 [sg(z_T), vx_true]，隐向量 MSE 与 vx MSE 等权相加。"""
        student = (obs[self.role_group] <= 0.5).squeeze(-1)
        if not bool(student.any()):
            zero = self.estimator(self._normalized_history(obs)).sum() * 0.0
            return zero, {}
        with torch.no_grad():
            target = self.teacher_latent(obs)[student]
        pred = self.student_latent(obs)[student]
        error = pred - target
        latent_mse = error[:, : self.cts_latent_dim].square().mean()
        vx_mse = error[:, self.cts_latent_dim :].square().mean()
        # 两项各自取均值再相加：若对 33 维拼接整体取均值（官方代码写法），vx 只占 1/33 的权重，
        # 而 vx 是 VxObserver 验证过的主要收益来源，这里保持它与 VxObserver 相同的权重。
        loss = latent_mse + vx_mse
        return loss, {"cts_latent_mse": latent_mse.detach(), "estimator_vx_mse": vx_mse.detach()}

    def estimate_vx(self, obs: TensorDict) -> torch.Tensor:
        """学生路径的 v̂x（诊断用）。"""
        return self.estimator(self._normalized_history(obs))[:, self.cts_latent_dim :]

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("CTSVxObserverModel 只支持 ONNX 导出")

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxCTSStudent(self, verbose)


class _OnnxCTSStudent(nn.Module):
    """ONNX 导出：只走学生路径，输入展平历史、输出确定性动作，与训练时学生 env 的前向逐项相同。"""

    is_recurrent: bool = False

    def __init__(self, model: CTSVxObserverModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.estimator = copy.deepcopy(model.estimator)
        self.mlp = copy.deepcopy(model.mlp)
        self.register_buffer("latest_frame_index", model.latest_frame_index.clone())
        self.latent_dim = model.cts_latent_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history = self.obs_normalizer(x)
        out = self.estimator(history)
        z = F.normalize(out[:, : self.latent_dim], dim=-1)
        latent = torch.cat(
            [history.index_select(-1, self.latest_frame_index), z, out[:, self.latent_dim :]],
            dim=-1,
        )
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


__all__ = ["CTSVxObserverModel"]
