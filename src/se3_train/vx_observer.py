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

自适应气弹簧前馈（2026-10-06，用户定，`estimate_spring_force=True`）：估计器再输出左右弹簧力
（(F − 额定值) / KNEE_GAS_SPRING_ESTIMATE_UNIT_N），换算成 N 后由 runner 每个 policy tick 写进动作项，经限幅、限速后
替代固定 300 N 做前馈补偿。补偿越准、历史里关于 F 的信号越弱，只看历史的估计器在闭环下不可辨识（F 不同的两台车
若都估准，净弹簧力矩都为 0、历史相同、估计相同，矛盾），所以 actor 观测带上一拍下发值 spring_force_prev（随 16 帧
历史一起进估计器），估计器学"上一拍 + 按残差修正"。spring_force_prev 不进 policy：policy 的单帧输入用
policy_frame_term_mask 去掉它，使相对 VxObserver 只差"补偿改为估计值"这一个变量。
ONNX 额外输出 spring_force [1, 2]（N，限幅限速前的原始估计）。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict

from se3_shared import KNEE_GAS_SPRING_ESTIMATE_UNIT_N
from se3_shared import RobotConfig as SharedRobotConfig

_NOMINAL_KNEE_SPRING_FORCE = SharedRobotConfig().knee_gas_spring_force


def _latest_frame_index(
    frame_term_dims: tuple[int, ...], history_length: int, term_mask: tuple[bool, ...]
) -> torch.Tensor:
    """term-major 历史展平后，被选中 term 最新一帧在展平向量中的下标（按 term 顺序拼成单帧观测）。"""
    index: list[int] = []
    offset = 0
    for dim, keep in zip(frame_term_dims, term_mask, strict=True):
        block = dim * history_length
        if keep:
            index.extend(range(offset + block - dim, offset + block))
        offset += block
    return torch.tensor(index, dtype=torch.long)


def _spring_force_newton(normalized: torch.Tensor) -> torch.Tensor:
    return _NOMINAL_KNEE_SPRING_FORCE + KNEE_GAS_SPRING_ESTIMATE_UNIT_N * normalized


class VxObserverMLPModel(MLPModel):
    """actor：估计器从历史估 vx（可选再估左右弹簧力），policy 只看最新一帧 + detach 的 v̂x。"""

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
        policy_frame_term_mask: tuple[bool, ...] | list[bool] = (),
        estimate_spring_force: bool = False,
    ) -> None:
        # 父类 __init__ 里会调用 _get_latent_dim() 建 policy MLP，所以单帧宽度必须先于 super().__init__ 设好
        # （与 rsl_rl CNNModel 先设 cnn_latent_dim 的写法一致）。
        self.history_length = int(history_length)
        self.frame_term_dims = tuple(int(d) for d in frame_term_dims)
        self.frame_dim = sum(self.frame_term_dims)
        self.policy_frame_term_mask = (
            tuple(bool(m) for m in policy_frame_term_mask)
            if policy_frame_term_mask
            else (True,) * len(self.frame_term_dims)
        )
        if len(self.policy_frame_term_mask) != len(self.frame_term_dims):
            raise ValueError("policy_frame_term_mask 与 frame_term_dims 长度不一致")
        self.policy_frame_dim = sum(
            d
            for d, keep in zip(self.frame_term_dims, self.policy_frame_term_mask, strict=True)
            if keep
        )
        self.estimates_spring_force = bool(estimate_spring_force)
        # 估计器输出：[vx (m/s)]，或 [vx (m/s), F_L, F_R ((F − 额定值) / KNEE_GAS_SPRING_ESTIMATE_UNIT_N)]。
        # estimator_output_units 把各列换算回物理单位，只用于日志。
        if self.estimates_spring_force:
            self.estimator_output_names: tuple[str, ...] = ("vx", "spring_l", "spring_r")
            self.estimator_output_units: tuple[float, ...] = (
                1.0,
                KNEE_GAS_SPRING_ESTIMATE_UNIT_N,
                KNEE_GAS_SPRING_ESTIMATE_UNIT_N,
            )
        else:
            self.estimator_output_names = ("vx",)
            self.estimator_output_units = (1.0,)
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
        if self.estimates_spring_force and (
            self.frame_term_dims[-1] != 2 or self.policy_frame_term_mask[-1]
        ):
            raise ValueError("估计弹簧力时 actor 最后一项必须是 2 维、不进 policy 的 spring_force_prev")
        if self.frame_dim * self.history_length != self.obs_dim:
            raise ValueError(
                f"观测宽度 {self.obs_dim} 与 sum(frame_term_dims)={self.frame_dim} × "
                f"history_length={self.history_length} 不一致"
            )
        self.register_buffer(
            "latest_frame_index",
            _latest_frame_index(
                self.frame_term_dims, self.history_length, self.policy_frame_term_mask
            ),
            persistent=False,
        )
        self.estimator = MLP(
            self.obs_dim, len(self.estimator_output_names), estimator_hidden_dims, activation
        )

    def _get_latent_dim(self) -> int:
        """policy MLP 输入：最新一帧（被选中的 term）+ v̂x。"""
        return self.policy_frame_dim + 1

    def _normalized_history(self, obs: TensorDict) -> torch.Tensor:
        return self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))

    def estimate(self, obs: TensorDict) -> torch.Tensor:
        """估计器前向（带梯度），供 Se3PPO 的 MSE 更新用；形状 (N, len(estimator_output_names))。"""
        return self.estimator(self._normalized_history(obs))

    @torch.no_grad()
    def spring_force_estimate(self, obs: TensorDict) -> torch.Tensor:
        """左右弹簧力估计（N，限幅限速前），形状 (N, 2)；runner 每个 policy tick 写进动作项。"""
        if not self.estimates_spring_force:
            raise RuntimeError("estimate_spring_force=False 的模型没有弹簧力输出")
        return _spring_force_newton(self.estimate(obs)[:, 1:3])

    @torch.no_grad()
    def spring_feedback_diagnostics(self, obs: TensorDict, target: torch.Tensor) -> dict[str, float]:
        """从 rollout 原始观测读下发值诊断（N）：下发值相对真值的误差、每拍变化量。

        spring_force_prev 是 actor 最后一项，展平历史的最后 4 列是它的上一帧与最新帧（均为未归一化观测）。
        """
        history = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        newest = history[..., -2:]
        previous = history[..., -4:-2]
        unit = KNEE_GAS_SPRING_ESTIMATE_UNIT_N
        error = (newest - target[..., 1:3]) * unit
        step = (newest - previous).abs() * unit
        return {
            "spring_applied_rmse": float(error.square().mean().sqrt()),
            "spring_applied_abs_err_p90": float(error.abs().flatten().quantile(0.9)),
            "spring_step_abs_mean": float(step.mean()),
        }

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """policy 输入 [ô_t, sg(v̂x)]：PPO 梯度不回传到估计器。"""
        history = self._normalized_history(obs)
        vx_hat = self.estimator(history)[:, :1].detach()
        return torch.cat([history.index_select(-1, self.latest_frame_index), vx_hat], dim=-1)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("VxObserverMLPModel 只支持 ONNX 导出")

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxVxObserverMLPModel(self, verbose)


class _OnnxVxObserverMLPModel(nn.Module):
    """ONNX 导出：输入展平历史，输出确定性动作（及弹簧力估计）；与训练前向逐项相同（无 detach，推理时无意义）。"""

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
        self.estimates_spring_force = model.estimates_spring_force

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        history = self.obs_normalizer(x)
        estimate = self.estimator(history)
        latent = torch.cat(
            [history.index_select(-1, self.latest_frame_index), estimate[:, :1]], dim=-1
        )
        actions = self.deterministic_output(self.mlp(latent))
        if not self.estimates_spring_force:
            return actions
        return actions, _spring_force_newton(estimate[:, 1:3])

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "spring_force"] if self.estimates_spring_force else ["actions"]


__all__ = ["VxObserverMLPModel"]
