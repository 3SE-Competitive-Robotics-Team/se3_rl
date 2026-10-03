"""带独立 critic 学习率的 PPO。

背景（2026-09-05，D4 诊断）：rsl_rl 的 KL 自适应学习率作用在 actor 与 critic 共用的唯一
optimizer 上。σ 缩小后梯度方向变得一致，KL 规则把 actor 的 LR 压到 1e-5 地板，critic 的 LR 被一起压到
地板：D4 从 3580 轮起 97% 的轮次贴地板，4250 轮后 Loss/value 出现最高 27 的尖峰，critic 跟不上回报变化。
critic 的回归目标与策略信任域无关，不该被同一条规则限速。

做法：`critic_learning_rate` 非 None 时把 optimizer 换成两个 param group（actor / critic）的 Adam，
每次 `step()` 前把 critic group 的 lr 恢复为固定值；KL 规则仍照常改写全部 group 的 lr，因此 actor
行为与原版 PPO 逐位相同。`critic_learning_rate=None` 时不做任何改动，等价于 rsl_rl.PPO。

多卡梯度噪声尺度（2026-10-03）：多卡时额外记录 actor 的梯度噪声尺度 B_noise（McCandlish et al. 2018,
"An Empirical Model of Large-Batch Training" 附录 A），只读梯度、不改更新。单卡 mini-batch 梯度（样本数 B_小）
与各卡平均后的梯度（B_大 = 卡数 × B_小）给出两种 batch 下的 |g|²，由
    |G|² ≈ (B_大·|g_大|² − B_小·|g_小|²) / (B_大 − B_小)，tr(Σ) ≈ (|g_小|² − |g_大|²) / (1/B_小 − 1/B_大)
解出真实梯度模长与噪声，B_noise = tr(Σ) / |G|² 是"再加样本仍明显有用"的 batch 量级。两项先按整轮
mini-batch 求均值再相除（单个 mini-batch 的估计噪声大）。日志键 Loss/grad_noise_scale_actor（样本数），
Loss/grad_sq_small_actor、Loss/grad_sq_big_actor 供复核。单卡训练不记录。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed
from rsl_rl.algorithms import PPO


class ActorCriticAdam(torch.optim.Adam):
    """param_groups[0] 为 actor（lr 由 KL 自适应规则改写），param_groups[1] 为 critic（lr 固定）。"""

    def __init__(
        self,
        actor_params,
        critic_params,
        *,
        lr: float,
        critic_lr: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            [
                {"params": list(actor_params)},
                {"params": list(critic_params), "lr": float(critic_lr)},
            ],
            lr=float(lr),
            **kwargs,
        )
        self.critic_lr = float(critic_lr)

    def step(self, closure=None):  # type: ignore[override]
        """每次更新前恢复 critic 的固定 lr，抵消 PPO.update 对全部 group 的改写。"""
        self.param_groups[1]["lr"] = self.critic_lr
        return super().step(closure)


class Se3PPO(PPO):
    """rsl_rl.PPO 的透明扩展：可选的固定 critic 学习率。"""

    def __init__(
        self, *args: Any, critic_learning_rate: float | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._noise_sq_small: list[float] = []
        self._noise_sq_big: list[float] = []
        self.critic_learning_rate = (
            None if critic_learning_rate is None else float(critic_learning_rate)
        )
        if self.critic_learning_rate is None:
            return
        if kwargs.get("optimizer", "adam") != "adam":
            raise ValueError("critic_learning_rate 目前只支持 optimizer='adam'")
        if self.critic_learning_rate <= 0.0:
            raise ValueError(f"critic_learning_rate 必须为正数，收到 {self.critic_learning_rate}")
        self.optimizer = ActorCriticAdam(
            self.actor.parameters(),
            self.critic.parameters(),
            lr=self.learning_rate,
            critic_lr=self.critic_learning_rate,
        )

    def _actor_grad_sq(self) -> torch.Tensor:
        grads = [p.grad.reshape(-1) for p in self.actor.parameters() if p.grad is not None]
        return torch.cat(grads).square().sum()

    def reduce_parameters(self) -> None:
        """多卡梯度平均前后各取一次 actor 梯度模长平方，供 update() 汇总噪声尺度。"""
        sq_small = self._actor_grad_sq()
        torch.distributed.all_reduce(sq_small, op=torch.distributed.ReduceOp.SUM)
        sq_small /= self.gpu_world_size
        super().reduce_parameters()
        self._noise_sq_small.append(float(sq_small))
        self._noise_sq_big.append(float(self._actor_grad_sq()))

    def update(self) -> dict[str, float]:
        """原版 update；多卡时附带本轮 actor 梯度噪声尺度。"""
        b_small = (
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        self._noise_sq_small.clear()
        self._noise_sq_big.clear()
        loss_dict = super().update()
        if self.is_multi_gpu and self._noise_sq_small:
            b_big = b_small * self.gpu_world_size
            sq_small = sum(self._noise_sq_small) / len(self._noise_sq_small)
            sq_big = sum(self._noise_sq_big) / len(self._noise_sq_big)
            grad_sq = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
            trace = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
            loss_dict["grad_sq_small_actor"] = sq_small
            loss_dict["grad_sq_big_actor"] = sq_big
            loss_dict["grad_noise_scale_actor"] = trace / grad_sq if grad_sq > 0.0 else float("nan")
        return loss_dict


__all__ = ["ActorCriticAdam", "Se3PPO"]
