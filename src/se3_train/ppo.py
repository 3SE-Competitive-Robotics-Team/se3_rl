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

显式 vx 估计器（2026-10-05，见 se3_train.vx_observer）：`estimator_learning_rate` 非 None 时 actor 必须是
VxObserverMLPModel。估计器用独立 Adam，在 PPO 各 epoch 结束后用同一批 rollout 做 MSE 回归（epoch 数、mini-batch
数与 PPO 相同），目标是观测组 `estimator_target_group`（同一时刻的真实机身系 vx）。放在 PPO 之后是为了让 PPO 各
epoch 的 log prob 与 rollout 时用的是同一个估计器，ratio 从 1 开始；rsl_rl 的 storage.clear() 只复位写指针、不清数据，
所以 PPO update 之后仍可读同一批 rollout。估计器参数也在 PPO optimizer 里（它属于 actor），但 v̂x 在 policy 前
detach，PPO 损失对它的梯度恒为 None，Adam 跳过，不会被 PPO 更新。日志 Loss/estimator_vx_mse、Loss/estimator_vx_rmse。
估计器多输出时（如左右弹簧力，见 vx_observer），损失是各列归一化单位下的 MSE 均值，另记各列物理单位的 RMSE
（Loss/estimator_<列名>_rmse）。
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
    """rsl_rl.PPO 的透明扩展：可选的固定 critic 学习率、可选的显式 vx 估计器监督。"""

    def __init__(
        self,
        *args: Any,
        critic_learning_rate: float | None = None,
        estimator_learning_rate: float | None = None,
        estimator_target_group: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._noise_sq_small: list[float] = []
        self._noise_sq_big: list[float] = []
        self.estimator_target_group = estimator_target_group
        self.estimator_optimizer: torch.optim.Adam | None = None
        if estimator_learning_rate is not None:
            if not hasattr(self._raw_actor, "estimate"):
                raise ValueError("estimator_learning_rate 需要 actor 为 VxObserverMLPModel")
            if estimator_target_group is None:
                raise ValueError("estimator_learning_rate 需要同时给 estimator_target_group")
            self.estimator_optimizer = torch.optim.Adam(
                self._raw_actor.estimator.parameters(), lr=float(estimator_learning_rate)
            )
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
        if self.estimator_optimizer is not None:
            loss_dict.update(self._update_estimator())
        return loss_dict

    def _update_estimator(self) -> dict[str, float]:
        """用本轮 rollout 对估计器做 MSE 回归；多卡时梯度跨卡平均，保证各卡估计器一致。"""
        assert self.estimator_optimizer is not None
        actor = self._raw_actor
        params = list(actor.estimator.parameters())
        names = tuple(actor.estimator_output_names)
        mse_sum = torch.zeros((), device=self.device)
        column_sq_sum = torch.zeros(len(names), device=self.device)
        num_updates = 0
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            target = batch.observations[self.estimator_target_group]
            error = actor.estimate(batch.observations) - target
            loss = error.square().mean()
            self.estimator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                grads = torch.cat([p.grad.reshape(-1) for p in params])
                torch.distributed.all_reduce(grads, op=torch.distributed.ReduceOp.SUM)
                grads /= self.gpu_world_size
                offset = 0
                for p in params:
                    p.grad.copy_(grads[offset : offset + p.numel()].view_as(p))
                    offset += p.numel()
            self.estimator_optimizer.step()
            mse_sum += loss.detach()
            column_sq_sum += error.detach().square().mean(dim=0)
            num_updates += 1
        # 梯度置 None，下一轮 PPO 的噪声尺度统计与 reduce_parameters 只看到 PPO 自己的梯度。
        self.estimator_optimizer.zero_grad(set_to_none=True)
        count = max(num_updates, 1)
        column_mse = (column_sq_sum / count).tolist()
        log = {"estimator_mse": float(mse_sum) / count}
        for name, unit, value in zip(names, actor.estimator_output_units, column_mse, strict=True):
            log[f"estimator_{name}_rmse"] = unit * value**0.5
        log["estimator_vx_mse"] = column_mse[0]
        if getattr(actor, "estimates_spring_force", False):
            log.update(
                actor.spring_feedback_diagnostics(
                    self.storage.observations, self.storage.observations[self.estimator_target_group]
                )
            )
        return log

    def save(self) -> dict:
        saved_dict = super().save()
        if self.estimator_optimizer is not None:
            saved_dict["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_optimizer = load_cfg is None or bool(load_cfg.get("optimizer"))
        if (
            self.estimator_optimizer is not None
            and load_optimizer
            and "estimator_optimizer_state_dict" in loaded_dict
        ):
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        return load_iteration


__all__ = ["ActorCriticAdam", "Se3PPO"]
