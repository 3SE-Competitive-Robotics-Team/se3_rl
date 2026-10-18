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
估计器带隐向量时（latent_dim > 0）隐向量不 detach，PPO 梯度经它进入估计器、由 PPO optimizer 一起更新；MSE 只监督 v̂x 一列。
另记 Loss/estimator_latent_std（隐向量各维 batch 标准差均值，看是否塌缩）。
actor 自带 estimator_loss(observations) 时（CTS，见 se3_train.cts_observer）改用它给出的损失（学生编码器的隐向量重建），
并按 cts_role 分别记教师 / 学生 env 的平均单步奖励（Loss/cts_reward_teacher、Loss/cts_reward_student）：
训练端其余指标被占多数的教师 env 主导，学生 env 才代表部署路径。

换学生编码器续训（2026-10-09，单片机部署版 128/128）：`load` 发现 checkpoint 的 estimator 形状与当前不一致时，
只丢弃 `estimator.*` 参数与估计器独立优化器状态，其余（教师编码器、policy、critic、PPO 优化器）照常加载。
PPO 优化器能原样加载：估计器参数虽在 actor 里，但从未收到 PPO 梯度，Adam 里没有它们的状态，参数个数也不变。
`estimator_warmup_iterations` > 0 时本进程前若干轮跳过 PPO 更新、只训练估计器（日志 Loss/estimator_warmup = 1），
避免随机初始化的学生编码器给 25% 学生 env 的垃圾隐向量把共享 policy 带偏。
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
        estimator_warmup_iterations: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.estimator_warmup_remaining = int(estimator_warmup_iterations)
        self._noise_sq_small: list[float] = []
        self._noise_sq_big: list[float] = []
        self.estimator_target_group = estimator_target_group
        self.estimator_optimizer: torch.optim.Adam | None = None
        if estimator_learning_rate is not None:
            if not hasattr(self._raw_actor, "estimate_vx"):
                raise ValueError("estimator_learning_rate 需要 actor 为 VxObserverMLPModel")
            if estimator_target_group is None:
                raise ValueError("estimator_learning_rate 需要同时给 estimator_target_group")
            self.estimator_optimizer = torch.optim.Adam(
                self._raw_actor.estimator.parameters(), lr=float(estimator_learning_rate)
            )
        if self.estimator_warmup_remaining > 0 and self.estimator_optimizer is None:
            raise ValueError("estimator_warmup_iterations 需要同时给 estimator_learning_rate")
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
        """原版 update；多卡时附带本轮 actor 梯度噪声尺度。预热期只训练估计器。"""
        if self.estimator_warmup_remaining > 0:
            self.estimator_warmup_remaining -= 1
            loss_dict = self._update_estimator()
            loss_dict["estimator_warmup"] = 1.0
            self.storage.clear()
            return loss_dict
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
        """用本轮 rollout 对 vx 估计器做 MSE 回归；多卡时梯度跨卡平均，保证各卡估计器一致。"""
        assert self.estimator_optimizer is not None
        actor = self._raw_actor
        params = list(actor.estimator.parameters())
        custom_loss = getattr(actor, "estimator_loss", None)
        mse_sum = torch.zeros((), device=self.device)
        extra_sums: dict[str, torch.Tensor] = {}
        num_updates = 0
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            if custom_loss is not None:
                loss, extra = custom_loss(batch.observations)
                for key, value in extra.items():
                    extra_sums[key] = extra_sums.get(key, 0.0) + value
            else:
                target = batch.observations[self.estimator_target_group]
                loss = torch.nn.functional.mse_loss(actor.estimate_vx(batch.observations), target)
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
            num_updates += 1
        # 梯度置 None，下一轮 PPO 的噪声尺度统计与 reduce_parameters 只看到 PPO 自己的梯度。
        self.estimator_optimizer.zero_grad(set_to_none=True)
        mse = float(mse_sum) / max(num_updates, 1)
        log = {"estimator_vx_mse": mse, "estimator_vx_rmse": mse**0.5}
        if custom_loss is not None:
            log = {"estimator_mse": mse}
            for key, value in extra_sums.items():
                log[key] = float(value) / max(num_updates, 1)
            if "estimator_vx_mse" in log:
                log["estimator_vx_rmse"] = log["estimator_vx_mse"] ** 0.5
        role_group = getattr(actor, "role_group", None)
        if role_group is not None:
            teacher = self.storage.observations[role_group] > 0.5
            rewards = self.storage.rewards
            log["cts_reward_teacher"] = float(rewards[teacher].mean())
            log["cts_reward_student"] = float(rewards[~teacher].mean())
        if getattr(actor, "latent_dim", 0) > 0:
            log["estimator_latent_std"] = actor.latent_std(batch.observations)
        return log

    def save(self) -> dict:
        saved_dict = super().save()
        if self.estimator_optimizer is not None:
            saved_dict["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return saved_dict

    def _estimator_shape_changed(self, actor_state: dict) -> bool:
        """checkpoint 的 estimator.* 与当前 actor 的形状或键集合是否不同（换学生编码器续训）。"""
        if self.estimator_optimizer is None:
            return False
        current = {
            k: v.shape
            for k, v in self._raw_actor.state_dict().items()
            if k.startswith("estimator.")
        }
        saved = {k: v.shape for k, v in actor_state.items() if k.startswith("estimator.")}
        return current != saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
        reset_estimator = bool(load_cfg.get("actor")) and self._estimator_shape_changed(
            loaded_dict["actor_state_dict"]
        )
        if reset_estimator:
            state = {
                k: v
                for k, v in loaded_dict["actor_state_dict"].items()
                if not k.startswith("estimator.")
            }
            missing, unexpected = self._raw_actor.load_state_dict(state, strict=False)
            bad = [k for k in missing if not k.startswith("estimator.")] + list(unexpected)
            if bad:
                raise RuntimeError(f"换学生编码器续训只允许 estimator.* 缺失，实际还缺 / 多：{bad}")
            print(
                "[Se3PPO] checkpoint 的学生编码器形状与当前不同：estimator.* 随机初始化，其余照常加载"
            )
            load_cfg = {**load_cfg, "actor": False}
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_optimizer = bool(load_cfg.get("optimizer"))
        if (
            self.estimator_optimizer is not None
            and load_optimizer
            and not reset_estimator
            and "estimator_optimizer_state_dict" in loaded_dict
        ):
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        return load_iteration


__all__ = ["ActorCriticAdam", "Se3PPO"]
