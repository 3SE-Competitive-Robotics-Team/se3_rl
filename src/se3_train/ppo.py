"""带独立 critic 学习率的 PPO。

背景（2026-09-05，D4 诊断）：rsl_rl 的 KL 自适应学习率作用在 actor 与 critic 共用的唯一
optimizer 上。σ 缩小后梯度方向变得一致，KL 规则把 actor 的 LR 压到 1e-5 地板，critic 的 LR 被一起压到
地板：D4 从 3580 轮起 97% 的轮次贴地板，4250 轮后 Loss/value 出现最高 27 的尖峰，critic 跟不上回报变化。
critic 的回归目标与策略信任域无关，不该被同一条规则限速。

做法：`critic_learning_rate` 非 None 时把 optimizer 换成两个 param group（actor / critic）的 Adam，
每次 `step()` 前把 critic group 的 lr 恢复为固定值；KL 规则仍照常改写全部 group 的 lr，因此 actor
行为与原版 PPO 逐位相同。`critic_learning_rate=None` 时不做任何改动，等价于 rsl_rl.PPO。
"""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.algorithms import PPO

from se3_train.amp import AMP


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
    """rsl_rl.PPO 的透明扩展：可选的固定 critic 学习率、可选的 AMP 风格奖励。

    AMP（se3_train.amp，移植自 BioInnov/rsl_rl_bioin 的扩展模块）：`amp_cfg` 非 None 时，
    process_env_step 在写 storage 之前把风格奖励加进 transition.rewards（与 fork 的钩子位置一致：
    在 time-out bootstrap 之前）；update 在 storage 清空之前用 rollout 更新判别器。
    actor/critic 网络与观测组不变，导出的 ONNX 不受影响。
    """

    def __init__(
        self,
        *args: Any,
        critic_learning_rate: float | None = None,
        amp_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.amp: AMP | None = None
        if amp_cfg:
            amp_cfg = dict(amp_cfg)
            step_dt = amp_cfg.pop("step_dt", None)
            if step_dt is None:
                raise ValueError("amp_cfg 缺少 step_dt；应由 Se3PPO.construct_algorithm 从 env 填入")
            self.amp = AMP(
                num_envs=int(self.storage.observations.shape[1]),
                step_dt=float(step_dt),
                obs=self.storage.observations[0],
                cfg=amp_cfg,
                device=str(self.device),
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

    # ---- AMP 钩子 ----
    @staticmethod
    def construct_algorithm(obs: Any, env: Any, cfg: dict, device: str) -> PPO:
        """在 PPO.construct_algorithm 之前把 env 的控制周期填进 amp_cfg。"""
        amp = cfg.get("algorithm", {}).get("amp_cfg")
        if amp:
            unwrapped = getattr(env, "unwrapped", env)
            amp.setdefault("step_dt", float(unwrapped.step_dt))
        return PPO.construct_algorithm(obs, env, cfg, device)

    def process_env_step(self, obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        if self.amp is None:
            super().process_env_step(obs, rewards, dones, extras)
            return
        # 照抄 rsl_rl 5.4.0 PPO.process_env_step，在 bootstrap 之前插入 AMP 钩子（与 fork 一致）。
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        self.amp.process_env_step(obs, self.transition, extras)
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def update(self) -> dict[str, float]:
        amp_metrics = self.amp.individual_update(self.storage) if self.amp is not None else {}
        loss_dict = super().update()  # 末尾会清空 storage，所以判别器更新放在它之前
        loss_dict.update(amp_metrics)
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        if self.amp is not None:
            self.amp.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.amp is not None:
            self.amp.eval()

    def save(self) -> dict:
        saved = super().save()
        if self.amp is not None:
            saved["ext_state_dict"] = {"amp": self.amp.save()}
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        if self.amp is not None and (load_cfg is None or load_cfg.get("ext", True)):
            state = loaded_dict.get("ext_state_dict", {}).get("amp")
            if state is not None:
                self.amp.load(state, strict=strict)
        return resumed


__all__ = ["ActorCriticAdam", "Se3PPO"]
