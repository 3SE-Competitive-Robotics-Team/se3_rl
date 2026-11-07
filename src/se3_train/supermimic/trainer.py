"""世界模型拟合（论文 Eq. 1）与穿过冻结世界模型的短视野 actor–critic（§3.3）。

移植自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的
`world_model/wm_trainer.py`、`imagination/policy_trainer.py`。第 1 阶段单卡，没有梯度 allreduce。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .config import ImaginationCfg, WorldModelCfg
from .heads import ObservationHead, RewardHeads
from .imagination import gae_targets, imagine, policy_objective
from .replay import ReplayBuffer
from .state import FEAT_DIM, XI_DIM
from .world_model import COMMAND_DIM, MLP, WorldModel


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, cfg: WorldModelCfg, learned_index: list[int]):
        self.wm = wm
        self.cfg = cfg
        self.learned_index = list(learned_index)
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.learning_rate)

    def set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = lr

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            targets = w.reward_terms[..., self.learned_index]
            self.wm.update_reward_scale(targets)
            loss, metrics = self.wm.window_loss(
                w.states, w.actions, w.xi, w.done, w.command, targets
            )
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.wm.parameters(), self.cfg.grad_clip_norm
            )
            self.opt.step()
            metrics["loss"] = loss.detach()
            metrics["grad_norm"] = grad_norm.detach()
            for key, value in metrics.items():
                agg[key] = agg.get(key, 0.0) + value
        # 每次更新只同步一次主机
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024) -> torch.Tensor:
        """缓存窗口上的开环 k 步误差 (k,)。"""
        w = buffer.sample_windows(n, self.cfg.window)
        return self.wm.rollout_error(w.states, w.actions, w.xi)


class Critic(nn.Module):
    """V(feat(s), cmd, ξ)，只在想象 transition 上拟合。"""

    def __init__(self, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.mlp = MLP(FEAT_DIM + COMMAND_DIM + XI_DIM, tuple(hidden_dims), 1)

    def forward(self, feat: torch.Tensor, cmd: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, cmd, xi_norm], dim=-1)).squeeze(-1)


class ActorCriticTrainer:
    def __init__(
        self,
        wm: WorldModel,
        obs_head: ObservationHead,
        reward_heads: RewardHeads,
        actor: nn.Module,
        policy_fn: Callable[[torch.Tensor], torch.Tensor],
        critic: Critic,
        cfg: ImaginationCfg,
    ):
        self.wm, self.obs_head, self.reward_heads = wm, obs_head, reward_heads
        self.actor, self.policy_fn, self.critic, self.cfg = actor, policy_fn, critic, cfg
        self.actor_opt = torch.optim.Adam(
            actor.parameters(), lr=cfg.actor_learning_rate, betas=tuple(cfg.actor_betas)
        )
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_learning_rate)

    def set_lr(self, actor_lr: float, critic_lr: float) -> None:
        for group in self.actor_opt.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_opt.param_groups:
            group["lr"] = critic_lr

    def _critic_inputs(self, states: torch.Tensor, commands: torch.Tensor):
        return self.wm.spec.canonical(states), self.wm.normalize_command(commands)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.horizon)

        # ---- actor：BPTT 穿过冻结的世界模型 -----------------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            ro = imagine(
                self.wm,
                self.obs_head,
                self.reward_heads,
                self.policy_fn,
                start,
                cfg.horizon,
                observation_noise=cfg.observation_noise,
            )
            feat, cmd = self._critic_inputs(ro.states, ro.commands)
            v_terminal = self.critic(feat[-1], cmd[-1], ro.xi_norm)
            objective = policy_objective(ro, v_terminal, cfg.gamma)
            loss = -objective / cfg.horizon
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), cfg.actor_grad_clip_norm
            )
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), cfg.actor_grad_clip_value)
            self.actor_opt.step()
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": objective.detach(),
            "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
            "imag_reward_mean": ro.rewards.mean().detach(),
            "imag_survival_H": ro.survival[-1].mean().detach(),
            "imag_done_mean": ro.done_prob[1:].mean().detach(),
            "actor_grad_norm": grad_norm.detach(),
            "action_abs_mean": ro.actions.abs().mean().detach(),
        }

        # ---- critic：拟合本 rollout 的 GAE 目标（目标固定）--------------------------------
        with torch.no_grad():
            feat_d, cmd_d = feat.detach(), cmd.detach()
            h1, n, _ = feat_d.shape
            xi_rep = ro.xi_norm[None].expand(h1, n, -1)
            values = self.critic(feat_d, cmd_d, xi_rep)
            targets = gae_targets(
                ro.rewards.detach(), ro.done_prob[1:].detach(), values, cfg.gamma, cfg.gae_lambda
            )
            x_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
            x_cmd = cmd_d[:-1].reshape(-1, cmd_d.shape[-1])
            x_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
            y = targets.reshape(-1)
        total = y.shape[0]
        mb = total // cfg.critic_minibatches
        critic_loss = torch.zeros((), device=y.device)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(total, device=y.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                pred = self.critic(x_feat[idx], x_cmd[idx], x_xi[idx])
                closs = torch.nn.functional.mse_loss(pred, y[idx])
                self.critic_opt.zero_grad(set_to_none=True)
                closs.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip_norm)
                self.critic_opt.step()
                critic_loss = closs.detach()
        metrics["critic_loss"] = critic_loss
        metrics["critic_value_mean"] = values.mean()
        metrics["critic_target_mean"] = targets.mean()
        out = {k: float(v) for k, v in metrics.items()}
        for name, value in ro.analytic_means.items():
            out[f"imag_r/{name}"] = value
        for name, value in zip(
            self.reward_heads.learned_names, ro.learned_means.tolist(), strict=True
        ):
            out[f"imag_r/{name}"] = value
        return out


__all__ = ["ActorCriticTrainer", "Critic", "WorldModelTrainer"]
