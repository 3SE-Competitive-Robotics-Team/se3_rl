"""穿过冻结世界模型的想象 rollout，以及策略目标（论文 Eq. 2）与 critic 的 GAE 目标（Eq. 3）。

移植自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的
`imagination/{rollout,objective}.py`；动作跟踪目标换成本仓库的速度指令，观测与奖励走 heads.py 的解析头 +
世界模型的学习头。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .heads import HeadContext, ObservationHead, RewardHeads
from .replay import StartBatch
from .world_model import WorldModel


@dataclass
class ImaginedRollout:
    states: torch.Tensor  # (H+1, n, S)
    actions: torch.Tensor  # (H, n, A)
    rewards: torch.Tensor  # (H, n) r_{t+1}（已乘 dt）
    done_prob: torch.Tensor  # (H+1, n) 到达状态的预测终止概率（d_0 = 0）
    survival: torch.Tensor  # (H+1, n) 软存活权重 c_t
    commands: torch.Tensor  # (H+1, n, C)
    xi_norm: torch.Tensor  # (n, X)
    analytic_means: dict[str, float]  # 各解析项在 rollout 上的均值（诊断）
    learned_means: torch.Tensor  # (K,) 各学习项的均值（诊断）


def imagine(
    wm: WorldModel,
    obs_head: ObservationHead,
    reward_heads: RewardHeads,
    policy: Callable[[torch.Tensor], torch.Tensor],
    start: StartBatch,
    horizon: int,
    observation_noise: bool = True,
) -> ImaginedRollout:
    """从缓存起点出发，用策略在世界模型里滚 H 步；梯度经世界模型雅可比回传到策略。"""
    n = start.state.shape[0]
    device = start.state.device
    xi_norm = wm.normalize_xi(start.xi)
    s = start.state
    a_prev = start.prev_action
    a_prev2 = start.prev_prev_action
    c = torch.ones(n, device=device)
    states, actions, rewards = [s], [], []
    dones, survivals = [torch.zeros(n, device=device)], [c]
    analytic_sum: dict[str, torch.Tensor] = {}
    learned_sum = torch.zeros(max(wm.num_learned, 1), device=device)

    for t in range(horizon):
        cmd = start.commands[t]
        obs_ctx = HeadContext(
            wm.spec, s, a_prev, a_prev, a_prev, cmd, start.xi, start.episode_step + t
        )
        obs = obs_head.clean(obs_ctx)
        if observation_noise:
            obs = obs_head.add_noise(obs)
        a = policy(obs)
        s_next = wm.step(s, a, xi_norm)
        d = torch.sigmoid(wm.termination_logits(s_next, xi_norm, cmd))
        learned = wm.learned_rewards(s, s_next, a, xi_norm, cmd)
        rew_ctx = HeadContext(
            wm.spec, s_next, a, a_prev, a_prev2, cmd, start.xi, start.episode_step + t
        )
        analytic = reward_heads.analytic_terms(rew_ctx)
        r = learned.sum(-1) + reward_heads.fall_weight * d
        for name, value in analytic.items():
            r = r + value
            analytic_sum[name] = analytic_sum.get(name, 0.0) + value.detach().mean()
        r = r * reward_heads.step_dt
        learned_sum = learned_sum + learned.detach().reshape(n, -1).mean(0)
        c = c * (1.0 - d)

        a_prev2, a_prev, s = a_prev, a, s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        survivals.append(c)

    return ImaginedRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        done_prob=torch.stack(dones),
        survival=torch.stack(survivals),
        commands=start.commands,
        xi_norm=xi_norm,
        analytic_means={k: float(v) / horizon for k, v in analytic_sum.items()},
        learned_means=learned_sum[: wm.num_learned] / horizon,
    )


def policy_objective(
    ro: ImaginedRollout, terminal_values: torch.Tensor | None, gamma: float
) -> torch.Tensor:
    """Ĵ = mean_n [ Σ_t γ^t c_t r_{t+1} + γ^H c_H V_H ]（V_H 对 s_H 保留梯度，SHAC 式）。"""
    horizon = ro.rewards.shape[0]
    disc = gamma ** torch.arange(horizon, device=ro.rewards.device, dtype=ro.rewards.dtype)
    objective = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        objective = objective + (gamma**horizon) * ro.survival[-1] * terminal_values
    return objective.mean()


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor,
    done_next: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """R_t = r_{t+1} + γ(1 − d_{t+1}) [(1 − λ) V_{t+1} + λ R_{t+1}]，R_H = V_H。返回 (H, n)。"""
    horizon = rewards.shape[0]
    ret = values[horizon].clone()
    out = torch.empty_like(rewards)
    for t in range(horizon - 1, -1, -1):
        ret = rewards[t] + gamma * (1.0 - done_next[t]) * ((1.0 - lam) * values[t + 1] + lam * ret)
        out[t] = ret
    return out


__all__ = ["ImaginedRollout", "gae_targets", "imagine", "policy_objective"]
