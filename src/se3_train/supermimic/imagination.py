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
from .world_model import WorldModel, compile_mode_arg, mark_cudagraph_step


@dataclass
class ImaginedRollout:
    states: torch.Tensor  # (H+1, n, S)
    actions: torch.Tensor  # (H, n, A)
    rewards: torch.Tensor  # (H, n) r_{t+1}（已乘 dt）
    done_prob: torch.Tensor  # (H+1, n) 到达状态的预测终止概率（d_0 = 0）
    survival: torch.Tensor  # (H+1, n) 软存活权重 c_t
    commands: torch.Tensor  # (H+1, n, C)
    xi_norm: torch.Tensor  # (n, X)
    analytic_means: (
        torch.Tensor
    )  # (T_a,) 各解析项在 rollout 上的均值（诊断，顺序同 analytic_names）
    learned_means: torch.Tensor  # (K,) 各学习项的均值（诊断）


class ImaginationStep:
    """想象中的一步：观测头 → 策略 → 世界模型一步 → 终止概率 → 学习 / 解析奖励，整体编译成一张图。

    `step(s, a_prev, a_prev2, cmd, xi, xi_norm, ep, noise)` 返回
    (a_t, s_{t+1}, d_{t+1}, r_{t+1}（已乘 dt）, 解析项 (T_a, n), 学习项 (n, K))；noise 是加到观测上的噪声（不加噪时传 0）。
    策略必须是纯张量函数（runner 直接调 actor 的 obs 归一化与 MLP，不经 TensorDict），这样才能和前后两段放进同一张图。
    不编译时每步上千个逐元素小算子；拆成多张图时每次调用 CPU 侧约 1.3–2.4 ms（见 WorldModel.enable_compile 注释）。
    """

    def __init__(
        self,
        wm: WorldModel,
        obs_head: ObservationHead,
        reward_heads: RewardHeads,
        policy: Callable[[torch.Tensor], torch.Tensor],
        compile: bool,
        compile_mode: str = "default",
    ):
        self.wm, self.obs_head, self.reward_heads, self.policy = wm, obs_head, reward_heads, policy
        self.analytic_names = list(reward_heads.analytic_names)
        self.step = (
            torch.compile(self._step, dynamic=False, mode=compile_mode_arg(compile_mode))
            if compile
            else self._step
        )

    def _step(self, s, a_prev, a_prev2, cmd, xi, xi_norm, ep, noise):
        wm, heads = self.wm, self.reward_heads
        obs = self.obs_head.clean(HeadContext(wm.spec, s, a_prev, a_prev, a_prev, cmd, xi, ep))
        a = self.policy(obs + noise)
        s_next = wm.step(s, a, xi_norm)
        d = torch.sigmoid(wm.termination_logits(s_next, xi_norm, cmd))
        learned = wm.learned_rewards(s, s_next, a, xi_norm, cmd)
        analytic = heads.analytic_terms(
            HeadContext(wm.spec, s_next, a, a_prev, a_prev2, cmd, xi, ep)
        )
        terms = torch.stack([analytic[name] for name in self.analytic_names], dim=0)
        r = (learned.sum(-1) + heads.fall_weight * d + terms.sum(0)) * heads.step_dt
        return a, s_next, d, r, terms, learned


def imagine(
    step: ImaginationStep,
    start: StartBatch,
    horizon: int,
    observation_noise: bool = True,
) -> ImaginedRollout:
    """从缓存起点出发，用策略在世界模型里滚 H 步；梯度经世界模型雅可比回传到策略。"""
    wm = step.wm
    n = start.state.shape[0]
    device = start.state.device
    xi_norm = wm.normalize_xi(start.xi)
    s = start.state
    a_prev = start.prev_action
    a_prev2 = start.prev_prev_action
    c = torch.ones(n, device=device)
    states, actions, rewards = [s], [], []
    dones, survivals = [torch.zeros(n, device=device)], [c]
    analytic_sum = torch.zeros(len(step.analytic_names), device=device)
    learned_sum = torch.zeros(max(wm.num_learned, 1), device=device)
    zero_noise = torch.zeros(n, step.obs_head.dim, device=device)
    mark_cudagraph_step()  # 新一次 rollout：上一次回放的输出不再需要

    for t in range(horizon):
        noise = step.obs_head.sample_noise(zero_noise) if observation_noise else zero_noise
        a, s_next, d, r, terms, learned = step.step(
            s, a_prev, a_prev2, start.commands[t], start.xi, xi_norm, start.episode_step + t, noise
        )
        analytic_sum = analytic_sum + terms.detach().mean(-1)
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
        analytic_means=analytic_sum / horizon,
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


__all__ = ["ImaginationStep", "ImaginedRollout", "gae_targets", "imagine", "policy_objective"]
