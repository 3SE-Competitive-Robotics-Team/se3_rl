"""世界模型：学习的动力学 + 学习的终止头 + 学习的奖励头（解析观测 / 奖励头见 heads.py）。

移植自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的
`world_model/{dynamics,world_model}.py`；改动：
- 终止头改为在规范化特征 + ξ 上学（SerialLeg 的终止含腿部接触，不是只看根部偏离）；
- 新增学习奖励头，回归接触 / 力矩类奖励项的逐项值（论文 Eq. 1 的 r̂ 项，只对解析不出来的项）；
- ξ 归一化改成按首次采到的全体 env 统计量（DR 是 startup 事件，每 env 常量）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WorldModelCfg
from .state import ACTION_DIM, DELTA_DIM, FEAT_DIM, XI_DIM, StateSpec

COMMAND_DIM = 8
_COMMAND_SCALE = (2.0, 0.25, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0)


def compile_mode_arg(mode: str) -> str | None:
    """配置里的 compile_mode → torch.compile 的 mode 参数。"""
    if mode not in ("default", "reduce-overhead"):
        raise ValueError(f"compile_mode 只支持 default / reduce-overhead，实际 {mode}")
    return None if mode == "default" else mode


def mark_cudagraph_step() -> None:
    """CUDA Graphs（reduce-overhead）下标记新一轮的开始：此前回放的输出可被覆盖。default 模式下是空操作。"""
    torch.compiler.cudagraph_mark_step_begin()


class MLP(nn.Module):
    """SiLU MLP；zero_last 时最后一层零初始化，last_gain 时正交初始化最后一层。"""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        zero_last: bool = False,
        last_gain: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModel(nn.Module):
    """ŝ_{t+1} = integrate(ŝ_t, f(feat(ŝ_t), a_t, ξ))；d̂_{t+1} = σ(term(feat(ŝ_{t+1}), ξ))；
    r̂_learned = rew(feat(ŝ_t), feat(ŝ_{t+1}), a_t, cmd_t, ξ) × 逐项尺度。"""

    def __init__(
        self,
        cfg: WorldModelCfg,
        spec: StateSpec,
        num_learned_rewards: int,
        action_clip: float | None,
        device: torch.device | str,
    ):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.action_clip = action_clip
        self.num_learned = int(num_learned_rewards)
        self.dynamics = MLP(
            FEAT_DIM + ACTION_DIM + XI_DIM, tuple(cfg.hidden_dims), DELTA_DIM, zero_last=True
        )
        self.termination = MLP(
            FEAT_DIM + XI_DIM + COMMAND_DIM, tuple(cfg.termination_hidden_dims), 1
        )
        self.reward = MLP(
            2 * FEAT_DIM + ACTION_DIM + XI_DIM + COMMAND_DIM,
            tuple(cfg.reward_hidden_dims),
            max(self.num_learned, 1),
            zero_last=True,
        )
        self.register_buffer("xi_mean", torch.zeros(XI_DIM))
        self.register_buffer("xi_std", torch.ones(XI_DIM))
        self.register_buffer("xi_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("reward_scale", torch.ones(max(self.num_learned, 1)))
        self.register_buffer("command_scale", torch.tensor(_COMMAND_SCALE))
        self._window_fn = None
        if cfg.precision not in ("tf32", "bf16"):
            raise ValueError(f"world_model.precision 只支持 tf32 / bf16，实际 {cfg.precision}")
        self._bf16 = cfg.precision == "bf16"
        self.to(device)

    # ------------------------------------------------------------------ 归一化

    @torch.no_grad()
    def fit_xi(self, xi: torch.Tensor) -> None:
        """用全体 env 的 ξ 定归一化（只做一次，之后随 checkpoint 保存）。"""
        if bool(self.xi_ready):
            return
        self.xi_mean.copy_(xi.mean(0))
        self.xi_std.copy_(xi.std(0).clamp(min=1.0e-3))
        self.xi_ready.fill_(True)

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_mean) / self.xi_std

    def normalize_command(self, cmd: torch.Tensor) -> torch.Tensor:
        return cmd * self.command_scale

    @torch.no_grad()
    def update_reward_scale(self, targets: torch.Tensor, momentum: float = 0.99) -> None:
        """学习项目标的逐项 RMS（EMA），恒 0 的项落到下限 1e-6，不影响损失。"""
        if self.num_learned == 0:
            return
        rms = targets.reshape(-1, self.num_learned).square().mean(0).sqrt().clamp(min=1.0e-6)
        self.reward_scale.mul_(momentum).add_((1.0 - momentum) * rms)

    def _mlp(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """网络前向；precision=bf16 时在 autocast 里算，输出转回 fp32 再进积分 / 损失。"""
        if not self._bf16:
            return net(x)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            return net(x).float()

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_clip is None:
            return action
        return torch.clamp(action, -self.action_clip, self.action_clip)

    # ------------------------------------------------------------------ 前向

    def step(self, s: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """一步动力学；action 为 ActionManager 语义（未裁剪），模型内按动作项同样裁剪。"""
        a = self.clip_action(action)
        delta = self._mlp(self.dynamics, torch.cat([self.spec.canonical(s), a, xi_norm], dim=-1))
        return self.spec.integrate(s, delta, a)

    def termination_logits(
        self, s_next: torch.Tensor, xi_norm: torch.Tensor, cmd: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([self.spec.canonical(s_next), xi_norm, self.normalize_command(cmd)], dim=-1)
        return self._mlp(self.termination, x).squeeze(-1)

    def learned_rewards_normalized(
        self,
        s: torch.Tensor,
        s_next: torch.Tensor,
        action: torch.Tensor,
        xi_norm: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.spec.canonical(s),
                self.spec.canonical(s_next),
                self.clip_action(action),
                xi_norm,
                self.normalize_command(cmd),
            ],
            dim=-1,
        )
        return self._mlp(self.reward, x)[..., : self.num_learned]

    def learned_rewards(self, s, s_next, action, xi_norm, cmd) -> torch.Tensor:
        """学习项的加权值（weight × 项值，未乘 dt）(…, K)。"""
        out = self.learned_rewards_normalized(s, s_next, action, xi_norm, cmd)
        return out * self.reward_scale[: self.num_learned]

    # ------------------------------------------------------------------ Eq. 1

    def window_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        xi: torch.Tensor,
        done: torch.Tensor,
        cmd: torch.Tensor,
        learned_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """k 步自回归损失。

        states (k+1, n, S) 记录状态；actions (k, n, 6) 执行动作；xi (n, X) 原始 ξ；done (k, n) 到达状态的终止标签；
        cmd (k, n, 8) 各步指令；learned_targets (k, n, K) 学习项的加权值。
        """
        k = actions.shape[0]
        xi_norm = self.normalize_xi(xi)
        inv_scale = 1.0 / self.reward_scale[: self.num_learned]
        window_fn = self._window_fn or self._window
        state_loss, term_loss, reward_loss, first, last, acc = window_fn(
            states, actions, xi_norm, done, cmd, learned_targets, inv_scale
        )
        loss = (
            state_loss / k
            + self.cfg.termination_loss_weight * term_loss / k
            + self.cfg.reward_loss_weight * reward_loss / k
        )
        metrics = {
            "state_loss": (state_loss / k).detach(),
            "termination_bce": (term_loss / k).detach(),
            "reward_mse": (reward_loss / k).detach(),
            "nse_1step": first.detach(),
            "nse_kstep": last.detach(),
            "termination_acc": (acc / k).detach(),
        }
        return loss, metrics

    def _window(self, states, actions, xi_norm, done, cmd, targets, inv_scale):
        """k 步窗口整体（编译时展开成一张图）：返回 (状态损失和, 终止 BCE 和, 奖励 MSE 和, 首步误差, 末步误差, 准确率和)。"""
        s = states[0]
        state_loss = term_loss = reward_loss = acc = torch.zeros((), device=states.device)
        first = last = torch.zeros((), device=states.device)
        for j in range(actions.shape[0]):
            s, sl, tl, rl, nse, ok = self._loss_step(
                s, actions[j], xi_norm, states[j + 1], done[j], cmd[j], targets[j], inv_scale
            )
            state_loss = state_loss + sl
            term_loss = term_loss + tl
            reward_loss = reward_loss + rl
            acc = acc + ok
            if j == 0:
                first = nse
            last = nse
        return state_loss, term_loss, reward_loss, first, last, acc

    def _loss_step(self, s, action, xi_norm, s_true, done, cmd, target, inv_scale):
        """Eq. 1 的一步自回归：返回 (ŝ_{j+1}, 状态损失, 终止 BCE, 奖励 MSE, 归一化误差, 终止准确率)。"""
        s_next = self.step(s, action, xi_norm)
        sq = self.spec.error_vector(s_next, s_true).square()
        logits = self.termination_logits(s_next, xi_norm, cmd)
        bce = F.binary_cross_entropy_with_logits(logits, done)
        if self.num_learned:
            pred = self.learned_rewards_normalized(s, s_next, action, xi_norm, cmd)
            mse = F.mse_loss(pred, target * inv_scale)
        else:
            mse = torch.zeros((), device=s.device)
        nse = sq.detach().mean()
        ok = ((logits.detach() > 0).float() == done).float().mean()
        return s_next, sq.sum(-1).mean(), bce, mse, nse, ok

    def enable_compile(self, enabled: bool, mode: str = "default") -> None:
        """把单步损失编译成一个融合图（固定 batch 形状）。

        不编译时每步是几百个逐元素小算子，2026-11-07 nulltask1 A800 剖析：一次更新 2.2 s 里矩阵乘只占 117 ms，
        其余是 kernel launch 与 Python 开销。复现仓库同样编译了这一步。
        """
        # 整个 k 步窗口编译成一张图：逐步编译时每批要调 k 次前向 + k 次反传的编译函数，每次 CPU 侧
        # autograd / launch 约 1.3–2.4 ms（2026-11-07 剖析：世界模型更新 GPU 176 ms、墙钟 220 ms）
        self._window_fn = (
            torch.compile(self._window, dynamic=False, mode=compile_mode_arg(mode))
            if enabled
            else None
        )

    @torch.no_grad()
    def rollout_error(
        self, states: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        """开环 k 步逐步归一化平方误差 (k,)，按 batch 平均。"""
        xi_norm = self.normalize_xi(xi)
        s = states[0]
        out = []
        for j in range(actions.shape[0]):
            s = self.step(s, actions[j], xi_norm)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)


__all__ = ["COMMAND_DIM", "MLP", "WorldModel"]
