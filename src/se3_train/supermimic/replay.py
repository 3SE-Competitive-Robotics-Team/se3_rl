"""GPU 环形回放缓存，布局 [env, time]。

移植自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的 `data/replay_buffer.py`。

每行是控制步开始时的状态 s_t 以及之后那次 transition：干净动作 a_t、执行动作 a_exec_t（ActionManager 语义，
未裁剪）、指令 cmd_t、逐项奖励（RewardManager._step_reward，weight × 项值）、奖励计算时的 episode 步数。
episode 结束时额外写一行**终止行**存 s_{T+1}（没有从它出发的 transition），`done` 只在终止行上标该状态是否终止，
transition 行一律 0。相对原实现的改动：原实现把终止前最后一个 transition 行也标成了终止，该行状态被当作窗口
中间目标时终止头会拿到错标签；这里只给终止行打标签。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, n, S)
    actions: torch.Tensor  # (k, n, A) 执行动作
    xi: torch.Tensor  # (n, X)
    done: torch.Tensor  # (k, n) 到达状态 1..k 的终止标签
    command: torch.Tensor  # (k, n, C)
    reward_terms: torch.Tensor  # (k, n, T)


@dataclass
class StartBatch:
    state: torch.Tensor  # (n, S)
    xi: torch.Tensor  # (n, X)
    prev_action: torch.Tensor  # (n, A) 上一步执行动作（episode 首步为 0）
    prev_prev_action: torch.Tensor  # (n, A)
    commands: torch.Tensor  # (H+1, n, C) 起点及之后的记录指令，出 episode 后保持最后一帧
    episode_step: torch.Tensor  # (n,) 起点那一步 transition 的奖励计算时 episode 步数


class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        capacity: int,
        state_dim: int,
        action_dim: int,
        xi_dim: int,
        command_dim: int,
        num_reward_terms: int,
        device: torch.device | str,
    ):
        self.E, self.T = int(num_envs), int(capacity)
        self.device = torch.device(device)
        E, T = self.E, self.T

        def f(*shape):
            return torch.zeros(E, T, *shape, device=self.device)

        self.s = f(state_dim)
        self.a = f(action_dim)
        self.a_exec = f(action_dim)
        self.xi = f(xi_dim)
        self.cmd = f(command_dim)
        self.rew = f(num_reward_terms)
        self.done = f()
        self.ep_step = torch.zeros(E, T, dtype=torch.long, device=self.device)
        self.ep_id = torch.full((E, T), -1, dtype=torch.long, device=self.device)
        self.terminal = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ptr = torch.zeros(E, dtype=torch.long, device=self.device)
        self.ep_counter = torch.zeros(E, dtype=torch.long, device=self.device)
        self.rows_written = 0

    # ------------------------------------------------------------------ 写

    @torch.no_grad()
    def add(self, env_ids, *, s, a, a_exec, xi, cmd, rew, done, ep_step, terminal) -> None:
        p = self.ptr[env_ids]
        self.s[env_ids, p] = s
        self.a[env_ids, p] = a
        self.a_exec[env_ids, p] = a_exec
        self.xi[env_ids, p] = xi
        self.cmd[env_ids, p] = cmd
        self.rew[env_ids, p] = rew
        self.done[env_ids, p] = done
        self.ep_step[env_ids, p] = ep_step
        self.ep_id[env_ids, p] = self.ep_counter[env_ids]
        self.terminal[env_ids, p] = terminal
        self.valid[env_ids, p] = True
        self.ptr[env_ids] = (p + 1) % self.T
        self.rows_written += int(env_ids.numel())

    @torch.no_grad()
    def end_episode(self, env_ids) -> None:
        self.ep_counter[env_ids] += 1

    @property
    def size(self) -> int:
        return int(self.valid.sum().item())

    # ------------------------------------------------------------------ 采样

    def _rolled(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """x[:, (t + j) % T]。"""
        return torch.roll(x, shifts=-j, dims=1) if j else x

    def _dist_to_ptr(self) -> torch.Tensor:
        t = torch.arange(self.T, device=self.device)[None, :]
        return (self.ptr[:, None] - t - 1) % self.T  # 0 = 最新一行

    @torch.no_grad()
    def window_mask(self, k: int) -> torch.Tensor:
        """(E, T)：t..t+k 都有效、同一 episode，且 t..t+k−1 都不是终止行。"""
        ok = self.valid & ~self.terminal
        for j in range(1, k + 1):
            v = self._rolled(self.valid, j) & (self._rolled(self.ep_id, j) == self.ep_id)
            if j < k:
                v &= ~self._rolled(self.terminal, j)
            ok &= v
        ok &= self._dist_to_ptr() >= k  # 窗口不能跨写指针
        return ok

    @torch.no_grad()
    def _sample(self, mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = mask.reshape(-1).float()
        if not bool(flat.any()):
            raise RuntimeError("回放缓存里还没有可采样的行")
        idx = torch.multinomial(flat, n, replacement=True)
        return idx // self.T, idx % self.T

    @torch.no_grad()
    def sample_windows(self, n: int, k: int) -> WindowBatch:
        e, t = self._sample(self.window_mask(k), n)
        tt = (t[None, :] + torch.arange(k + 1, device=self.device)[:, None]) % self.T
        ee = e[None, :].expand(k + 1, n)
        return WindowBatch(
            states=self.s[ee, tt],
            actions=self.a_exec[ee[:-1], tt[:-1]],
            xi=self.xi[e, t],
            done=self.done[ee[1:], tt[1:]],
            command=self.cmd[ee[:-1], tt[:-1]],
            reward_terms=self.rew[ee[:-1], tt[:-1]],
        )

    @torch.no_grad()
    def sample_starts(self, n: int, horizon: int) -> StartBatch:
        mask = self.valid & ~self.terminal
        mask &= self._dist_to_ptr() < self.T - 3  # 前两行历史不能是刚被覆盖的
        e, t = self._sample(mask, n)
        same = self.ep_id[e, t]

        def back(j: int) -> torch.Tensor:
            tj = (t - j) % self.T
            ok = self.valid[e, tj] & (self.ep_id[e, tj] == same) & ~self.terminal[e, tj]
            return torch.where(
                ok[:, None], self.a_exec[e, tj], torch.zeros_like(self.a_exec[e, tj])
            )

        # 起点之后的指令：同 episode 的非终止行，超出后保持最后一个有效帧
        offs = torch.arange(horizon + 1, device=self.device)
        tt = (t[None, :] + offs[:, None]) % self.T
        ee = e[None, :].expand(horizon + 1, n)
        ok = (
            self.valid[ee, tt]
            & (self.ep_id[ee, tt] == same[None, :])
            & ~self.terminal[ee, tt]
            & (self._dist_to_ptr()[ee, tt] <= self._dist_to_ptr()[e, t][None, :])
        )
        ok = torch.cumprod(ok.long(), dim=0).bool()  # 第一次失效之后全部失效
        last = ok.long().sum(0) - 1  # ≥ 0（起点自身有效）
        hold = torch.minimum(offs[:, None].expand(-1, n), last[None, :])
        t_cmd = (t[None, :] + hold) % self.T
        return StartBatch(
            state=self.s[e, t],
            xi=self.xi[e, t],
            prev_action=back(1),
            prev_prev_action=back(2),
            commands=self.cmd[ee, t_cmd],
            episode_step=self.ep_step[e, t],
        )


__all__ = ["ReplayBuffer", "StartBatch", "WindowBatch"]
