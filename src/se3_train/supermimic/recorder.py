"""把模拟器 transition 写进回放缓存的 mjlab RecorderTerm。

移植自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的 `data/recorder.py`。

钩子时序（mjlab 1.5.3 `ManagerBasedRlEnv.step`）：
- `record_post_reset(ids)`：新 episode 起点，记下 s_0、ξ、指令。
- `record_post_step()`：未 reset 的 env 写一行 (s_t, a_t, a_exec_t, cmd_t, 逐项奖励, …)，再记下 s_{t+1} 作为下一行的状态。
- `record_pre_reset(ids)`：终止 / 超时那一步，先写最后一个 transition 行，再写一行终止行存 s_{T+1}
  与它的终止标签（超时为 0）。此时 qpos / qvel、动作项与执行器滞后状态都还没被 reset 清掉。

干净动作由 runner 在 env.step 之前通过 `set_clean_action` 交给 recorder（env 只看到加噪后的执行动作）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from .replay import ReplayBuffer
from .state import ACTION_DIM, STATE_DIM, XI_DIM, SimStateReader
from .world_model import COMMAND_DIM

RECORDER_NAME = "supermimic"
"""env 配置里 recorder 的名字，runner 按它取回放缓存。"""


class SuperMimicRecorder(RecorderTerm):
    def __init__(self, cfg: SuperMimicRecorderCfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.command_name = str(p.get("command_name", "velocity_height"))
        self.reader = SimStateReader(env, str(p.get("action_name", "delayed_action")))
        n = env.num_envs
        self.num_reward_terms = len(env.reward_manager.active_terms)
        self.buffer = ReplayBuffer(
            n,
            int(p["capacity"]),
            STATE_DIM,
            ACTION_DIM,
            XI_DIM,
            COMMAND_DIM,
            self.num_reward_terms,
            env.device,
        )
        self.prev_s = torch.zeros(n, STATE_DIM, device=env.device)
        self.prev_xi = torch.zeros(n, XI_DIM, device=env.device)
        self.prev_cmd = torch.zeros(n, COMMAND_DIM, device=env.device)
        self.clean_action = torch.zeros(n, ACTION_DIM, device=env.device)
        self.enabled = True
        self._primed = False

    def set_clean_action(self, action: torch.Tensor) -> None:
        self.clean_action.copy_(action)

    def _command(self) -> torch.Tensor:
        cmd = self._env.command_manager.get_command(self.command_name)
        if cmd.shape[-1] != COMMAND_DIM:
            raise RuntimeError(f"SuperMimic 假定 {COMMAND_DIM} 维指令，实际 {cmd.shape[-1]}")
        return cmd

    def _remember(self, ids: torch.Tensor, *, new_episode: bool) -> None:
        self.prev_s[ids] = self.reader.read_state()[ids]
        self.prev_cmd[ids] = self._command()[ids]
        if new_episode:
            # ξ 在 episode 内是常量（DR 是 startup 事件，延迟与轮滞后 τ 在 reset 时重采），只在起点读
            self.prev_xi[ids] = self.reader.read_xi()[ids]

    def prime(self) -> None:
        """runner 在第一次 step 之前调用：env.reset() 已经做完，记下全体 env 的起点。"""
        ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._remember(ids, new_episode=True)
        self._primed = True

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled or env_ids is None or isinstance(env_ids, slice):
            return
        self._remember(env_ids, new_episode=True)

    def _write_transition(self, ids: torch.Tensor) -> None:
        env = self._env
        zeros = torch.zeros(ids.numel(), device=env.device)
        self.buffer.add(
            ids,
            s=self.prev_s[ids],
            a=self.clean_action[ids],
            a_exec=env.action_manager.action[ids],
            xi=self.prev_xi[ids],
            cmd=self.prev_cmd[ids],
            rew=env.reward_manager._step_reward[ids],
            done=zeros,
            ep_step=env.episode_length_buf[ids],
            terminal=torch.zeros(ids.numel(), dtype=torch.bool, device=env.device),
        )

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        if not self.enabled or not self._primed or env_ids.numel() == 0:
            return
        env = self._env
        self._write_transition(env_ids)
        n = env_ids.numel()
        zeros_a = torch.zeros(n, ACTION_DIM, device=env.device)
        self.buffer.add(
            env_ids,
            s=self.reader.read_state()[env_ids],
            a=zeros_a,
            a_exec=zeros_a,
            xi=self.prev_xi[env_ids],
            cmd=self.prev_cmd[env_ids],
            rew=torch.zeros(n, self.num_reward_terms, device=env.device),
            done=env.reset_terminated[env_ids].float(),
            ep_step=env.episode_length_buf[env_ids],
            terminal=torch.ones(n, dtype=torch.bool, device=env.device),
        )
        self.buffer.end_episode(env_ids)

    def record_post_step(self) -> None:
        if not self.enabled or not self._primed:
            return
        ids = (~self._env.reset_buf).nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        self._write_transition(ids)
        self._remember(ids, new_episode=False)


@dataclass(kw_only=True)
class SuperMimicRecorderCfg(RecorderTermCfg):
    func: type = SuperMimicRecorder
    params: dict = field(default_factory=lambda: {"capacity": 1000})


__all__ = ["RECORDER_NAME", "SuperMimicRecorder", "SuperMimicRecorderCfg"]
