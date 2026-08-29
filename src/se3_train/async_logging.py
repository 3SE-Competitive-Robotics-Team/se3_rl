"""SE3 训练的主机侧异步日志累积工具。"""

from __future__ import annotations

import os
import statistics
from collections import deque
from time import perf_counter
from typing import Any

import torch


def async_host_logger_enabled() -> bool:
    """读取是否启用主机侧延迟日志累积。"""
    raw = os.environ.get("SE3_ASYNC_HOST_LOGGER")
    if raw is None:
        return True
    return raw.lower() not in {"0", "false", "no", "off"}


def expand_episode_log_buffers(logger: Any, min_size: int) -> int:
    """把 RSL-RL 的短 episode 窗口扩到至少覆盖一轮环境池。"""
    target_size = max(100, int(min_size))
    for name in ("rewbuffer", "lenbuffer", "erewbuffer", "irewbuffer"):
        buffer = getattr(logger, name, None)
        if isinstance(buffer, deque) and (buffer.maxlen or 0) < target_size:
            setattr(logger, name, deque(buffer, maxlen=target_size))
    return target_size


class Se3AsyncHostLogger:
    """把 episode reward/length 的 CUDA 到 CPU 传输推迟到迭代末尾。

    RSL-RL 默认在每个 env step 中对 done env 执行 `.cpu().numpy()`，这会在
    rollout collection 内制造主机同步点。本类保持 PPO 存储与梯度更新不变，只
    把纯日志缓冲改成 GPU 侧累积、迭代末尾批量搬运。
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._enabled = logger.writer is not None
        self._reward_batches: list[torch.Tensor] = []
        self._length_batches: list[torch.Tensor] = []
        self._extrinsic_reward_batches: list[torch.Tensor] = []
        self._intrinsic_reward_batches: list[torch.Tensor] = []

    @property
    def enabled(self) -> bool:
        """返回当前 logger 是否真的会写入日志。"""
        return self._enabled

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        """记录一步日志信息，但不在 rollout 内做 CPU 拷贝。"""
        if not self._enabled:
            return

        if "episode" in extras:
            self._logger.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self._logger.ep_extras.append(extras["log"])

        reward_vec = rewards.reshape(-1)
        done_mask = dones.reshape(-1) > 0

        if intrinsic_rewards is not None:
            intrinsic_vec = intrinsic_rewards.reshape(-1)
            self._logger.cur_ereward_sum += reward_vec
            self._logger.cur_ireward_sum += intrinsic_vec
            self._logger.cur_reward_sum += reward_vec + intrinsic_vec
        else:
            self._logger.cur_reward_sum += reward_vec
        self._logger.cur_episode_length += 1

        self._reward_batches.append(self._logger.cur_reward_sum[done_mask].detach().clone())
        self._length_batches.append(self._logger.cur_episode_length[done_mask].detach().clone())
        if intrinsic_rewards is not None:
            self._extrinsic_reward_batches.append(
                self._logger.cur_ereward_sum[done_mask].detach().clone()
            )
            self._intrinsic_reward_batches.append(
                self._logger.cur_ireward_sum[done_mask].detach().clone()
            )

        self._logger.cur_reward_sum[done_mask] = 0
        self._logger.cur_episode_length[done_mask] = 0
        if intrinsic_rewards is not None:
            self._logger.cur_ereward_sum[done_mask] = 0
            self._logger.cur_ireward_sum[done_mask] = 0

    def flush(self) -> float:
        """把本轮完成 episode 的统计值批量写入 RSL-RL logger 缓冲区。"""
        if not self._enabled:
            return 0.0

        start = perf_counter()
        self._extend_buffer(self._logger.rewbuffer, self._reward_batches)
        self._extend_buffer(self._logger.lenbuffer, self._length_batches)
        if self._extrinsic_reward_batches:
            self._extend_buffer(self._logger.erewbuffer, self._extrinsic_reward_batches)
            self._extend_buffer(self._logger.irewbuffer, self._intrinsic_reward_batches)
        return perf_counter() - start

    @staticmethod
    def _extend_buffer(buffer: Any, batches: list[torch.Tensor]) -> None:
        if not batches:
            return
        values = torch.cat([batch.reshape(-1) for batch in batches])
        finite = torch.isfinite(values)
        if finite.any():
            buffer.extend(values[finite].cpu().numpy().tolist())
        batches.clear()


class Se3RolloutMetricsLogger:
    """记录不受 RSL-RL 最近 100 个 episode 缓冲影响的训练指标。"""

    def __init__(self, logger: Any, env: Any) -> None:
        self._logger = logger
        self._env = env.unwrapped
        self._enabled = logger.writer is not None
        self._rollout_reward_sum = torch.zeros((), device=env.device)
        self._rollout_reward_count = 0
        self._episode_reward_sum = torch.zeros(env.num_envs, device=env.device)
        self._episode_length = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        raw_group_ids = getattr(self._env, "env_group_ids", None)
        self._group_ids = raw_group_ids if isinstance(raw_group_ids, torch.Tensor) else None
        self._group_names = self._resolve_group_names()
        self._reward_batches: dict[int, list[torch.Tensor]] = {
            group_id: [] for group_id in self._group_names
        }
        self._length_batches: dict[int, list[torch.Tensor]] = {
            group_id: [] for group_id in self._group_names
        }
        self._termination_counts: dict[int, dict[str, torch.Tensor]] = {
            group_id: {} for group_id in self._group_names
        }
        self._reward_windows: dict[int, deque[float]] = {}
        self._length_windows: dict[int, deque[float]] = {}
        if self._group_ids is not None:
            group_counts = torch.bincount(
                self._group_ids, minlength=max(self._group_names, default=-1) + 1
            ).cpu()
            for group_id in self._group_names:
                window_size = max(1, int(group_counts[group_id].item()))
                self._reward_windows[group_id] = deque(maxlen=window_size)
                self._length_windows[group_id] = deque(maxlen=window_size)

    def _resolve_group_names(self) -> dict[int, str]:
        """统一环境保存的 tuple 或 dict 分组名称。"""
        if self._group_ids is None:
            return {}
        raw_names = getattr(self._env, "env_group_names", None)
        if isinstance(raw_names, dict):
            return {int(group_id): str(name) for group_id, name in raw_names.items()}
        if isinstance(raw_names, (tuple, list)):
            return {group_id: str(name) for group_id, name in enumerate(raw_names)}
        return {
            int(group_id): f"group_{int(group_id)}"
            for group_id in torch.unique(self._group_ids).cpu().tolist()
        }

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        """在 GPU 上累积 rollout 与分组 episode 统计。"""
        if not self._enabled:
            return

        reward_vec = rewards.reshape(-1)
        if intrinsic_rewards is not None:
            reward_vec = reward_vec + intrinsic_rewards.reshape(-1)
        done_mask = dones.reshape(-1) > 0

        self._rollout_reward_sum += reward_vec.sum()
        self._rollout_reward_count += reward_vec.numel()
        self._episode_reward_sum += reward_vec
        self._episode_length += 1

        if self._group_ids is not None:
            termination_manager = getattr(self._env, "termination_manager", None)
            for group_id in self._group_names:
                completed = done_mask & (self._group_ids == group_id)
                self._reward_batches[group_id].append(
                    self._episode_reward_sum[completed].detach().clone()
                )
                self._length_batches[group_id].append(
                    self._episode_length[completed].detach().clone()
                )
                if termination_manager is not None:
                    group_counts = self._termination_counts[group_id]
                    for term_name in termination_manager.active_terms:
                        term_done = termination_manager.get_term(term_name)
                        count = torch.count_nonzero(completed & term_done).detach()
                        group_counts[term_name] = group_counts.get(term_name, 0) + count

        self._episode_reward_sum[done_mask] = 0
        self._episode_length[done_mask] = 0

    def log(self, iteration: int) -> None:
        """在迭代末写入 rollout 均值和按组 episode 指标。"""
        writer = self._logger.writer
        if not self._enabled or writer is None:
            return

        if self._rollout_reward_count > 0:
            rollout_mean = self._rollout_reward_sum / self._rollout_reward_count
            writer.add_scalar("Train/rollout_mean_reward_per_step", rollout_mean.item(), iteration)
        self._rollout_reward_sum.zero_()
        self._rollout_reward_count = 0

        for group_id, group_name in self._group_names.items():
            rewards = self._finite_values(self._reward_batches[group_id])
            lengths = self._finite_values(self._length_batches[group_id])
            completed_count = len(rewards)
            writer.add_scalar(f"Episode_Count_Group/{group_name}", completed_count, iteration)
            if completed_count == 0:
                self._clear_termination_counts(group_id)
                continue

            reward_window = self._reward_windows[group_id]
            length_window = self._length_windows[group_id]
            reward_window.extend(rewards)
            length_window.extend(lengths)

            writer.add_scalar(
                f"Episode_Reward_Group/{group_name}/mean_reward",
                statistics.fmean(rewards),
                iteration,
            )
            writer.add_scalar(
                f"Episode_Length_Group/{group_name}",
                statistics.fmean(lengths),
                iteration,
            )
            writer.add_scalar(
                f"Train_Group/{group_name}/mean_episode_return",
                statistics.fmean(reward_window),
                iteration,
            )
            writer.add_scalar(
                f"Train_Group/{group_name}/mean_episode_length",
                statistics.fmean(length_window),
                iteration,
            )

            for term_name, count in self._termination_counts[group_id].items():
                writer.add_scalar(
                    f"Episode_Termination_Group/{group_name}/{term_name}_rate",
                    count.item() / completed_count,
                    iteration,
                )
            self._clear_termination_counts(group_id)

    @staticmethod
    def _finite_values(batches: list[torch.Tensor]) -> list[float]:
        """批量搬运有限值到主机，并清空本轮缓存。"""
        if not batches:
            return []
        values = torch.cat([batch.reshape(-1) for batch in batches])
        batches.clear()
        values = values[torch.isfinite(values)]
        return values.cpu().tolist()

    def _clear_termination_counts(self, group_id: int) -> None:
        """清空指定组的本轮 termination 计数。"""
        self._termination_counts[group_id].clear()
