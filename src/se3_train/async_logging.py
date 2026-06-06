"""SE3 训练的主机侧异步日志累积工具。"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import torch


def async_host_logger_enabled() -> bool:
    """读取是否启用主机侧延迟日志累积。"""
    raw = os.environ.get("SE3_ASYNC_HOST_LOGGER")
    if raw is None:
        return True
    return raw.lower() not in {"0", "false", "no", "off"}


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
