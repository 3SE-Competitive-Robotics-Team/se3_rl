"""SE3 轮腿机器人的终止函数。

仅超时终止——原始设计允许从任何姿态恢复。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """当 episode 长度超过最大值时终止。"""
    return env.episode_length_buf >= env.max_episode_length
