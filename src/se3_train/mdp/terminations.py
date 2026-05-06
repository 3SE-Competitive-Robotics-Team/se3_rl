"""SE3 轮腿机器人的终止函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """当 episode 长度超过最大值时终止。"""
    return env.episode_length_buf >= env.max_episode_length


def base_contact(env: ManagerBasedRlEnv, tilt_threshold: float = 1.0) -> torch.Tensor:
    """当机器人倾斜超过阈值时终止(倒地检测)。

    tilt_threshold: projected_gravity_z 的阈值。
    pg_z = -1 为完全直立, pg_z > 0 为超过 90 度倾倒。
    默认 tilt_threshold=1.0 意味着 pg_z > -cos(60deg) 时终止,
    即倾斜超过约 60 度就判定为倒地。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    # pg_z < -threshold 表示直立(接近 -1), pg_z > -threshold 表示倒了
    return pg_z > -0.5
