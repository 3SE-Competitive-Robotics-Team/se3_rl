"""SE3 轮腿机器人的终止函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(env: ManagerBasedRlEnv, limit_angle: float = 0.873) -> torch.Tensor:
    """倾斜超过阈值即时终止（默认 50° = 0.873 rad）。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt_angle = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    return tilt_angle > limit_angle
