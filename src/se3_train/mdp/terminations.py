"""SE3 轮腿机器人的终止函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


class BadOrientationDelayed:
    """倾斜超过阈值连续 max_steps 步才终止(给恢复机会)。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self, env: ManagerBasedRlEnv, limit_angle: float = 0.5236, max_steps: int = 100
    ) -> torch.Tensor:
        if self._fail_count is None:
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        robot = env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt_angle = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        bad = tilt_angle > limit_angle

        self._fail_count[bad] += 1
        self._fail_count[~bad] = 0
        self._fail_count[env.episode_length_buf <= 1] = 0

        return self._fail_count > max_steps

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


bad_orientation_delayed = BadOrientationDelayed()
