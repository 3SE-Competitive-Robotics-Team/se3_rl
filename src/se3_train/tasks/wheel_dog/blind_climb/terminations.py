"""WheelDog 盲爬任务终止条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.contact_utils import finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """episode 时间到达上限。"""
    return env.episode_length_buf >= env.max_episode_length


class BadOrientationDelayed:
    """倾斜连续超过阈值后终止。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        limit_angle: float = 0.7854,
        max_steps: int = 60,
    ) -> torch.Tensor:
        if self._fail_count is None or self._fail_count.shape[0] != env.num_envs:
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        robot = env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        bad = tilt > float(limit_angle)
        self._fail_count[bad] += 1
        self._fail_count[~bad] = 0
        self._fail_count[env.episode_length_buf <= 1] = 0
        return self._fail_count > int(max_steps)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """重置指定环境的计数器。"""
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


class LowBaseHeightDelayed:
    """base_link 连续低于阈值后终止。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        min_height: float = 0.24,
        max_steps: int = 30,
    ) -> torch.Tensor:
        if self._fail_count is None or self._fail_count.shape[0] != env.num_envs:
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        sensor = env.scene[sensor_name]
        low = sensor.data.heights[:, 0] < float(min_height)
        self._fail_count[low] += 1
        self._fail_count[~low] = 0
        self._fail_count[env.episode_length_buf <= 1] = 0
        return self._fail_count > int(max_steps)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """重置指定环境的计数器。"""
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


def body_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """非轮子 body 触地后终止。"""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force = finite_contact_force_norm(sensor.data.force)
    return torch.max(force, dim=1).values > float(force_threshold)


bad_orientation_delayed = BadOrientationDelayed()
low_base_height_delayed = LowBaseHeightDelayed()


__all__ = [
    "bad_orientation_delayed",
    "body_contact",
    "low_base_height_delayed",
    "time_out",
]
