"""WheelDog 盲爬任务终止条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.tasks.wheel_dog.blind_climb import terrain_progress

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


def root_height_bounds(
    env: ManagerBasedRlEnv,
    min_relative_height: float = -0.08,
    max_relative_height: float = 2.0,
) -> torch.Tensor:
    """root 高度明显越界时终止，避免掉入坑底后的异常轨迹污染训练。"""
    robot = env.scene["robot"]
    rel_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (rel_height < float(min_relative_height)) | (rel_height > float(max_relative_height))


def corridor_bounds(
    env: ManagerBasedRlEnv,
    hard_half_width: float = terrain_progress.CORRIDOR_HARD_HALF_WIDTH,
) -> torch.Tensor:
    """base_link 离开中心通道时终止，避免绕过坑坡拿奖励。"""
    return torch.abs(terrain_progress.lateral_offset(env)) > float(hard_half_width)


class BodyContactDelayed:
    """非轮子 body 连续触地一段时间后终止。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 1.0,
        max_steps: int = 20,
        ignore_initial_steps: int = 2,
    ) -> torch.Tensor:
        if self._fail_count is None or self._fail_count.shape[0] != env.num_envs:
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        sensor: ContactSensor = env.scene[sensor_name]
        if sensor.data.force is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        force = finite_contact_force_norm(sensor.data.force)
        contact = torch.max(force, dim=1).values > float(force_threshold)
        self._fail_count[contact] += 1
        self._fail_count[~contact] = 0
        self._fail_count[env.episode_length_buf <= int(ignore_initial_steps)] = 0
        return self._fail_count > int(max_steps)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """重置指定环境的计数器。"""
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


def nonfinite_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """机器人核心状态出现非有限值时终止。"""
    robot = env.scene["robot"]
    finite = (
        torch.isfinite(robot.data.root_link_pos_w).all(dim=1)
        & torch.isfinite(robot.data.root_link_lin_vel_b).all(dim=1)
        & torch.isfinite(robot.data.root_link_ang_vel_b).all(dim=1)
        & torch.isfinite(robot.data.projected_gravity_b).all(dim=1)
        & torch.isfinite(robot.data.joint_pos).all(dim=1)
        & torch.isfinite(robot.data.joint_vel).all(dim=1)
    )
    return ~finite


bad_orientation_delayed = BadOrientationDelayed()
body_contact_delayed = BodyContactDelayed()
low_base_height_delayed = LowBaseHeightDelayed()


__all__ = [
    "bad_orientation_delayed",
    "body_contact",
    "body_contact_delayed",
    "corridor_bounds",
    "low_base_height_delayed",
    "nonfinite_state",
    "root_height_bounds",
    "time_out",
]
