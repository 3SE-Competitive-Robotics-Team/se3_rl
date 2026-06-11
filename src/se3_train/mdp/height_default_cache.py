"""缓存随高度指令变化的默认腿部姿态。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import FourbarRobotConfig, policy_default_from_height_torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_FOURBAR_ROBOT = FourbarRobotConfig()
_CACHE_POSE_ATTR = "_se3_height_conditioned_policy_default"
_CACHE_HEIGHT_ATTR = "_se3_height_conditioned_policy_default_height"


def update_policy_default_from_height_cache(
    env: ManagerBasedRlEnv,
    command_name: str,
    env_ids: torch.Tensor | None = None,
    command: torch.Tensor | None = None,
) -> torch.Tensor:
    """按当前 command height 更新 policy 默认腿型缓存。"""
    cmd = command if command is not None else env.command_manager.get_command(command_name)
    if cmd.shape[1] <= 4:
        raise ValueError(f"{command_name} command 维度不足，无法读取 height: {cmd.shape}")

    heights = cmd[:, 4].detach()
    cache = getattr(env, _CACHE_POSE_ATTR, None)
    height_cache = getattr(env, _CACHE_HEIGHT_ATTR, None)
    cache_invalid = (
        not isinstance(cache, torch.Tensor)
        or cache.shape != (env.num_envs, 4)
        or cache.device != heights.device
    )
    height_cache_invalid = (
        not isinstance(height_cache, torch.Tensor)
        or height_cache.shape != (env.num_envs,)
        or height_cache.device != heights.device
    )

    if cache_invalid or height_cache_invalid:
        cache = policy_default_from_height_torch(heights, _FOURBAR_ROBOT)
        height_cache = heights.clone()
        setattr(env, _CACHE_POSE_ATTR, cache)
        setattr(env, _CACHE_HEIGHT_ATTR, height_cache)
        return cache

    if env_ids is None:
        cache[:] = policy_default_from_height_torch(heights, _FOURBAR_ROBOT)
        height_cache[:] = heights
    else:
        if len(env_ids) == 0:
            return cache
        env_ids = env_ids.to(device=heights.device, dtype=torch.long)
        cache[env_ids] = policy_default_from_height_torch(heights[env_ids], _FOURBAR_ROBOT)
        height_cache[env_ids] = heights[env_ids]
    return cache


def get_policy_default_from_height_cache(
    env: ManagerBasedRlEnv,
    command_name: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """读取高度条件默认腿型；缓存缺失时按当前 command height 初始化。"""
    target_device = torch.device(device)
    cache = getattr(env, _CACHE_POSE_ATTR, None)
    if (
        not isinstance(cache, torch.Tensor)
        or cache.shape != (env.num_envs, 4)
        or cache.device != target_device
    ):
        cache = update_policy_default_from_height_cache(env, command_name)
    return cache.to(device=target_device, dtype=dtype)
