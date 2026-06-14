"""倒地自启训练的 episode 状态缓存。

这些缓存只描述训练时的 recovery episode，不改变机器人模型或动作语义。
旧代码仍使用 ``_recovery_reset_mask``，这里把它保留为“当前仍在恢复中”的
active mask；额外的 ``_recovery_episode_mask`` 用于日志和分桶统计。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def ensure_bool_buffer(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    """创建或读取 bool 缓存。"""
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        values = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, name, values)
    return values


def ensure_float_buffer(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    """创建或读取 float 缓存。"""
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        values = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, name, values)
    return values


def ensure_long_buffer(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    """创建或读取 long 缓存。"""
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        values = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        setattr(env, name, values)
    return values


def recovery_episode_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回本 episode 是否由 recovery reset 开始。"""
    mask = getattr(env, "_recovery_episode_mask", None)
    if isinstance(mask, torch.Tensor) and mask.shape[0] == env.num_envs:
        return mask.to(device=env.device, dtype=torch.bool)
    active = getattr(env, "_recovery_reset_mask", None)
    if isinstance(active, torch.Tensor) and active.shape[0] == env.num_envs:
        return active.to(device=env.device, dtype=torch.bool)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def recovery_active_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回当前仍处于 recovery 控制/奖励门控的 env。"""
    mask = getattr(env, "_recovery_reset_mask", None)
    if isinstance(mask, torch.Tensor) and mask.shape[0] == env.num_envs:
        return mask.to(device=env.device, dtype=torch.bool)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def set_recovery_episode(env: ManagerBasedRlEnv, env_ids: torch.Tensor, mask: torch.Tensor) -> None:
    """在 reset 时初始化 recovery episode 与 active 状态。"""
    episode = ensure_bool_buffer(env, "_recovery_episode_mask")
    active = ensure_bool_buffer(env, "_recovery_reset_mask")
    stable_steps = ensure_long_buffer(env, "_recovery_success_steps")
    time_to_success = ensure_long_buffer(env, "_recovery_time_to_success_steps")
    cache_reset = ensure_bool_buffer(env, "_recovery_cache_reset_mask")
    cache_type = ensure_long_buffer(env, "_recovery_cache_type")
    success_updated_step = ensure_long_buffer(env, "_recovery_success_updated_step")
    completed = ensure_bool_buffer(env, "_recovery_success_completed")
    completed_latched = ensure_bool_buffer(env, "_recovery_success_completed_latched")

    episode[env_ids] = mask
    active[env_ids] = mask
    stable_steps[env_ids] = 0
    time_to_success[env_ids] = -1
    cache_reset[env_ids] = False
    cache_type[env_ids] = 0
    success_updated_step[env_ids] = -1
    completed[env_ids] = False
    completed_latched[env_ids] = False


def mark_cache_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    cache_mask: torch.Tensor,
    cache_type_values: torch.Tensor | None = None,
) -> None:
    """记录哪些 recovery episode 来自离线倒地状态缓存。"""
    cache_reset = ensure_bool_buffer(env, "_recovery_cache_reset_mask")
    cache_type = ensure_long_buffer(env, "_recovery_cache_type")
    cache_reset[env_ids] = cache_mask
    if cache_type_values is not None and cache_mask.any():
        cache_type[env_ids[cache_mask]] = cache_type_values.to(env.device, dtype=torch.long)


def deactivate_recovered(
    env: ManagerBasedRlEnv,
    success: torch.Tensor,
    stable_steps_required: int,
) -> torch.Tensor:
    """累计稳定步数，达到阈值后退出 recovery active 模式。"""
    active = recovery_active_mask(env)
    stable_steps = ensure_long_buffer(env, "_recovery_success_steps")
    time_to_success = ensure_long_buffer(env, "_recovery_time_to_success_steps")

    stable_steps[active & success] += 1
    stable_steps[active & ~success] = 0

    completed = active & (stable_steps >= int(stable_steps_required))
    first_completed = completed & (time_to_success < 0)
    time_to_success[first_completed] = env.episode_length_buf[first_completed].to(torch.long)

    active_buffer = ensure_bool_buffer(env, "_recovery_reset_mask")
    active_buffer[completed] = False
    return completed


def update_success_window(
    env: ManagerBasedRlEnv,
    success: torch.Tensor,
    stable_steps_required: int,
    min_episode_steps: int,
) -> torch.Tensor:
    """按连续成功窗口更新完成状态；同一仿真步内重复调用保持幂等。"""
    active = recovery_active_mask(env)
    stable_steps = ensure_long_buffer(env, "_recovery_success_steps")
    time_to_success = ensure_long_buffer(env, "_recovery_time_to_success_steps")
    updated_step = ensure_long_buffer(env, "_recovery_success_updated_step")
    completed = ensure_bool_buffer(env, "_recovery_success_completed")
    completed_latched = ensure_bool_buffer(env, "_recovery_success_completed_latched")

    common_step = int(getattr(env, "common_step_counter", 0))
    needs_update = updated_step != common_step
    if needs_update.any():
        valid_success = success.to(device=env.device, dtype=torch.bool) & active
        stable_steps[needs_update & valid_success] += 1
        stable_steps[needs_update & ~valid_success] = 0

        enough_time = env.episode_length_buf >= int(min_episode_steps)
        reached_window = active & enough_time & (stable_steps >= int(stable_steps_required))
        first_completed = reached_window & ~completed_latched
        completed[:] = first_completed
        completed_latched[first_completed] = True
        time_to_success[first_completed] = env.episode_length_buf[first_completed].to(torch.long)
        updated_step[needs_update] = common_step

    return completed & active
