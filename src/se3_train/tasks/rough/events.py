"""本任务使用的事件函数：平地那套 + 速度课程信号掩码 + 逐项奖励的分列拆分。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.tasks.flat.events import *  # noqa: F403
from se3_train.tasks.flat.events import __all__ as _FLAT_ALL

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

CURRICULUM_ENV_MASK_ATTR = "_se3_curriculum_env_mask"

# 逐项奖励分列日志的键前缀；走 Rough/ 命名空间，log_filter 整段白名单。
REWARD_SPLIT_LOG_PREFIX = "Rough/rw_"


def set_curriculum_env_mask(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = ("flat",),
) -> None:
    """startup 事件：把“只按这些子地形列的 env 评估速度课程”的掩码挂到 env 上。

    rewards.tracking_lin_vel 据此额外记 `Locomotion/tracking_lin_vel_reward_curriculum`，
    curriculums.commands_vel_adaptive 用它推进平地速度课程。地形列强制前向指令、跟踪分偏低，
    混进全体均值会把平地列的课程钉死在 vx=0（R3）。非课程地形（没有分列）时不挂，课程退回全体均值。
    """
    del env_ids
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if generator is None or terrain_types is None or not generator.curriculum:
        return
    names = list(generator.sub_terrains.keys())
    cols = [names.index(n) for n in terrain_type_names if n in names]
    if not cols:
        return
    types = terrain_types.to(device=env.device, dtype=torch.long)
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for col in cols:
        mask |= types == col
    setattr(env, CURRICULUM_ENV_MASK_ATTR, mask)


def log_reward_split_by_column(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    suffix: str = "stairs",
) -> None:
    """每步把奖励表里**每一项**在指定子地形列上的均值记一份。

    `Episode_Reward/*` 是全体均值，台阶列的定价被平地列稀释了看不出来——A8 就是卡在这：
    只知道 `command_velocity_error` 在台阶列是 −2.48/s（因为它本来就只在那一列生效），
    其余罚项各是多少完全没法从日志读出来。

    直接读 RewardManager 已经算好的逐 env 逐项缓冲 `_step_reward`（存的是 value×weight，
    即每秒贡献，与 Episode_Reward 同口径），只做一次掩码均值，不重算任何奖励、不碰奖励数学。
    该缓冲是上一次 `compute()` 的结果，对日志来说落后一个控制步无所谓。
    """
    del env_ids
    reward_manager = getattr(env, "reward_manager", None)
    step_reward = getattr(reward_manager, "_step_reward", None)
    if reward_manager is None or step_reward is None:
        return
    names = list(reward_manager.active_terms)
    if step_reward.shape != (env.num_envs, len(names)):
        return

    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if generator is None or terrain_types is None or not generator.curriculum:
        return
    sub_names = list(generator.sub_terrains.keys())
    cols = [sub_names.index(n) for n in terrain_type_names if n in sub_names]
    if not cols:
        return
    types = terrain_types.to(device=env.device, dtype=torch.long)
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for col in cols:
        mask |= types == col

    weights = mask.to(dtype=step_reward.dtype).unsqueeze(1)
    means = (step_reward * weights).sum(dim=0) / weights.sum().clamp(min=1.0)
    log = env.extras.setdefault("log", {})
    for index, name in enumerate(names):
        log[f"{REWARD_SPLIT_LOG_PREFIX}{name}_{suffix}"] = means[index]


__all__ = [
    *_FLAT_ALL,
    "CURRICULUM_ENV_MASK_ATTR",
    "REWARD_SPLIT_LOG_PREFIX",
    "log_reward_split_by_column",
    "set_curriculum_env_mask",
]
