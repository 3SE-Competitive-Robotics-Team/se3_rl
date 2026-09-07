"""本任务使用的事件函数：平地那套 + 速度课程信号掩码。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.tasks.flat.events import *  # noqa: F403
from se3_train.tasks.flat.events import __all__ as _FLAT_ALL

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

CURRICULUM_ENV_MASK_ATTR = "_se3_curriculum_env_mask"


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


__all__ = [*_FLAT_ALL, "CURRICULUM_ENV_MASK_ATTR", "set_curriculum_env_mask"]
