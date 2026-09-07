"""本任务使用的终止条件：平地那套 + 墙阻挡转 time_out + 清块出界转 time_out。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.tasks.flat.terminations import *  # noqa: F403
from se3_train.tasks.flat.terminations import __all__ as _FLAT_ALL

from .commands import WALL_BLOCKED_ATTR
from .terrains import ROUGH_TERRAIN_EXIT_DISTANCE_M

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def wall_blocked(env: ManagerBasedRlEnv) -> torch.Tensor:
    """前方障碍高到跨不过去时结束该 episode。

    对应参考仓库 `StepUpStateMachine.apply_done_masks` 的
    `time_out |= wall_reset`：地形不可通过不是策略失败，注册时必须带
    `time_out=True`，让 PPO 按截断 bootstrap 而不是按失败记 0 值。
    标志由 `StepUpCommandTerm` 每步写入 env 属性。
    """
    flag = getattr(env, WALL_BLOCKED_ATTR, None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return flag


def terrain_cleared(
    env: ManagerBasedRlEnv,
    exit_distance_m: float = ROUGH_TERRAIN_EXIT_DISTANCE_M,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """机身走到本块边框上时截断 episode。

    切比雪夫距离 max(|dx|, |dy|) 超过 `exit_distance_m`（默认 = 块半边长 − 0.25 m）即触发。
    这不是策略失败：注册时必须带 `time_out=True` 按截断 bootstrap。作用是让课程
    （curriculums.terrain_levels）在清块后立刻结算，经验不串进邻块的难度行，
    也避免边框与邻块的高差撞出 wall_blocked。参考 extreme-parkour 走完全部 goal 即截断的做法。
    """
    asset = env.scene[asset_cfg.name]
    offset = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return offset.abs().max(dim=1).values > float(exit_distance_m)


__all__ = [*_FLAT_ALL, "terrain_cleared", "wall_blocked"]
