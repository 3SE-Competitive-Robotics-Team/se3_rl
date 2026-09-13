"""本任务的课程：平地热身。地形难度的升降级用 mjlab 官方 `terrain_levels_vel`（见 env_cfg.py）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .events import set_curriculum_env_mask

# 平地热身状态挂在 env 上的属性名。
FLAT_WARMUP_ORIGINAL_TYPES_ATTR = "_se3_flat_warmup_original_types"
FLAT_WARMUP_DONE_ATTR = "_se3_flat_warmup_done"
FLAT_WARMUP_THRESHOLD_ATTR = "_se3_flat_warmup_threshold"

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _refresh_terrain_dependent_masks(env: ManagerBasedRlEnv, command_name: str) -> None:
    """env 换列后刷新按列计算的两个掩码：地形列前向指令覆盖、平地速度课程信号掩码。"""
    term = env.command_manager.get_term(command_name)
    refresh = getattr(term, "refresh_terrain_override", None)
    if callable(refresh):
        refresh()
    set_curriculum_env_mask(env, None)


def flat_warmup(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    iterations: int = 500,
    ramp_iterations: int = 0,
    steps_per_policy_iter: int = 24,
    flat_name: str = "flat",
) -> dict[str, torch.Tensor]:
    """前 `iterations` 轮全部 env 放在平地列第 0 行，之后逐 env 在 reset 时换回原列第 0 行。

    R7 引入、A7 加 ramp：地形课程 250 轮就把还不会稳走的策略推上 12 cm 台阶，学成原地站着
    （R4–R6）；一刀切换列时伤害集中在换列后那 100 轮（A5/A6 的 sim2x 回放）。所以 ramp 期内
    地形 env 的比例从 0 线性涨到 1：每个 env 建表时分到一个固定阈值（均匀铺在 [0,1) 上再随机置换，
    任何 env 数下比例都严格线性、各列同步迁移），进度越过自己的阈值才换列，迁移单调、不会回平地，
    终态与一刀切相同。

    与官方 `terrain_levels_vel` 的配合：本项必须排在它**之后**。升降级按 episode 末位置到出生点
    的距离结算；若先换列再结算，换列那次 reset 会拿旧地块上的位置减新出生点，把每个刚迁移的 env
    无条件升一级。先结算再换列就没有这个问题：热身期平地列各行几何相同，升降只是换一块平地，
    迁移时 level 归 0。换列后同步刷新地形列前向指令覆盖与平地速度课程信号掩码（它们按列计算）。
    """
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    names = list(generator.sub_terrains.keys())
    flat_col = names.index(flat_name)

    original = getattr(env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, None)
    if original is None:
        original = terrain.terrain_types.clone()
        setattr(env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, original)
        setattr(
            env,
            FLAT_WARMUP_DONE_ATTR,
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
        order = torch.randperm(env.num_envs, device=env.device)
        setattr(env, FLAT_WARMUP_THRESHOLD_ATTR, order.float() / float(env.num_envs))
    done: torch.Tensor = getattr(env, FLAT_WARMUP_DONE_ATTR)
    threshold: torch.Tensor = getattr(env, FLAT_WARMUP_THRESHOLD_ATTR)

    iteration = int(env.common_step_counter) // max(1, int(steps_per_policy_iter))
    ramp = max(int(ramp_iterations), 0)
    if ramp > 0:
        progress = (iteration - int(iterations)) / float(ramp)
    else:
        progress = 1.0 if iteration >= int(iterations) else 0.0
    progress = min(max(progress, 0.0), 1.0)

    changed = False
    pending = ~done[env_ids]
    hold = env_ids[pending & (threshold[env_ids] >= progress)]
    move = env_ids[pending & (threshold[env_ids] < progress)]
    if hold.numel() > 0:
        # 还没轮到自己迁移：留在平地列第 0 行。
        stay = hold[terrain.terrain_types[hold] != flat_col]
        if stay.numel() > 0:
            terrain.terrain_types[stay] = flat_col
            terrain.terrain_levels[stay] = 0
            changed = True
    if move.numel() > 0:
        terrain.terrain_types[move] = original[move]
        terrain.terrain_levels[move] = 0
        done[move] = True
        changed = True
    if changed:
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        _refresh_terrain_dependent_masks(env, command_name)
    return {
        "active": (~done).float().mean(),
        "progress": torch.tensor(progress, device=env.device),
    }


__all__ = ["flat_warmup"]
