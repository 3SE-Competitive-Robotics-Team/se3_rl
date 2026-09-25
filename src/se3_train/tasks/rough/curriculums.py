"""本任务的课程：平地热身 + 二级台阶门控。地形难度的升降级用 mjlab 官方 `terrain_levels_vel`（见 env_cfg.py）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .events import set_curriculum_env_mask

# 平地热身状态挂在 env 上的属性名。
FLAT_WARMUP_ORIGINAL_TYPES_ATTR = "_se3_flat_warmup_original_types"
FLAT_WARMUP_DONE_ATTR = "_se3_flat_warmup_done"
FLAT_WARMUP_THRESHOLD_ATTR = "_se3_flat_warmup_threshold"
# 二级台阶门控状态。
TWO_STEP_GATE_ORIGINAL_TYPES_ATTR = "_se3_two_step_gate_original_types"
TWO_STEP_GATE_OPENED_ATTR = "_se3_two_step_gate_opened"

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


def two_step_gate(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    gate_terrain_name: str = "stairs_up",
    gate_level: float = 5.0,
    gated_columns: tuple[tuple[str, str], ...] = (
        ("stairs_two_step_up", "stairs_up"),
        ("stairs_two_step_down", "flat"),
    ),
) -> dict[str, torch.Tensor]:
    """`gate_terrain_name` 的平均难度等级达到 `gate_level` 之前，把被门控的列的 env 暂放到各自的"母列"。

    M24（2026-09-21 用户定）：二级台阶（那道 0.15 m 窄棱）比普通台阶难一档，一开始就放出来会让策略
    在还不会爬普通台阶时就被它拖住。所以等 `stairs_up` 均级到 5（9 级里的中段、约 11 cm 阶高）再开放。
    `gated_columns` 的每一项是 (被门控的列, 门控期去哪一列)：上行那列去 stairs_up（它本来就该练台阶），
    下行那列去 flat（它按平地待遇）。

    **一次性放开、不再回收**：等级会随策略波动，反复迁移会让这些 env 的 episode 统计与课程等级来回重置。

    与 `flat_warmup` 的配合：本项必须排在它**之后**，且只处理已经结束热身的 env——热身期全体都在平地列，
    此时迁移会把还在热身的 env 提前拽到 stairs_up。原始列名也优先复用热身记下的那份，
    否则本项第一次运行时 clone 到的是"热身把大家都改成 flat 之后"的快照。
    """
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    names = list(generator.sub_terrains.keys())

    original = getattr(env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, None)
    if original is None:
        original = getattr(env, TWO_STEP_GATE_ORIGINAL_TYPES_ATTR, None)
        if original is None:
            original = terrain.terrain_types.clone()
            setattr(env, TWO_STEP_GATE_ORIGINAL_TYPES_ATTR, original)
    opened = getattr(env, TWO_STEP_GATE_OPENED_ATTR, None)
    if opened is None:
        opened = torch.zeros((), dtype=torch.bool, device=env.device)
        setattr(env, TWO_STEP_GATE_OPENED_ATTR, opened)

    # 门控判据按**当前所在列**统计，与官方 `Curriculum/terrain_levels/<列名>` 同口径。不能按原始列归属取：
    # 热身期原生 stairs_up env 被放在平地列，官方升降级照常按平地上走的距离给它们升级，读到的是平地等级，
    # M24 因此在热身期（< 500 轮）就开了门。门控期迁进来的二级上行 env 与原生 env 同列同难度，一并统计。
    level = torch.zeros((), device=env.device)
    if gate_terrain_name in names:
        gate_mask = terrain.terrain_types == names.index(gate_terrain_name)
        if bool(gate_mask.any()):
            level = terrain.terrain_levels[gate_mask].float().mean()
    if not bool(opened) and float(level) >= float(gate_level):
        opened.fill_(True)

    pairs = [
        (names.index(gated), names.index(fallback))
        for gated, fallback in gated_columns
        if gated in names and fallback in names
    ]
    warmup_done = getattr(env, FLAT_WARMUP_DONE_ATTR, None)
    ids = env_ids if warmup_done is None else env_ids[warmup_done[env_ids]]

    changed = False
    if ids.numel() > 0:
        for gated_col, fallback_col in pairs:
            owned = ids[original[ids] == gated_col]
            if owned.numel() == 0:
                continue
            target = gated_col if bool(opened) else fallback_col
            move = owned[terrain.terrain_types[owned] != target]
            if move.numel() > 0:
                terrain.terrain_types[move] = target
                terrain.terrain_levels[move] = 0
                changed = True
    if changed:
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        _refresh_terrain_dependent_masks(env, command_name)
    return {"opened": opened.float(), "gate_level": level}


__all__ = ["flat_warmup", "two_step_gate"]
