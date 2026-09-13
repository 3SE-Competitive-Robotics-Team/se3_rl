"""本任务使用的课程函数：平地那套 + 地形难度课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp.curriculums import commands_vel, push_disturbance

from .events import set_curriculum_env_mask
from .terrains import ROUGH_TERRAIN_CLEARED_DISTANCE_M

# 平地热身状态挂在 env 上的属性名。
FLAT_WARMUP_ORIGINAL_TYPES_ATTR = "_se3_flat_warmup_original_types"
FLAT_WARMUP_DONE_ATTR = "_se3_flat_warmup_done"
FLAT_WARMUP_THRESHOLD_ATTR = "_se3_flat_warmup_threshold"

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


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
    target_terrain_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """前 `iterations` 轮全部 env 放在平地列、第 0 行，之后逐 env 在 reset 时换回原列、第 0 行。

    目的：先把 Flat 基线练出来再进地形。R4/R5/R6 里地形课程 250 轮就把策略推上 12 cm 下台阶和 25% 坡，
    策略在还不会稳走时学成原地站着。逐 env 在 reset 时换列，不在 episode 中途改出生点。
    换列后同步刷新地形列前向指令覆盖与平地速度课程信号掩码（它们按列计算）。
    必须排在 terrain_levels 之前；还留在平地列的 env，terrain_levels 不给它升级。

    `ramp_iterations > 0` 时不再一刀切：地形 env 的比例在
    `iterations → iterations + ramp_iterations` 之间从 0 线性涨到 1。每个 env 建表时分到一个
    固定阈值（均匀铺在 [0,1) 上，保证任何 env 数下比例都严格线性），进度越过自己的阈值才换列，
    所以迁移单调、不会回平地，终态与一刀切相同。
    2026-09-08 用户定（A7）：A5/A6 的本机回放显示伤害集中在换列后那 100 轮
    （A6 model_500 在 2 m/s 上误差 0.02，model_600 掉到 0.93，且 A5 同型），
    一次性把 75% 的 env 扔进跟不上的指令里，共享 actor 连平地一起退化。
    """
    # 指定目标列时热身后全员进入该列，默认仍恢复原始分配。
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    names = list(generator.sub_terrains.keys())
    flat_col = names.index(flat_name)

    original = getattr(env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, None)
    if original is None:
        original = terrain.terrain_types.clone()
        if target_terrain_name is not None:
            original.fill_(names.index(target_terrain_name))
        setattr(env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, original)
        setattr(
            env,
            FLAT_WARMUP_DONE_ATTR,
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
        env._rough_terrain_migrated_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        # 逐 env 的迁移阈值：把 [0,1) 均匀铺开再随机置换，任何 env 数下比例都严格线性，
        # 且阈值与列号无关（各列同步迁移，不会先把台阶列整列放出去）。
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
        # 同次 reset 仍是旧地形上的机器人位置，不得用它减去新出生点结算升级。
        env._rough_terrain_migrated_step[move] = int(env.common_step_counter)
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


def terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    clear_distance_m: float = ROUGH_TERRAIN_CLEARED_DISTANCE_M,
    max_level: int | None = None,
) -> dict[str, torch.Tensor]:
    """清掉本块全部台阶就升一级；只升不降。

    升级判据：episode 结束时出生点到机身的切比雪夫距离 max(|dx|, |dy|) ≥ `clear_distance_m`
    （默认 = 平台半宽 + 台阶数 × 踏面，见 terrains.ROUGH_TERRAIN_CLEARED_DISTANCE_M），
    即越过最外一级台阶，与朝向和指令速度无关。配合 terminations.terrain_cleared 在出块时截断，
    一个 episode 最多记一次升级，且经验不会串到邻块的难度行。

    不用 mjlab `terrain_levels_vel` 的欧氏位移：金字塔是正方形，欧氏门槛在对角线方向少算台阶数。
    降级也去掉了：原判据用 episode 末段的 |vx| 反推应走距离，末段静站时永不降、末段高速时几乎必降，
    与地形能力无关；对称随机指令下净位移本身是随机游走，升降各半会把课程钉在低位
    （R2，W&B 32eentyo 的平地列也只到 1.6）。
    `max_level` 限制可到达的最高行号，保留原地形几何和逐级升级；到上限后留在该行。
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    del command  # 只升不降后不再需要指令反推应走距离；保留取值以校验指令项存在。

    offset = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = offset.abs().max(dim=1).values
    move_up = distance >= float(clear_distance_m)
    move_down = torch.zeros_like(move_up)

    # 首次 reset 发生在任何一步之前，机器人还在模型默认位姿（世界原点附近）而不是出生点，
    # 到 env_origins 的距离可达几十米，会把全员无条件升一级（R3 首轮 level 全为 1.0 即此故障）。
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
    # 平地热身期不升级（平地列每行都一样，升了也只是把 env 挪到另一块平地）。
    warmup_done = getattr(env, FLAT_WARMUP_DONE_ATTR, None)
    if isinstance(warmup_done, torch.Tensor):
        move_up = move_up & warmup_done[env_ids]
    migrated = getattr(env, "_rough_terrain_migrated_step", None)
    if isinstance(migrated, torch.Tensor):
        move_up &= migrated[env_ids] != int(env.common_step_counter)

    if max_level is not None:
        if not 0 <= max_level < terrain_generator.num_rows:
            raise ValueError("max_level 必须位于地形行号范围内")
        # 先收回超限行，再阻止上限处升级，避免最高行溢出触发随机重分配。
        terrain.terrain_levels[env_ids] = terrain.terrain_levels[env_ids].clamp(max=max_level)
        move_up &= terrain.terrain_levels[env_ids] < max_level
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    # 课程模式下每种子地形独占一列，列号即子地形名，可以分地形看难度爬到哪。
    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    if terrain_origins.shape[1] == len(sub_terrain_names):
        types = terrain.terrain_types
        for index, name in enumerate(sub_terrain_names):
            mask = types == index
            if mask.any():
                result[name] = torch.mean(levels[mask])
    return result


__all__ = ["commands_vel", "flat_warmup", "push_disturbance", "terrain_levels"]
