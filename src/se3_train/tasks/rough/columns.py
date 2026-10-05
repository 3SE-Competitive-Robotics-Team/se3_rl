"""按子地形列取 env 掩码，rough 线所有"分列"逻辑共用这一处。

课程模式（`TerrainGeneratorCfg.curriculum=True`）下每种子地形独占一列，`terrain_types` 即列号，
列号对应 `sub_terrains` 的键序。非课程地形、平面地形或列名对不上时返回 None，调用方按
"没有分列信息"退化：奖励不分列、指令不覆盖、诊断不记。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _column_indices(
    env: ManagerBasedRlEnv, terrain_type_names: tuple[str, ...]
) -> tuple[list[int], torch.Tensor] | None:
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if generator is None or terrain_types is None or not generator.curriculum:
        return None
    names = list(generator.sub_terrains.keys())
    cols = [names.index(name) for name in terrain_type_names if name in names]
    if not cols:
        return None
    return cols, terrain_types.to(device=env.device, dtype=torch.long)


def column_mask(env: ManagerBasedRlEnv, terrain_type_names: tuple[str, ...]) -> torch.Tensor | None:
    """返回"在这些子地形列上"的 env 掩码 [N]；返回的是共享缓存张量，调用方不得原地修改。

    rough 每个 env.step 有十几个奖励 / 指令项各自要同一批列掩码。`terrain_types` 只在 reset、课程升降级、
    平地热身换列时被 torch 原地写入（版本号随之加一），所以按 (列名, 张量身份, 版本号) 缓存，结果与重算逐位相同。
    """
    if not terrain_type_names:
        return None
    names = tuple(terrain_type_names)
    terrain_types = getattr(getattr(env.scene, "terrain", None), "terrain_types", None)
    cache: dict = env.__dict__.setdefault("_se3_column_mask_cache", {})
    stamp = (id(terrain_types), getattr(terrain_types, "_version", None))
    hit = cache.get(names)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    resolved = _column_indices(env, names)
    if resolved is None:
        mask = None
    else:
        cols, types = resolved
        mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for col in cols:
            mask |= types == col
    cache[names] = (stamp, mask)
    return mask


def non_flat_column_mask(
    env: ManagerBasedRlEnv, flat_type_names: tuple[str, ...] = ("flat",)
) -> torch.Tensor | None:
    """返回"不在平地列"的 env 掩码；凡是要爬升的列都算，新增子地形时不用点名。"""
    mask = column_mask(env, flat_type_names)
    return None if mask is None else ~mask


__all__ = ["column_mask", "non_flat_column_mask"]
