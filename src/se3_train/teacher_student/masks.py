"""teacher loss 使用的样本级 mask。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .config import MaskConfig


@dataclass(frozen=True)
class TerrainMetadata:
    """从 MJLab env 中读取出的 terrain 元数据。"""

    terrain_types: torch.Tensor | None
    terrain_type_names: tuple[str, ...]
    terrain_levels: torch.Tensor | None = None
    is_curriculum: bool | None = None

    @classmethod
    def from_env(cls, env: Any) -> TerrainMetadata:
        """从 env 读取 terrain_types/levels；失败时返回空元数据。"""
        scene = getattr(env, "scene", None)
        terrain = getattr(scene, "terrain", None)
        if terrain is None:
            return cls(terrain_types=None, terrain_type_names=())

        cfg = getattr(terrain, "cfg", None)
        generator_cfg = getattr(cfg, "terrain_generator", None)
        sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
        return cls(
            terrain_types=getattr(terrain, "terrain_types", None),
            terrain_type_names=tuple(str(name) for name in sub_terrains),
            terrain_levels=getattr(terrain, "terrain_levels", None),
            is_curriculum=getattr(generator_cfg, "curriculum", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回不含大 tensor 内容的诊断信息。"""
        data = asdict(self)
        data["terrain_types"] = (
            None if self.terrain_types is None else tuple(self.terrain_types.shape)
        )
        data["terrain_levels"] = (
            None if self.terrain_levels is None else tuple(self.terrain_levels.shape)
        )
        return data


@dataclass(frozen=True)
class TeacherMasks:
    """当前 batch 的 stair teacher mask。"""

    stair: torch.Tensor
    recovery: torch.Tensor
    raw_stair: torch.Tensor

    def rates(self) -> dict[str, float]:
        """返回 mask 占比，便于写入日志。"""
        return {
            "Teacher/recovery_mask_rate": _mean_float(self.recovery),
            "Teacher/stair_mask_rate": _mean_float(self.stair),
            "Teacher/raw_stair_terrain_rate": _mean_float(self.raw_stair),
        }


def build_teacher_masks(
    obs: Any,
    metadata: TerrainMetadata | None = None,
    config: MaskConfig | None = None,
) -> TeacherMasks:
    """生成 stair teacher mask；倒地 recovery 样本优先从 stair mask 中排除。"""
    cfg = config or MaskConfig()
    projected_gravity = get_projected_gravity(obs, cfg)
    pg_z = projected_gravity[..., 2]
    recovery = pg_z > cfg.recovery_pg_z_threshold
    raw_stair = _terrain_mask_like(pg_z, metadata, cfg.stair_type_names, cfg)
    stair = raw_stair & ~recovery
    return TeacherMasks(stair=stair, recovery=recovery, raw_stair=raw_stair)


def get_projected_gravity(obs: Any, config: MaskConfig | None = None) -> torch.Tensor:
    """从 TensorDict/dict/拼接 actor obs 中取 projected_gravity。"""
    cfg = config or MaskConfig()
    direct = _get_obs_value(obs, cfg.projected_gravity_key)
    if direct is not None:
        if direct.shape[-1] != 3:
            raise ValueError(f"projected_gravity 最后一维应为 3，实际为 {direct.shape[-1]}")
        return direct

    actor_obs = _get_obs_value(obs, cfg.actor_group)
    if actor_obs is None:
        if isinstance(obs, torch.Tensor):
            actor_obs = obs
        else:
            raise KeyError(f"obs 中找不到 {cfg.projected_gravity_key!r} 或 {cfg.actor_group!r}")

    start = cfg.projected_gravity_z_index - 2
    stop = cfg.projected_gravity_z_index + 1
    if actor_obs.shape[-1] < stop:
        raise ValueError(
            f"actor obs 维度不足，无法按 index={cfg.projected_gravity_z_index} 取 projected_gravity"
        )
    return actor_obs[..., start:stop]


def _terrain_mask_like(
    reference: torch.Tensor,
    metadata: TerrainMetadata | None,
    selected_names: tuple[str, ...],
    config: MaskConfig,
) -> torch.Tensor:
    """按 terrain_types 生成 mask；元数据不可用时返回全 False。"""
    if metadata is None or metadata.terrain_types is None or not metadata.terrain_type_names:
        return torch.zeros_like(reference, dtype=torch.bool)
    if metadata.is_curriculum is False and not config.trust_non_curriculum_terrain_types:
        return torch.zeros_like(reference, dtype=torch.bool)

    terrain_types = metadata.terrain_types.to(device=reference.device)
    if terrain_types.shape != reference.shape:
        terrain_types = terrain_types.reshape(-1)
        if terrain_types.numel() == 1:
            terrain_types = terrain_types.expand(reference.numel()).reshape(reference.shape)
        elif terrain_types.numel() == reference.numel():
            terrain_types = terrain_types.reshape(reference.shape)
        else:
            return torch.zeros_like(reference, dtype=torch.bool)

    mask = torch.zeros_like(reference, dtype=torch.bool)
    selected = set(selected_names)
    for terrain_index, terrain_name in enumerate(metadata.terrain_type_names):
        if terrain_name in selected:
            mask = mask | (terrain_types == terrain_index)
    return mask


def _get_obs_value(obs: Any, key: str) -> torch.Tensor | None:
    """从 dict-like obs 中读取指定 key。"""
    if hasattr(obs, "keys") and key in obs:
        value = obs[key]
        if isinstance(value, torch.Tensor):
            return value
    return None


def _mean_float(mask: torch.Tensor) -> float:
    """安全计算 bool mask 均值。"""
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().detach().cpu().item())
