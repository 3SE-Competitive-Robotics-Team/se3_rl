"""Recovery-Stand 腿部几何对齐工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def wheel_alignment_metrics(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回机身坐标系下左右轮横向距离和前后错位。"""
    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    delta_w = wheel_pos_w[:, 0, :] - wheel_pos_w[:, 1, :]
    delta_b = quat_apply_inverse(robot.data.root_link_quat_w, delta_w)
    lateral_distance = torch.abs(delta_b[:, 1])
    fore_aft_offset = torch.abs(delta_b[:, 0])
    return lateral_distance, fore_aft_offset


def wheel_alignment_ok(
    env: ManagerBasedRlEnv,
    min_lateral_distance: float = 0.40,
    max_lateral_distance: float = 0.46,
    max_fore_aft_offset: float = 0.03,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """判断左右轮是否处在可交接站立几何范围内。"""
    lateral_distance, fore_aft_offset = wheel_alignment_metrics(env, asset_cfg)
    ok = (
        (lateral_distance >= float(min_lateral_distance))
        & (lateral_distance <= float(max_lateral_distance))
        & (fore_aft_offset <= float(max_fore_aft_offset))
    )
    return ok, lateral_distance, fore_aft_offset


def wheel_alignment_penalty(
    env: ManagerBasedRlEnv,
    min_lateral_distance: float = 0.40,
    max_lateral_distance: float = 0.46,
    max_fore_aft_offset: float = 0.03,
    lateral_scale: float = 0.04,
    fore_aft_scale: float = 0.03,
    fore_aft_weight: float = 1.5,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算轮距和前后错位惩罚，前后劈叉主要由 fore_aft_offset 捕获。"""
    ok, lateral_distance, fore_aft_offset = wheel_alignment_ok(
        env,
        min_lateral_distance=min_lateral_distance,
        max_lateral_distance=max_lateral_distance,
        max_fore_aft_offset=max_fore_aft_offset,
        asset_cfg=asset_cfg,
    )
    lateral_low = torch.clamp(float(min_lateral_distance) - lateral_distance, min=0.0)
    lateral_high = torch.clamp(lateral_distance - float(max_lateral_distance), min=0.0)
    lateral_error = lateral_low + lateral_high
    fore_aft_error = torch.clamp(fore_aft_offset - float(max_fore_aft_offset), min=0.0)
    penalty = (lateral_error / float(lateral_scale)) ** 2 + float(fore_aft_weight) * (
        fore_aft_error / float(fore_aft_scale)
    ) ** 2
    return torch.clamp(penalty, max=float(max_penalty)), lateral_distance, fore_aft_offset, ok


def _wheel_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """缓存左右轮 body 在 articulation 内的局部索引。"""
    attr_name = f"_recovery_wheel_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids
