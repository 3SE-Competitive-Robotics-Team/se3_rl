"""WheelDog 盲爬课程几何目标。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.terrains import (
    BLIND_CLIMB_PIT_LENGTH_RANGE,
    BLIND_CLIMB_RAMP_ANGLE_RANGE_DEG,
    BLIND_CLIMB_RAMP_HEIGHT_RANGE,
    GapRampFacilitySpec,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_SPEC = GapRampFacilitySpec()
POST_RAMP_RUNOUT = 1.0
"""到达坡底后仍需继续前进的距离。"""

FINAL_SUCCESS_DISTANCE = (
    _SPEC.left_platform_length * 0.5
    + BLIND_CLIMB_PIT_LENGTH_RANGE[1]
    + _SPEC.ramp_horizontal_run
    + POST_RAMP_RUNOUT
)
"""最高难度下的完整路线目标距离：起跑、越坑上坡、落地后继续前进。"""


def current_success_distance(
    env: ManagerBasedRlEnv,
    final_success_distance: float = FINAL_SUCCESS_DISTANCE,
    post_ramp_margin: float = POST_RAMP_RUNOUT,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """按当前地形等级计算通过目标距离。"""
    ramp_low_x = ramp_low_progress(env, env_ids=env_ids)
    final_target = torch.full_like(ramp_low_x, float(final_success_distance))
    return torch.minimum(ramp_low_x + float(post_ramp_margin), final_target)


def ramp_high_progress(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回坡顶竖直高边相对 env origin 的 x 坐标。"""
    difficulty = current_difficulty(env, env_ids=env_ids)
    pit_length = _lerp_tensor(BLIND_CLIMB_PIT_LENGTH_RANGE, difficulty)
    return float(_SPEC.left_platform_length * 0.5) + pit_length


def ramp_low_progress(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回坡底低端相对 env origin 的 x 坐标。"""
    difficulty = current_difficulty(env, env_ids=env_ids)
    ramp_height = _lerp_tensor(BLIND_CLIMB_RAMP_HEIGHT_RANGE, difficulty)
    ramp_angle_deg = _lerp_tensor(BLIND_CLIMB_RAMP_ANGLE_RANGE_DEG, difficulty)
    ramp_angle = torch.deg2rad(ramp_angle_deg)
    ramp_run = ramp_height / torch.clamp(torch.tan(ramp_angle), min=1.0e-6)
    return ramp_high_progress(env, env_ids=env_ids) + ramp_run


def obstacle_window(
    env: ManagerBasedRlEnv,
    before_high_edge: float = 0.25,
    after_high_edge: float = 0.35,
    env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前难度下需要抬升的进度窗口。"""
    high_edge = ramp_high_progress(env, env_ids=env_ids)
    start = torch.clamp(high_edge - float(before_high_edge), min=0.05)
    end = high_edge + float(after_high_edge)
    return start, end


def current_difficulty(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """把 terrain level 转成近似 difficulty，使用每个 row 的中点。"""
    levels, num_rows = current_terrain_levels(env, env_ids=env_ids)
    return torch.clamp((levels.float() + 0.5) / max(float(num_rows), 1.0), 0.0, 1.0)


def current_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """读取当前环境对应的 terrain level；无课程地形时返回 0。"""
    terrain = getattr(env.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(terrain_levels, torch.Tensor) and terrain_levels.numel() >= env.num_envs:
        levels = terrain_levels[env_ids].to(device=env.device, dtype=torch.long)
    else:
        levels = torch.zeros(len(env_ids), device=env.device, dtype=torch.long)
    if isinstance(terrain_origins, torch.Tensor) and terrain_origins.ndim >= 2:
        num_rows = int(terrain_origins.shape[0])
    else:
        num_rows = 1
    return levels, num_rows


def _lerp_tensor(value_range: tuple[float, float], progress: torch.Tensor) -> torch.Tensor:
    start, end = value_range
    return float(start) + (float(end) - float(start)) * progress


__all__ = [
    "FINAL_SUCCESS_DISTANCE",
    "current_difficulty",
    "current_success_distance",
    "current_terrain_levels",
    "obstacle_window",
    "ramp_high_progress",
    "ramp_low_progress",
]
