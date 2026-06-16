"""WheelDog 盲爬课程几何目标。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from se3_train.terrains import (
    BLIND_CLIMB_FACILITY_TERRAIN_TYPE,
    BLIND_CLIMB_FLAT_TERRAIN_TYPE,
    BLIND_CLIMB_MAX_DIFFICULTY,
    BLIND_CLIMB_PIT_LENGTH_RANGE,
    BLIND_CLIMB_RAMP_ANGLE_RANGE_DEG,
    BLIND_CLIMB_RAMP_HEIGHT_RANGE,
    BLIND_CLIMB_REFERENCE_NUM_ROWS,
    GapRampFacilitySpec,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_SPEC = GapRampFacilitySpec()
POST_RAMP_RUNOUT = 1.0
"""到达坡底后仍需继续前进的距离。"""

CORRIDOR_SOFT_HALF_WIDTH = 0.35
"""中心通道软半宽；超过后开始惩罚横向绕路。"""

CORRIDOR_SUCCESS_HALF_WIDTH = 0.55
"""有效完成判定的中心通道半宽。"""

CORRIDOR_HARD_HALF_WIDTH = 0.75
"""中心通道硬半宽；超过后终止 episode。"""


def _lerp_scalar(value_range: tuple[float, float], progress: float) -> float:
    start, end = value_range
    return float(start) + (float(end) - float(start)) * float(progress)


_MAX_PIT_LENGTH = _lerp_scalar(BLIND_CLIMB_PIT_LENGTH_RANGE, BLIND_CLIMB_MAX_DIFFICULTY)
_MAX_RAMP_HEIGHT = _lerp_scalar(BLIND_CLIMB_RAMP_HEIGHT_RANGE, BLIND_CLIMB_MAX_DIFFICULTY)
_MAX_RAMP_ANGLE_DEG = _lerp_scalar(BLIND_CLIMB_RAMP_ANGLE_RANGE_DEG, BLIND_CLIMB_MAX_DIFFICULTY)
_MAX_RAMP_RUN = _MAX_RAMP_HEIGHT / math.tan(math.radians(_MAX_RAMP_ANGLE_DEG))

FINAL_SUCCESS_DISTANCE = (
    _SPEC.left_platform_length * 0.5 + _MAX_PIT_LENGTH + _MAX_RAMP_RUN + POST_RAMP_RUNOUT
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


def lateral_offset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 base_link 相对设施中心线的横向偏移。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    robot = env.scene["robot"]
    offset = robot.data.root_link_pos_w[env_ids, 1] - env.scene.env_origins[env_ids, 1]
    return torch.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0)


def corridor_gate(
    env: ManagerBasedRlEnv,
    soft_half_width: float = CORRIDOR_SOFT_HALF_WIDTH,
    hard_half_width: float = CORRIDOR_HARD_HALF_WIDTH,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回中心通道门控，中心为 1，硬边界外为 0。"""
    abs_y = torch.abs(lateral_offset(env, env_ids=env_ids))
    soft = float(soft_half_width)
    hard = max(float(hard_half_width), soft + 1.0e-6)
    return torch.clamp((hard - abs_y) / (hard - soft), min=0.0, max=1.0)


def within_corridor(
    env: ManagerBasedRlEnv,
    half_width: float = CORRIDOR_SUCCESS_HALF_WIDTH,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """判断 base_link 是否仍在有效中心通道内。"""
    return torch.abs(lateral_offset(env, env_ids=env_ids)) <= float(half_width)


def current_terrain_types(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor | None:
    """读取当前环境对应的 terrain type；没有课程地形时返回 None。"""
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return None
    if terrain_types.numel() < env.num_envs:
        return None
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    return terrain_types[env_ids].to(device=env.device, dtype=torch.long)


def _terrain_type_mask(
    env: ManagerBasedRlEnv,
    terrain_type: int,
    env_ids: torch.Tensor | slice | None = None,
    *,
    default_value: bool,
) -> torch.Tensor:
    """按 terrain type 生成布尔掩码。"""
    terrain_types = current_terrain_types(env, env_ids=env_ids)
    if terrain_types is None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        return torch.full(
            (len(env_ids),),
            bool(default_value),
            device=env.device,
            dtype=torch.bool,
        )
    return terrain_types == int(terrain_type)


def is_flat_terrain(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
    """判断当前环境是否属于平地列。"""
    return _terrain_type_mask(
        env,
        BLIND_CLIMB_FLAT_TERRAIN_TYPE,
        env_ids=env_ids,
        default_value=False,
    )


def is_facility_terrain(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
    """判断当前环境是否属于坑坡设施列。"""
    return _terrain_type_mask(
        env,
        BLIND_CLIMB_FACILITY_TERRAIN_TYPE,
        env_ids=env_ids,
        default_value=True,
    )


def current_difficulty(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """把 terrain level 转成 difficulty，保持 level 39 对应旧满难度。"""
    play_difficulty = _play_terrain_difficulty(env)
    if play_difficulty is not None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        return torch.full(
            (len(env_ids),),
            float(play_difficulty),
            device=env.device,
            dtype=torch.float,
        )
    levels, num_rows = current_terrain_levels(env, env_ids=env_ids)
    del num_rows
    reference_rows = max(float(BLIND_CLIMB_REFERENCE_NUM_ROWS), 1.0)
    return torch.clamp((levels.float() + 0.5) / reference_rows, 0.0, BLIND_CLIMB_MAX_DIFFICULTY)


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


def _play_terrain_difficulty(env: ManagerBasedRlEnv) -> float | None:
    """读取 se3-play 注入的固定地形难度。"""
    terrain = getattr(env.scene, "terrain", None)
    cfg = getattr(terrain, "cfg", None)
    value = getattr(cfg, "play_terrain_difficulty", None)
    if value is None:
        return None
    return max(0.0, min(float(value), float(BLIND_CLIMB_MAX_DIFFICULTY)))


__all__ = [
    "CORRIDOR_HARD_HALF_WIDTH",
    "CORRIDOR_SOFT_HALF_WIDTH",
    "CORRIDOR_SUCCESS_HALF_WIDTH",
    "FINAL_SUCCESS_DISTANCE",
    "corridor_gate",
    "current_difficulty",
    "current_success_distance",
    "current_terrain_levels",
    "current_terrain_types",
    "is_facility_terrain",
    "is_flat_terrain",
    "lateral_offset",
    "obstacle_window",
    "ramp_high_progress",
    "ramp_low_progress",
    "within_corridor",
]
