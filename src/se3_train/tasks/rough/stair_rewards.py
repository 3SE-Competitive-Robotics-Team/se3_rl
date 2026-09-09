"""倒金字塔上台阶的新增进度与逐阶双轮支撑奖励，不接管策略动作。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .rewards import terrain_column_mask

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _buffer(env: ManagerBasedRlEnv, name: str, initial: float) -> torch.Tensor:
    """按环境保存本轮累计量；首次观测作为基准，避免 reset 位置被奖励。"""
    if not hasattr(env, name):
        setattr(env, name, torch.full((env.num_envs,), initial, device=env.device))
    return getattr(env, name)


def reset_stair_rewards(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """只清零 reset 环境的进度、已领奖阶数和接触保持计数。"""
    ids = slice(None) if env_ids is None else env_ids
    for name, initial in (
        ("_rough_stair_max_progress", -1.0),
        ("_rough_stair_paid_steps", -1.0),
        ("_rough_stair_candidate", 0.0),
        ("_rough_stair_hold", 0.0),
    ):
        if hasattr(env, name):
            getattr(env, name)[ids] = initial


def _geometry(env: ManagerBasedRlEnv, terrain_type_names: tuple[str, ...]):
    """按实际列配置与行难度计算阶高、中心平台边界和可奖励的最大进度。"""
    terrain = env.scene.terrain
    mask = terrain_column_mask(env, terrain_type_names)
    height = torch.ones(env.num_envs, device=env.device)
    start = torch.zeros_like(height)
    length = torch.zeros_like(height)
    count = torch.zeros_like(height)
    if mask is None:
        return torch.zeros_like(height, dtype=torch.bool), height, start, length, count
    generator = terrain.cfg.terrain_generator
    alpha = terrain.terrain_levels.float() / max(1, generator.num_rows - 1)
    for index, (name, cfg) in enumerate(generator.sub_terrains.items()):
        if name not in terrain_type_names:
            continue
        selected = terrain.terrain_types == index
        n = max(
            0,
            int((min(cfg.size) - 2 * cfg.border_width - cfg.platform_width) / (2 * cfg.step_width)),
        )
        inner_half = min(cfg.size) / 2 - cfg.border_width
        # 最外侧边框比最后踏面再高一阶，故总抬升次数是 n+1。
        height[selected] = cfg.step_height_range[0] + alpha[selected] * (
            cfg.step_height_range[1] - cfg.step_height_range[0]
        )
        start[selected] = inner_half - n * cfg.step_width
        length[selected] = n * cfg.step_width
        count[selected] = n + 1
    return mask, height, start, length, count


def _upright(env: ManagerBasedRlEnv) -> torch.Tensor:
    pg_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    return torch.nan_to_num((-pg_z).clamp(0, 0.7) / 0.7, nan=0.0)


def stair_climb_progress(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
) -> torch.Tensor:
    """只奖励平台外新增的最大切比雪夫进度；后退再走回不会重复收酬。"""
    mask, _, start, length, _ = _geometry(env, terrain_type_names)
    offset = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    distance = offset.abs().amax(dim=1)
    progress = torch.minimum((distance - start).clamp(min=0), length)
    progress = torch.where(mask & torch.isfinite(progress), progress, 0.0)
    previous = _buffer(env, "_rough_stair_max_progress", -1.0)
    initialized = previous >= 0
    updated = torch.maximum(previous, progress)
    delta = torch.where(initialized & mask, updated - previous, 0.0)
    previous.copy_(updated.detach())
    return delta / env.step_dt * _upright(env)


def stair_support_height(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    height_sensor_name: str = "stair_reward_height",
    contact_sensor_name: str = "stair_reward_contact",
    contact_force_threshold_n: float = 5.0,
    wheel_radius_m: float = 0.06,
    wheel_clearance_tol_m: float = 0.025,
    height_tolerance_m: float = 0.015,
    hold_time_s: float = 0.10,
) -> torch.Tensor:
    """双轮持续支撑更高踏面后，每阶每 episode 只奖励一次，站住或反复上下不刷分。"""
    mask, step_height, _, _, count = _geometry(env, terrain_type_names)
    robot = env.scene["robot"]
    wheel_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    heights = env.scene[height_sensor_name].data.heights.reshape(env.num_envs, 2)
    wheel_z = robot.data.body_link_pos_w[:, wheel_ids, 2]
    rise = wheel_z - heights - env.scene.env_origins[:, 2:3]
    sensor = env.scene[contact_sensor_name].data
    force = sensor.force.reshape(env.num_envs, 2, -1, 3)
    normal = sensor.normal.reshape(env.num_envs, 2, -1, 3)
    contact = torch.linalg.vector_norm(force, dim=-1) >= contact_force_threshold_n
    contact &= sensor.found.reshape(env.num_envs, 2, -1) > 0
    top = (contact & (normal[..., 2].abs() > 0.5)).any(dim=-1)
    riser = (contact & (normal[..., 2].abs() <= 0.5)).any(dim=-1)
    near = (
        torch.isfinite(heights)
        & (heights >= 0)
        & (heights <= wheel_radius_m + wheel_clearance_tol_m)
    )
    supported = (top & ~riser & near & torch.isfinite(rise)).all(dim=1)
    # 低阶高时限制容差比例，避免 2cm 台阶被 1.5cm 容差吞掉。
    tolerance = torch.minimum(torch.full_like(step_height, height_tolerance_m), step_height * 0.25)
    steps = torch.floor((rise.amin(dim=1) + tolerance) / step_height.clamp(min=1e-6)).clamp(min=0)
    upright = _upright(env)
    candidate = torch.where(mask & supported & (upright >= 1.0), torch.minimum(steps, count), 0.0)
    paid = _buffer(env, "_rough_stair_paid_steps", -1.0)
    previous_candidate = _buffer(env, "_rough_stair_candidate", 0.0)
    hold = _buffer(env, "_rough_stair_hold", 0.0)
    initialized = paid >= 0
    paid[~initialized] = candidate[~initialized]
    hold.copy_(
        torch.where(
            (candidate > 0) & (candidate == previous_candidate), hold + 1, (candidate > 0).float()
        )
    )
    previous_candidate.copy_(candidate.detach())
    ready = hold >= max(1, math.ceil(hold_time_s / env.step_dt))
    gain = torch.where(ready & initialized, (candidate - paid).clamp(min=0), 0.0)
    paid.add_(gain.detach())
    log = env.extras.setdefault("log", {})
    denom = mask.sum().clamp(min=1)
    log["Rough/stair_supported_steps"] = candidate.sum() / denom
    log["Rough/stair_new_supported_steps"] = gain.sum() / denom
    return gain / env.step_dt
