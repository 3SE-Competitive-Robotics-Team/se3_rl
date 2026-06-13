"""台阶任务使用的课程函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.curriculums import commands_height, commands_vel, push_disturbance

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height


def stair_terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    asset_name: str = "robot",
    standing_height: float = _DEFAULT_STANDING_HEIGHT,
    move_up_distance_ratio: float = 0.35,
    move_down_distance_ratio: float = 0.12,
    upright_threshold: float = -0.5,
    terrain_type_names: tuple[str, ...] = ("inv_pyramid_stairs",),
    walking_phase_iterations: int = 0,
    flat_terrain_type_name: str = "flat",
    steps_per_policy_iter: int = 64,
) -> dict[str, torch.Tensor]:
    """仅根据倒金字塔完整出坑表现升降地形 row。"""
    terrain = env.scene.terrain
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
        return _zero_log(env)

    ids = _env_ids_tensor(env, env_ids)
    if ids.numel() == 0:
        return _zero_log(env)

    iteration = int(getattr(env, "common_step_counter", 0)) // max(1, int(steps_per_policy_iter))
    if iteration < max(0, int(walking_phase_iterations)):
        _set_walking_phase_terrain(
            env,
            terrain,
            ids,
            flat_terrain_type_name=flat_terrain_type_name,
        )
        logs = _zero_log(env)
        logs["walking_phase"] = torch.tensor(1.0, device=env.device)
        logs["iteration"] = torch.tensor(float(iteration), device=env.device)
        return logs

    newly_entered = _restore_training_terrain_types(env, terrain, ids)
    robot = env.scene[asset_name]
    origins = env.scene.env_origins[ids]
    root_pos = robot.data.root_link_pos_w[ids]
    pg_z = robot.data.projected_gravity_b[ids, 2]
    valid_episode = (env.episode_length_buf[ids] > 0) & (~newly_entered)
    stair_mask = _terrain_type_mask(terrain, ids, terrain_type_names, env.device)

    height_gain = (root_pos[:, 2] - origins[:, 2]) - float(standing_height)
    distance = torch.norm(root_pos[:, :2] - origins[:, :2], dim=1)
    terrain_size_x = float(
        getattr(getattr(terrain.cfg, "terrain_generator", None), "size", (8.0, 8.0))[0]
    )
    move_up_distance = terrain_size_x * float(move_up_distance_ratio)
    move_down_distance = terrain_size_x * float(move_down_distance_ratio)

    upright = pg_z < float(upright_threshold)
    move_up = stair_mask & valid_episode & upright & (distance > move_up_distance)
    move_down = stair_mask & valid_episode & ((~upright) | (distance < move_down_distance))
    move_down &= ~move_up

    _update_env_origins(terrain, ids, move_up=move_up, move_down=move_down)

    levels = terrain.terrain_levels[ids].float()
    valid_stair = stair_mask & valid_episode
    level_max = (
        torch.max(levels[stair_mask])
        if torch.any(stair_mask)
        else torch.tensor(0.0, device=env.device)
    )
    return {
        "level_mean": _masked_mean(levels, stair_mask),
        "level_max": level_max,
        "move_up_rate": _masked_mean(move_up.float(), valid_stair),
        "move_down_rate": _masked_mean(move_down.float(), valid_stair),
        "height_gain_mean": _masked_mean(height_gain, valid_stair),
        "distance_mean": _masked_mean(distance, valid_stair),
        "stair_env_rate": torch.mean(stair_mask.float()),
        "walking_phase": torch.tensor(0.0, device=env.device),
        "iteration": torch.tensor(float(iteration), device=env.device),
    }


def _set_walking_phase_terrain(
    env: ManagerBasedRlEnv,
    terrain,
    ids: torch.Tensor,
    *,
    flat_terrain_type_name: str,
) -> None:
    original_types = getattr(env, "_stair_training_terrain_types", None)
    if not isinstance(original_types, torch.Tensor):
        env._stair_training_terrain_types = terrain.terrain_types.clone()
        env._stair_training_phase_started = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    flat_type = _terrain_type_index(terrain, flat_terrain_type_name)
    terrain.terrain_levels[ids] = 0
    terrain.terrain_types[ids] = flat_type
    terrain.env_origins[ids] = terrain.terrain_origins[0, flat_type]
    env._stair_training_phase_started[ids] = False


def _restore_training_terrain_types(
    env: ManagerBasedRlEnv,
    terrain,
    ids: torch.Tensor,
) -> torch.Tensor:
    original_types = getattr(env, "_stair_training_terrain_types", None)
    phase_started = getattr(env, "_stair_training_phase_started", None)
    if not isinstance(original_types, torch.Tensor) or not isinstance(phase_started, torch.Tensor):
        return torch.zeros(ids.shape, device=env.device, dtype=torch.bool)

    newly_entered = ~phase_started[ids]
    terrain.terrain_types[ids] = original_types[ids]
    terrain.env_origins[ids] = terrain.terrain_origins[
        terrain.terrain_levels[ids],
        terrain.terrain_types[ids],
    ]
    phase_started[ids] = True
    return newly_entered


def _terrain_type_index(terrain, terrain_type_name: str) -> int:
    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_names = tuple((getattr(generator_cfg, "sub_terrains", {}) or {}).keys())
    if terrain_type_name not in terrain_names:
        raise ValueError(f"未知地形类型: {terrain_type_name}")
    return terrain_names.index(terrain_type_name)


def _update_env_origins(
    terrain,
    ids: torch.Tensor,
    *,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
) -> None:
    if hasattr(terrain, "update_env_origins"):
        terrain.update_env_origins(ids, move_up=move_up, move_down=move_down)
        return

    num_rows = int(terrain.terrain_origins.shape[0])
    selected = ids[move_up | move_down]
    if selected.numel() == 0:
        return
    levels = terrain.terrain_levels[selected]
    levels[move_up[move_up | move_down]] += 1
    levels[move_down[move_up | move_down]] -= 1
    terrain.terrain_levels[selected] = torch.clamp(levels, 0, num_rows - 1)
    terrain.env_origins[selected] = terrain.terrain_origins[
        terrain.terrain_levels[selected],
        terrain.terrain_types[selected],
    ]


def _env_ids_tensor(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)


def _terrain_type_mask(
    terrain,
    env_ids: torch.Tensor,
    terrain_type_names: tuple[str, ...],
    device: torch.device | str,
) -> torch.Tensor:
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return torch.zeros(env_ids.shape, device=device, dtype=torch.bool)

    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    selected = {str(name) for name in terrain_type_names}
    env_terrain_types = terrain_types.to(device=device)[env_ids]
    mask = torch.zeros(env_ids.shape, device=device, dtype=torch.bool)
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if str(terrain_name) in selected:
            mask |= env_terrain_types == terrain_index
    return mask


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return torch.tensor(0.0, device=value.device)
    return value[mask].float().mean()


def _zero_log(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, device=env.device)
    return {
        "level_mean": zero,
        "level_max": zero,
        "move_up_rate": zero,
        "move_down_rate": zero,
        "height_gain_mean": zero,
        "distance_mean": zero,
        "stair_env_rate": zero,
        "walking_phase": zero,
        "iteration": zero,
    }


__all__ = [
    "commands_height",
    "commands_vel",
    "push_disturbance",
    "stair_terrain_levels",
]
