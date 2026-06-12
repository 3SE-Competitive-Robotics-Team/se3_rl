"""台阶任务奖励和诊断函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.tasks.flat.rewards import *  # noqa: F403
from se3_train.tasks.flat.rewards import __all__ as _FLAT_REWARD_ALL

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_STAIR_TERRAIN_TYPES = ("pyramid_stairs",)
_BASELINE_Z_ATTR = "_se3_stair_episode_base_z"


def _upright_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回直立程度门控，0 表示明显翻倒，1 表示接近直立。"""
    robot = env.scene["robot"]
    return torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def _terrain_type_mask(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...],
) -> torch.Tensor:
    """返回指定地形类型的环境掩码。"""
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    if terrain_types is None or not sub_terrains:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    selected = {str(name) for name in terrain_type_names}
    env_terrain_types = terrain_types.to(device=env.device)
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if str(terrain_name) in selected:
            mask |= env_terrain_types == terrain_index
    return mask


def _episode_baseline_z(env: ManagerBasedRlEnv, root_z: torch.Tensor) -> torch.Tensor:
    """返回每个 episode 的初始 base 高度，用作真实爬升基准。"""
    baseline = getattr(env, _BASELINE_Z_ATTR, None)
    if not isinstance(baseline, torch.Tensor) or baseline.shape != root_z.shape:
        baseline = root_z.detach().clone()
        setattr(env, _BASELINE_Z_ATTR, baseline)

    reset_mask = env.episode_length_buf <= 1
    if reset_mask.any():
        baseline[reset_mask] = root_z.detach()[reset_mask]
    return baseline


def stair_height_gain(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """机器人 base 相对出生站立高度的爬升增量。"""
    del command_name
    robot = env.scene["robot"]
    root_z = robot.data.root_link_pos_w[:, 2]
    gain = root_z - _episode_baseline_z(env, root_z)
    gain = torch.clamp(gain, min=0.0)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    return torch.where(terrain_mask, gain, torch.zeros_like(gain)) * _upright_gate(env)


def stair_steps_climbed(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    step_height: float | None = None,
    step_height_range: tuple[float, float] = (0.05, 0.20),
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """按当前 row 台阶高度估算已爬台阶级数。"""
    height_gain = stair_height_gain(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
    )
    if step_height is None:
        terrain_level = stair_terrain_level(env)
        terrain_generator = getattr(
            getattr(getattr(env.scene, "terrain", None), "cfg", None),
            "terrain_generator",
            None,
        )
        num_rows = max(1, int(getattr(terrain_generator, "num_rows", 10)) - 1)
        step_height_tensor = float(step_height_range[0]) + (
            torch.clamp(terrain_level, min=0.0, max=float(num_rows))
            / float(num_rows)
            * (float(step_height_range[1]) - float(step_height_range[0]))
        )
    else:
        step_height_tensor = torch.full_like(height_gain, float(step_height))
    return height_gain / torch.clamp(step_height_tensor, min=1.0e-6)


def stair_terrain_level(env: ManagerBasedRlEnv) -> torch.Tensor:
    """当前台阶课程 row，0 最低，数值越大台阶越高。"""
    terrain = getattr(env.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if isinstance(terrain_levels, torch.Tensor):
        return terrain_levels.to(device=env.device).float()
    return torch.zeros(env.num_envs, device=env.device)


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """台阶场景的前进速度跟踪，不惩罚爬升所需的 z 速度。"""
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = (robot.data.root_link_lin_vel_b[:, 0] - command[:, 0]) ** 2
    return torch.exp(-error / float(sigma)) * _upright_gate(env)


def stair_diagnostics(
    env: ManagerBasedRlEnv,
    command_name: str | None = "velocity_height",
    terrain_type_names: tuple[str, ...] = _STAIR_TERRAIN_TYPES,
) -> torch.Tensor:
    """把台阶关键指标写入训练日志，不直接改变奖励。"""
    height_gain = stair_height_gain(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
    )
    steps_climbed = stair_steps_climbed(
        env,
        command_name=command_name,
        terrain_type_names=terrain_type_names,
    )
    terrain_level = stair_terrain_level(env)
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/obs_steps_climbed": steps_climbed.mean().item(),
                "Stair/obs_height_gain": height_gain.mean().item(),
                "Stair/obs_terrain_level": terrain_level.mean().item(),
                "Stair/diag_stair_env_rate": terrain_mask.float().mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


__all__ = [
    *_FLAT_REWARD_ALL,
    "stair_diagnostics",
    "stair_forward_progress",
    "stair_height_gain",
    "stair_steps_climbed",
    "stair_terrain_level",
]
