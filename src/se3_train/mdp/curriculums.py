"""SE3 轮腿机器人的课程学习函数。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from se3_train.mdp.commands import VelocityHeightCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_STEPS_PER_POLICY_ITER = 64


def _curriculum_progress(
    env: ManagerBasedRlEnv,
    *,
    use_iterations: bool,
    steps_per_policy_iter: int,
    offset_iter: int = 0,
) -> int:
    """返回课程进度；recovery 任务使用 PPO iter，普通任务沿用 policy step。"""
    step = int(getattr(env, "common_step_counter", 0))
    if not use_iterations:
        return step

    steps_per_iter = max(1, int(steps_per_policy_iter))
    return max(0, step // steps_per_iter - int(offset_iter))


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
) -> dict[str, torch.Tensor]:
    """按课程进度阶梯式扩大速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    threshold_key = "iteration" if use_iterations else "step"
    for stage in velocity_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            if "lin_vel_x_range" in stage:
                cfg.lin_vel_x_range = stage["lin_vel_x_range"]
            if "ang_vel_yaw_range" in stage:
                cfg.ang_vel_yaw_range = stage["ang_vel_yaw_range"]
    return {
        "step_counter": torch.tensor(float(getattr(env, "common_step_counter", 0))),
        "progress": torch.tensor(float(progress)),
        "lin_vel_x_max": torch.tensor(cfg.lin_vel_x_range[1]),
        "ang_vel_yaw_max": torch.tensor(cfg.ang_vel_yaw_range[1]),
    }


def commands_height(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    height_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
) -> dict[str, torch.Tensor]:
    """按课程进度逐步放开高度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    threshold_key = "iteration" if use_iterations else "step"
    for stage in height_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            if "height_range" in stage:
                cfg.height_range = stage["height_range"]
            if "standing_height_range" in stage:
                cfg.standing_height_range = stage["standing_height_range"]
            elif "height_range" in stage:
                cfg.standing_height_range = stage["height_range"]
    return {
        "step_counter": torch.tensor(float(getattr(env, "common_step_counter", 0))),
        "progress": torch.tensor(float(progress)),
        "height_min": torch.tensor(cfg.height_range[0]),
        "height_max": torch.tensor(cfg.height_range[1]),
        "standing_height_min": torch.tensor(cfg.standing_height_range[0]),
        "standing_height_max": torch.tensor(cfg.standing_height_range[1]),
    }


def push_disturbance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    push_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
) -> dict[str, torch.Tensor]:
    """按训练进度逐步增大推扰动强度。

    修改 env 上存储的 push velocity_range 配置。
    push_stages 格式:
    [{"step": 0, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, ...]
    """
    del env_ids
    step = int(getattr(env, "common_step_counter", 0))
    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    threshold_key = "iteration" if use_iterations else "step"
    current_max = 0.0

    for stage in push_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            velocity_range = stage["velocity_range"]
            current_max = max(max(abs(low), abs(high)) for low, high in velocity_range.values())
            env._push_velocity_range = velocity_range

    return {
        "step_counter": torch.tensor(float(step)),
        "progress": torch.tensor(float(progress)),
        "push_vel_max": torch.tensor(current_max),
    }


def terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    level_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
    terrain_type_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """按训练进度放开地形 row 难度，不依赖策略当前表现。"""
    env_ids = _as_env_id_tensor(env, env_ids)
    device = env.device
    zero = torch.tensor(0.0, device=device)
    if env_ids.numel() == 0:
        return {
            "level_mean": zero,
            "scheduled_max_level": zero,
            "scheduled_max_difficulty": zero,
            "active_rate": zero,
            "progress": zero,
        }

    terrain = getattr(env.scene, "terrain", None)
    terrain_levels_buf = getattr(terrain, "terrain_levels", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    env_origins = getattr(terrain, "env_origins", None)
    if (
        not isinstance(terrain_levels_buf, torch.Tensor)
        or not isinstance(terrain_origins, torch.Tensor)
        or not isinstance(terrain_types, torch.Tensor)
        or not isinstance(env_origins, torch.Tensor)
    ):
        return {
            "level_mean": zero,
            "scheduled_max_level": zero,
            "scheduled_max_difficulty": zero,
            "active_rate": zero,
            "progress": zero,
        }

    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    threshold_key = "iteration" if use_iterations else "step"
    max_level = 0
    max_difficulty = 0.0
    for stage in level_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            if "max_difficulty" in stage:
                max_difficulty = min(max(float(stage["max_difficulty"]), 0.0), 1.0)
                max_level = math.ceil(max_difficulty * int(terrain_origins.shape[0])) - 1
            else:
                max_level = int(stage["max_level"])
                max_difficulty = max_level / max(1, int(terrain_origins.shape[0]) - 1)

    max_level = max(0, min(max_level, int(terrain_origins.shape[0]) - 1))
    active = _terrain_type_mask(env, terrain_type_names, env_ids)
    active_env_ids = env_ids[active]
    if active_env_ids.numel() > 0:
        sampled_levels = torch.randint(
            0,
            max_level + 1,
            (active_env_ids.numel(),),
            device=device,
            dtype=terrain_levels_buf.dtype,
        )
        terrain_levels_buf[active_env_ids] = sampled_levels
        env_origins[active_env_ids] = terrain_origins[
            terrain_levels_buf[active_env_ids], terrain_types[active_env_ids]
        ]

    terrain_levels = getattr(terrain, "terrain_levels", terrain_levels_buf)
    level_mean = terrain_levels[env_ids].float().mean()
    return {
        "level_mean": level_mean,
        "scheduled_max_level": torch.tensor(float(max_level), device=device),
        "scheduled_max_difficulty": torch.tensor(float(max_difficulty), device=device),
        "active_rate": active.float().mean(),
        "progress": torch.tensor(float(progress), device=device),
    }


def _as_env_id_tensor(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    """把 manager 传入的 env_ids 统一成一维 long tensor。"""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def _terrain_type_mask(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...] | list[str] | None,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """根据 terrain type 名称筛选需要参与课程推进的 env。"""
    if not terrain_type_names:
        return torch.ones(env_ids.shape, device=env.device, dtype=torch.bool)

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return torch.zeros(env_ids.shape, device=env.device, dtype=torch.bool)

    cfg = getattr(terrain, "cfg", None)
    generator_cfg = getattr(cfg, "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    selected = {str(name) for name in terrain_type_names}
    mask = torch.zeros(env_ids.shape, device=env.device, dtype=torch.bool)
    env_terrain_types = terrain_types.to(device=env.device)[env_ids]
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if str(terrain_name) in selected:
            mask = mask | (env_terrain_types == terrain_index)
    return mask
