"""SE3 轮腿机器人的课程学习函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from se3_train.mdp.commands import VelocityHeightCommandCfg

_FLOW_MATCH_GRU_STEPS_PER_ENV = 64
_DEFAULT_STEPS_PER_POLICY_ITER = 64


def _lerp(
    progress_step: int,
    start_step: int,
    ramp_steps: int,
    initial_value: float,
    final_value: float,
) -> float:
    """按 step 线性插值。"""
    if progress_step < start_step:
        return float(initial_value)
    if ramp_steps <= 0:
        return float(final_value)
    progress = min(1.0, max(0.0, (progress_step - start_step) / ramp_steps))
    return float(initial_value) + (float(final_value) - float(initial_value)) * progress


def _stage_progress(stage: dict) -> int:
    """读取课程阶段阈值，优先使用 PPO iter，兼容旧 step 键。"""
    if "iter" in stage:
        return int(stage["iter"])
    return int(stage["step"])


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


def _set_reward_weight(env: ManagerBasedRlEnv, term_name: str, weight: float) -> float | None:
    """运行期更新 RewardManager 中指定项权重；缺项时跳过。"""
    reward_manager = getattr(env, "reward_manager", None)
    if reward_manager is None or term_name not in reward_manager.active_terms:
        return None
    reward_manager.get_term_cfg(term_name).weight = float(weight)
    return float(weight)


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
    step = int(getattr(env, "common_step_counter", 0))
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
        "step_counter": torch.tensor(float(step)),
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
    step = int(getattr(env, "common_step_counter", 0))
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
        "step_counter": torch.tensor(float(step)),
        "progress": torch.tensor(float(progress)),
        "height_min": torch.tensor(cfg.height_range[0]),
        "height_max": torch.tensor(cfg.height_range[1]),
        "standing_height_min": torch.tensor(cfg.standing_height_range[0]),
        "standing_height_max": torch.tensor(cfg.standing_height_range[1]),
    }


def wheel_expert_motion_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    profile_stages: list[dict],
    reward_weight_stages: list[dict],
    steps_per_iter: int = _FLOW_MATCH_GRU_STEPS_PER_ENV,
) -> dict[str, torch.Tensor]:
    """WHEEL 专家课程：按 PPO iter 同步速度、画像比例和专项惩罚权重。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg = term.cfg
    step = env.common_step_counter
    policy_iter = int(step) // max(int(steps_per_iter), 1)

    for stage in velocity_stages:
        if policy_iter >= _stage_progress(stage):
            if "lin_vel_x_range" in stage:
                cfg.lin_vel_x_range = stage["lin_vel_x_range"]
            if "ang_vel_yaw_range" in stage:
                cfg.ang_vel_yaw_range = stage["ang_vel_yaw_range"]

    current_profiles = getattr(cfg, "wheel_profile_probabilities", (1.0, 0.0, 0.0))
    for stage in profile_stages:
        if policy_iter >= _stage_progress(stage):
            current_profiles = stage["wheel_profile_probabilities"]
    cfg.wheel_profile_probabilities = tuple(float(value) for value in current_profiles)

    logs: dict[str, torch.Tensor] = {
        "step_counter": torch.tensor(float(step)),
        "policy_iter": torch.tensor(float(policy_iter)),
        "lin_vel_x_max": torch.tensor(float(cfg.lin_vel_x_range[1])),
        "ang_vel_yaw_max": torch.tensor(float(cfg.ang_vel_yaw_range[1])),
        "wheel_profile_mixed": torch.tensor(float(cfg.wheel_profile_probabilities[0])),
        "wheel_profile_straight": torch.tensor(float(cfg.wheel_profile_probabilities[1])),
        "wheel_profile_turn": torch.tensor(float(cfg.wheel_profile_probabilities[2])),
    }

    for stage in reward_weight_stages:
        term_name = stage["term_name"]
        start_progress = int(stage.get("start_iter", stage.get("start_step", 0)))
        ramp_progress = int(stage.get("ramp_iters", stage.get("ramp_steps", 0)))
        weight = _lerp(
            int(policy_iter),
            start_progress,
            ramp_progress,
            float(stage["initial_weight"]),
            float(stage["final_weight"]),
        )
        applied = _set_reward_weight(env, term_name, weight)
        if applied is not None:
            logs[f"{term_name}_weight"] = torch.tensor(applied)

    return logs


def commands_vel_linear(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start_step: int,
    end_step: int,
    start_lin_vel_x_range: tuple[float, float],
    end_lin_vel_x_range: tuple[float, float],
    ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0),
) -> dict[str, torch.Tensor]:
    """按训练步数线性扩大前向速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    step = env.common_step_counter
    span = max(end_step - start_step, 1)
    progress = min(max((step - start_step) / span, 0.0), 1.0)
    cfg.lin_vel_x_range = (
        start_lin_vel_x_range[0] + progress * (end_lin_vel_x_range[0] - start_lin_vel_x_range[0]),
        start_lin_vel_x_range[1] + progress * (end_lin_vel_x_range[1] - start_lin_vel_x_range[1]),
    )
    cfg.ang_vel_yaw_range = ang_vel_yaw_range
    return {
        "step_counter": torch.tensor(float(step)),
        "lin_vel_x_min": torch.tensor(cfg.lin_vel_x_range[0]),
        "lin_vel_x_max": torch.tensor(cfg.lin_vel_x_range[1]),
        "ang_vel_yaw_max": torch.tensor(cfg.ang_vel_yaw_range[1]),
        "progress": torch.tensor(progress),
    }


def terrain_distribution_linear(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    start_proportions: tuple[float, ...],
    end_proportions: tuple[float, ...],
    start_max_level: int = 0,
    end_max_level: int | None = None,
) -> dict[str, torch.Tensor]:
    """按训练步数线性调整地形类型占比和最大难度等级。"""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter)),
            "progress": torch.tensor(0.0),
            "non_flat_ratio": torch.tensor(0.0),
            "max_level": torch.tensor(0.0),
        }

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return {
            "step_counter": torch.tensor(float(env.common_step_counter)),
            "progress": torch.tensor(0.0),
            "non_flat_ratio": torch.tensor(0.0),
            "max_level": torch.tensor(0.0),
        }

    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    if len(start_proportions) != num_cols or len(end_proportions) != num_cols:
        raise ValueError(
            "地形占比数量必须和 terrain columns 一致: "
            f"start={len(start_proportions)}, end={len(end_proportions)}, cols={num_cols}"
        )

    step = env.common_step_counter
    span = max(end_step - start_step, 1)
    progress = min(max((step - start_step) / span, 0.0), 1.0)

    start_probs = torch.tensor(start_proportions, device=env.device, dtype=torch.float)
    end_probs = torch.tensor(end_proportions, device=env.device, dtype=torch.float)
    probs = start_probs + progress * (end_probs - start_probs)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / torch.clamp(probs.sum(), min=1.0e-6)

    target_end_level = num_rows - 1 if end_max_level is None else min(end_max_level, num_rows - 1)
    max_level = round(start_max_level + progress * (target_end_level - start_max_level))
    max_level = max(0, min(max_level, num_rows - 1))

    terrain.terrain_types[env_ids] = torch.multinomial(probs, len(env_ids), replacement=True)
    terrain.terrain_levels[env_ids] = torch.randint(
        0,
        max_level + 1,
        (len(env_ids),),
        device=env.device,
    )
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    terrain_types = terrain.terrain_types[env_ids]
    non_flat = terrain_types != 0
    obstacle = terrain_types >= 2
    extras = {
        "step_counter": torch.tensor(float(step)),
        "progress": torch.tensor(progress),
        "flat_ratio": torch.mean((terrain_types == 0).float()),
        "rough_ratio": torch.mean((terrain_types == 1).float()),
        "obstacle_ratio": torch.mean(obstacle.float()),
        "non_flat_ratio": torch.mean(non_flat.float()),
        "max_level": torch.tensor(float(max_level)),
    }
    if num_cols >= 3:
        extras["box_ratio"] = torch.mean((terrain_types == 2).float())
    if num_cols >= 4:
        extras["stairs_up_ratio"] = torch.mean((terrain_types == 3).float())
        extras["stairs_ratio"] = torch.mean((terrain_types >= 3).float())
    if num_cols >= 5:
        extras["stairs_down_ratio"] = torch.mean((terrain_types == 4).float())
    return extras


def gait_terrain_distribution_linear(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    start_proportions: tuple[float, ...],
    end_proportions: tuple[float, ...],
    start_max_level: int = 0,
    end_max_level: int | None = None,
) -> dict[str, torch.Tensor]:
    """兼容旧 GAIT 课程名，实际使用通用地形占比课程。"""
    return terrain_distribution_linear(
        env=env,
        env_ids=env_ids,
        start_step=start_step,
        end_step=end_step,
        start_proportions=start_proportions,
        end_proportions=end_proportions,
        start_max_level=start_max_level,
        end_max_level=end_max_level,
    )


def push_disturbance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    push_stages: list[dict],
) -> dict[str, torch.Tensor]:
    """按训练步数逐步增大推扰动强度。

    修改 env 上存储的 push velocity_range 配置。
    push_stages 格式: [{"step": 0, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, ...]
    """
    del env_ids
    step = env.common_step_counter
    current_velocity_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

    for stage in push_stages:
        if step >= stage["step"]:
            current_velocity_range = stage["velocity_range"]

    # interval push 事件读取这个动态配置，避免课程项只更新监控值而不影响实际扰动。
    env._push_velocity_range = current_velocity_range
    current_max = max(
        max(abs(axis_range[0]), abs(axis_range[1]))
        for axis_range in current_velocity_range.values()
    )

    return {
        "step_counter": torch.tensor(float(step)),
        "push_vel_max": torch.tensor(current_max),
    }
