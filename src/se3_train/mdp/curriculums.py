"""SE3 轮腿机器人的课程学习函数。"""

from __future__ import annotations

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
    current_max = 0.5

    for stage in push_stages:
        if step >= stage["step"]:
            velocity_range = stage["velocity_range"]
            current_max = max(abs(velocity_range["x"][0]), abs(velocity_range["x"][1]))
            if hasattr(env, "_push_velocity_range"):
                env._push_velocity_range = velocity_range

    return {
        "step_counter": torch.tensor(float(step)),
        "push_vel_max": torch.tensor(current_max),
    }
