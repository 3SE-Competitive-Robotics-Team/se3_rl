"""本任务的 critic 特权观测：倒地约束与倒地教师的状态。

倒地约束（2026-10-22 起，见 rewards.fall_lagrange_penalty）：倒地出生、尚未起身的 env（M）奖励被 1/(1+Σλ) 缩放并带拉格朗日罚，
走路时摔倒的 env 不缩放、不罚；同样的倒地姿态两种奖励口径，critic 不知道属于哪种就会估混。
倒地教师（2026-10-23 起，见 fall_teacher.py）：教师接管时执行动作被混入教师动作、另有模仿奖励，α 随轮次退火，
critic 同样要知道当前是否被接管以及 α。这些维度只进 critic 组，不进 actor 与 CTS 教师编码器（部署时没有 M 与教师）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .fall_teacher import fall_teacher_of
from .rewards import fall_constraint_active, fall_lagrange_scale

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def fall_constraint_active_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：本 env 是否在 M 里（倒地出生、尚未起身）。"""
    return fall_constraint_active(env).float().unsqueeze(1)


def fall_lagrange_scale_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：当前 M 内奖励缩放 1/(1+Σλ)（所有 env 同值）。"""
    return torch.full((env.num_envs, 1), fall_lagrange_scale(env), device=env.device)


def fall_teacher_active_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：本步倒地教师是否接管本 env；没有教师时为 0。"""
    teacher = fall_teacher_of(env)
    if teacher is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return teacher.active.float().unsqueeze(1)


def fall_teacher_alpha_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：当前教师混合系数 α（所有 env 同值）；没有教师时为 0。"""
    teacher = fall_teacher_of(env)
    alpha = 0.0 if teacher is None else float(teacher.alpha_value)
    return torch.full((env.num_envs, 1), alpha, device=env.device)


__all__ = [
    "fall_constraint_active_obs",
    "fall_lagrange_scale_obs",
    "fall_teacher_active_obs",
    "fall_teacher_alpha_obs",
]
