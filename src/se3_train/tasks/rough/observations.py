"""本任务的特权观测：倒地教师的状态（critic），以及没有 recover 组时的起身掩码（拉格朗日倒地约束的 M 标记与缩放两维
2026-10-25 随约束一起删除）。

倒地教师（2026-10-23 起，见 fall_teacher.py）：教师接管时执行动作被混入教师动作、另有模仿奖励，α 随轮次退火，
critic 同样要知道当前是否被接管以及 α。这些维度只进 critic 组，不进 actor 与 CTS 教师编码器（部署时没有 M 与教师）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .fall_teacher import fall_teacher_of
from .rewards import fall_constraint_active

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def fall_recovering_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：本 env 是否倒地出生、尚未起身（M，见 rewards.fall_constraint_active）。没有 recover 组时作起身掩码
    （env_cfg.ROUGH_RECOVER_MASK_GROUP），给 AMP 与网络老师蒸馏读；不进 actor / critic / CTS 教师组。"""
    return fall_constraint_active(env).float().unsqueeze(1)


def fall_teacher_active_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：下一步倒地教师是否接管执行本 env（观测时刻已准备好）；没有教师或 α = 0 时为 0。"""
    teacher = fall_teacher_of(env)
    if teacher is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    teacher.prepare()
    executing = teacher.active & (float(teacher.alpha_value) > 0.0)
    return executing.float().unsqueeze(1)


def fall_teacher_alpha_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1]：当前教师混合系数 α（所有 env 同值）；没有教师时为 0。"""
    teacher = fall_teacher_of(env)
    if teacher is not None:
        teacher.prepare()
    alpha = 0.0 if teacher is None else float(teacher.alpha_value)
    return torch.full((env.num_envs, 1), alpha, device=env.device)


def fall_teacher_target_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 8]：PPO 模仿损失的监督目标 [接管标记, 监督权重, 教师动作 6 维（未按策略对齐的原始表示）]，与本步观测同一时刻。

    监督权重即 α（模仿损失乘 α）；教师 supervise_after_retire 时恒为 1（α 退到 0 后照常监督，模仿损失不乘 α）。

    只进单独的观测组（env_cfg.ROUGH_FALL_TEACHER_TARGET_GROUP），不进 actor / critic / CTS 教师编码器；没有教师时全 0。
    """
    teacher = fall_teacher_of(env)
    if teacher is None:
        return torch.zeros(env.num_envs, 8, device=env.device)
    teacher.prepare()
    weight = 1.0 if teacher.cfg.supervise_after_retire else float(teacher.alpha_value)
    alpha = torch.full((env.num_envs, 1), weight, device=env.device)
    return torch.cat((teacher.active.float().unsqueeze(1), alpha, teacher.teacher_raw), dim=1)


__all__ = [
    "fall_recovering_obs",
    "fall_teacher_active_obs",
    "fall_teacher_alpha_obs",
    "fall_teacher_target_obs",
]
