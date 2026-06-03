"""跳跃任务课程调度器。

课程：
1. jump_prob_curriculum：调度 JumpCommandTerm 的每步启动概率 jump_prob（0 → final_prob）
2. jump_height_curriculum：阶段2 fine-tune 扩大目标高度范围（[0.1,0.3] → [0.1,0.6]）
3. jump_quality_weight_curriculum：PostTrain 先保起跳，再逐步加强 tracking/姿态/yaw/对称约束
4. jump_pretrain_constraint_weight_curriculum：PreTrain 先学主动起跳，再逐步收紧坏行为惩罚
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from se3_train.mdp.jump_commands import JumpCommandTerm

# GRU rollout 步数，对应 jump/jump_pretrain task rl_cfg.py 的 num_steps_per_env=64
# 如果修改了 rl_cfg 里的 num_steps_per_env，需要同步更新此常量
_GRU_STEPS_PER_ENV = 64

_JUMP_CURRICULUM_STEP_BASE_ATTR = "_jump_curriculum_step_base"


def _current_iter(env: ManagerBasedRlEnv) -> int:
    """根据本次 run 的相对 common_step_counter 估算当前训练迭代数。

    common_step_counter 每次 env.step()（policy step）加 1，
    与 num_envs 和 decimation 无关（并行 env 共享同一个 counter）。
    每个 policy iter = steps_per_env 次 env.step()。

    跳跃课程必须从当前 run 的首次课程调用开始计算。base checkpoint 只用于
    warm-start 网络权重，不应把旧 run 的训练进度带入课程。
    """
    if not hasattr(env, _JUMP_CURRICULUM_STEP_BASE_ATTR):
        setattr(env, _JUMP_CURRICULUM_STEP_BASE_ATTR, int(env.common_step_counter))
    step_base = getattr(env, _JUMP_CURRICULUM_STEP_BASE_ATTR)
    relative_steps = max(0, int(env.common_step_counter) - int(step_base))
    steps_per_policy_iter = _GRU_STEPS_PER_ENV
    return relative_steps // max(steps_per_policy_iter, 1)


def jump_prob_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    warmup_iters: int = 300,
    rampup_iters: int = 200,
    initial_prob: float = 0.0,
    final_prob: float = 0.05,
) -> dict[str, Any]:
    """跳跃每步启动概率课程：从 initial_prob 线性爬升到 final_prob。

    调度逻辑：
        iter < warmup_iters:                     jump_prob = initial_prob
        rampup_iters <= 0:                       jump_prob = final_prob
        warmup_iters ≤ iter < warmup+rampup:    线性插值 initial_prob → final_prob
        iter ≥ warmup + rampup:                  jump_prob = final_prob
    """
    current_iter = _current_iter(env)

    term = env.command_manager.get_term(command_name)
    if not isinstance(term, JumpCommandTerm):
        return {}

    if current_iter < warmup_iters:
        new_prob = initial_prob
    elif rampup_iters <= 0:
        new_prob = final_prob
    elif current_iter < warmup_iters + rampup_iters:
        progress = (current_iter - warmup_iters) / rampup_iters
        new_prob = initial_prob + (final_prob - initial_prob) * progress
    else:
        new_prob = final_prob

    new_prob = max(0.0, min(1.0, new_prob))
    term.cfg.jump_prob = new_prob
    return {"jump_prob": new_prob}


def jump_height_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    expand_iter: int = 1000,
    ramp_iters: int = 0,
    initial_range: tuple[float, float] = (0.1, 0.3),
    final_range: tuple[float, float] = (0.1, 0.6),
) -> dict[str, Any]:
    """目标跳跃高度课程：先用保守范围，再平滑扩大到完整范围。

    用于 fine-tune 阶段，让策略先学低跳再挑战高跳。
    """
    current_iter = _current_iter(env)

    term = env.command_manager.get_term(command_name)
    if not isinstance(term, JumpCommandTerm):
        return {}

    if current_iter < expand_iter:
        new_hi = initial_range[1]
    elif ramp_iters <= 0:
        new_hi = final_range[1]
    else:
        progress = min(1.0, (current_iter - expand_iter) / ramp_iters)
        new_hi = initial_range[1] + (final_range[1] - initial_range[1]) * progress

    term.cfg.jump_height_range = (initial_range[0], new_hi)

    return {"jump_height_hi": term.cfg.jump_height_range[1]}


def _lerp_weight(
    current_iter: int, start_iter: int, ramp_iters: int, lo: float, hi: float
) -> float:
    """按 iter 线性插值 reward weight。"""
    if current_iter < start_iter:
        return lo
    if ramp_iters <= 0:
        return hi
    progress = min(1.0, max(0.0, (current_iter - start_iter) / ramp_iters))
    return lo + (hi - lo) * progress


def _set_reward_weight(env: ManagerBasedRlEnv, term_name: str, weight: float) -> float | None:
    """运行期更新 RewardManager 中指定项权重；缺项时静默跳过。"""
    reward_manager = getattr(env, "reward_manager", None)
    if reward_manager is None or term_name not in reward_manager.active_terms:
        return None
    reward_manager.get_term_cfg(term_name).weight = float(weight)
    return float(weight)


def jump_quality_weight_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_iter: int = 800,
    ramp_iters: int = 2000,
) -> dict[str, Any]:
    """PostTrain 跳跃质量权重课程：先保起跳速度，再逐步收轨迹、姿态和对称性。

    静态强姿态/强对称约束会把策略推向“稳定但跳不起来”；静态强起跳奖励又会
    产生 tilt/yaw 偏大。强轨迹 tracking 在早期也会让策略学到“低速贴轨迹”。
    这里把权重拆成两阶段：
    - 0 ~ start_iter：很弱 tracking + 弱质量约束，先保主动蹬地速度
    - start_iter 之后：缓慢拉到完整 tracking 和强质量约束，让策略在会跳之后再学稳
    """
    del env_ids

    current_iter = _current_iter(env)
    schedule = {
        "jump_orientation": (-6.0, -30.0),
        "jump_tilt_barrier": (-1.0, -10.0),
        "jump_ang_vel_xy": (-1.0, -3.5),
        "jump_ang_vel_z": (-3.0, -6.0),
        "jump_joint_mirror": (-12.0, -16.0),
        "wheel_distance": (-5.0, -80.0),
        "jump_action_mirror": (-1.2, -2.0),
        "jump_action_rate": (-0.04, -0.08),
        "jump_wheel_counterspin": (-0.04, -0.08),
        "jump_vel_encourage": (60.0, 30.0),
        "jump_takeoff_vz_tracking": (120.0, 105.0),
        "traj_base_pose_6d": (12.0, 50.0),
        "traj_vz": (3.0, 12.0),
        "traj_joint_pos": (5.0, 20.0),
        "jump_wheel_clr_tracking": (-3.0, -8.0),
        "landing_symmetry": (-1.5, -6.0),
        "jump_landing_recovery": (6.0, 18.0),
        "jump_landing_base_height": (-4.0, -30.0),
    }

    logs: dict[str, Any] = {"quality_weight_progress": 0.0}
    if ramp_iters <= 0 and current_iter >= start_iter:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, (current_iter - start_iter) / max(ramp_iters, 1)))
    logs["quality_weight_progress"] = progress

    for term_name, (initial_weight, final_weight) in schedule.items():
        weight = _lerp_weight(
            current_iter,
            start_iter,
            ramp_iters,
            initial_weight,
            final_weight,
        )
        applied = _set_reward_weight(env, term_name, weight)
        if applied is not None:
            logs[f"{term_name}_weight"] = applied

    return logs


def jump_pretrain_constraint_weight_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_iter: int = 150,
    ramp_iters: int = 650,
) -> dict[str, Any]:
    """PreTrain 质量权重课程：先保主动起跳，再逐步收紧坏行为惩罚。

    PreTrain 的主要目标是学会良好发力。滑移、落地水平运动、提前抬轮、
    原地轮组抖动和双轮前后错位都仍然按坏行为惩罚；早期使用较弱权重，
    避免策略在还不会起跳时被这些惩罚推向“少动、弱跳”。
    """
    del env_ids

    current_iter = _current_iter(env)
    schedule = {
        "idle_wheel_motion": (-2.0, -6.0),
        "flat_wheel_contact": (-3.0, -10.0),
        "flat_action_smoothness": (-0.04, -0.08),
        "wheel_distance": (-4.0, -8.0),
        "jump_pre_takeoff_wheel_lift": (-6.0, -18.0),
        "jump_wheel_ground_slip": (-6.0, -24.0),
        "jump_landing_horizontal_motion": (-3.0, -12.0),
    }

    if ramp_iters <= 0 and current_iter >= start_iter:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, (current_iter - start_iter) / max(ramp_iters, 1)))

    logs: dict[str, Any] = {"pretrain_constraint_weight_progress": progress}
    for term_name, (initial_weight, final_weight) in schedule.items():
        weight = _lerp_weight(
            current_iter,
            start_iter,
            ramp_iters,
            initial_weight,
            final_weight,
        )
        applied = _set_reward_weight(env, term_name, weight)
        if applied is not None:
            logs[f"{term_name}_weight"] = applied

    return logs
