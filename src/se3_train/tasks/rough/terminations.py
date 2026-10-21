"""本任务专属终止项：倒地约束的 CaT（Constraints as Terminations，Chane-Sane 等 2024）。

2026-10-21 用户定：倒地平顺性从罚分改成概率终止。罚分版（b3odjuk6）收到 s=1 后起身仍是腿速 20 rad/s、
机身角速度 10.4 rad/s、功率 1.3 kW——倒地每秒比站着少约 124 分，猛甩省下的时间够付罚款。终止丢掉的是
全部未来回报，付不掉；前提是每步奖励非负（rewards.total_reward_floor），否则倒地状态回报为负、终止反成逃生。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import JointGroup
from se3_train.mdp.rewards import _policy_leg_torque_and_vel
from se3_train.mdp.terminations import _should_log_step

from .rewards import fall_constraint_level, fall_weight

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _wheel_joint_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    cached = getattr(env, "_se3_wheel_joint_ids", None)
    if cached is None:
        ids, _ = env.scene["robot"].find_joints(JointGroup.WHEEL_NAMES, preserve_order=True)
        cached = torch.tensor(ids, device=env.device, dtype=torch.long)
        env._se3_wheel_joint_ids = cached
    return cached


def fall_constraint_violation(
    env: ManagerBasedRlEnv,
    limits: dict[str, tuple[float, float]],
    p_max_range: tuple[float, float] = (0.0, 0.25),
) -> torch.Tensor:
    """按约束越界程度随机终止：p = p_max · max_i clip((c_i − limit_i) / margin_i, 0, 1)。

    limits：约束名 → (limit, margin)，越界 margin 时终止概率到 p_max。可用约束：
    - `leg_speed`：四个腿电机转速绝对值最大值（rad/s），全程生效（行走最多约 3.7）；
    - `body_ang_vel`：机身角速度模长（rad/s），乘倒地程度 fall_weight；
    - `leg_power`：四个腿电机机械功率 Σ|τ·q̇|（W），乘倒地程度；
    - `wheel_speed`：两轮转速绝对值最大值（rad/s），乘倒地程度。
    p_max 按课程的收紧程度 s 在 p_max_range 两端插值。本项须从 fall_penalty 的计数里排除（只丢未来回报，不另扣终止罚）。
    日志：`FallConstraint/p_max`、各约束越界率（超过 limit 的 env 占比）、本步终止率。
    """
    robot = env.scene["robot"]
    torque, leg_vel = _policy_leg_torque_and_vel(env, robot)
    fallen = fall_weight(env)
    values = {
        "leg_speed": leg_vel.abs().amax(dim=1),
        "body_ang_vel": torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1),
        "leg_power": (torque * leg_vel).abs().sum(dim=1),
        "wheel_speed": robot.data.joint_vel[:, _wheel_joint_ids(env)].abs().amax(dim=1),
    }
    fall_gated = {"body_ang_vel", "leg_power", "wheel_speed"}
    lo, hi = (float(v) for v in p_max_range)
    p_max = lo + (hi - lo) * fall_constraint_level(env)
    severity = torch.zeros(env.num_envs, device=env.device)
    over: dict[str, torch.Tensor] = {}
    for name, (limit, margin) in limits.items():
        if name not in values:
            raise ValueError(f"未知倒地约束 {name}，可用 {sorted(values)}")
        v = torch.clamp((values[name] - float(limit)) / float(margin), 0.0, 1.0)
        if name in fall_gated:
            v = v * fallen
        over[name] = v > 0.0
        severity = torch.maximum(severity, v)
    terminated = torch.rand(env.num_envs, device=env.device) < p_max * severity
    if hasattr(env, "extras") and _should_log_step(env):
        log = env.extras.setdefault("log", {})
        log["FallConstraint/p_max"] = torch.tensor(p_max, device=env.device)
        log["FallConstraint/terminated"] = terminated.float().mean()
        log["FallConstraint/severity_fallen"] = (severity * fallen).sum() / fallen.sum().clamp(
            min=1.0
        )
        for name, mask in over.items():
            log[f"FallConstraint/over_{name}"] = mask.float().mean()
    return terminated


__all__ = ["fall_constraint_violation"]
