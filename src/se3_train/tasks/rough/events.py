"""本任务的事件：平地速度课程的信号掩码 + 逐项奖励的分列诊断。

两个都是按列的口径。mjlab 的 MetricsManager 只会对全部 reset 的 env 求均值，做不出
"只看台阶列"的均值，所以分列诊断仍走 interval 事件直接写 `extras["log"]`。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .columns import column_mask

if TYPE_CHECKING:
    import torch
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

CURRICULUM_ENV_MASK_ATTR = "_se3_curriculum_env_mask"

# 逐项奖励分列日志的键前缀；走 Rough/ 命名空间，log_filter 整段白名单。
REWARD_SPLIT_LOG_PREFIX = "Rough/rw_"


def set_curriculum_env_mask(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = ("flat",),
) -> None:
    """startup 事件：把"只按这些子地形列的 env 评估速度课程"的掩码挂到 env 上。

    mdp.rewards.tracking_lin_vel 据此额外记 `Locomotion/tracking_lin_vel_reward_curriculum`，
    commands_vel_adaptive 用它推进平地速度课程。地形列强制前向指令、跟踪分天然偏低，混进全体
    均值会把平地列的课程钉死在 vx=0（R3）。非课程地形（没有分列）时不挂，课程退回全体均值。
    """
    del env_ids
    mask = column_mask(env, terrain_type_names)
    if mask is not None:
        setattr(env, CURRICULUM_ENV_MASK_ATTR, mask)


def log_reward_split_by_column(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    suffix: str = "stairs",
) -> None:
    """每步把奖励表里每一项在指定子地形列上的均值记一份（`Rough/rw_<项名>_<suffix>`）。

    `Episode_Reward/*` 是全体均值，台阶列的定价被平地列稀释了看不出来（A8 只能读出本来就只在
    台阶列生效的 command_velocity_error）。这里直接读 RewardManager 上一次 `compute()` 算好的
    逐 env 逐项缓冲 `_step_reward`（value×weight，每秒贡献，与 Episode_Reward 同口径），
    只做一次掩码均值，不重算任何奖励；落后一个控制步对日志无所谓。
    """
    del env_ids
    reward_manager = getattr(env, "reward_manager", None)
    step_reward = getattr(reward_manager, "_step_reward", None)
    if reward_manager is None or step_reward is None:
        return
    names = list(reward_manager.active_terms)
    if step_reward.shape != (env.num_envs, len(names)):
        return
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return
    weights = mask.to(dtype=step_reward.dtype).unsqueeze(1)
    means = (step_reward * weights).sum(dim=0) / weights.sum().clamp(min=1.0)
    log = env.extras.setdefault("log", {})
    for index, name in enumerate(names):
        log[f"{REWARD_SPLIT_LOG_PREFIX}{name}_{suffix}"] = means[index]


__all__ = [
    "CURRICULUM_ENV_MASK_ATTR",
    "REWARD_SPLIT_LOG_PREFIX",
    "log_reward_split_by_column",
    "set_curriculum_env_mask",
]
