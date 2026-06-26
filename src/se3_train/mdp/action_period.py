"""训练端 action 周期读取工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from se3_shared import FRONT_ACTION_INDICES, front_action_periods_from_scales

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def front_action_periods_from_env(env: ManagerBasedRlEnv) -> tuple[float, float] | None:
    """从当前 action term 的 leg scale 推出 LF0/RF0 的 action 周期。"""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        periods = getattr(term, "front_action_periods", None)
        if periods is not None:
            return _as_period_pair(periods)
        cfg = getattr(term, "cfg", None)
        leg_scales = getattr(cfg, "leg_scales", None)
        if leg_scales is not None:
            return front_action_periods_from_scales(leg_scales)
    return None


def primary_front_action_period_from_env(env: ManagerBasedRlEnv) -> float | None:
    """返回单个前杆周期；左右不一致时取平均以维持镜像奖励的标量接口。"""
    periods = front_action_periods_from_env(env)
    if periods is None:
        return None
    return 0.5 * (float(periods[0]) + float(periods[1]))


def _as_period_pair(periods) -> tuple[float, float]:
    if isinstance(periods, (int, float)):
        value = float(periods)
        return value, value
    flat = tuple(float(value) for value in periods)
    if len(flat) == 1:
        return flat[0], flat[0]
    if len(flat) == 2:
        return flat
    return flat[FRONT_ACTION_INDICES[0]], flat[FRONT_ACTION_INDICES[1]]
