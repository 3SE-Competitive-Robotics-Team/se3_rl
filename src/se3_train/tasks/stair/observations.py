"""台阶任务观测函数。"""

from __future__ import annotations

import torch

from se3_train.mdp.observations import _finite_clamp
from se3_train.tasks.flat.observations import *  # noqa: F403
from se3_train.tasks.flat.observations import __all__ as _FLAT_OBS_ALL

_CTBC_OBS_SCALE = 0.01


def last_actions_obs(env):
    """上一步策略输出的 6 维 clipped action，不包含 CTBC 前馈。"""
    action_term = env.action_manager.get_term("delayed_action")
    policy_action = getattr(action_term, "policy_action", None)
    if isinstance(policy_action, torch.Tensor):
        return _finite_clamp(policy_action)
    return _finite_clamp(action_term.raw_action)


def ctbc_action_delta_obs(env):
    """单独暴露 CTBC 注入量；默认不接入 actor，以保持 34 维 checkpoint contract。"""
    action_term = env.action_manager.get_term("delayed_action")
    delta = getattr(action_term, "ctbc_action_delta", None)
    if isinstance(delta, torch.Tensor):
        return _finite_clamp(delta)
    return torch.zeros(env.num_envs, 6, device=env.device)


def ctbc_phase_obs(env):
    """复用 jump 观测槽输出 CTBC 左右摆动相位和触发位。"""
    state = getattr(env, "stair_climb_state", None)
    obs = torch.zeros(env.num_envs, 3, device=env.device)
    if state is None:
        return obs

    phase_steps = state.ff_phase
    active = phase_steps >= 0
    period = max(1, int(getattr(state, "ff_period_steps", 1)))
    phase = torch.where(
        active,
        torch.clamp((phase_steps.float() + 1.0) / float(period), 0.0, 1.0),
        torch.zeros_like(phase_steps, dtype=torch.float32),
    )
    obs[:, :2] = phase[:, :2] * _CTBC_OBS_SCALE
    obs[:, 2] = active.any(dim=-1).float() * _CTBC_OBS_SCALE
    return _finite_clamp(obs, limit=_CTBC_OBS_SCALE)


__all__ = [*_FLAT_OBS_ALL, "ctbc_action_delta_obs", "ctbc_phase_obs", "last_actions_obs"]
