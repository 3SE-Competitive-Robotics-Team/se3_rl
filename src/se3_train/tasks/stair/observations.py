"""台阶任务观测函数。"""

from __future__ import annotations

import torch

from se3_train.mdp.observations import _finite_clamp


def last_actions_obs(env):
    """上一步策略输出的 6 维 clipped action，不包含 CTBC 前馈。"""
    action_term = env.action_manager.get_term("delayed_action")
    policy_action = getattr(action_term, "policy_action", None)
    if isinstance(policy_action, torch.Tensor):
        return _finite_clamp(policy_action)
    return _finite_clamp(action_term.raw_action)


__all__ = ["last_actions_obs"]
