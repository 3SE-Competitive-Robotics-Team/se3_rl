"""纯倒地自起站立任务 PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.recovery.rl_cfg import rl_cfg as recovery_rl_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """纯倒地自起到固定站立姿态的 GRU PPO 配置，从零训练。"""
    cfg = recovery_rl_cfg(smoke=smoke)
    cfg.resume = False
    return cfg


__all__ = ["rl_cfg"]
