"""倒地自启 Discovery 阶段 PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.recovery.rl_cfg import rl_cfg as recovery_rl_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Discovery 阶段从零训练，沿用 recovery GRU PPO 配置。"""
    return recovery_rl_cfg(smoke=smoke)
