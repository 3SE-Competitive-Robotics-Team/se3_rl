"""倒地自启 Discovery 阶段 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.recovery.rl_cfg import rl_cfg as recovery_rl_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Discovery 阶段从零训练，沿用 recovery GRU PPO 配置。"""
    cfg = recovery_rl_cfg(smoke=smoke)
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        return cfg
    cfg.max_iterations = int(os.environ.get("SE3_RECOVERY_DISCOVERY_MAX_ITERATIONS", "1500"))
    return cfg
