"""Rough terrain discovery GRU 任务 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.recovery_discovery.rl_cfg import rl_cfg as discovery_rl_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """从 recovery-discovery checkpoint warm-start 的 GRU PPO 配置。"""
    cfg = discovery_rl_cfg(smoke=smoke)
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        return cfg

    cfg.max_iterations = int(os.environ.get("SE3_ROUGH_DISCOVERY_MAX_ITERATIONS", "3000"))
    cfg.resume = True
    cfg.load_run = os.environ.get("SE3_ROUGH_DISCOVERY_LOAD_RUN", ".*")
    cfg.load_checkpoint = os.environ.get("SE3_ROUGH_DISCOVERY_LOAD_CHECKPOINT", "model_.*\\.pt")
    cfg.algorithm.learning_rate = float(
        os.environ.get("SE3_ROUGH_DISCOVERY_LEARNING_RATE", "3.0e-5")
    )
    return cfg
