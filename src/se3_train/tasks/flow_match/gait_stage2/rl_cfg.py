"""GAIT Stage2 task PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flow_match.common import single_label_gru_ppo_runner_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """构造 GAIT Stage2 GRU PPO 配置。"""
    cfg = single_label_gru_ppo_runner_cfg("gait_stage2", smoke=smoke)
    cfg.algorithm.learning_rate = 2.0e-4
    cfg.algorithm.desired_kl = 0.006
    cfg.algorithm.entropy_coef = 0.003
    if not (smoke or os.environ.get("SE3_SMOKE", "0") == "1"):
        cfg.max_iterations = 3000
    return cfg
