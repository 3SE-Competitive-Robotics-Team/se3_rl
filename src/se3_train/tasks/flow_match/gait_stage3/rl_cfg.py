"""GAIT Stage3 task PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flow_match.common import single_label_gru_ppo_runner_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """构造 GAIT Stage3 GRU PPO 配置。"""
    cfg = single_label_gru_ppo_runner_cfg("gait_stage3", smoke=smoke)
    cfg.algorithm.learning_rate = 2.0e-4
    cfg.algorithm.desired_kl = 0.006
    cfg.algorithm.entropy_coef = 0.003
    return cfg
