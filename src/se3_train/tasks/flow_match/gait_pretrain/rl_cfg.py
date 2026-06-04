"""纯 GAIT PreTrain task PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flow_match.common import gait_pretrain_gru_ppo_runner_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """构造纯 GAIT PreTrain GRU PPO 配置。"""
    return gait_pretrain_gru_ppo_runner_cfg(smoke=smoke)
