"""纯 GAIT FineTune task PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flow_match.common import gait_finetune_gru_ppo_runner_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """构造纯 GAIT FineTune GRU PPO 配置。"""
    return gait_finetune_gru_ppo_runner_cfg(smoke=smoke)
