"""FlowMatch 脚本切换 task PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flow_match.common import single_label_gru_ppo_runner_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """构造 FlowMatch 脚本切换 GRU PPO 配置。"""
    return single_label_gru_ppo_runner_cfg("loco_script", smoke=smoke)
