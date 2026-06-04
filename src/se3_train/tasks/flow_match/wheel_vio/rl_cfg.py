"""FlowMatch WHEEL 单标签普通 VIO task PPO 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flow_match.common import single_label_vio_ppo_runner_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """构造 FlowMatch WHEEL 单标签普通 VIO PPO 配置。"""
    return single_label_vio_ppo_runner_cfg("wheel", smoke=smoke)
