"""纯 GAIT FineTune task 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.tasks.flow_match.common import gait_finetune_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造纯 GAIT FineTune 训练环境。"""
    return gait_finetune_env_cfg(play=play)
