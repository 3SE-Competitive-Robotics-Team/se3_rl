"""FlowMatch WHEEL 单标签 task 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_shared import TaskMode
from se3_train.tasks.flow_match.common import single_label_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 FlowMatch WHEEL 单标签训练环境。"""
    return single_label_env_cfg(TaskMode.WHEEL, play=play)
