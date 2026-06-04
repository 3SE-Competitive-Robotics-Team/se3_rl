"""FlowMatch WHEEL 单标签普通 VIO task 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.tasks.flow_match.wheel.env_cfg import env_cfg as wheel_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 FlowMatch WHEEL 单标签普通 VIO 训练环境。"""
    return wheel_env_cfg(play=play)
