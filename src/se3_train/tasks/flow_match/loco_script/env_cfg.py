"""FlowMatch 脚本切换 play 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.tasks.flow_match.common import loco_script_play_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 FlowMatch 脚本切换 play 环境。"""
    return loco_script_play_env_cfg(play=play)
