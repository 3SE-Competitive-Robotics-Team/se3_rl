"""SE3 轮腿机器人的动作配置。

迁移后使用 MJLab 内置 JointPositionAction（腿部）+ JointVelocityAction（轮子），
不再需要自定义 ActionTerm。本模块仅重导出配置类以保持 mdp 包接口稳定。
"""

from mjlab.envs.mdp.actions import (
    JointPositionActionCfg as JointPositionActionCfg,
)
from mjlab.envs.mdp.actions import (
    JointVelocityActionCfg as JointVelocityActionCfg,
)

__all__ = ["JointPositionActionCfg", "JointVelocityActionCfg"]
