"""跳跃专属终止条件。

在行走三个终止条件基础上，新增膝关节冲击终止：
防止着陆时膝关节高速撞到硬限位造成模型不稳定。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_shared import JointGroup

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def knee_hyperextension(
    env: ManagerBasedRlEnv,
    command_name: str,
    limit_ratio: float = 0.98,
    vel_threshold: float = 5.0,
    probability: float = 0.7,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """膝关节冲击终止：超软限位 98% 范围且速度 > vel_threshold 时以 probability 概率终止。

    防止着陆时膝关节高速撞限位导致数值爆炸。
    vel_threshold=5.0 rad/s 对应约 48 rpm，着陆冲击时典型可达范围。
    概率化终止（< 1.0）以提供更平滑的训练梯度。
    仅对 jump_flag=1 的 env 生效。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5

    robot = env.scene[asset_cfg.name]
    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # 膝关节：lf1（index 1）和 rf1（index 3）
    knee_indices = [JointGroup.LEGS[1], JointGroup.LEGS[3]]
    q = robot.data.joint_pos[:, knee_indices]
    q_vel = robot.data.joint_vel[:, knee_indices]
    limits = soft_limits[:, knee_indices]

    q_range = limits[:, :, 1] - limits[:, :, 0]
    q_lower_threshold = limits[:, :, 0] + q_range * (1.0 - limit_ratio)
    q_upper_threshold = limits[:, :, 1] - q_range * (1.0 - limit_ratio)

    # 接近限位（下限 2% 以内或上限 2% 以内）
    near_limit = (q < q_lower_threshold) | (q > q_upper_threshold)
    high_vel = torch.abs(q_vel) > vel_threshold

    # 任一膝关节同时满足近限位 + 高速
    triggered = (near_limit & high_vel).any(dim=1) & jump_flag

    # 概率化终止
    rand = torch.rand(env.num_envs, device=env.device)
    return triggered & (rand < probability)
