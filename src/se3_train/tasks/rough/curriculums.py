"""本任务使用的课程函数：平地那套 + 地形难度课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp.curriculums import commands_vel, push_disturbance

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> dict[str, torch.Tensor]:
    """按本 episode 走出的距离升降地形难度等级。

    与 mjlab `tasks.velocity.mdp.terrain_levels_vel` 同一套判据：走够半块地形升一级，
    走不到指令距离的一半降一级。不能直接复用那个实现，因为它按
    `norm(command[:, :2])` 取指令速度，而本仓库的指令布局是
    `[lin_vel_x, ang_vel_yaw, pitch, roll, height]`，第 1 维是角速度（rad/s），
    混进模长会把要求的行进距离算错。
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    move_up = distance > terrain_generator.size[0] / 2
    required = torch.abs(command[env_ids, 0]) * env.max_episode_length_s * 0.5
    move_down = (distance < required) & ~move_up

    # 首次 reset 发生在任何一步之前，distance 还是出生点到出生点的 0，
    # 会把所有 env 无条件降级，抹掉 max_init_terrain_level。
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    # 课程模式下每种子地形独占一列，列号即子地形名，可以分地形看难度爬到哪。
    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    if terrain_origins.shape[1] == len(sub_terrain_names):
        types = terrain.terrain_types
        for index, name in enumerate(sub_terrain_names):
            mask = types == index
            if mask.any():
                result[name] = torch.mean(levels[mask])
    return result


__all__ = ["commands_vel", "push_disturbance", "terrain_levels"]
