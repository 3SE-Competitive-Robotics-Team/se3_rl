"""本任务使用的课程函数：平地那套 + 地形难度课程。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp.curriculums import commands_vel, push_disturbance

from .terrains import ROUGH_TERRAIN_CLEARED_DISTANCE_M

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    clear_distance_m: float = ROUGH_TERRAIN_CLEARED_DISTANCE_M,
) -> dict[str, torch.Tensor]:
    """清掉本块全部台阶就升一级；只升不降。

    升级判据：episode 结束时出生点到机身的切比雪夫距离 max(|dx|, |dy|) ≥ `clear_distance_m`
    （默认 = 平台半宽 + 台阶数 × 踏面，见 terrains.ROUGH_TERRAIN_CLEARED_DISTANCE_M），
    即越过最外一级台阶，与朝向和指令速度无关。配合 terminations.terrain_cleared 在出块时截断，
    一个 episode 最多记一次升级，且经验不会串到邻块的难度行。

    不用 mjlab `terrain_levels_vel` 的欧氏位移：金字塔是正方形，欧氏门槛在对角线方向少算台阶数。
    降级也去掉了：原判据用 episode 末段的 |vx| 反推应走距离，末段静站时永不降、末段高速时几乎必降，
    与地形能力无关；对称随机指令下净位移本身是随机游走，升降各半会把课程钉在低位
    （R2，W&B 32eentyo 的平地列也只到 1.6）。
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    del command  # 只升不降后不再需要指令反推应走距离；保留取值以校验指令项存在。

    offset = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = offset.abs().max(dim=1).values
    move_up = distance >= float(clear_distance_m)
    move_down = torch.zeros_like(move_up)

    # 首次 reset 发生在任何一步之前，机器人还在模型默认位姿（世界原点附近）而不是出生点，
    # 到 env_origins 的距离可达几十米，会把全员无条件升一级（R3 首轮 level 全为 1.0 即此故障）。
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)

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
