"""崎岖地形行走任务环境配置。"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainEntityCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """带地形课程的崎岖地形环境配置。"""

    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    cfg = flat_env_cfg(play=play)

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
    )

    return cfg
