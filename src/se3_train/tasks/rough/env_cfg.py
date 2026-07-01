"""崎岖地形行走任务环境配置。"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains import TerrainEntityCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """带地形课程的崎岖地形环境配置。"""

    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    cfg = flat_env_cfg(play=play)
    terrain_generator = deepcopy(ROUGH_TERRAINS_CFG)
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        sub_terrain = terrain_generator.sub_terrains.get(name)
        if sub_terrain is not None and hasattr(sub_terrain, "step_height_range"):
            sub_terrain.step_height_range = (0.0, 0.05)

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=5,
    )

    return cfg
