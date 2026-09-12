"""从 A22 同一检查点比较混合地形与全部上台阶采样。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from .a22_stair_command import a22_env_cfg


def a26_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """对照组保留 A22 的地形分布、指令和奖励。"""
    return a22_env_cfg(play=play)


def a27_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """仅保留上台阶列，避免零比例地形仍获分配环境。"""
    cfg = a26_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    stairs = generator.sub_terrains["stairs_up"]
    stairs.proportion = 1.0
    generator.sub_terrains = {"stairs_up": stairs}
    generator.num_cols = 1
    return cfg
