"""崎岖地形任务的地形集，移植自 scutrobotlab/wheeled-legged_RL 的 V14 rough 线。

参考仓库用 Isaac Lab 的 `TerrainGeneratorCfg`，本文件换成 MJLab 的同名接口，几何原语一一对应：
`HfPyramidStairsTerrainCfg` → `BoxPyramidStairsTerrainCfg`，`HfInvertedPyramidStairsTerrainCfg` →
`BoxInvertedPyramidStairsTerrainCfg`，斜坡与随机起伏同名。

课程语义与参考仓库一致：`curriculum=True` 时每种子地形独占一列，难度沿行递增，
`difficulty` 在各 `*_range` 内线性插值（row 0 取下界，最后一行取上界），每个 env 走通升一级、
走不动降一级。参考仓库把"轻微起伏"和"真台阶"拆成两个生成器（前者台阶只有 5-15 mm，
后者 180-220 mm 且关掉课程）；MJLab 的难度插值让两者可以合成一条课程，故本文件用
单个 `ROUGH_TERRAINS_CFG` 从平地一路升到比赛级台阶高度。

尺寸按本机器人标定：轮半径 0.06 m、轮距 0.433 m、base 指令高度 0.20-0.38 m。
台阶踏面取 1.5 m（远大于轮距，逐级上而非连续楼梯），与参考仓库的 `STAIR_FOR_RM1` 一致。
"""

from __future__ import annotations

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# 单块地形边长；配 1.5 m 踏面、2.0 m 顶平台、0.5 m 边框时金字塔每侧 2 级。
_PATCH_SIZE = (9.0, 9.0)
# 台阶高度课程区间。上界 0.20 m 是赛场台阶量级，也是参考仓库 STAIR_FOR_RM1 的取值；
# 下界 0.02 m 让 row 0 近似平地，避免课程起点就把策略卡死。
_STEP_HEIGHT_RANGE = (0.02, 0.20)
_STEP_WIDTH = 1.5
_PLATFORM_WIDTH = 2.0
_STAIR_BORDER_WIDTH = 0.5


def rough_terrains_cfg(*, num_rows: int = 10) -> TerrainGeneratorCfg:
    """返回带课程的崎岖地形集：平地、上/下台阶、上/下斜坡、随机起伏。

    上行与下行成对出现（正金字塔与反金字塔），保证上台阶和下台阶、上坡和下坡都练到，
    这是参考仓库的做法。`proportion` 在课程模式下只决定各列的 env 分配比例，不决定列数。
    """
    return TerrainGeneratorCfg(
        size=_PATCH_SIZE,
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        # 课程模式下生成器忽略该值，按子地形种类数取列，这里填同值只为可读。
        num_cols=6,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.25, size=_PATCH_SIZE),
            "stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=0.2,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
            "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
            "slope_up": HfPyramidSlopedTerrainCfg(
                proportion=0.15,
                size=_PATCH_SIZE,
                slope_range=(0.05, 0.30),
                platform_width=2.0,
                border_width=0.25,
            ),
            "slope_down": HfPyramidSlopedTerrainCfg(
                proportion=0.15,
                size=_PATCH_SIZE,
                slope_range=(0.05, 0.30),
                platform_width=2.0,
                border_width=0.25,
                inverted=True,
            ),
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=0.05,
                size=_PATCH_SIZE,
                noise_range=(0.0, 0.05),
                noise_step=0.005,
                border_width=0.25,
            ),
        },
        add_lights=False,
    )


def stair_only_terrains_cfg(*, num_rows: int = 10) -> TerrainGeneratorCfg:
    """只含上/下台阶与平地的地形集，用于台阶能力的定向评测。

    对应参考仓库的 `RM_ROUGH_STAIRS_CFG`；那边关掉课程随机采难度，这里保留课程，
    以便按 terrain level 定位策略能上到多高的台阶。
    """
    return TerrainGeneratorCfg(
        size=_PATCH_SIZE,
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        num_cols=3,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.1, size=_PATCH_SIZE),
            "stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=0.45,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
            "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.45,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
        },
        add_lights=False,
    )


__all__ = ["rough_terrains_cfg", "stair_only_terrains_cfg"]
