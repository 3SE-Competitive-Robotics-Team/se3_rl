"""崎岖地形任务的地形集，移植自 scutrobotlab/wheeled-legged_RL 的 V14 rough 线。

参考仓库用 Isaac Lab 的 `TerrainGeneratorCfg`，本文件换成 MJLab 的同名接口，几何原语一一对应：
`HfPyramidStairsTerrainCfg` → `BoxPyramidStairsTerrainCfg`，`HfInvertedPyramidStairsTerrainCfg` →
`BoxInvertedPyramidStairsTerrainCfg`，斜坡与随机起伏同名。

课程语义与参考仓库一致：`curriculum=True` 时每种子地形独占一列，难度沿行递增，
`difficulty` 在各 `*_range` 内线性插值（row 0 取下界，最后一行取上界），每个 env 走通升一级，
不降级（见 curriculums.terrain_levels）。

列名按**机器人从出生点出发实际经历的方向**命名：MJLab 把正金字塔的出生点放在顶部平台、
反金字塔放在底部凹坑（`BoxPyramidStairsTerrainCfg` 的 origin z 为 +(n+1)h，
`BoxInvertedPyramidStairsTerrainCfg` 为 -(n+1)h；hf 斜坡同理），所以 `stairs_up`/`slope_up`
用的是反金字塔（出生在低处、向外爬升），`stairs_down`/`slope_down` 用正金字塔（出生在高处、向外下行）。参考仓库把"轻微起伏"和"真台阶"拆成两个生成器（前者台阶只有 5-15 mm，
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

# heightfield 的水平分辨率(m)。MuJoCo 的凸体-hfield 碰撞按 AABB 覆盖的格子逐格生成
# 2 个三角棱柱，单对上限 mjMAXCONPAIR=50，即最多 25 格。机身最大碰撞块 0.52 m 宽，
# 默认 0.1 m 格子要占 7x7=49 格 = 98 个三角形，接触被直接丢弃，机身会穿进地形
# （2026-09-06 R1 崩溃的物理侧诱因）。0.2 m 格子只占 4x4=16 格 = 32 个三角形，留 36% 余量。
# 斜坡是平面，棱柱三角化在任何分辨率下都精确还原，粗化不损失几何；
# random_rough 的鼓包宽度会从 0.1 m 变成 0.2 m。
_HF_HORIZONTAL_SCALE = 0.2

# 台阶数与 mjlab BoxPyramidStairs 的算法一致：(边长 - 2 边框 - 平台) / (2 踏面) 取整。
_NUM_STEPS = int((_PATCH_SIZE[0] - 2 * _STAIR_BORDER_WIDTH - _PLATFORM_WIDTH) / (2 * _STEP_WIDTH))

# 地形课程的几何门槛，用切比雪夫距离（max(|dx|, |dy|)）量：金字塔是正方形，台阶边界是 L∞ 等距线，
# 欧氏距离在对角线方向会少算台阶数（欧氏 4.5 m 对角只到 L∞ 3.2 m，两级只爬了一级）。
# 清块：越过最外一级台阶的外沿 = 平台半宽 + 台阶数 × 踏面 = 4.0 m。斜坡块坡面到 4.25 m，同一门槛等于爬完 94%。
ROUGH_TERRAIN_CLEARED_DISTANCE_M = _PLATFORM_WIDTH / 2 + _NUM_STEPS * _STEP_WIDTH
# 出块：走到边框上就截断 episode（time_out），不进邻块——邻块是另一行难度，经验会记错行，
# 边框与邻块的高差还会撞出 wall_blocked。留 0.25 m 给机身半宽。
ROUGH_TERRAIN_EXIT_DISTANCE_M = _PATCH_SIZE[0] / 2 - 0.25


def rough_terrains_cfg(*, num_rows: int = 10) -> TerrainGeneratorCfg:
    """返回带课程的崎岖地形集：平地、上/下台阶、上/下斜坡、随机起伏。

    上行与下行成对出现（正金字塔与反金字塔），这是参考仓库的做法。`proportion` 在课程模式下只决定各列的
    env 分配比例，不决定列数。2026-09-09 用户定（A9）：只留平地与上台阶——flat 30%、stairs_up 70%，
    slope/下行/随机起伏全部设 0（mjlab 仍给每列至少 1 个 env，列保留，日志键不变）。
    A8 的诊断：slope_up 已经爬到 6.4 级、正常清块，但它在 `Rough/*_terrain` 这些"非平地"口径里
    把 stairs_up 的数字整个稀释掉了，看不出台阶列到底卡在哪；去掉之后非平地口径实质上就等于台阶列。
    此前分配为 flat 25 / stairs_up 40 / slope_up 30 / random_rough 5（A4）。
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
            "flat": BoxFlatTerrainCfg(proportion=0.30, size=_PATCH_SIZE),
            # 出生在凹坑底部，向外爬升。
            "stairs_up": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.70,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
            # 出生在顶部平台，向外下行。
            "stairs_down": BoxPyramidStairsTerrainCfg(
                proportion=0.0,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
            "slope_up": HfPyramidSlopedTerrainCfg(
                proportion=0.0,
                size=_PATCH_SIZE,
                slope_range=(0.05, 0.30),
                platform_width=2.0,
                border_width=0.25,
                inverted=True,
                horizontal_scale=_HF_HORIZONTAL_SCALE,
            ),
            "slope_down": HfPyramidSlopedTerrainCfg(
                proportion=0.0,
                size=_PATCH_SIZE,
                slope_range=(0.05, 0.30),
                platform_width=2.0,
                border_width=0.25,
                horizontal_scale=_HF_HORIZONTAL_SCALE,
            ),
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=0.0,
                size=_PATCH_SIZE,
                noise_range=(0.0, 0.05),
                noise_step=0.005,
                border_width=0.25,
                horizontal_scale=_HF_HORIZONTAL_SCALE,
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
            "stairs_up": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.45,
                size=_PATCH_SIZE,
                step_height_range=_STEP_HEIGHT_RANGE,
                step_width=_STEP_WIDTH,
                platform_width=_PLATFORM_WIDTH,
                border_width=_STAIR_BORDER_WIDTH,
            ),
            "stairs_down": BoxPyramidStairsTerrainCfg(
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


__all__ = [
    "ROUGH_TERRAIN_CLEARED_DISTANCE_M",
    "ROUGH_TERRAIN_EXIT_DISTANCE_M",
    "rough_terrains_cfg",
    "stair_only_terrains_cfg",
]
