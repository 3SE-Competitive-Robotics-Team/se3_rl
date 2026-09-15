"""崎岖地形任务的地形集：用 mjlab 官方 preset 组合，几何按本机器人标定。

课程语义是 mjlab `TerrainGeneratorCfg(curriculum=True)` 的：每种子地形独占一列，难度沿行插值
（row 0 取各 `*_range` 下界，最后一行取上界），升降级由官方 `terrain_levels_vel` 管
（见 env_cfg.py）。

列名按机器人从出生点出发实际经历的方向命名：mjlab 把正金字塔的出生点放在顶部平台、
反金字塔放在底部凹坑，所以 `stairs_up` 用反金字塔（出生在低处、向外爬升），
`stairs_down` 用正金字塔（出生在高处、向外下行）。

训练地形五列（2026-09-15）：flat、stairs_up、stairs_down、slope_up、slope_down。
2026-09-13 曾收到只剩 flat + stairs_up 两列，因为 A9 起那四列比例为 0 却仍各分 1 个 env、白占 160 个 geom
（实测每轮多 0.6 s，见 docs/plan/rough_iteration_time_20260913.md）。**比例为 0 的列不要放进
ROUGH_TERRAIN_PROPORTIONS**——关不掉，只会白花钱；要么给正比例，要么删掉。

若以后再加 hfield 地形（斜坡、随机起伏），`horizontal_scale` 取 0.2 m：MuJoCo 凸体-hfield 碰撞按 AABB
覆盖的格子逐格生成三角棱柱、单对上限 50 个三角形，机身最大碰撞块 0.52 m 宽在 0.1 m 格子下要 98 个
三角形，接触被丢弃、机身穿进地形（R1 崩溃的物理侧诱因）；0.2 m 格子只要 32 个。

尺寸按本机器人标定：轮半径 0.06 m、轮距 0.433 m、base 指令高度 0.20–0.38 m。台阶踏面 0.7 m
（2026-09-14 用户定；M1–M3 是 1.5 m 每侧 2 级、M4 试过 0.5 m 每侧 6 级但用户判定太窄）。0.7 m 时每侧
(9 − 1 − 2) / (2 × 0.7) 取整 4 级，一块地形最高爬 4 × 0.20 = 0.8 m；台阶高课程 0.02–0.20 m，上界是赛场
台阶量级，下界让 row 0 近似平地，课程起点不会把策略卡死。踏面越窄每块 box 越多、迭代越慢
（1.5 m 13 个 box、M3 1.85 s/轮；0.5 m 29 个、M4 2.20 s/轮），见 docs/plan/rough_iteration_time_20260913.md。
"""

from __future__ import annotations

from mjlab.terrains.config import (
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# 单块地形边长；配 0.7 m 踏面、2.0 m 中央平台、0.5 m 边框时金字塔每侧 (9 − 1 − 2) / (2 × 0.7) 取整 4 级。
ROUGH_PATCH_SIZE = (9.0, 9.0)
ROUGH_STEP_HEIGHT_RANGE = (0.02, 0.20)
# 踏面宽 0.7 m（M5，2026-09-14 用户定；M1–M3 为 1.5 m，M4 为 0.5 m）。评测场景 serialleg_stairs_1to9.xml 由它生成，
# 改了要重新生成；旧几何留在 serialleg_stairs_1to9_tread1p5.xml（M1–M3）与 _tread0p5.xml（M4）供回放。
ROUGH_STEP_WIDTH = 0.7
ROUGH_PLATFORM_WIDTH = 2.0
_STAIR_BORDER_WIDTH = 0.5

# env 分配比例（2026-09-09 用户定，A9）：只留平地与上台阶。这里的键就是训练地形的全部列；
# `proportion` 只管 env 分配，不管列数，所以比例为 0 的列不能靠设 0 关掉，只能不放进来。
# 2026-09-15 用户定：在上台阶之外补下台阶与上下坡。比例让台阶仍占主力（55%）。
# 实测 geom 代价（9 m 块 × 10 行，scripts 见 .scratch/terrain_cost.py）：
# flat 14、stairs_up/stairs_down 各 214、slope_up/slope_down 各 14（hfield 类几乎免费）。
# 不选的：格宽 0.5 的 box_random_grid 要 2454 个 geom（十倍于台阶）、stepping_stones 784 且石块间是
# 2 m 深坑对轮距 0.433 是断崖难度、discrete_obstacles 只能绕而地形列 yaw 指令绕不开。
ROUGH_TERRAIN_PROPORTIONS: dict[str, float] = {
    "flat": 0.15,
    "stairs_up": 0.35,
    "stairs_down": 0.20,
    "slope_up": 0.15,
    "slope_down": 0.15,
}

# 坡度（rise/run）：上坡 0.4 = 21.8°，摩擦 1.0 下轮式可行；下坡有重力助推更易失控，上界收到 0.35。
ROUGH_SLOPE_UP_RANGE = (0.0, 0.4)
ROUGH_SLOPE_DOWN_RANGE = (0.0, 0.35)
# hfield 水平分辨率必须 0.2 m：MuJoCo 凸体-hfield 按 AABB 覆盖的格子逐格生成三角棱柱、单对上限 50 个，
# 机身最大碰撞块 0.52 m 宽在 0.1 m 格子下要 98 个三角形，接触会被丢弃、机身穿进地形（R1 崩溃的物理侧诱因）。
ROUGH_HFIELD_HORIZONTAL_SCALE = 0.2


def _slope(preset, *, proportion: float, slope_range: tuple[float, float]):
    return preset(
        proportion=proportion,
        size=ROUGH_PATCH_SIZE,
        slope_range=slope_range,
        platform_width=ROUGH_PLATFORM_WIDTH,
        border_width=_STAIR_BORDER_WIDTH,
        horizontal_scale=ROUGH_HFIELD_HORIZONTAL_SCALE,
    )


def _stairs(preset, *, proportion: float):
    return preset(
        proportion=proportion,
        size=ROUGH_PATCH_SIZE,
        step_height_range=ROUGH_STEP_HEIGHT_RANGE,
        step_width=ROUGH_STEP_WIDTH,
        platform_width=ROUGH_PLATFORM_WIDTH,
        border_width=_STAIR_BORDER_WIDTH,
    )


def rough_terrains_cfg(*, num_rows: int = 10) -> TerrainGeneratorCfg:
    """训练地形集：平地与上台阶各一列，env 按 ROUGH_TERRAIN_PROPORTIONS 分配。"""
    p = ROUGH_TERRAIN_PROPORTIONS
    return TerrainGeneratorCfg(
        size=ROUGH_PATCH_SIZE,
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        # 课程模式下生成器忽略该值，按子地形种类数取列，这里填同值只为可读。
        num_cols=len(p),
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": flat(proportion=p["flat"], size=ROUGH_PATCH_SIZE),
            "stairs_up": _stairs(pyramid_stairs_inv, proportion=p["stairs_up"]),
            "stairs_down": _stairs(pyramid_stairs, proportion=p["stairs_down"]),
            "slope_up": _slope(
                hf_pyramid_slope_inv, proportion=p["slope_up"], slope_range=ROUGH_SLOPE_UP_RANGE
            ),
            "slope_down": _slope(
                hf_pyramid_slope, proportion=p["slope_down"], slope_range=ROUGH_SLOPE_DOWN_RANGE
            ),
        },
        add_lights=False,
    )


def stair_only_terrains_cfg(*, num_rows: int = 10) -> TerrainGeneratorCfg:
    """只含上/下台阶与平地的地形集，用于台阶能力的定向评测（保留课程，按 terrain level 定位能上多高）。"""
    return TerrainGeneratorCfg(
        size=ROUGH_PATCH_SIZE,
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        num_cols=3,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": flat(proportion=0.1, size=ROUGH_PATCH_SIZE),
            "stairs_up": _stairs(pyramid_stairs_inv, proportion=0.45),
            "stairs_down": _stairs(pyramid_stairs, proportion=0.45),
        },
        add_lights=False,
    )


__all__ = [
    "ROUGH_PATCH_SIZE",
    "ROUGH_PLATFORM_WIDTH",
    "ROUGH_STEP_HEIGHT_RANGE",
    "ROUGH_STEP_WIDTH",
    "ROUGH_HFIELD_HORIZONTAL_SCALE",
    "ROUGH_SLOPE_DOWN_RANGE",
    "ROUGH_SLOPE_UP_RANGE",
    "ROUGH_TERRAIN_PROPORTIONS",
    "rough_terrains_cfg",
    "stair_only_terrains_cfg",
]
