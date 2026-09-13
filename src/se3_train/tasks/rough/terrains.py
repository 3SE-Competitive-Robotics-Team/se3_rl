"""崎岖地形任务的地形集：用 mjlab 官方 preset 组合，几何按本机器人标定。

课程语义是 mjlab `TerrainGeneratorCfg(curriculum=True)` 的：每种子地形独占一列，难度沿行插值
（row 0 取各 `*_range` 下界，最后一行取上界），升降级由官方 `terrain_levels_vel` 管
（见 env_cfg.py）。

列名按机器人从出生点出发实际经历的方向命名：mjlab 把正金字塔的出生点放在顶部平台、
反金字塔放在底部凹坑，所以 `stairs_up` / `slope_up` 用反金字塔（出生在低处、向外爬升），
`stairs_down` / `slope_down` 用正金字塔（出生在高处、向外下行）。

尺寸按本机器人标定：轮半径 0.06 m、轮距 0.433 m、base 指令高度 0.20–0.38 m。台阶踏面 1.5 m
（远大于轮距，逐级上而非连续楼梯）；台阶高课程 0.02–0.20 m，上界是赛场台阶量级，下界让 row 0
近似平地，课程起点不会把策略卡死。
"""

from __future__ import annotations

from mjlab.terrains.config import (
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# 单块地形边长；配 1.5 m 踏面、2.0 m 中央平台、0.5 m 边框时金字塔每侧 2 级。
ROUGH_PATCH_SIZE = (9.0, 9.0)
ROUGH_STEP_HEIGHT_RANGE = (0.02, 0.20)
ROUGH_STEP_WIDTH = 1.5
ROUGH_PLATFORM_WIDTH = 2.0
_STAIR_BORDER_WIDTH = 0.5
_SLOPE_RANGE = (0.05, 0.30)

# heightfield 的水平分辨率(m)。MuJoCo 凸体-hfield 碰撞按 AABB 覆盖的格子逐格生成三角棱柱，
# 单对上限 50 个三角形；机身最大碰撞块 0.52 m 宽，0.1 m 格子要 98 个三角形，接触被丢弃、
# 机身穿进地形（R1 崩溃的物理侧诱因）；0.2 m 格子只要 32 个。斜坡是平面，粗化不损失几何。
_HF_HORIZONTAL_SCALE = 0.2

# env 分配比例（2026-09-09 用户定，A9）：只留平地与上台阶。mjlab 在课程模式下仍给比例为 0 的列
# 至少 1 个 env，列保留、日志键不变；`proportion` 只管 env 分配，不管列数。
ROUGH_TERRAIN_PROPORTIONS: dict[str, float] = {
    "flat": 0.30,
    "stairs_up": 0.70,
    "stairs_down": 0.0,
    "slope_up": 0.0,
    "slope_down": 0.0,
    "random_rough": 0.0,
}


def _stairs(preset, *, proportion: float):
    return preset(
        proportion=proportion,
        size=ROUGH_PATCH_SIZE,
        step_height_range=ROUGH_STEP_HEIGHT_RANGE,
        step_width=ROUGH_STEP_WIDTH,
        platform_width=ROUGH_PLATFORM_WIDTH,
        border_width=_STAIR_BORDER_WIDTH,
    )


def _slope(preset, *, proportion: float):
    return preset(
        proportion=proportion,
        size=ROUGH_PATCH_SIZE,
        slope_range=_SLOPE_RANGE,
        platform_width=2.0,
        border_width=0.25,
        horizontal_scale=_HF_HORIZONTAL_SCALE,
    )


def rough_terrains_cfg(*, num_rows: int = 10) -> TerrainGeneratorCfg:
    """训练地形集：平地、上/下台阶、上/下斜坡、随机起伏各一列，env 按 ROUGH_TERRAIN_PROPORTIONS 分配。"""
    p = ROUGH_TERRAIN_PROPORTIONS
    return TerrainGeneratorCfg(
        size=ROUGH_PATCH_SIZE,
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        # 课程模式下生成器忽略该值，按子地形种类数取列，这里填同值只为可读。
        num_cols=6,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": flat(proportion=p["flat"], size=ROUGH_PATCH_SIZE),
            "stairs_up": _stairs(pyramid_stairs_inv, proportion=p["stairs_up"]),
            "stairs_down": _stairs(pyramid_stairs, proportion=p["stairs_down"]),
            "slope_up": _slope(hf_pyramid_slope_inv, proportion=p["slope_up"]),
            "slope_down": _slope(hf_pyramid_slope, proportion=p["slope_down"]),
            "random_rough": random_rough(
                proportion=p["random_rough"],
                size=ROUGH_PATCH_SIZE,
                noise_range=(0.0, 0.05),
                noise_step=0.005,
                border_width=0.25,
                horizontal_scale=_HF_HORIZONTAL_SCALE,
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
    "ROUGH_TERRAIN_PROPORTIONS",
    "rough_terrains_cfg",
    "stair_only_terrains_cfg",
]
