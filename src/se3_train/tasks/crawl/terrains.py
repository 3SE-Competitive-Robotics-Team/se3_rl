"""趴地斜坡版的训练地形：只有上坡（倒金字塔斜坡，出生在坑底平台、四周往上）。

2026-10-17 用户定：坡度 0–0.15（rise/run，约 8.5°），只要上坡。坡度上限的依据（docs/plan/crawl_guide_wheels_20261017.md）：
- 几何：机身底部凸包分段后名义趴姿接近角 6.6°、纵向通过角 12.1°（真车视觉 mesh 7.8° / 17.6°），
  0.15 坡底的凹折角会让车头轻擦，与真车一致；
- 牵引：导轮自由滚动不出刹车力，下滑力全靠主轮摩擦，tanθ ≤ μ × 主轮承重占比。

块大小、hfield 水平分辨率与边框沿用 Rough（9 m、0.2 m：凸体-hfield 单对最多 50 个三角棱柱，0.1 m 格子会丢接触）。
课程模式下生成器按子地形种类数取列，这里只有一种，所以只有一列；难度沿行从 0 插值到 0.15。
"""

from __future__ import annotations

from mjlab.terrains.config import hf_pyramid_slope_inv
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

CRAWL_SLOPE_UP_RANGE = (0.0, 0.15)
CRAWL_SLOPE_PATCH_SIZE = (9.0, 9.0)
CRAWL_SLOPE_PLATFORM_WIDTH = 2.0
CRAWL_SLOPE_HFIELD_HORIZONTAL_SCALE = 0.2
CRAWL_SLOPE_NUM_ROWS = 10


def crawl_slope_terrains_cfg(*, num_rows: int = CRAWL_SLOPE_NUM_ROWS) -> TerrainGeneratorCfg:
    """只有上坡一列的课程地形。"""
    return TerrainGeneratorCfg(
        size=CRAWL_SLOPE_PATCH_SIZE,
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        num_cols=1,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "slope_up": hf_pyramid_slope_inv(
                proportion=1.0,
                size=CRAWL_SLOPE_PATCH_SIZE,
                slope_range=CRAWL_SLOPE_UP_RANGE,
                platform_width=CRAWL_SLOPE_PLATFORM_WIDTH,
                border_width=0.5,
                horizontal_scale=CRAWL_SLOPE_HFIELD_HORIZONTAL_SCALE,
            ),
        },
        add_lights=False,
    )


__all__ = ["CRAWL_SLOPE_UP_RANGE", "crawl_slope_terrains_cfg"]
