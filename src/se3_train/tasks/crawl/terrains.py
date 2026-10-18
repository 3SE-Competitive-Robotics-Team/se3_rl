"""趴地斜坡版的训练地形：自建「平地 → 上坡 → 平台」直线坡，沿世界系 +x 只上不下。

2026-10-18 用户定：不用 mjlab 自带的倒金字塔斜坡，改为平地接斜坡接平地，坡度最陡 15°。
- 每块 9 m 见方，出生点在块中心的低平地上（terrain_edge_reached / out_of_terrain_bounds 都按块中心算位移）；
  坡脚在出生点前方 CRAWL_RAMP_START_AHEAD，坡水平长度 CRAWL_RAMP_RUN 固定，坡度随难度从 0 线性插值到 15°，
  15° 时坡高 0.40 m；坡顶平台一直铺到块的 +x 边缘。RMUC 门洞前的坡约 13.7°（rise 0.18 / run 0.74）。
- 坡与平台用长方体：凸体对长方体的接触是精确平面，不受 hfield 三角棱柱分辨率限制，坡面法线也干净。
- 网格行沿 +x 排列，平台在块边缘与下一块的低平地之间是一道落差；出块截断阈值取 0.9 倍半块，
  机器人在平台上距离块边缘还有 0.45 m 时就截断，向后开时在撞到上一块平台立面之前截断（见 env_cfg）。
- 升级判据换成沿 +x 的前进距离（curriculums.terrain_levels_forward）：官方 terrain_levels_vel 按位移模长升级，
  在只往一个方向上坡的地形上，倒车开远也会升级。

几何依据（docs/plan/crawl_guide_wheels_20261017.md）：机身底部分段凸包名义趴姿接近角 6.6°、纵向通过角 12.1°，
15° 远超趴姿接近角，必须靠改变姿态（抬车头）才能上坡；牵引上限 tanθ ≤ μ × 主轮承重占比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import make_plane

CRAWL_SLOPE_DEG_RANGE = (0.0, 15.0)
CRAWL_SLOPE_PATCH_SIZE = (9.0, 9.0)
CRAWL_RAMP_START_AHEAD = 1.0
CRAWL_RAMP_RUN = 1.5
CRAWL_SLOPE_NUM_ROWS = 10
# 机身完全上到平台才算爬上去：坡顶到出生点距离 + 主轮在机身后方约 0.23 m 的余量。
CRAWL_SLOPE_PROMOTE_DISTANCE = CRAWL_RAMP_START_AHEAD + CRAWL_RAMP_RUN + 0.4
_RAMP_HALF_THICKNESS = 0.25
_MIN_RISE = 1.0e-3


@dataclass(kw_only=True)
class FlatRampFlatTerrainCfg(SubTerrainCfg):
    """块中心出生的低平地，前方 ramp_start_ahead 处起坡，坡顶平台铺到块的 +x 边缘。"""

    slope_deg_range: tuple[float, float] = CRAWL_SLOPE_DEG_RANGE
    ramp_start_ahead: float = CRAWL_RAMP_START_AHEAD
    ramp_run: float = CRAWL_RAMP_RUN

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        size_x, size_y = (float(v) for v in self.size)
        center_x, center_y = 0.5 * size_x, 0.5 * size_y
        geometries = [
            TerrainGeometry(
                geom=make_plane(body, self.size, 0.0, center_zero=False)[0],
                color=(0.5, 0.5, 0.5, 1.0),
            )
        ]

        low, high = (float(v) for v in self.slope_deg_range)
        theta = math.radians(low + float(difficulty) * (high - low))
        run = float(self.ramp_run)
        rise = run * math.tan(theta)
        x0 = center_x + float(self.ramp_start_ahead)
        x1 = x0 + run
        assert x1 < size_x, "坡顶必须落在块内"
        if rise > _MIN_RISE:
            # 坡：绕 y 轴转 -theta 的长方体，顶面恰好从 (x0, 0) 连到 (x1, rise)；两端面垂直于坡面，
            # 坡脚一端伸进地面以下、坡顶一端伸进平台内部，不会露出棱角。
            t = _RAMP_HALF_THICKNESS
            ramp = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.5 * run / math.cos(theta), 0.5 * size_y, t),
                pos=(
                    0.5 * (x0 + x1) + t * math.sin(theta),
                    center_y,
                    0.5 * rise - t * math.cos(theta),
                ),
                quat=(math.cos(-0.5 * theta), 0.0, math.sin(-0.5 * theta), 0.0),
            )
            geometries.append(TerrainGeometry(geom=ramp, color=(0.85, 0.55, 0.25, 1.0)))
            plateau = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.5 * (size_x - x1), 0.5 * size_y, 0.5 * rise),
                pos=(0.5 * (x1 + size_x), center_y, 0.5 * rise),
            )
            geometries.append(TerrainGeometry(geom=plateau, color=(0.35, 0.55, 0.75, 1.0)))

        origin = np.array([center_x, center_y, 0.0], dtype=np.float64)
        return TerrainOutput(origin=origin, geometries=geometries)


def crawl_slope_terrains_cfg(*, num_rows: int = CRAWL_SLOPE_NUM_ROWS) -> TerrainGeneratorCfg:
    """只有「平地-上坡-平台」一列的课程地形，难度沿行把坡度从 0 插值到 15°。"""
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
            "flat_ramp_flat": FlatRampFlatTerrainCfg(proportion=1.0, size=CRAWL_SLOPE_PATCH_SIZE),
        },
        add_lights=False,
    )


__all__ = [
    "CRAWL_SLOPE_DEG_RANGE",
    "CRAWL_SLOPE_PROMOTE_DISTANCE",
    "FlatRampFlatTerrainCfg",
    "crawl_slope_terrains_cfg",
]
