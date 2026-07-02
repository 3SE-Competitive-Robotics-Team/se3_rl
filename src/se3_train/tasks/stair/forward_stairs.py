"""沿世界系 +x 方向排列的台阶地形。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane


@dataclass(kw_only=True)
class BoxForwardStairsTerrainCfg(SubTerrainCfg):
    """生成与 sim2sim stair terrain 对齐的直线台阶。"""

    step_height_range: tuple[float, float] = (0.05, 0.20)
    step_depth: float = 0.80
    step_count: int = 6
    stair_start_x: float = 1.0
    spawn_x: float = 1.0
    half_width: float = 3.0

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        floor_geom = make_plane(body, self.size, 0.0, center_zero=False)[0]
        geometries.append(TerrainGeometry(geom=floor_geom, color=(0.5, 0.5, 0.5, 1.0)))

        low, high = (float(value) for value in self.step_height_range)
        step_height = low + float(difficulty) * (high - low)
        step_depth = float(self.step_depth)
        step_count = max(1, int(self.step_count))
        half_width = max(1.0e-6, float(self.half_width))
        origin_x = float(self.spawn_x)
        start_x = origin_x + float(self.stair_start_x)
        center_y = 0.5 * float(self.size[1])

        for idx in range(step_count):
            height = step_height * float(idx + 1)
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.5 * step_depth, half_width, 0.5 * height),
                pos=(start_x + idx * step_depth + 0.5 * step_depth, center_y, 0.5 * height),
            )
            t = idx / max(step_count - 1, 1)
            color = (0.25 + 0.35 * t, 0.45 + 0.20 * t, 0.85 - 0.30 * t, 1.0)
            geometries.append(TerrainGeometry(geom=geom, color=color))

        origin = np.array([origin_x, center_y, 0.0], dtype=np.float64)
        return TerrainOutput(origin=origin, geometries=geometries)


__all__ = ["BoxForwardStairsTerrainCfg"]
