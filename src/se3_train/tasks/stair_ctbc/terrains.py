"""CTBC 台阶任务专用地形。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class BoxStageStairsTerrainCfg(SubTerrainCfg):
    """按实物台阶尺寸拼接的单向台阶地形。"""

    approach_length: float = 1.5
    """第一级台阶前的平地长度。"""
    first_platform_length: float = 0.8
    """第一级台阶后的平台长度。"""
    second_platform_length: float = 0.15
    """第二级台阶后的平台长度。"""
    final_platform_length: float = 2.0
    """下落 50 mm 后的平台长度。"""
    first_height: float = 0.2
    """第一级平台绝对高度。"""
    second_rise: float = 0.15
    """第二级相对第一级继续上升的高度。"""
    final_drop: float = 0.05
    """第三级相对第二级下落的高度。"""
    walkway_width: float = 2.0
    """台阶走廊宽度。"""
    base_thickness: float = 1.0
    """地形盒体向下延伸厚度，避免薄片接触不稳定。"""
    spawn_base_offset: float = 0.1
    """出生 base 相对第一级竖面的 x 偏移，使轮子前缘贴近台阶。"""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """生成 200 mm、150 mm、-50 mm 的三段台阶。"""
        del difficulty, rng

        body = spec.body("terrain")
        total_length = (
            self.approach_length
            + self.first_platform_length
            + self.second_platform_length
            + self.final_platform_length
        )
        start_x = max(0.0, 0.5 * (self.size[0] - total_length))
        center_y = 0.5 * self.size[1]
        walkway_width = min(float(self.walkway_width), float(self.size[1]))

        first_riser_x = start_x + self.approach_length
        first_end_x = first_riser_x + self.first_platform_length
        second_end_x = first_end_x + self.second_platform_length
        final_end_x = second_end_x + self.final_platform_length

        first_top_z = self.first_height
        second_top_z = self.first_height + self.second_rise
        final_top_z = second_top_z - self.final_drop

        geometries = [
            self._add_box(
                body,
                center=(0.5 * self.size[0], center_y, -0.5 * self.base_thickness),
                size=(0.5 * self.size[0], 0.5 * self.size[1], 0.5 * self.base_thickness),
                color=(0.42, 0.42, 0.42, 1.0),
            ),
            self._add_platform(
                body,
                x0=first_riser_x,
                x1=first_end_x,
                center_y=center_y,
                width=walkway_width,
                top_z=first_top_z,
                color=(0.62, 0.35, 0.24, 1.0),
            ),
            self._add_platform(
                body,
                x0=first_end_x,
                x1=second_end_x,
                center_y=center_y,
                width=walkway_width,
                top_z=second_top_z,
                color=(0.70, 0.42, 0.24, 1.0),
            ),
            self._add_platform(
                body,
                x0=second_end_x,
                x1=final_end_x,
                center_y=center_y,
                width=walkway_width,
                top_z=final_top_z,
                color=(0.48, 0.52, 0.34, 1.0),
            ),
        ]

        origin = np.array([first_riser_x + self.spawn_base_offset, center_y, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)

    def _add_platform(
        self,
        body: mujoco.MjsBody,
        *,
        x0: float,
        x1: float,
        center_y: float,
        width: float,
        top_z: float,
        color: tuple[float, float, float, float],
    ) -> TerrainGeometry:
        length = max(float(x1 - x0), 1.0e-6)
        return self._add_box(
            body,
            center=(0.5 * (x0 + x1), center_y, 0.5 * (top_z - self.base_thickness)),
            size=(0.5 * length, 0.5 * width, 0.5 * (top_z + self.base_thickness)),
            color=color,
        )

    @staticmethod
    def _add_box(
        body: mujoco.MjsBody,
        *,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ) -> TerrainGeometry:
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=center,
            size=(
                max(float(size[0]), 1.0e-6),
                max(float(size[1]), 1.0e-6),
                max(float(size[2]), 1.0e-6),
            ),
        )
        return TerrainGeometry(geom=geom, color=color)
