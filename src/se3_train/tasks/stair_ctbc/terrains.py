"""CTBC 台阶任务专用地形。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class BoxRampTerrainCfg(SubTerrainCfg):
    """沿世界系 +x 方向上升的斜坡地形。"""

    slope_deg: float
    """坡面角度，单位为度。"""
    height: float
    """坡顶相对入口平地的高度。"""
    min_height_scale: float = 0.2
    """最低课程难度下的坡高比例。"""
    approach_length: float = 1.5
    """斜坡前平地长度。"""
    top_platform_length: float = 2.0
    """坡顶平台长度。"""
    walkway_width: float = 2.0
    """斜坡走廊宽度。"""
    wedge_depth: float = 0.5
    """斜坡楔形体向下延伸深度，避免薄片接触不稳定。"""
    base_thickness: float = 1.0
    """平地和平台向下延伸厚度。"""
    spawn_base_offset: float = -0.08
    """出生 base 相对坡底的 x 偏移，使轮子在坡面入口前。"""

    @property
    def run_length(self) -> float:
        """返回斜坡水平投影长度。"""
        return float(self.height) / np.tan(np.deg2rad(float(self.slope_deg)))

    @property
    def slope_length(self) -> float:
        """返回斜坡坡面长度。"""
        return float(self.height) / np.sin(np.deg2rad(float(self.slope_deg)))

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """生成指定坡度和高度的单向斜坡。"""
        del rng

        body = spec.body("terrain")
        height = self._scaled_height(difficulty)
        run_length = self.run_length
        total_length = self.approach_length + run_length + self.top_platform_length
        start_x = max(0.0, 0.5 * (self.size[0] - total_length))
        center_y = 0.5 * self.size[1]
        walkway_width = min(float(self.walkway_width), float(self.size[1]))
        ramp_start_x = start_x + self.approach_length
        ramp_end_x = ramp_start_x + run_length

        geometries = [
            self._add_box(
                body,
                center=(0.5 * self.size[0], center_y, -0.5 * self.base_thickness),
                size=(0.5 * self.size[0], 0.5 * self.size[1], 0.5 * self.base_thickness),
                color=(0.42, 0.42, 0.42, 1.0),
            ),
            self._add_wedge(
                spec,
                body,
                x0=ramp_start_x,
                x1=ramp_end_x,
                center_y=center_y,
                width=walkway_width,
                height=height,
                depth=float(self.wedge_depth),
                color=(0.58, 0.42, 0.25, 1.0),
            ),
        ]
        if self.top_platform_length > 0.0:
            geometries.append(
                self._add_box(
                    body,
                    center=(
                        ramp_end_x + 0.5 * self.top_platform_length,
                        center_y,
                        0.5 * (height - self.base_thickness),
                    ),
                    size=(
                        0.5 * self.top_platform_length,
                        0.5 * walkway_width,
                        0.5 * (height + self.base_thickness),
                    ),
                    color=(0.40, 0.50, 0.34, 1.0),
                )
            )

        origin = np.array([ramp_start_x + self.spawn_base_offset, center_y, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)

    def _scaled_height(self, difficulty: float) -> float:
        """按课程难度缩放坡高，最高难度等于配置目标高度。"""
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        scale = float(self.min_height_scale) + (1.0 - float(self.min_height_scale)) * difficulty
        return float(self.height) * scale

    @staticmethod
    def _add_wedge(
        spec: mujoco.MjSpec,
        body: mujoco.MjsBody,
        *,
        x0: float,
        x1: float,
        center_y: float,
        width: float,
        height: float,
        depth: float,
        color: tuple[float, float, float, float],
    ) -> TerrainGeometry:
        """添加坡面楔形 mesh，入口和出口端面保持世界系竖直。"""
        half_width = 0.5 * float(width)
        y0 = float(center_y) - half_width
        y1 = float(center_y) + half_width
        z_bottom = -float(depth)
        vertices = [
            x0,
            y0,
            z_bottom,
            x1,
            y0,
            z_bottom,
            x1,
            y1,
            z_bottom,
            x0,
            y1,
            z_bottom,
            x0,
            y0,
            0.0,
            x1,
            y0,
            height,
            x1,
            y1,
            height,
            x0,
            y1,
            0.0,
        ]
        faces = [
            0,
            2,
            1,
            0,
            3,
            2,
            0,
            4,
            7,
            0,
            7,
            3,
            1,
            2,
            6,
            1,
            6,
            5,
            0,
            1,
            5,
            0,
            5,
            4,
            3,
            7,
            6,
            3,
            6,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
        ]
        mesh_name = f"ramp_wedge_{len(spec.meshes)}"
        spec.add_mesh(
            name=mesh_name,
            uservert=vertices,
            userface=faces,
            maxhullvert=8,
        )
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
        )
        return TerrainGeometry(geom=geom, color=color)

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
    min_height_scale: float = 0.2
    """最低课程难度下的台阶高度比例。"""
    walkway_width: float = 2.0
    """台阶走廊宽度。"""
    base_thickness: float = 1.0
    """地形盒体向下延伸厚度，避免薄片接触不稳定。"""
    spawn_base_offset: float = -0.08
    """出生 base 相对第一级竖面的 x 偏移，使轮子在台阶前。"""

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        """生成 200 mm、150 mm、-50 mm 的三段台阶。"""
        del rng

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

        height_scale = self._height_scale(difficulty)
        first_top_z = self.first_height * height_scale
        second_top_z = first_top_z + self.second_rise * height_scale
        final_top_z = second_top_z - self.final_drop * height_scale

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

    def _height_scale(self, difficulty: float) -> float:
        """按课程难度缩放台阶高度，最高难度保持实物尺寸。"""
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        return float(self.min_height_scale) + (1.0 - float(self.min_height_scale)) * difficulty

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
