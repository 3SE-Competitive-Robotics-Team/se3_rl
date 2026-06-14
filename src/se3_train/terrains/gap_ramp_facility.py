"""左平台-坑-反飞坡固定设施地形。

这个地形面向反向跳跃训练：
- 初始化在左侧长平台，先向前行驶一段
- 越过中间坑和反向三角坡高边
- 沿坡面落到右侧等高长平台，并继续向前行驶

尺寸来源于用户给出的毫米级草图。代码内统一换算成米。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import count

import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

_MM = 0.001
_TERRAIN_COLOR_LEFT = (0.45, 0.56, 0.40, 1.0)
_TERRAIN_COLOR_RAMP = (0.38, 0.45, 0.55, 1.0)
_TERRAIN_COLOR_TOP = (0.72, 0.68, 0.60, 1.0)
_FACILITY_INSTANCE_COUNTER = count()
BLIND_CLIMB_NUM_ROWS = 40
BLIND_CLIMB_PIT_LENGTH_RANGE = (0.05, 0.65)
BLIND_CLIMB_RAMP_HEIGHT_RANGE = (0.03, 0.35)
BLIND_CLIMB_RAMP_ANGLE_RANGE_DEG = (8.0, 17.0)


@dataclass(frozen=True, kw_only=True)
class GapRampFacilitySpec:
    """固定设施的几何参数，单位均为米。"""

    left_platform_height: float = 200 * _MM
    """左侧平台顶面高度。"""

    left_platform_length: float = 1800 * _MM
    """左侧起跑平台长度。"""

    pit_length: float = 650 * _MM
    """左侧平台到反飞坡高边之间的坑宽。"""

    ramp_angle_deg: float = 17.0
    """主坡角度。"""

    ramp_height: float = 350 * _MM
    """三角坡高边相对平台顶面的垂直高差。"""

    ramp_base_height: float = 200 * _MM
    """右侧坡脚/平台顶面高度，应与左侧平台近似等高。"""

    ramp_top_lip_length: float = 151 * _MM
    """坡顶短平台长度。"""

    right_platform_length: float = 1800 * _MM
    """右侧落地后继续行驶的平台长度。"""

    track_width: float = 2.00
    """设施横向宽度。"""

    platform_thickness: float = 0.20
    """左侧平台厚度。"""

    right_platform_thickness: float = 200 * _MM
    """右侧平台厚度。"""

    ramp_thickness: float = 0.08
    """主坡实体厚度。"""

    border_margin_x: float = 0.35
    """设施左右预留边距。"""

    border_margin_y: float = 0.35
    """设施前后预留边距。"""

    @property
    def ramp_angle_rad(self) -> float:
        return math.radians(self.ramp_angle_deg)

    @property
    def ramp_surface_length(self) -> float:
        """主坡斜面长度，由高差和角度反推。"""
        return self.ramp_height / math.sin(self.ramp_angle_rad)

    @property
    def ramp_horizontal_run(self) -> float:
        """主坡在水平面的投影长度。"""
        return self.ramp_height / math.tan(self.ramp_angle_rad)

    @property
    def total_length(self) -> float:
        """设施沿 x 轴的总长度。"""
        return (
            self.left_platform_length
            + self.pit_length
            + self.ramp_horizontal_run
            + self.ramp_top_lip_length
            + self.right_platform_length
        )

    @property
    def terrain_size(self) -> tuple[float, float]:
        """给 TerrainGenerator 用的 patch 尺寸。"""
        return (
            self.total_length + 2.0 * self.border_margin_x,
            self.track_width + 2.0 * self.border_margin_y,
        )

    @property
    def right_platform_height(self) -> float:
        """右侧平台顶面高度。"""
        return self.ramp_base_height

    @property
    def ramp_low_height(self) -> float:
        """坡脚低端高度，与两侧平台顶面一致。"""
        return self.ramp_base_height

    @property
    def ramp_high_height(self) -> float:
        """靠绿色平台一侧的坡顶高边高度。"""
        return self.ramp_low_height + self.ramp_height


@dataclass(kw_only=True)
class GapRampFacilityTerrainCfg(SubTerrainCfg):
    """固定设施地形配置。"""

    facility: GapRampFacilitySpec = field(default_factory=GapRampFacilitySpec)
    use_difficulty: bool = False
    """是否用 difficulty 从简化设施插值到完整设施。"""

    pit_length_range: tuple[float, float] = BLIND_CLIMB_PIT_LENGTH_RANGE
    """课程坑宽范围。"""

    ramp_height_range: tuple[float, float] = BLIND_CLIMB_RAMP_HEIGHT_RANGE
    """课程坡顶高差范围。"""

    ramp_angle_range_deg: tuple[float, float] = BLIND_CLIMB_RAMP_ANGLE_RANGE_DEG
    """课程坡面角度范围。"""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng

        body = spec.body("terrain")
        prefix = f"gap_ramp_{next(_FACILITY_INSTANCE_COUNTER)}"
        f = (
            _facility_for_difficulty(
                self.facility,
                difficulty,
                pit_length_range=self.pit_length_range,
                ramp_height_range=self.ramp_height_range,
                ramp_angle_range_deg=self.ramp_angle_range_deg,
            )
            if self.use_difficulty
            else self.facility
        )

        local_center_y = self.size[1] / 2.0
        left_start_x = f.border_margin_x
        left_end_x = left_start_x + f.left_platform_length
        pit_end_x = left_end_x + f.pit_length

        terrain_geoms: list[TerrainGeometry] = []

        left_center_x = left_start_x + f.left_platform_length / 2.0
        left_center_z = f.left_platform_height - f.platform_thickness / 2.0
        left_box = body.add_geom(
            name=f"{prefix}_left_platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                f.left_platform_length / 2.0,
                f.track_width / 2.0,
                f.platform_thickness / 2.0,
            ),
            pos=(left_center_x, local_center_y, left_center_z),
            contype=2,
            conaffinity=1,
        )
        left_box.rgba[:] = _TERRAIN_COLOR_LEFT
        terrain_geoms.append(TerrainGeometry(geom=left_box, color=_TERRAIN_COLOR_LEFT))

        support_center_x = (
            pit_end_x
            + f.ramp_horizontal_run
            + (f.ramp_top_lip_length + f.right_platform_length) / 2.0
        )
        support_center_z = f.right_platform_height - f.right_platform_thickness / 2.0
        support_box = body.add_geom(
            name=f"{prefix}_right_approach_platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                (f.ramp_top_lip_length + f.right_platform_length) / 2.0,
                f.track_width / 2.0,
                f.right_platform_thickness / 2.0,
            ),
            pos=(support_center_x, local_center_y, support_center_z),
            contype=2,
            conaffinity=1,
        )
        support_box.rgba[:] = _TERRAIN_COLOR_TOP
        terrain_geoms.append(TerrainGeometry(geom=support_box, color=_TERRAIN_COLOR_TOP))

        ramp_mesh_name = f"{prefix}_reverse_triangular_prism_mesh"
        spec.add_mesh(
            name=ramp_mesh_name,
            uservert=_reverse_ramp_vertices(
                run=f.ramp_horizontal_run,
                width=f.track_width,
                low_height=f.ramp_low_height,
                high_height=f.ramp_high_height,
            ),
            userface=_reverse_ramp_faces(),
        )
        ramp_geom = body.add_geom(
            name=f"{prefix}_main_slope",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=ramp_mesh_name,
            pos=(
                pit_end_x + f.ramp_horizontal_run / 2.0,
                local_center_y,
                0.0,
            ),
            contype=2,
            conaffinity=1,
        )
        ramp_geom.rgba[:] = _TERRAIN_COLOR_RAMP
        terrain_geoms.append(TerrainGeometry(geom=ramp_geom, color=_TERRAIN_COLOR_RAMP))

        origin = np.array(
            [
                left_center_x,
                local_center_y,
                f.left_platform_height,
            ],
            dtype=np.float64,
        )
        return TerrainOutput(origin=origin, geometries=terrain_geoms)


def _reverse_ramp_vertices(
    *, run: float, width: float, low_height: float, high_height: float
) -> list[float]:
    """生成直接落在地面上的反飞坡实体顶点。

    x 负方向是靠绿色平台/坑的一侧，高边为竖直面；x 正方向接右侧等高平台。
    """
    half_run = run / 2.0
    half_width = width / 2.0
    vertices = (
        (-half_run, -half_width, 0.0),
        (-half_run, half_width, 0.0),
        (half_run, -half_width, 0.0),
        (half_run, half_width, 0.0),
        (-half_run, -half_width, high_height),
        (-half_run, half_width, high_height),
        (half_run, -half_width, low_height),
        (half_run, half_width, low_height),
    )
    return [coord for vertex in vertices for coord in vertex]


def _reverse_ramp_faces() -> list[int]:
    """反飞坡实体的三角面索引。"""
    faces = (
        (0, 2, 3),
        (0, 3, 1),  # 底面
        (0, 1, 5),
        (0, 5, 4),  # 靠绿色平台的竖直高面
        (2, 6, 7),
        (2, 7, 3),  # 右侧低端竖直面
        (4, 5, 7),
        (4, 7, 6),  # 斜坡面
        (0, 4, 6),
        (0, 6, 2),  # 前侧面
        (1, 3, 7),
        (1, 7, 5),  # 后侧面
    )
    return [index for face in faces for index in face]


def gap_ramp_facility_terrain_cfg() -> TerrainGeneratorCfg:
    """返回可直接挂到 TerrainEntityCfg 的固定设施生成器。"""
    spec = GapRampFacilitySpec()
    return TerrainGeneratorCfg(
        curriculum=False,
        size=spec.terrain_size,
        border_width=0.0,
        border_height=0.0,
        num_rows=1,
        num_cols=1,
        color_scheme="height",
        sub_terrains={
            "gap_ramp_facility": GapRampFacilityTerrainCfg(facility=spec),
        },
        difficulty_range=(0.0, 0.0),
        add_lights=True,
    )


def gap_ramp_blind_climb_terrain_cfg(
    num_rows: int = BLIND_CLIMB_NUM_ROWS,
) -> TerrainGeneratorCfg:
    """返回盲爬训练用的课程化坑坡设施生成器。"""
    spec = GapRampFacilitySpec()
    return TerrainGeneratorCfg(
        curriculum=True,
        size=spec.terrain_size,
        border_width=0.0,
        border_height=0.0,
        num_rows=num_rows,
        num_cols=1,
        color_scheme="height",
        sub_terrains={
            "blind_climb_facility": GapRampFacilityTerrainCfg(
                proportion=1.0,
                facility=spec,
                use_difficulty=True,
            ),
        },
        difficulty_range=(0.0, 1.0),
        add_lights=True,
    )


def gap_ramp_facility_entity_cfg(num_envs: int = 1) -> TerrainEntityCfg:
    """返回固定设施对应的 TerrainEntityCfg。"""
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=gap_ramp_facility_terrain_cfg(),
        max_init_terrain_level=0,
        num_envs=num_envs,
    )


def gap_ramp_blind_climb_entity_cfg(num_envs: int = 1024) -> TerrainEntityCfg:
    """返回盲爬训练课程地形对应的 TerrainEntityCfg。"""
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=gap_ramp_blind_climb_terrain_cfg(),
        max_init_terrain_level=0,
        num_envs=num_envs,
    )


def _facility_for_difficulty(
    facility: GapRampFacilitySpec,
    difficulty: float,
    *,
    pit_length_range: tuple[float, float],
    ramp_height_range: tuple[float, float],
    ramp_angle_range_deg: tuple[float, float],
) -> GapRampFacilitySpec:
    """按课程难度插值设施几何，难度 1 对应完整场景。"""
    progress = min(1.0, max(0.0, float(difficulty)))
    return replace(
        facility,
        pit_length=_lerp(pit_length_range[0], pit_length_range[1], progress),
        ramp_height=_lerp(ramp_height_range[0], ramp_height_range[1], progress),
        ramp_angle_deg=_lerp(ramp_angle_range_deg[0], ramp_angle_range_deg[1], progress),
    )


def _lerp(start: float, end: float, progress: float) -> float:
    """线性插值。"""
    return float(start) + (float(end) - float(start)) * float(progress)
