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

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.config import (
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
)
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import make_border

# 单块地形边长；配 0.7 m 踏面、2.0 m 中央平台、0.5 m 边框时金字塔每侧 (9 − 1 − 2) / (2 × 0.7) 取整 4 级。
ROUGH_PATCH_SIZE = (9.0, 9.0)
ROUGH_STEP_HEIGHT_RANGE = (0.02, 0.20)
# 踏面宽 0.7 m（M5，2026-09-14 用户定；M1–M3 为 1.5 m，M4 为 0.5 m）。评测场景 serialleg_stairs_1to9.xml 由它生成，
# 改了要重新生成；旧几何留在 serialleg_stairs_1to9_tread1p5.xml（M1–M3）与 _tread0p5.xml（M4）供回放。
ROUGH_STEP_WIDTH = 0.7
ROUGH_PLATFORM_WIDTH = 2.0
_STAIR_BORDER_WIDTH = 0.5

# 二级台阶列（M24，2026-09-21 用户定）：复刻复旦 sim2sim 场景那道「20 cm/0.6 m + 15 cm/0.15 m」的凸棱。
# 这里是"哪些列算上台阶"的**唯一定义**，commands.py 与 env_cfg.py 的台阶开关全部引用它，
# 免得以后再加列时漏掉某一处（台阶专项奖励、台阶指令、地形感知高度下限、窄核权重、支撑面高度参考、
# 轮高差罚、宽核 σ、接触税豁免，共八处）。
ROUGH_TWO_STEP_UP_COLUMN = "stairs_two_step_up"
ROUGH_TWO_STEP_DOWN_COLUMN = "stairs_two_step_down"
# 上行的二级台阶与 stairs_up 同待遇（台阶专项奖励、台阶指令、地形感知高度下限、窄核 w=1、
# 支撑面高度参考、轮高度差罚、宽核 σ、接触税豁免）；下行那列按平地待遇，只豁免接触税（同 stairs_down）。
ROUGH_STAIR_LIKE_COLUMNS: tuple[str, ...] = ("stairs_up", ROUGH_TWO_STEP_UP_COLUMN)
# 二级台阶的课程：难度 0 是两级小坎（5 cm + 4 cm、踏面都 0.6 m），难度 1 精确等于源几何
# （20 cm 踏面 0.6 m → 15 cm 踏面 0.15 m → 下 5 cm）。第二级踏面收窄是核心课程量。
ROUGH_TWO_STEP_FIRST_HEIGHT_RANGE = (0.05, 0.20)
ROUGH_TWO_STEP_SECOND_HEIGHT_RANGE = (0.04, 0.15)
ROUGH_TWO_STEP_FIRST_WIDTH = 0.60
ROUGH_TWO_STEP_SECOND_WIDTH_RANGE = (0.60, 0.15)
ROUGH_TWO_STEP_OUTER_DROP = 0.05
# 门控（用户定）：stairs_up 的平均等级达到这个值之前，两条二级台阶列的 env 暂放到各自的"母列"
# （上行去 stairs_up、下行去 flat），达标后一次性放开且不再回收（见 curriculums.two_step_gate）。
ROUGH_TWO_STEP_GATE_LEVEL = 5.0

# env 分配比例（2026-09-09 用户定，A9）：只留平地与上台阶。这里的键就是训练地形的全部列；
# `proportion` 只管 env 分配，不管列数，所以比例为 0 的列不能靠设 0 关掉，只能不放进来。
# 2026-09-15 用户定：在上台阶之外补下台阶与上下坡。比例让台阶仍占主力（55%）。
# 实测 geom 代价（9 m 块 × 10 行，scripts 见 .scratch/terrain_cost.py）：
# flat 14、stairs_up/stairs_down 各 214、slope_up/slope_down 各 14（hfield 类几乎免费）。
# 不选的：格宽 0.5 的 box_random_grid 要 2454 个 geom（十倍于台阶）、stepping_stones 784 且石块间是
# 2 m 深坑对轮距 0.433 是断崖难度、discrete_obstacles 只能绕而地形列 yaw 指令绕不开。
# M24（2026-09-21 用户定）：新增上行/下行两条二级台阶列，比例从 stairs_up、stairs_down 与两条坡匀出来。
# 代价：每块 17 个 box（stairs_up 是 21），七列合计 geom 450 → 790，单轮预计从 2.6 s 涨到 3.2 s 上下。
ROUGH_TERRAIN_PROPORTIONS: dict[str, float] = {
    "flat": 0.15,
    "stairs_up": 0.28,
    ROUGH_TWO_STEP_UP_COLUMN: 0.15,
    ROUGH_TWO_STEP_DOWN_COLUMN: 0.10,
    "stairs_down": 0.14,
    "slope_up": 0.09,
    "slope_down": 0.09,
}

# 坡度（rise/run）：上坡 0.4 = 21.8°，摩擦 1.0 下轮式可行；下坡有重力助推更易失控，上界收到 0.35。
ROUGH_SLOPE_UP_RANGE = (0.0, 0.4)
ROUGH_SLOPE_DOWN_RANGE = (0.0, 0.35)
# hfield 水平分辨率必须 0.2 m：MuJoCo 凸体-hfield 按 AABB 覆盖的格子逐格生成三角棱柱、单对上限 50 个，
# 机身最大碰撞块 0.52 m 宽在 0.1 m 格子下要 98 个三角形，接触会被丢弃、机身穿进地形（R1 崩溃的物理侧诱因）。
ROUGH_HFIELD_HORIZONTAL_SCALE = 0.2


# 复旦 sim2sim 场景那道二级台阶的配色：中心平台灰、第一级蓝、第二级橙（突出那道窄棱）、外圈浅蓝。
_TWO_STEP_PLATFORM_RGBA = (0.45, 0.47, 0.50, 1.0)
_TWO_STEP_FIRST_RGBA = (0.16, 0.62, 0.78, 1.0)
_TWO_STEP_SECOND_RGBA = (0.92, 0.55, 0.18, 1.0)
_TWO_STEP_OUTER_RGBA = (0.35, 0.68, 0.82, 1.0)
_TWO_STEP_BORDER_RGBA = (0.28, 0.30, 0.33, 1.0)


@dataclass(kw_only=True)
class TwoStepStairsTerrainCfg(SubTerrainCfg):
    """复旦 sim2sim 场景那道「二级台阶」的训练版：中心平台向外两级**不等高**台阶，外圈比第二级低一档。

    源几何（`assets/robots/serialleg/mjcf/serialleg_fudan_sim2sim_steps.xml`，机器人沿 +x 走）：
    地面 → 上 0.20 m（踏面 0.60 m）→ 上 0.15 m（踏面 **0.15 m**）→ 下 0.05 m 进入长平台。
    第二级不是普通台阶而是一道 0.15 m 宽的凸棱，轮距 0.433 m 决定了两只轮不可能同时站在上面，
    所以它考的是「跨棱」而不是「爬楼梯」，与 stairs_up 的等高宽踏面金字塔是两类问题。

    环形铺法与 `stairs_up`（反金字塔）一致：机器人生在中心，向任意方向走都遇到同一组台阶；
    外圈顶面对齐 z=0 与相邻地形块平滑衔接。难度 0 给矮台阶 + 宽踏面（近似两级小坎），难度 1 精确等于源几何。

    `descending=False`（上行，默认）：中心位于负高度，向外依次 +h1（踏面 `step_width`）、+h2（踏面 `second_*`）、−`outer_drop`。
    `descending=True`（下行）：把同一道剖面反过来走——中心是源场景那块 0.30 m 平台，向外依次
    +`outer_drop`（那道 0.15 m 窄棱的顶）、−h2、−h1 落到 z=0。对应源场景里机器人从平台往台阶方向走的顺序：
    先上 5 cm 小坎、再下 15 cm、再下 20 cm。
    """

    step_height_range: tuple[float, float] = (0.05, 0.20)
    """第一级台阶高度(m)，随难度线性插值。字段名沿用 mjlab 约定：指令侧的地形感知高度下限
    （`mdp/commands.py` 的 `terrain_step_height_type_names`）直接 getattr 读它，取两级里更高的那一级才安全。"""
    second_step_height_range: tuple[float, float] = (0.04, 0.15)
    """第二级台阶高度(m)，随难度线性插值。"""
    step_width: float = 0.60
    """第一级踏面深度(m)，取源几何的 0.60，不随难度变。"""
    second_step_width_range: tuple[float, float] = (0.60, 0.15)
    """第二级踏面深度(m)：难度 0 取前者、难度 1 取后者（收窄到源几何的 0.15）。这是本地形的核心课程量。"""
    outer_drop: float = 0.05
    """第二级顶面到外围平台的下降(m)，取源几何的 0.35 → 0.30。"""
    platform_width: float = 2.0
    """中心平台边长(m)，机器人生在这里。"""
    border_width: float = 0.5
    """外边框宽度(m)，顶面与 z=0 齐平。"""
    floor_thickness: float = 0.5
    """所有台阶 box 向下延伸的厚度(m)，只为保证实心无缝，不参与课程。"""
    descending: bool = False
    """True 时中心高、向外下行（机器人从源场景那块平台往台阶方向走）。"""

    def dims(self, difficulty):
        """返回该难度下的 (第一级高, 第二级高, 第二级踏面深)；difficulty 可以是 float 或张量。"""
        lo1, hi1 = self.step_height_range
        lo2, hi2 = self.second_step_height_range
        w_lo, w_hi = self.second_step_width_range
        first = lo1 + difficulty * (hi1 - lo1)
        second = lo2 + difficulty * (hi2 - lo2)
        width = w_lo + difficulty * (w_hi - w_lo)
        return first, second, width

    def se3_stair_geometry(self, difficulty):
        """台阶奖励用的等效几何：(阶高, 第一道立面半径, 爬升段长度, 可奖励级数)。

        阶高取**第二级**：`stair_support_height` 用 floor(rise / 阶高) 数级数，而 rise 在第一级是 h1、
        在第二级是 h1+h2；取 h2 能让两级分别数成 1 和 2（难度 1 时 0.20/0.15→1、0.35/0.15→2）。
        取 h1 会把第二级也数成 1，第二级就白爬了。
        """
        first, second, width = self.dims(difficulty)
        start = 0.5 * self.platform_width + 0.0 * first
        length = self.step_width + width
        count = 2.0 + 0.0 * first
        return second, start, length, count

    def profile(self, difficulty: float) -> list[tuple[float, float]]:
        """返回从中心向外的 [(环外半径, 顶面高度)] 剖面，供几何生成与测试共用。"""
        first, second, width = self.dims(difficulty)
        r0 = 0.5 * self.platform_width
        inner_x = self.size[0] - 2.0 * self.border_width
        inner_y = self.size[1] - 2.0 * self.border_width
        r_outer = 0.5 * min(inner_x, inner_y)
        if not self.descending:
            # 外圈锚在 0，向内逐级下挖。
            z_outer = 0.0
            z_second = z_outer + self.outer_drop
            z_first = z_second - second
            z_center = z_first - first
            return [
                (r0, z_center),
                (r0 + self.step_width, z_first),
                (r0 + self.step_width + width, z_second),
                (r_outer, z_outer),
            ]
        # 下行：外圈仍锚在 0，中心是源场景那块平台。窄棱在内、宽踏面在外。
        z_outer = 0.0
        z_wide = z_outer + first
        z_ridge = z_wide + second
        z_center = z_ridge - self.outer_drop
        return [
            (r0, z_center),
            (r0 + width, z_ridge),
            (r0 + width + self.step_width, z_wide),
            (r_outer, z_outer),
        ]

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # 几何完全由难度决定，不随机。
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []
        rings = self.profile(float(difficulty))
        (r_platform, z_center), (r_first, z_first), (r_second, z_second), (r_outer, z_outer) = rings
        z_floor = min(z for _, z in rings) - self.floor_thickness

        center = (0.5 * self.size[0], 0.5 * self.size[1])
        inner_x = self.size[0] - 2.0 * self.border_width
        inner_y = self.size[1] - 2.0 * self.border_width

        def ring(outer_r: float, inner_r: float, top_z: float, rgba):
            height = max(top_z - z_floor, 1e-3)
            boxes = make_border(
                body,
                (2.0 * outer_r, 2.0 * outer_r),
                (2.0 * inner_r, 2.0 * inner_r),
                height,
                (center[0], center[1], top_z - 0.5 * height),
            )
            for box in boxes:
                geometries.append(TerrainGeometry(geom=box, color=rgba))

        # 中心平台（机器人出生处）。
        platform_h = max(z_center - z_floor, 1e-3)
        platform = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(r_platform, r_platform, 0.5 * platform_h),
            pos=(center[0], center[1], z_center - 0.5 * platform_h),
        )
        geometries.append(TerrainGeometry(geom=platform, color=_TWO_STEP_PLATFORM_RGBA))

        first_rgba = _TWO_STEP_SECOND_RGBA if self.descending else _TWO_STEP_FIRST_RGBA
        second_rgba = _TWO_STEP_FIRST_RGBA if self.descending else _TWO_STEP_SECOND_RGBA
        ring(r_first, r_platform, z_first, first_rgba)
        ring(r_second, r_first, z_second, second_rgba)
        if r_outer > r_second:
            ring(r_outer, r_second, z_outer, _TWO_STEP_OUTER_RGBA)

        if self.border_width > 0.0:
            border_height = max(z_outer - z_floor, 1e-3)
            boxes = make_border(
                body,
                (self.size[0], self.size[1]),
                (inner_x, inner_y),
                border_height,
                (center[0], center[1], z_outer - 0.5 * border_height),
            )
            for box in boxes:
                geometries.append(TerrainGeometry(geom=box, color=_TWO_STEP_BORDER_RGBA))

        origin = np.array([center[0], center[1], z_center])
        return TerrainOutput(origin=origin, geometries=geometries)


def _slope(preset, *, proportion: float, slope_range: tuple[float, float]):
    return preset(
        proportion=proportion,
        size=ROUGH_PATCH_SIZE,
        slope_range=slope_range,
        platform_width=ROUGH_PLATFORM_WIDTH,
        border_width=_STAIR_BORDER_WIDTH,
        horizontal_scale=ROUGH_HFIELD_HORIZONTAL_SCALE,
    )


def _stairs(
    preset,
    *,
    proportion: float,
    step_height_range: tuple[float, float] = ROUGH_STEP_HEIGHT_RANGE,
):
    return preset(
        proportion=proportion,
        size=ROUGH_PATCH_SIZE,
        step_height_range=step_height_range,
        step_width=ROUGH_STEP_WIDTH,
        platform_width=ROUGH_PLATFORM_WIDTH,
        border_width=_STAIR_BORDER_WIDTH,
    )


def _two_step(*, proportion: float, descending: bool = False) -> TwoStepStairsTerrainCfg:
    return TwoStepStairsTerrainCfg(
        proportion=proportion,
        descending=descending,
        size=ROUGH_PATCH_SIZE,
        step_height_range=ROUGH_TWO_STEP_FIRST_HEIGHT_RANGE,
        second_step_height_range=ROUGH_TWO_STEP_SECOND_HEIGHT_RANGE,
        step_width=ROUGH_TWO_STEP_FIRST_WIDTH,
        second_step_width_range=ROUGH_TWO_STEP_SECOND_WIDTH_RANGE,
        outer_drop=ROUGH_TWO_STEP_OUTER_DROP,
        platform_width=ROUGH_PLATFORM_WIDTH,
        border_width=_STAIR_BORDER_WIDTH,
    )


def rough_terrains_cfg(
    *,
    num_rows: int = 10,
    stairs_up_step_height_range: tuple[float, float] = ROUGH_STEP_HEIGHT_RANGE,
) -> TerrainGeneratorCfg:
    """训练地形集：七列地形，env 按 ROUGH_TERRAIN_PROPORTIONS 分配。

    stairs_up_step_height_range：只改上台阶列的阶高范围（第 0 级取下限、第 9 级取上限），下台阶列不变。
    M32（2026-09-26 用户定）把下限从 2 cm 提到 5 cm，对齐复旦台阶最低一级（5 cm，轮半径 6 cm）。
    """
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
            "stairs_up": _stairs(
                pyramid_stairs_inv,
                proportion=p["stairs_up"],
                step_height_range=stairs_up_step_height_range,
            ),
            ROUGH_TWO_STEP_UP_COLUMN: _two_step(proportion=p[ROUGH_TWO_STEP_UP_COLUMN]),
            ROUGH_TWO_STEP_DOWN_COLUMN: _two_step(
                proportion=p[ROUGH_TWO_STEP_DOWN_COLUMN], descending=True
            ),
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
    "ROUGH_HFIELD_HORIZONTAL_SCALE",
    "ROUGH_PATCH_SIZE",
    "ROUGH_PLATFORM_WIDTH",
    "ROUGH_SLOPE_DOWN_RANGE",
    "ROUGH_SLOPE_UP_RANGE",
    "ROUGH_STAIR_LIKE_COLUMNS",
    "ROUGH_STEP_HEIGHT_RANGE",
    "ROUGH_STEP_WIDTH",
    "ROUGH_TERRAIN_PROPORTIONS",
    "ROUGH_TWO_STEP_DOWN_COLUMN",
    "ROUGH_TWO_STEP_FIRST_HEIGHT_RANGE",
    "ROUGH_TWO_STEP_FIRST_WIDTH",
    "ROUGH_TWO_STEP_GATE_LEVEL",
    "ROUGH_TWO_STEP_OUTER_DROP",
    "ROUGH_TWO_STEP_SECOND_HEIGHT_RANGE",
    "ROUGH_TWO_STEP_SECOND_WIDTH_RANGE",
    "ROUGH_TWO_STEP_UP_COLUMN",
    "TwoStepStairsTerrainCfg",
    "rough_terrains_cfg",
    "stair_only_terrains_cfg",
]
