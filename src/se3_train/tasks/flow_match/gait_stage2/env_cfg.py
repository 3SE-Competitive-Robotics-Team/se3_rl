"""GAIT Stage2 低矮随机地形环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxOpenStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    BoxRandomSpreadTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_train.tasks.flow_match.gait_stage1.env_cfg import GAIT_MAX_SPEED, base_gait_env_cfg

from . import curriculums, events

_SPEED_END_STEP = 64_000
_TERRAIN_START_STEP = 0
_TERRAIN_END_STEP = 128_000
_PUSH_STAGE_STEPS = (128_000, 160_000)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 Stage2 低矮地形 GAIT 训练环境。"""
    cfg = base_gait_env_cfg(play=play)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=stage2_terrain_cfg(),
        max_init_terrain_level=0,
    )
    cfg.commands["velocity_height"].lin_vel_x_range = (
        (0.05, GAIT_MAX_SPEED) if play else (0.05, 0.12)
    )

    if not play:
        cfg.events["push_robots"] = EventTermCfg(
            func=events.push_robots,
            mode="interval",
            interval_range_s=(4.0, 5.0),
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        cfg.curriculum["command_vel_linear"] = CurriculumTermCfg(
            func=curriculums.commands_vel_linear,
            params={
                "command_name": "velocity_height",
                "start_step": 0,
                "end_step": _SPEED_END_STEP,
                "start_lin_vel_x_range": (0.05, 0.12),
                "end_lin_vel_x_range": (0.05, GAIT_MAX_SPEED),
                "ang_vel_yaw_range": (0.0, 0.0),
            },
        )
        cfg.curriculum["terrain_distribution"] = CurriculumTermCfg(
            func=curriculums.terrain_distribution_linear,
            params={
                "start_step": _TERRAIN_START_STEP,
                "end_step": _TERRAIN_END_STEP,
                "start_proportions": (0.85, 0.10, 0.03, 0.02, 0.00),
                "end_proportions": (0.45, 0.25, 0.15, 0.10, 0.05),
                "start_max_level": 0,
                "end_max_level": 3,
            },
        )
        cfg.curriculum["push_disturbance"] = CurriculumTermCfg(
            func=curriculums.push_disturbance,
            params={
                "push_stages": [
                    {
                        "step": 0,
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                    },
                    {
                        "step": _PUSH_STAGE_STEPS[0],
                        "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                    },
                    {
                        "step": _PUSH_STAGE_STEPS[1],
                        "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)},
                    },
                ],
            },
        )
    return cfg


def stage2_terrain_cfg() -> TerrainGeneratorCfg:
    """Stage2 低矮随机地形和最高 8cm 台阶。"""
    return TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=1.0,
        num_rows=4,
        num_cols=5,
        color_scheme="height",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.45),
            "random_grid": BoxRandomGridTerrainCfg(
                proportion=0.25,
                grid_width=0.55,
                grid_height_range=(0.010, 0.080),
                platform_width=1.4,
                merge_similar_heights=True,
                height_merge_threshold=0.006,
                max_merge_distance=3,
                border_width=0.25,
            ),
            "random_spread_boxes": BoxRandomSpreadTerrainCfg(
                proportion=0.15,
                num_boxes=14,
                box_width_range=(0.08, 0.22),
                box_length_range=(0.08, 0.35),
                box_height_range=(0.010, 0.080),
                platform_width=1.2,
                border_width=0.25,
            ),
            "open_stairs_up": BoxOpenStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.030, 0.080),
                step_width_range=(0.65, 1.00),
                platform_width=1.4,
                border_width=0.25,
                step_thickness=0.05,
                inverted=True,
            ),
            "open_stairs_down": BoxOpenStairsTerrainCfg(
                proportion=0.05,
                step_height_range=(0.030, 0.080),
                step_width_range=(0.65, 1.00),
                platform_width=1.4,
                border_width=0.25,
                step_thickness=0.05,
                inverted=False,
            ),
        },
        difficulty_range=(0.0, 1.0),
        add_lights=True,
    )


__all__ = ["env_cfg", "stage2_terrain_cfg"]
