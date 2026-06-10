"""恢复奖励 + 台阶 teacher 的四地形综合任务配置。"""

from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_train.mdp import curriculums, events
from se3_train.tasks.stair_ctbc.env_cfg import (
    env_cfg as stair_ctbc_env_cfg,
)
from se3_train.tasks.stair_ctbc.env_cfg import (
    play_terrain_difficulty_from_training,
)
from se3_train.tasks.stair_ctbc.terrains import BoxRampTerrainCfg, BoxStageStairsTerrainCfg

_STAIR_TERRAIN_TYPE_NAMES = ("stage_stairs", "inv_pyramid_stairs")
_PLAY_NUM_ENVS = 4
_RECOVERY_COMMAND_HEIGHT = 0.30


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；空值按默认值处理。"""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成四地形 universal stair teacher 环境。"""
    cfg = stair_ctbc_env_cfg(play=play)
    cfg.scene.terrain = _four_terrain_cfg(play)
    _configure_commands_and_curriculum(cfg, play)
    _configure_reference_reset_mix(cfg, play)
    _configure_ctbc_anneal(cfg)
    _configure_rewards(cfg)
    cfg.episode_length_s = 8.0 if not play else 9999.0
    return cfg


def _four_terrain_cfg(play: bool) -> TerrainEntityCfg:
    """只保留二级台阶、金字塔台阶、17 度坡、43 度坡四类地形。"""
    play_difficulty = play_terrain_difficulty_from_training() if play else None
    sub_terrains = {
        "stage_stairs": BoxStageStairsTerrainCfg(proportion=0.25, size=(8.0, 8.0)),
        "inv_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            size=(8.0, 8.0),
            step_height_range=(0.05, 0.20),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
        ),
        "ramp_43deg_400mm": BoxRampTerrainCfg(
            proportion=0.25,
            size=(8.0, 8.0),
            slope_deg=43.0,
            height=0.40,
        ),
        "ramp_17deg_350mm": BoxRampTerrainCfg(
            proportion=0.25,
            size=(8.0, 8.0),
            slope_deg=17.0,
            height=0.35,
            top_platform_length=0.0,
        ),
    }
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=True,
            size=(8.0, 8.0),
            border_width=20.0,
            border_height=1.0,
            num_rows=1 if play else 10,
            num_cols=_PLAY_NUM_ENVS if play else 20,
            difficulty_range=(play_difficulty, play_difficulty) if play else (0.0, 1.0),
            add_lights=True,
            sub_terrains=sub_terrains,
        ),
        max_init_terrain_level=0,
    )


def _configure_commands_and_curriculum(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
    """配置正冲台阶速度课程；始终保持 0.8m/s 以上且不随机 yaw。"""
    command_cfg = cfg.commands["velocity_height"]
    command_cfg.height_range = (0.38, 0.39)
    command_cfg.standing_height_range = (0.38, 0.39)
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = 0.0
    command_cfg.jump_prob = 0.0
    command_cfg.enable_jump_lifecycle = False
    command_cfg.enable_jump_metrics = False
    if play:
        command_cfg.lin_vel_x_range = (0.8, 0.8)
        command_cfg.ang_vel_yaw_range = (0.0, 0.0)
        return

    command_cfg.lin_vel_x_range = (0.80, 1.20)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    cfg.curriculum = {
        "commands_vel": CurriculumTermCfg(
            func=curriculums.commands_vel,
            params={
                "command_name": "velocity_height",
                "use_iterations": True,
                "steps_per_policy_iter": 64,
                "velocity_stages": [
                    {
                        "iteration": 0,
                        "lin_vel_x_range": (0.80, 1.20),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 300,
                        "lin_vel_x_range": (0.80, 1.80),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 600,
                        "lin_vel_x_range": (0.80, 2.50),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                ],
            },
        )
    }


def _configure_reference_reset_mix(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
    """配置参考代码风格的 75% 正常 + 25% 倒地混合 reset。"""
    if play:
        return

    root_reset = cfg.events["reset_root_state"]
    root_reset.func = events.reset_root_state_full
    root_reset.params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "recovery_prob": 0.25,
        "recovery_roll_range": (-0.35, 0.35),
        "recovery_pitch_range": (-0.35, 0.35),
        "recovery_height_range": (0.24, 0.34),
        "recovery_fallen_pose_prob": 1.0,
        "recovery_fallen_roll_pose_prob": 0.5,
        "recovery_fallen_roll_abs_range": (1.35, 1.75),
        "recovery_fallen_pitch_abs_range": (2.65, math.pi),
        "recovery_fallen_coupled_range": (-0.35, 0.35),
        "recovery_fallen_height_range": (0.12, 0.24),
        "recovery_lin_vel_range": (-0.15, 0.15),
        "recovery_ang_vel_range": (-0.8, 0.8),
        "recovery_command_height": _RECOVERY_COMMAND_HEIGHT,
    }

    joint_reset = cfg.events["reset_joints"]
    joint_reset.params.update(
        {
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "full_joint_randomization": False,
            "recovery_joint_offset_range": 0.20 if not play else 0.0,
            "recovery_joint_vel_range": (-0.8, 0.8) if not play else (0.0, 0.0),
            "height_conditioned_default": True,
            "command_name": "velocity_height",
        }
    )


def _configure_ctbc_anneal(cfg: ManagerBasedRlEnvCfg) -> None:
    """按参考 universal 任务配置 CTBC 退火，并只允许台阶地形触发。"""
    init_event = cfg.events["init_stair_climb_state"]
    init_event.params.update(
        {
            "ann_start_iter": _env_int("SE3_UNIVERSAL_STAIR_CTBC_ANN_START", 800),
            "ann_end_iter": _env_int("SE3_UNIVERSAL_STAIR_CTBC_ANN_END", 1800),
            "phantom_trigger_iter": _env_int("SE3_UNIVERSAL_STAIR_CTBC_PHANTOM_ITER", 1000),
        }
    )
    step_event = cfg.events["step_stair_climb_state"]
    step_event.params.update(
        {
            "stair_terrain_type_names": _STAIR_TERRAIN_TYPE_NAMES,
            "disable_during_recovery": True,
        }
    )


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """加强台阶爬升和高度语义，避免把高度误差当奖励。"""
    if "tracking_height" in cfg.rewards:
        cfg.rewards["tracking_height"].weight = 5.0
        cfg.rewards["tracking_height"].params.update(
            {
                "sigma": 0.04,
                "kernel": "exp",
                "use_upright_gate": False,
                "use_pose_end_gate": False,
            }
        )
    if "stair_climb_height" in cfg.rewards:
        cfg.rewards["stair_climb_height"].weight = 8.0
        cfg.rewards["stair_climb_height"].params.update(
            {
                "forward_gate_start": 0.10,
                "forward_gate_width": 0.25,
                "terrain_type_names": _STAIR_TERRAIN_TYPE_NAMES,
            }
        )
    if "stair_forward_distance" in cfg.rewards:
        cfg.rewards["stair_forward_distance"].weight = 2.0
        cfg.rewards["stair_forward_distance"].params.update(
            {"terrain_type_names": _STAIR_TERRAIN_TYPE_NAMES}
        )
    if "stair_feet_clearance" in cfg.rewards:
        cfg.rewards["stair_feet_clearance"].weight = 1.0
    if "stair_feet_air_time" in cfg.rewards:
        cfg.rewards["stair_feet_air_time"].weight = 1.0
    if "stair_contact_number" in cfg.rewards:
        cfg.rewards["stair_contact_number"].weight = 1.0
    if "stair_wheel_swing_zero_vel" in cfg.rewards:
        cfg.rewards["stair_wheel_swing_zero_vel"].weight = 0.25
