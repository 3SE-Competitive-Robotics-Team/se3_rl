"""正金字塔台阶任务环境配置。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import curriculums as shared_curriculums
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import rewards

_ROBOT_DEFAULTS = SharedRobotConfig()
_RESOURCES = Path(__file__).resolve().parents[4] / "assets"
_STAIR_MJCF_PATH = (
    _RESOURCES
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_fourbar_surrogate_stair_visualbase_coacd_train.xml"
)
_STAIR_WHEEL_KD = 0.08
_STAIR_COMMAND_WHEEL_RADIUS = 0.060
_STAIR_COMMAND_HALF_TRACK = 0.200725
_STAIR_COMMAND_WHEEL_SPEED_FRACTION = 0.70
_STAIR_TERRAIN_TYPES = ("pyramid_stairs",)


@dataclass(kw_only=True)
class BottomSpawnPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
    """正金字塔台阶，但 reset origin 固定在台底左侧边框。"""

    spawn_x_fraction_in_border: float = 0.5
    """出生点在左侧底部边框内的相对位置，0.5 表示边框中线。"""

    def function(self, difficulty, spec, rng):
        """生成正金字塔台阶，并把出生点从台顶平台改到底部边框。"""
        output = super().function(difficulty, spec, rng)
        spawn_x = (
            self.border_width * self.spawn_x_fraction_in_border
            if self.border_width > 0.0
            else 0.5 * self.step_width
        )
        output.origin = np.array([spawn_x, 0.5 * self.size[1], 0.0])
        return output


def _stair_terrain_cfg() -> TerrainGeneratorCfg:
    """构造正金字塔台阶地形；机器人出生在台底，沿 +x 方向向中心上台阶。"""
    return TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=1.0,
        num_rows=10,
        num_cols=20,
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains={
            "pyramid_stairs": BottomSpawnPyramidStairsTerrainCfg(
                proportion=0.8,
                size=(8.0, 8.0),
                step_height_range=(0.05, 0.20),
                step_width=0.5,
                platform_width=2.0,
                border_width=1.0,
            ),
            "flat": BoxFlatTerrainCfg(proportion=0.2, size=(8.0, 8.0)),
        },
    )


def _sanitize_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """台阶 box 接触偶发 NaN 时，将观测清洗为有限值。"""
    for group_name in ("actor", "critic"):
        group_cfg = cfg.observations.get(group_name)
        if group_cfg is not None:
            cfg.observations[group_name] = replace(group_cfg, nan_policy="sanitize")


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造无 CTBC 的台阶爬升基础环境。"""
    cfg = flat_env_cfg(play=play)

    cfg.scene.entities["robot"] = get_serialleg_cfg(
        mjcf_path=_STAIR_MJCF_PATH,
        wheel_kd_override=_STAIR_WHEEL_KD,
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=_stair_terrain_cfg(),
        max_init_terrain_level=0,
    )
    cfg.scene.env_spacing = 4.0

    cfg.actions["delayed_action"].height_conditioned_action_default = True
    cfg.actions["delayed_action"].action_default_command_name = "velocity_height"

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.resampling_time_range = (5.0, 5.0)
    command_cfg.lin_vel_x_range = (0.2, 0.6)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    command_cfg.height_range = (0.24, 0.37)
    command_cfg.standing_height_range = (0.24, 0.37)
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = 0.0
    command_cfg.lin_vel_deadband = 0.05
    command_cfg.yaw_deadband = 0.05
    command_cfg.constrain_diff_drive_commands = True
    command_cfg.diff_drive_wheel_radius = _STAIR_COMMAND_WHEEL_RADIUS
    command_cfg.diff_drive_half_track = _STAIR_COMMAND_HALF_TRACK
    command_cfg.diff_drive_max_wheel_speed = _ROBOT_DEFAULTS.action_scale[
        JointGroup.WHEEL_ACTUATORS[0]
    ]
    command_cfg.diff_drive_wheel_speed_fraction = _STAIR_COMMAND_WHEEL_SPEED_FRACTION
    command_cfg.jump_prob = 0.0
    command_cfg.enable_jump_lifecycle = False
    command_cfg.enable_jump_metrics = False

    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.stair_forward_progress,
        weight=2.73,
        params={"command_name": "velocity_height", "sigma": 0.25},
    )
    cfg.rewards.pop("tracking_lin_yaw_joint", None)
    cfg.rewards.pop("flat_wheel_contact", None)
    cfg.rewards.pop("flat_leg_contact", None)
    cfg.rewards["tracking_orientation_l2"].weight = -6.0
    cfg.rewards["bad_tilt"].params = {
        "soft_limit_deg": 20.0,
        "hard_limit_deg": 40.0,
        "max_penalty": 4.0,
    }
    cfg.rewards["contact_forces"].weight = -5.0e-4
    cfg.rewards["contact_forces"].params["threshold"] = 45.0
    cfg.rewards["obs_steps_climbed"] = RewardTermCfg(
        func=rewards.stair_steps_climbed,
        weight=0.001,
        params={
            "command_name": "velocity_height",
            "step_height_range": (0.05, 0.20),
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["obs_height_gain"] = RewardTermCfg(
        func=rewards.stair_height_gain,
        weight=0.001,
        params={
            "command_name": "velocity_height",
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["obs_terrain_level"] = RewardTermCfg(
        func=rewards.stair_terrain_level,
        weight=0.001,
    )
    cfg.rewards["stair_diagnostics"] = RewardTermCfg(
        func=rewards.stair_diagnostics,
        weight=1.0,
        params={
            "command_name": "velocity_height",
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )

    if "bad_orientation" in cfg.terminations:
        cfg.terminations["bad_orientation"].params = {"limit_angle": 0.698, "max_steps": 100}
    if "leg_contact" in cfg.terminations:
        cfg.terminations["leg_contact"].params.update(
            {
                "force_threshold": 80.0,
                "terminate": False,
            }
        )

    if not play:
        cfg.events.pop("push_robots", None)
        cfg.curriculum.pop("push_disturbance", None)
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=shared_curriculums.commands_vel,
            params={
                "command_name": "velocity_height",
                "use_iterations": True,
                "steps_per_policy_iter": 64,
                "velocity_stages": [
                    {
                        "iteration": 0,
                        "lin_vel_x_range": (0.2, 0.6),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 250,
                        "lin_vel_x_range": (0.3, 1.0),
                        "ang_vel_yaw_range": (-0.25, 0.25),
                    },
                    {
                        "iteration": 500,
                        "lin_vel_x_range": (0.5, 1.6),
                        "ang_vel_yaw_range": (-0.50, 0.50),
                    },
                ],
            },
        )
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=shared_curriculums.terrain_levels,
            params={
                "use_iterations": True,
                "steps_per_policy_iter": 64,
                "terrain_type_names": _STAIR_TERRAIN_TYPES,
                "level_stages": [
                    {"iteration": 0, "max_level": 0},
                    {"iteration": 250, "max_level": 2},
                    {"iteration": 500, "max_level": 5},
                    {"iteration": 800, "max_level": 9},
                ],
            },
        )

    _sanitize_observations(cfg)
    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1040,
        mujoco=MujocoCfg(
            timestep=_ROBOT_DEFAULTS.sim_dt,
            iterations=12,
            ls_iterations=8,
            ccd_iterations=15,
            tolerance=1e-6,
        ),
    )
    cfg.clip_observations = 100.0
    return cfg
