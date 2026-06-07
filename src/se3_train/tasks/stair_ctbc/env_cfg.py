"""CTBC 台阶训练环境配置。"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxRandomStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_train.mdp import events, stair_rewards, terminations
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.stair_ctbc.terrains import BoxRampTerrainCfg, BoxStageStairsTerrainCfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成台阶 CTBC 环境，保持 recovery/flat 的 action 与 observation 语义。"""
    cfg = flat_env_cfg(play=play)

    stair_sub_terrains = (
        {
            "stage_stairs": BoxStageStairsTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
            ),
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
        if play
        else {
            "stage_stairs": BoxStageStairsTerrainCfg(
                proportion=0.175,
                size=(8.0, 8.0),
            ),
            "inv_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.175,
                size=(8.0, 8.0),
                step_height_range=(0.05, 0.20),
                step_width=0.5,
                platform_width=2.0,
                border_width=1.0,
            ),
            "ramp_43deg_400mm": BoxRampTerrainCfg(
                proportion=0.175,
                size=(8.0, 8.0),
                slope_deg=43.0,
                height=0.40,
            ),
            "ramp_17deg_350mm": BoxRampTerrainCfg(
                proportion=0.175,
                size=(8.0, 8.0),
                slope_deg=17.0,
                height=0.35,
                top_platform_length=0.0,
            ),
            "random_stairs": BoxRandomStairsTerrainCfg(
                proportion=0.15,
                size=(8.0, 8.0),
                step_height_range=(0.05, 0.20),
                step_width=0.6,
                platform_width=2.0,
                border_width=0.5,
            ),
            "flat": BoxFlatTerrainCfg(proportion=0.15, size=(8.0, 8.0)),
        }
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=True,
            size=(8.0, 8.0),
            border_width=20.0,
            border_height=1.0,
            num_rows=1 if play else 10,
            num_cols=1 if play else 20,
            difficulty_range=(1.0, 1.0) if play else (0.0, 1.0),
            add_lights=True,
            sub_terrains=stair_sub_terrains,
        ),
        max_init_terrain_level=0,
    )
    cfg.scene.env_spacing = 4.0

    wheel_riser_sensor = ContactSensorCfg(
        name="wheel_riser_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link|r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "normal", "tangent"),
        reduce="maxforce",
        num_slots=4,
        global_frame=True,
    )
    left_wheel_center_height_sensor = TerrainHeightSensorCfg(
        name="left_wheel_center_height_sensor",
        frame=ObjRef(type="body", name="l_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg(rings=(), include_center=True),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )
    right_wheel_center_height_sensor = TerrainHeightSensorCfg(
        name="right_wheel_center_height_sensor",
        frame=ObjRef(type="body", name="r_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg(rings=(), include_center=True),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )
    cfg.scene.sensors = (
        *tuple(cfg.scene.sensors or ()),
        left_wheel_center_height_sensor,
        right_wheel_center_height_sensor,
        wheel_riser_sensor,
    )

    cfg.actions["delayed_action"].height_conditioned_action_default = True
    cfg.actions["delayed_action"].action_default_command_name = "velocity_height"

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.2, 0.8)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    command_cfg.height_range = (0.31, 0.34)
    command_cfg.standing_height_range = (0.31, 0.34)
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = 0.0
    command_cfg.jump_prob = 0.0
    command_cfg.enable_jump_lifecycle = False
    command_cfg.enable_jump_metrics = False
    if play:
        command_cfg.lin_vel_x_range = (0.25, 0.25)
        command_cfg.height_range = (0.34, 0.34)
        command_cfg.standing_height_range = (0.34, 0.34)

    if "bad_orientation" in cfg.terminations:
        cfg.terminations["bad_orientation"] = replace(
            cfg.terminations["bad_orientation"],
            params={"limit_angle": 0.698, "max_steps": 100},
        )
    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact,
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 80.0,
            "command_name": "velocity_height",
            "terminate": False,
        },
    )

    new_events = dict(cfg.events)
    new_events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_robotlab_full_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_xy_range": (0.0, 0.0) if play else (-0.03, 0.03),
            "pos_xy_offset": (0.0, 0.0),
            "height_offset_range": (0.0, 0.0) if play else (0.0, 0.2),
            "roll_range": (0.0, 0.0),
            "pitch_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0) if play else (-math.pi, math.pi),
            "lin_vel_range": (0.0, 0.0),
            "ang_vel_range": (0.0, 0.0),
            "clearance_range": (0.0, 0.02) if play else (0.0, 0.05),
        },
    )
    new_events["reset_joints"] = EventTermCfg(
        func=events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "full_joint_randomization": False,
            "full_front_joint_offset_range": math.pi,
            "full_active_rod_angle_range": (0.0, 0.0),
            "align_root_height_to_wheels": True,
            "height_conditioned_default": True,
            "command_name": "velocity_height",
            "terrain_height_sensor_names": (
                "left_wheel_center_height_sensor",
                "right_wheel_center_height_sensor",
            ),
            "allow_wheel_clearance_lowering": True,
            "max_wheel_clearance_adjustment": 0.25,
        },
    )
    new_events["init_stair_climb_state"] = EventTermCfg(
        func=events.init_stair_climb_state,
        mode="startup",
        params={
            "contact_window": 3,
            "force_threshold_n": 30.0,
            "ff_period_s": 0.6,
            "cooldown_s": 0.3,
            "ann_start_iter": 0,
            "ann_end_iter": 1500,
            "phantom_trigger_iter": 0,
        },
    )
    new_events["step_stair_climb_state"] = EventTermCfg(
        func=events.step_stair_climb_state,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "sensor_name": "wheel_sensor",
            "riser_sensor_name": "wheel_riser_sensor",
            "riser_normal_z_max": 0.5,
            "num_steps_per_env": 64,
        },
    )
    new_events["reset_stair_climb_state"] = EventTermCfg(
        func=events.reset_stair_climb_state,
        mode="reset",
    )
    new_events.pop("push_robots", None)
    cfg.events = new_events

    cfg.curriculum = {}
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=stair_rewards.stair_forward_progress,
        weight=3.0,
        params={"command_name": "velocity_height", "sigma": 0.25},
    )
    if "flat_leg_contact" in cfg.rewards:
        cfg.rewards["flat_leg_contact"].weight = -2.0
    if "collision" in cfg.rewards:
        cfg.rewards["collision"].weight = -2.0
    if "contact_forces" in cfg.rewards:
        cfg.rewards["contact_forces"].weight = -2.0e-4
    if "tracking_height" in cfg.rewards:
        cfg.rewards["tracking_height"].weight = 3.0
        cfg.rewards["tracking_height"].params["sigma"] = 0.04

    cfg.rewards["stair_climb_height"] = RewardTermCfg(
        func=stair_rewards.stair_climb_height,
        weight=3.0,
        params={"command_name": "velocity_height", "max_gain": 0.35},
    )
    cfg.rewards["stair_contact_diagnostics"] = RewardTermCfg(
        func=stair_rewards.stair_contact_diagnostics,
        weight=0.001,
        params={
            "wheel_sensor_name": "wheel_sensor",
            "leg_sensor_name": "leg_contact_sensor",
            "collision_sensor_name": "collision_sensor",
            "force_threshold": 1.0,
        },
    )

    cfg.episode_length_s = 8.0 if not play else 9999.0
    return cfg
