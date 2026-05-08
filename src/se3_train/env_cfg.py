from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.rewards import is_terminated as mjlab_is_terminated
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from se3_train.mdp import events, observations, rewards, terminations
from se3_train.mdp.actions import SerialLegDelayedActionCfg
from se3_train.mdp.commands import VelocityHeightCommandCfg
from se3_train.mdp.curriculums import commands_vel as curriculum_commands_vel
from se3_train.mdp.curriculums import push_disturbance as curriculum_push_disturbance
from se3_train.robot_cfg import get_serialleg_cfg


def se3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """SerialLeg 轮腿机器人的平地环境配置。"""

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_serialleg_cfg()},
        num_envs=1024,
        env_spacing=3.0,
    )

    cfg = ManagerBasedRlEnvCfg(
        decimation=5,
        scene=scene,
    )

    collision_sensor_cfg = ContactSensorCfg(
        name="collision_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(base_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    leg_contact_sensor_cfg = ContactSensorCfg(
        name="leg_contact_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(lf0_Link|lf1_Link|rf0_Link|rf1_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    wheel_sensor_cfg = ContactSensorCfg(
        name="wheel_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link|r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    base_height_sensor_cfg = TerrainHeightSensorCfg(
        name="base_height_sensor",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=4),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )

    critic_height_sensor_cfg = TerrainHeightSensorCfg(
        name="critic_height_sensor",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.15, num_samples=8),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="mean",
    )

    cfg.scene.sensors = (
        collision_sensor_cfg,
        leg_contact_sensor_cfg,
        wheel_sensor_cfg,
        base_height_sensor_cfg,
        critic_height_sensor_cfg,
    )

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=observations.base_ang_vel_obs,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=observations.projected_gravity_obs,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "commands": ObservationTermCfg(func=observations.commands_obs),
        "leg_joint_pos": ObservationTermCfg(
            func=observations.leg_joint_pos_obs,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "leg_joint_vel": ObservationTermCfg(
            func=observations.leg_joint_vel_obs,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "wheel_pos": ObservationTermCfg(func=observations.wheel_pos_obs),
        "wheel_vel": ObservationTermCfg(func=observations.wheel_vel_obs),
        "last_actions": ObservationTermCfg(func=observations.last_actions_obs),
    }

    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=observations.base_lin_vel_obs),
        "wheel_contact_forces": ObservationTermCfg(
            func=observations.wheel_contact_force_obs,
            params={"sensor_name": "wheel_sensor"},
        ),
        "base_height": ObservationTermCfg(
            func=observations.base_height_obs,
            params={"sensor_name": "critic_height_sensor"},
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.actions = {
        "delayed_action": SerialLegDelayedActionCfg(entity_name="robot"),
    }

    cfg.commands = {
        "velocity_height": VelocityHeightCommandCfg(
            resampling_time_range=(5.0, 5.0),
        ),
    }

    cfg.rewards = {
        "tracking_lin_vel": RewardTermCfg(
            func=rewards.tracking_lin_vel,
            weight=2.73,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.25,
                "sigma_stand": 0.1,
                "vz_weight": 2.0,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=rewards.tracking_ang_vel,
            weight=1.73,
            params={"command_name": "velocity_height", "sigma": 0.25},
        ),
        "tracking_orientation": RewardTermCfg(
            func=rewards.tracking_orientation,
            weight=2.0,
            params={"command_name": "velocity_height", "sigma": 0.1},
        ),
        "tracking_height": RewardTermCfg(
            func=rewards.tracking_height,
            weight=2.49,
            params={
                "command_name": "velocity_height",
                "sigma": 0.05,
                "height_sensor_name": "base_height_sensor",
            },
        ),
        "upward": RewardTermCfg(func=rewards.upward, weight=0.607),
        "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.146),
        "angular_momentum": RewardTermCfg(
            func=rewards.angular_momentum,
            weight=-5.0e-5,
        ),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-2.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "wheel_torques": RewardTermCfg(
            func=rewards.wheel_torques,
            weight=-1.0e-4,
            params={"max_torque": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-1.0,
            params={
                "command_name": "velocity_height",
                "command_threshold": 0.1,
                "default_height": 0.27,
                "height_tolerance": 40.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-2.17e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-1.03e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.48),
        "joint_mirror": RewardTermCfg(
            func=rewards.joint_mirror,
            weight=-0.179,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=rewards.dof_pos_limits,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "collision": RewardTermCfg(
            func=rewards.collision,
            weight=-2.51,
            params={"sensor_name": "collision_sensor", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "contact_forces": RewardTermCfg(
            func=rewards.contact_forces,
            weight=-1.07e-3,
            params={
                "threshold": 35.0,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "feet_contact_without_cmd": RewardTermCfg(
            func=rewards.feet_contact_without_cmd,
            weight=0.386,
            params={
                "command_name": "velocity_height",
                "force_threshold": 1.0,
                "cmd_threshold": 0.1,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "is_terminated": RewardTermCfg(func=mjlab_is_terminated, weight=-200.0),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=terminations.bad_orientation_delayed,
            time_out=False,
            params={"limit_angle": 0.5236, "max_steps": 100},
        ),
        "leg_contact": TerminationTermCfg(
            func=terminations.leg_contact,
            time_out=False,
            params={"sensor_name": "leg_contact_sensor", "force_threshold": 1.0},
        ),
    }

    if not play:
        cfg.curriculum = {
            "command_vel": CurriculumTermCfg(
                func=curriculum_commands_vel,
                params={
                    "command_name": "velocity_height",
                    "velocity_stages": [
                        {
                            "step": 0,
                            "lin_vel_x_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 500,
                            "lin_vel_x_range": (-1.0, 1.0),
                            "ang_vel_yaw_range": (-0.5, 0.5),
                        },
                        {
                            "step": 1500,
                            "lin_vel_x_range": (-2.0, 2.0),
                            "ang_vel_yaw_range": (-1.5, 1.5),
                        },
                        {
                            "step": 2500,
                            "lin_vel_x_range": (-3.5, 3.5),
                            "ang_vel_yaw_range": (-2.5, 2.5),
                        },
                        {
                            "step": 3500,
                            "lin_vel_x_range": (-5.0, 5.0),
                            "ang_vel_yaw_range": (-3.0, 3.0),
                        },
                    ],
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculum_push_disturbance,
                params={
                    "push_stages": [
                        {
                            "step": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "step": 500,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "step": 1500,
                            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                        },
                        {
                            "step": 2500,
                            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                        },
                        {
                            "step": 3500,
                            "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
                        },
                    ],
                },
            ),
        }

    if play:
        cfg.events = {
            "reset_scene_to_default": EventTermCfg(
                func=lambda env, env_ids: None,
                mode="reset",
            ),
            "reset_root_state": EventTermCfg(
                func=events.reset_root_state_full,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "reset_joints": EventTermCfg(
                func=events.reset_joints,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
        }
        cfg.episode_length_s = 9999.0
    else:
        cfg.events = {
            "reset_scene_to_default": EventTermCfg(
                func=lambda env, env_ids: None,
                mode="reset",
            ),
            "reset_root_state": EventTermCfg(
                func=events.reset_root_state_full,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "reset_joints": EventTermCfg(
                func=events.reset_joints,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "friction": EventTermCfg(
                func=events.randomize_friction,
                mode="startup",
                params={"friction_range": (0.2, 1.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "restitution": EventTermCfg(
                func=events.randomize_restitution,
                mode="startup",
                params={"restitution_range": (0.0, 0.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "base_mass": EventTermCfg(
                func=events.randomize_base_mass,
                mode="startup",
                params={"mass_range": (-1.0, 3.0), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "inertia": EventTermCfg(
                func=events.randomize_inertia,
                mode="startup",
                params={"inertia_range": (0.8, 1.2), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "com": EventTermCfg(
                func=events.randomize_com,
                mode="startup",
                params={"com_range": 0.05, "asset_cfg": SceneEntityCfg("robot")},
            ),
            "pd_gains": EventTermCfg(
                func=events.randomize_pd_gains,
                mode="startup",
                params={
                    "kp_range": (0.8, 1.2),
                    "kd_range": (0.8, 1.2),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "default_dof_pos": EventTermCfg(
                func=events.randomize_default_dof_pos,
                mode="startup",
                params={
                    "offset_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "push_robots": EventTermCfg(
                func=events.push_robots,
                mode="interval",
                interval_range_s=(5.0, 6.0),
                params={
                    "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        }
        cfg.episode_length_s = 20.0

    cfg.scale_rewards_by_dt = True
    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1024,
        mujoco=MujocoCfg(timestep=0.002),
    )
    cfg.viewer = ViewerConfig()

    return cfg


def se3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """带地形课程的崎岖地形环境配置。"""

    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    cfg = se3_flat_env_cfg(play=play)

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
    )

    return cfg
