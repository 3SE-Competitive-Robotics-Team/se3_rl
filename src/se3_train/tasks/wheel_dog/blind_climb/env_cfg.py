"""WheelDog 盲爬坑坡环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
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
    GridPatternCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from se3_train.tasks.wheel_dog.robot_cfg import (
    DOG_ABAD_JOINT_IDS,
    DOG_BASE_HEIGHT,
    DOG_HIP_JOINT_IDS,
    DOG_KNEE_JOINT_IDS,
    DOG_LEG_JOINT_NAMES,
    DOG_WHEEL_JOINT_IDS,
    DOG_WHEEL_JOINT_NAMES,
    get_wheel_dog_cfg,
)
from se3_train.terrains import gap_ramp_blind_climb_entity_cfg

from . import commands, curriculums, events, observations, rewards, terminations
from .terrain_progress import FINAL_SUCCESS_DISTANCE

_COMMAND_NAME = "base_velocity"

_DOG_LEG_TARGET_CLIP = {
    ".*_abad_joint": (-0.55, 0.55),
    "fl_hip_joint|fr_hip_joint": (-2.20, 2.45),
    "hl_hip_joint|hr_hip_joint": (-2.45, 2.20),
    ".*_knee_joint": (-2.60, 2.60),
}
"""腿部位置目标限幅，给 MuJoCo 关节物理限位留出安全余量。"""

_DOG_WHEEL_TARGET_CLIP = {".*": (-15.0, 15.0)}
"""轮子速度目标限幅，避免策略用无意义的大速度目标顶饱和。"""


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 WheelDog 盲爬训练环境。"""
    scene = SceneCfg(
        terrain=gap_ramp_blind_climb_entity_cfg(num_envs=1024),
        entities={"robot": get_wheel_dog_cfg()},
        num_envs=1024,
        env_spacing=6.0,
    )

    cfg = ManagerBasedRlEnvCfg(decimation=4, scene=scene)
    cfg.episode_length_s = 20.0

    non_wheel_pattern = r"^(base_link|.*_abad_link|.*_hip_link|.*_thigh_link|.*_shank_link)$"
    wheel_pattern = r"^.*_wheel_link$"
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="body_contact_sensor",
            primary=ContactMatch(mode="body", pattern=non_wheel_pattern, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="wheel_sensor",
            primary=ContactMatch(mode="body", pattern=wheel_pattern, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
        TerrainHeightSensorCfg(
            name="base_height_sensor",
            frame=ObjRef(type="body", name="base_link", entity="robot"),
            ray_alignment="yaw",
            pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=4),
            max_distance=2.0,
            include_geom_groups=(0,),
            reduction="min",
        ),
        TerrainHeightSensorCfg(
            name="terrain_height_grid_sensor",
            frame=ObjRef(type="body", name="base_link", entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
            max_distance=3.0,
            include_geom_groups=(0,),
            reduction="none",
        ),
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
        "joint_vel": ObservationTermCfg(
            func=observations.joint_vel_obs,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
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
            params={"sensor_name": "base_height_sensor"},
        ),
        "terrain_height_grid": ObservationTermCfg(
            func=observations.terrain_height_grid_obs,
            params={"sensor_name": "terrain_height_grid_sensor", "target_height": DOG_BASE_HEIGHT},
        ),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
            nan_policy="sanitize",
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="sanitize",
        ),
    }

    cfg.actions = {
        "leg_position": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=DOG_LEG_JOINT_NAMES,
            scale={
                ".*_abad_joint": 0.125,
                ".*_hip_joint": 0.25,
                ".*_knee_joint": 0.25,
            },
            clip=_DOG_LEG_TARGET_CLIP,
            preserve_order=True,
            use_default_offset=True,
        ),
        "wheel_velocity": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=DOG_WHEEL_JOINT_NAMES,
            scale=5.0,
            clip=_DOG_WHEEL_TARGET_CLIP,
            preserve_order=True,
            use_default_offset=True,
        ),
    }

    cfg.commands = {
        _COMMAND_NAME: commands.DogVelocityCommandCfg(
            resampling_time_range=(6.0, 6.0),
            lin_vel_x_range=(0.35, 1.6) if play else (0.50, 1.10),
            lin_vel_y_range=(0.0, 0.0),
            ang_vel_yaw_range=(0.0, 0.0),
            standing_ratio=0.0,
            flat_lin_vel_x_range=(-4.0, 4.0) if play else (-1.0, 1.0),
            flat_lin_vel_y_range=(-1.5, 1.5) if play else (-0.3, 0.3),
            flat_ang_vel_yaw_range=(0.0, 0.0),
            flat_standing_ratio=0.08,
        ),
    }
    leg_joint_ids = DOG_ABAD_JOINT_IDS + DOG_HIP_JOINT_IDS + DOG_KNEE_JOINT_IDS
    cfg.rewards = {
        "tracking_lin_vel_xy": RewardTermCfg(
            func=rewards.tracking_lin_vel_xy,
            weight=2.0,
            params={"command_name": _COMMAND_NAME, "std": 0.5},
        ),
        "forward_velocity": RewardTermCfg(
            func=rewards.forward_velocity,
            weight=2.0,
            params={"command_name": _COMMAND_NAME, "max_velocity": 1.8},
        ),
        "progress_forward": RewardTermCfg(
            func=rewards.progress_forward,
            weight=3.0,
            params={"command_name": _COMMAND_NAME, "success_distance": FINAL_SUCCESS_DISTANCE},
        ),
        "obstacle_lift": RewardTermCfg(
            func=rewards.obstacle_lift,
            weight=1.5,
            params={
                "command_name": _COMMAND_NAME,
                "before_high_edge": 0.25,
                "after_high_edge": 0.35,
                "max_vertical_velocity": 0.7,
            },
        ),
        "success_progress": RewardTermCfg(
            func=rewards.success_progress,
            weight=3.0,
            params={"command_name": _COMMAND_NAME, "success_distance": FINAL_SUCCESS_DISTANCE},
        ),
        "run_stuck": RewardTermCfg(
            func=rewards.run_stuck,
            weight=-1.2,
            params={"command_name": _COMMAND_NAME, "min_speed": 0.25},
        ),
        "tracking_ang_vel_z": RewardTermCfg(
            func=rewards.tracking_ang_vel_z,
            weight=1.0,
            params={"command_name": _COMMAND_NAME, "std": 0.5},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=rewards.ang_vel_xy_l2, weight=-0.05),
        "flat_orientation_l2": RewardTermCfg(func=rewards.flat_orientation_l2, weight=-0.2),
        "base_height_l2": RewardTermCfg(
            func=rewards.base_height_l2,
            weight=-5.0,
            params={"sensor_name": "base_height_sensor", "target_height": DOG_BASE_HEIGHT},
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-1.0,
            params={"soft_limit_deg": 20.0, "hard_limit_deg": 50.0, "max_penalty": 4.0},
        ),
        "lateral_corridor": RewardTermCfg(
            func=rewards.lateral_corridor,
            weight=-4.0,
            params={
                "soft_half_width": 0.35,
                "hard_half_width": 0.75,
            },
        ),
        "joint_torques_l2": RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-5.0e-6,
            params={"joint_ids": leg_joint_ids},
        ),
        "wheel_torques_l2": RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-5.0e-6,
            params={"joint_ids": DOG_WHEEL_JOINT_IDS},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=rewards.joint_acc_l2,
            weight=-2.5e-7,
            params={"joint_ids": leg_joint_ids},
        ),
        "wheel_acc_l2": RewardTermCfg(
            func=rewards.joint_acc_l2,
            weight=-1.0e-7,
            params={"joint_ids": DOG_WHEEL_JOINT_IDS},
        ),
        "joint_power": RewardTermCfg(
            func=rewards.joint_power,
            weight=-2.0e-5,
            params={"joint_ids": leg_joint_ids},
        ),
        "wheel_power": RewardTermCfg(
            func=rewards.joint_power,
            weight=-1.0e-5,
            params={"joint_ids": DOG_WHEEL_JOINT_IDS},
        ),
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-0.5,
            params={"command_name": _COMMAND_NAME},
        ),
        "hipx_joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-0.5,
            params={"command_name": _COMMAND_NAME, "joint_ids": DOG_ABAD_JOINT_IDS},
        ),
        "hipy_joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-0.05,
            params={"command_name": _COMMAND_NAME, "joint_ids": DOG_HIP_JOINT_IDS},
        ),
        "knee_joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-0.05,
            params={"command_name": _COMMAND_NAME, "joint_ids": DOG_KNEE_JOINT_IDS},
        ),
        "action_rate_l2": RewardTermCfg(func=rewards.action_rate_l2, weight=-0.01),
        "action_l2": RewardTermCfg(func=rewards.action_l2, weight=-0.02),
        "dof_pos_limits": RewardTermCfg(func=rewards.dof_pos_limits, weight=-5.0),
        "undesired_contacts": RewardTermCfg(
            func=rewards.undesired_contacts,
            weight=-1.0,
            params={"sensor_name": "body_contact_sensor", "force_threshold": 1.0},
        ),
        "contact_forces": RewardTermCfg(
            func=rewards.contact_forces,
            weight=-1.0e-4,
            params={"sensor_name": "wheel_sensor", "threshold": 200.0},
        ),
        "upward": RewardTermCfg(func=rewards.upward, weight=0.05),
        "is_alive": RewardTermCfg(func=rewards.is_alive, weight=0.3),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=terminations.bad_orientation_delayed,
            time_out=False,
            params={"limit_angle": 0.7854, "max_steps": 60},
        ),
        "low_base_height": TerminationTermCfg(
            func=terminations.low_base_height_delayed,
            time_out=False,
            params={"sensor_name": "base_height_sensor", "min_height": 0.16, "max_steps": 30},
        ),
        "root_height_bounds": TerminationTermCfg(
            func=terminations.root_height_bounds,
            time_out=False,
            params={"min_relative_height": -0.08, "max_relative_height": 2.0},
        ),
        "corridor_bounds": TerminationTermCfg(
            func=terminations.corridor_bounds,
            time_out=False,
            params={"hard_half_width": 0.75},
        ),
        "nonfinite_state": TerminationTermCfg(
            func=terminations.nonfinite_state,
            time_out=False,
        ),
        "body_contact": TerminationTermCfg(
            func=terminations.body_contact_delayed,
            time_out=False,
            params={
                "sensor_name": "body_contact_sensor",
                "force_threshold": 1.0,
                "max_steps": 20,
            },
        ),
    }

    cfg.events = {
        "reset_scene_to_default": EventTermCfg(
            func=lambda env, env_ids: None,
            mode="reset",
        ),
        "reset_root_state": EventTermCfg(
            func=events.reset_root_state,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_height": DOG_BASE_HEIGHT,
                "runup_distance_range": (1.0, 3.0),
                "y_noise": 0.08,
                "yaw_range": (-0.12, 0.12),
            },
        ),
        "reset_joints": EventTermCfg(
            func=events.reset_joints,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "leg_pos_noise": 0.03},
        ),
    }

    if not play:
        cfg.events.update(
            {
                "friction": EventTermCfg(
                    func=events.randomize_friction,
                    mode="startup",
                    params={"friction_range": (0.2, 1.25), "asset_cfg": SceneEntityCfg("robot")},
                ),
                "restitution": EventTermCfg(
                    func=events.randomize_restitution,
                    mode="startup",
                    params={
                        "restitution_range": (0.0, 0.6),
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                ),
                "base_mass": EventTermCfg(
                    func=events.randomize_base_mass,
                    mode="startup",
                    params={"mass_range": (-1.0, 5.0), "asset_cfg": SceneEntityCfg("robot")},
                ),
                "inertia": EventTermCfg(
                    func=events.randomize_inertia,
                    mode="startup",
                    params={"inertia_range": (0.85, 1.15), "asset_cfg": SceneEntityCfg("robot")},
                ),
                "pd_gains": EventTermCfg(
                    func=events.randomize_pd_gains,
                    mode="reset",
                    params={
                        "kp_range": (0.9, 1.1),
                        "kd_range": (0.85, 1.15),
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                ),
                "push_robots": EventTermCfg(
                    func=events.push_robots,
                    mode="interval",
                    interval_range_s=(10.0, 15.0),
                    params={
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                ),
            }
        )
        cfg.curriculum = {
            "terrain_levels": CurriculumTermCfg(
                func=curriculums.terrain_levels,
                params={
                    "command_name": _COMMAND_NAME,
                    "success_distance": FINAL_SUCCESS_DISTANCE,
                    "min_progress_ratio": 0.50,
                    "top_sample_min_level": 35,
                    "high_level_threshold": 35,
                    "target_level_threshold": 39,
                    "success_half_width": 0.55,
                },
            ),
            "command_vel": CurriculumTermCfg(
                func=curriculums.commands_vel,
                params={
                    "command_name": _COMMAND_NAME,
                    "velocity_stages": [
                        {
                            "step": 0,
                            "lin_vel_x_range": (0.35, 0.80),
                            "lin_vel_y_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                            "flat_lin_vel_x_range": (-0.5, 0.5),
                            "flat_lin_vel_y_range": (0.0, 0.0),
                            "flat_ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 60000,
                            "lin_vel_x_range": (0.45, 1.00),
                            "lin_vel_y_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                            "flat_lin_vel_x_range": (-1.0, 1.0),
                            "flat_lin_vel_y_range": (-0.3, 0.3),
                            "flat_ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 160000,
                            "lin_vel_x_range": (0.55, 1.20),
                            "lin_vel_y_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                            "flat_lin_vel_x_range": (-1.5, 1.5),
                            "flat_lin_vel_y_range": (-0.45, 0.45),
                            "flat_ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 320000,
                            "lin_vel_x_range": (0.70, 1.60),
                            "lin_vel_y_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                            "flat_lin_vel_x_range": (-2.0, 2.0),
                            "flat_lin_vel_y_range": (-0.6, 0.6),
                            "flat_ang_vel_yaw_range": (0.0, 0.0),
                        },
                    ],
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculums.push_disturbance,
                params={
                    "push_stages": [
                        {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                        {
                            "step": 120000,
                            "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                        },
                        {
                            "step": 240000,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.35, 0.35)},
                        },
                    ],
                },
            ),
        }

    cfg.sim = SimulationCfg(
        nconmax=512,
        njmax=2048,
        mujoco=MujocoCfg(timestep=0.005),
    )
    cfg.viewer = ViewerConfig()
    return cfg


__all__ = ["env_cfg"]
