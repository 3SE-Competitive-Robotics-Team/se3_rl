"""WheelDog 平地速度跟随环境配置。"""

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
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
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

from . import commands, curriculums, events, observations, rewards, terminations

_COMMAND_NAME = "base_velocity"
_MAX_LIN_VEL_X = 4.0
_MAX_LIN_VEL_Y = 1.5

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
    """构造 WheelDog 平地速度跟随训练环境。"""
    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_wheel_dog_cfg()},
        num_envs=1024,
        env_spacing=3.0,
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
            resampling_time_range=(5.0, 5.0),
            lin_vel_x_range=(-_MAX_LIN_VEL_X, _MAX_LIN_VEL_X) if play else (0.0, 0.0),
            lin_vel_y_range=(-_MAX_LIN_VEL_Y, _MAX_LIN_VEL_Y) if play else (0.0, 0.0),
            ang_vel_yaw_range=(0.0, 0.0),
            standing_ratio=0.08,
        ),
    }
    if play:
        cfg.commands[_COMMAND_NAME].debug_vis = True

    leg_joint_ids = DOG_ABAD_JOINT_IDS + DOG_HIP_JOINT_IDS + DOG_KNEE_JOINT_IDS
    cfg.rewards = {
        "tracking_lin_vel_xy": RewardTermCfg(
            func=rewards.tracking_lin_vel_xy,
            weight=2.0,
            params={"command_name": _COMMAND_NAME, "std": 0.7071},
        ),
        "tracking_ang_vel_z": RewardTermCfg(
            func=rewards.tracking_ang_vel_z,
            weight=0.5,
            params={"command_name": _COMMAND_NAME, "std": 0.5},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=rewards.ang_vel_xy_l2, weight=-0.02),
        "flat_orientation_l2": RewardTermCfg(func=rewards.flat_orientation_l2, weight=-1.0),
        "base_height_l2": RewardTermCfg(
            func=rewards.base_height_l2,
            weight=-0.5,
            params={"sensor_name": "base_height_sensor", "target_height": DOG_BASE_HEIGHT},
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-2.0,
            params={"soft_limit_deg": 18.0, "hard_limit_deg": 45.0, "max_penalty": 4.0},
        ),
        "joint_torques_l2": RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-2.5e-5,
            params={"joint_ids": leg_joint_ids},
        ),
        "wheel_torques_l2": RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-1.0e-5,
            params={"joint_ids": DOG_WHEEL_JOINT_IDS},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=rewards.joint_acc_l2,
            weight=-2.0e-7,
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
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-2.0,
            params={"command_name": _COMMAND_NAME},
        ),
        "hipx_joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-0.4,
            params={"command_name": _COMMAND_NAME, "joint_ids": DOG_ABAD_JOINT_IDS},
        ),
        "hipy_joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-0.1,
            params={"command_name": _COMMAND_NAME, "joint_ids": DOG_HIP_JOINT_IDS},
        ),
        "knee_joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-0.1,
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
            weight=-1.5e-4,
            params={"sensor_name": "wheel_sensor", "threshold": 120.0},
        ),
        "feet_contact_without_cmd": RewardTermCfg(
            func=rewards.feet_contact_without_cmd,
            weight=0.1,
            params={"command_name": _COMMAND_NAME, "sensor_name": "wheel_sensor"},
        ),
        "upward": RewardTermCfg(func=rewards.upward, weight=0.08),
        "is_alive": RewardTermCfg(func=rewards.is_alive, weight=1.0),
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
        "body_contact": TerminationTermCfg(
            func=terminations.body_contact,
            time_out=False,
            params={"sensor_name": "body_contact_sensor", "force_threshold": 1.0},
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
            params={"asset_cfg": SceneEntityCfg("robot"), "base_height": DOG_BASE_HEIGHT},
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
                    params={"friction_range": (0.35, 1.5), "asset_cfg": SceneEntityCfg("robot")},
                ),
                "base_mass": EventTermCfg(
                    func=events.randomize_base_mass,
                    mode="startup",
                    params={"mass_range": (-0.5, 1.0), "asset_cfg": SceneEntityCfg("robot")},
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
                        "kd_range": (0.9, 1.1),
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
            "command_vel": CurriculumTermCfg(
                func=curriculums.commands_vel,
                params={
                    "command_name": _COMMAND_NAME,
                    "velocity_stages": [
                        {
                            "step": 0,
                            "lin_vel_x_range": (0.0, 0.0),
                            "lin_vel_y_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 500,
                            "lin_vel_x_range": (-0.75, 0.75),
                            "lin_vel_y_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 1500,
                            "lin_vel_x_range": (-1.5, 1.5),
                            "lin_vel_y_range": (-0.5, 0.5),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 2500,
                            "lin_vel_x_range": (-2.5, 2.5),
                            "lin_vel_y_range": (-0.9, 0.9),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 3500,
                            "lin_vel_x_range": (-_MAX_LIN_VEL_X, _MAX_LIN_VEL_X),
                            "lin_vel_y_range": (-_MAX_LIN_VEL_Y, _MAX_LIN_VEL_Y),
                            "ang_vel_yaw_range": (0.0, 0.0),
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
                            "step": 2000,
                            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        },
                        {
                            "step": 4000,
                            "velocity_range": {"x": (-0.6, 0.6), "y": (-0.45, 0.45)},
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
