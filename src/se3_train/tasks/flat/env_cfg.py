# ruff: noqa: F401
from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
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

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.actions import SerialLegDelayedActionCfg
from se3_train.robot_cfg import get_serialleg_cfg

from . import commands, curriculums, events, observations, rewards, terminations

_ROBOT_DEFAULTS = SharedRobotConfig()
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height
_STANDING_HEIGHT_RANGE = (0.20, 0.32)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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

    # 轮子离地高度传感器:从左轮 body 向下打射线,测量真实轮子离地距离
    # 用于 jump_wheel_clr_tracking,防止策略通过收腿套利 base_link 高度
    wheel_height_sensor_cfg = TerrainHeightSensorCfg(
        name="wheel_height_sensor",
        frame=ObjRef(type="body", name="l_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
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
        wheel_height_sensor_cfg,
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
        "jump_commands": ObservationTermCfg(func=observations.jump_commands_obs),
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
        "velocity_height": commands.JumpCommandCfg(
            resampling_time_range=(5.0, 5.0),
            jump_prob=0.0,  # 行走任务不触发跳跃
            height_range=_STANDING_HEIGHT_RANGE,
            standing_height_range=_STANDING_HEIGHT_RANGE,
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
        # 姿态相关项只使用惩罚语义:偏离目标姿态扣分,明显倾斜加重扣分。
        "tracking_orientation_l2": RewardTermCfg(
            func=rewards.tracking_orientation_l2,
            weight=-12.0,
            params={"command_name": "velocity_height"},
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
        "flat_base_height": RewardTermCfg(
            func=rewards.flat_base_height_penalty_no_jump,
            weight=-8.0,
            params={
                "command_name": "velocity_height",
                "height_sensor_name": "base_height_sensor",
                "sigma": 0.05,
            },
        ),
        "flat_base_lin_vel_z": RewardTermCfg(
            func=rewards.flat_base_lin_vel_z_no_jump,
            weight=-1.5,
            params={
                "command_name": "velocity_height",
                "low_speed_threshold": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-6.0,
            params={"soft_limit_deg": 10.0, "hard_limit_deg": 30.0, "max_penalty": 4.0},
        ),
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
                "default_height": _DEFAULT_STANDING_HEIGHT,
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
        "flat_wheel_ground_slip": RewardTermCfg(
            func=rewards.flat_wheel_ground_slip_no_jump,
            weight=-8.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "wheel_radius": 0.059,
                "contact_force_threshold": 1.0,
                "longitudinal_scale": 0.28,
                "lateral_scale": 0.20,
                "max_penalty": 9.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "flat_wheel_center_alignment": RewardTermCfg(
            func=rewards.flat_wheel_center_alignment_no_jump,
            weight=-3.0,
            params={
                "command_name": "velocity_height",
                "contact_sensor_name": "wheel_sensor",
                "contact_force_threshold": 1.0,
                "low_speed_threshold": 0.10,
                "center_lead_gain": 0.03,
                "center_tolerance": 0.05,
                "max_penalty": 4.0,
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
        # 业界标准:移除显式 termination 惩罚,改用 alive reward 隐式机制
        # 摔倒 → episode 结束 → 损失后续所有 alive 累积(隐式 penalty ~= alive * remaining_steps)
        # ETH/Unitree/CMU 所有框架 termination weight = 0,这是行业共识
        "is_alive": RewardTermCfg(func=rewards.is_alive, weight=1.0),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
        "catastrophic_state": TerminationTermCfg(
            func=terminations.catastrophic_state,
            time_out=False,
            params={
                "max_leg_pos_error": 3.0,
                "max_leg_vel": 120.0,
                "max_root_lin_vel": 80.0,
                "max_root_ang_vel": 500.0,
                "min_base_height": -0.5,
                "max_base_height": 3.0,
            },
        ),
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
                func=curriculums.commands_vel,
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
                            "lin_vel_x_range": (-0.5, 0.5),
                            "ang_vel_yaw_range": (-0.5, 0.5),
                        },
                        {
                            "step": 1500,
                            "lin_vel_x_range": (-1.0, 1.0),
                            "ang_vel_yaw_range": (-1.0, 1.0),
                        },
                        {
                            "step": 2500,
                            "lin_vel_x_range": (-1.5, 1.5),
                            "ang_vel_yaw_range": (-2.0, 2.0),
                        },
                        {
                            "step": 3500,
                            "lin_vel_x_range": (-2.0, 2.0),
                            "ang_vel_yaw_range": (-2.5, 2.5),
                        },
                        {
                            "step": 4500,
                            "lin_vel_x_range": (-2.5, 2.5),
                            "ang_vel_yaw_range": (-3.0, 3.0),
                        },
                    ],
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculums.push_disturbance,
                params={
                    "push_stages": [
                        {
                            "step": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "step": 2000,
                            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                        },
                        {
                            "step": 5000,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "step": 10000,
                            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                        },
                        {
                            "step": 20000,
                            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                        },
                        {
                            "step": 40000,
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
                params={"mass_range": (-0.5, 1.5), "asset_cfg": SceneEntityCfg("robot")},
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
                    "kp_range": (0.9, 1.1),  # 收窄:配合 stall_torque 上限,避免 kp 偏软时振荡跪地
                    "kd_range": (0.9, 1.1),
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
        njmax=1040,
        mujoco=MujocoCfg(timestep=0.002),
    )
    cfg.viewer = ViewerConfig()

    return cfg
