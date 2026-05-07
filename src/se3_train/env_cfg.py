from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from se3_train.mdp import events, observations, rewards, terminations
from se3_train.mdp.actions import VMCActionTermCfg
from se3_train.mdp.commands import VelocityHeightCommandCfg
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
        decimation=4,
        scene=scene,
    )

    # 碰撞传感器:惩罚躯干和大腿/小腿的接触(base + lf0/lf1 + rf0/rf1)。
    collision_sensor_cfg = ContactSensorCfg(
        name="collision_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(base_link|lf0_Link|lf1_Link|rf0_Link|rf1_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    # 轮子接触传感器:用于 contact_forces 和 feet_contact_without_cmd。
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

    cfg.scene.sensors = (collision_sensor_cfg, wheel_sensor_cfg)

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={
                "base_ang_vel": ObservationTermCfg(func=observations.base_ang_vel_obs),
                "projected_gravity": ObservationTermCfg(func=observations.projected_gravity_obs),
                "commands": ObservationTermCfg(func=observations.commands_obs),
                "theta0": ObservationTermCfg(func=observations.theta0_obs),
                "theta0_dot": ObservationTermCfg(func=observations.theta0_dot_obs),
                "L0": ObservationTermCfg(func=observations.L0_obs),
                "L0_dot": ObservationTermCfg(func=observations.L0_dot_obs),
                "wheel_pos": ObservationTermCfg(func=observations.wheel_pos_obs),
                "wheel_vel": ObservationTermCfg(func=observations.wheel_vel_obs),
                "last_actions": ObservationTermCfg(func=observations.last_actions_obs),
            },
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "base_ang_vel": ObservationTermCfg(func=observations.base_ang_vel_obs),
                "projected_gravity": ObservationTermCfg(func=observations.projected_gravity_obs),
                "commands": ObservationTermCfg(func=observations.commands_obs),
                "theta0": ObservationTermCfg(func=observations.theta0_obs),
                "theta0_dot": ObservationTermCfg(func=observations.theta0_dot_obs),
                "L0": ObservationTermCfg(func=observations.L0_obs),
                "L0_dot": ObservationTermCfg(func=observations.L0_dot_obs),
                "wheel_pos": ObservationTermCfg(func=observations.wheel_pos_obs),
                "wheel_vel": ObservationTermCfg(func=observations.wheel_vel_obs),
                "last_actions": ObservationTermCfg(func=observations.last_actions_obs),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.actions = {
        "vmc": VMCActionTermCfg(
            entity_name="robot",
        ),
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
            params={"command_name": "velocity_height", "sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=rewards.tracking_ang_vel,
            weight=1.73,
            params={"command_name": "velocity_height", "sigma": 0.25},
        ),
        "upward": RewardTermCfg(func=rewards.upward, weight=0.607),
        "lin_vel_z": RewardTermCfg(func=rewards.lin_vel_z, weight=-2.52),
        "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.146),
        "base_height": RewardTermCfg(
            func=rewards.base_height, weight=2.49, params={"target": 0.301}
        ),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-2.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_pos_penalty": RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=-1.03,
            params={"asset_cfg": SceneEntityCfg("robot")},
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
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.0476),
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-1.78,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
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
                "force_threshold": 1.0,
                "cmd_threshold": 0.1,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
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
                func=events.reset_joints_vmc,
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
                func=events.reset_joints_vmc,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "push_robot": EventTermCfg(
                func=events.push_robots,
                mode="interval",
                interval_range_s=(4.0, 4.0),
                params={
                    "velocity_range": {
                        "x": (-4.0, 4.0),
                        "y": (-4.0, 4.0),
                        "z": (-1.0, 1.0),
                    },
                    "asset_cfg": SceneEntityCfg("robot"),
                },
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
                params={"inertia_range": (0.7, 1.3), "asset_cfg": SceneEntityCfg("robot")},
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
                    "kp_range": (0.5, 2.0),
                    "kd_range": (0.5, 2.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "vmc_gains": EventTermCfg(
                func=events.randomize_vmc_gains,
                mode="startup",
                params={
                    "kp_range": (0.5, 2.0),
                    "kd_range": (0.5, 2.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "motor_torque": EventTermCfg(
                func=events.randomize_motor_torque,
                mode="startup",
                params={"torque_range": (0.7, 1.3), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "default_dof_pos": EventTermCfg(
                func=events.randomize_default_dof_pos,
                mode="startup",
                params={
                    "offset_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "action_delay": EventTermCfg(
                func=events.randomize_action_delay,
                mode="startup",
                params={"delay_range": (0.0, 0.02), "asset_cfg": SceneEntityCfg("robot")},
            ),
        }
        cfg.episode_length_s = 20.0

    cfg.scale_rewards_by_dt = True
    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1024,
        mujoco=MujocoCfg(timestep=0.005),
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
