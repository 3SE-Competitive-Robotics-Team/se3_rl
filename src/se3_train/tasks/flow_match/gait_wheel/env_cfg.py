"""FlowMatch GAIT_WHEEL 单标签 task 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.rewards import is_alive
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_shared import TaskMode
from se3_train.tasks.flow_match.common import single_label_env_cfg

from . import curriculums, rewards

_MODE = (int(TaskMode.GAIT_WHEEL),)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 FlowMatch GAIT_WHEEL 单标签训练环境。"""
    cfg = single_label_env_cfg(TaskMode.GAIT_WHEEL, play=play, use_light_terrain=True)
    cfg.rewards = _gait_wheel_rewards()

    command = cfg.commands["velocity_height"]
    command.lin_vel_x_range = (-1.2, 1.2)
    command.ang_vel_yaw_range = (-1.2, 1.2)

    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=curriculums.commands_vel,
            params={
                "command_name": "velocity_height",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x_range": (-0.25, 0.25),
                        "ang_vel_yaw_range": (-0.2, 0.2),
                    },
                    {
                        "step": 800,
                        "lin_vel_x_range": (-0.45, 0.45),
                        "ang_vel_yaw_range": (-0.4, 0.4),
                    },
                    {
                        "step": 1800,
                        "lin_vel_x_range": (-0.8, 0.8),
                        "ang_vel_yaw_range": (-0.8, 0.8),
                    },
                    {
                        "step": 3200,
                        "lin_vel_x_range": (-1.2, 1.2),
                        "ang_vel_yaw_range": (-1.2, 1.2),
                    },
                ],
            },
        )
    return cfg


def _gait_wheel_rewards() -> dict[str, RewardTermCfg]:
    """GAIT_WHEEL 的显式 reward 表：腿形成交替节律，轮子在支撑期滑行补速。"""
    return {
        "is_alive": RewardTermCfg(func=is_alive, weight=1.0),
        "gait_wheel_tracking_lin_vel": RewardTermCfg(
            func=rewards.mode_tracking_lin_vel,
            weight=1.4,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.25,
                "sigma_stand": 0.10,
                "vz_weight": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_wheel_tracking_ang_vel": RewardTermCfg(
            func=rewards.mode_tracking_ang_vel,
            weight=0.35,
            params={"command_name": "velocity_height", "sigma": 0.3, "modes": _MODE},
        ),
        "gait_wheel_velocity_assist": RewardTermCfg(
            func=rewards.gait_wheel_velocity_assist,
            weight=0.35,
            params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "gait_wheel_support_velocity_assist": RewardTermCfg(
            func=rewards.gait_wheel_support_velocity_assist,
            weight=0.8,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "min_command": 0.08,
                "contact_force_threshold": 1.0,
                "speed_scale": 0.45,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "gait_natural_swing_clearance": RewardTermCfg(
            func=rewards.gait_natural_swing_clearance,
            weight=2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "leg_sensor_name": "leg_contact_sensor",
                "target_clearance": 0.04,
                "balance_window_s": 0.6,
                "contact_force_threshold": 1.0,
                "leg_contact_force_threshold": 0.2,
                "asset_cfg": SceneEntityCfg("robot"),
                "modes": _MODE,
            },
        ),
        "gait_single_support_contact": RewardTermCfg(
            func=rewards.gait_single_support_contact,
            weight=1.2,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "contact_force_threshold": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_single_support_air_time": RewardTermCfg(
            func=rewards.gait_single_support_air_time,
            weight=2.5,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "target_air_time": 0.04,
                "min_command": 0.05,
                "contact_force_threshold": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_air_time": RewardTermCfg(
            func=rewards.gait_air_time,
            weight=5.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "leg_sensor_name": "leg_contact_sensor",
                "target_air_time": 0.04,
                "max_reward_air_time": 0.25,
                "min_command": 0.05,
                "contact_force_threshold": 1.0,
                "leg_contact_force_threshold": 0.2,
                "modes": _MODE,
            },
        ),
        "gait_alternating_air_time": RewardTermCfg(
            func=rewards.gait_alternating_air_time,
            weight=8.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "leg_sensor_name": "leg_contact_sensor",
                "target_air_time": 0.04,
                "max_reward_air_time": 0.25,
                "min_command": 0.05,
                "contact_force_threshold": 1.0,
                "leg_contact_force_threshold": 0.2,
                "modes": _MODE,
            },
        ),
        "gait_stuck_stance_penalty": RewardTermCfg(
            func=rewards.gait_stuck_stance_penalty,
            weight=-6.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "grace_time_s": 0.24,
                "contact_force_threshold": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_swing_side_balance_penalty": RewardTermCfg(
            func=rewards.gait_swing_side_balance_penalty,
            weight=-10.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "window_s": 0.6,
                "min_single_support_ratio": 0.1,
                "contact_force_threshold": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_short_air_time_penalty": RewardTermCfg(
            func=rewards.gait_short_air_time_penalty,
            weight=-1.5,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "min_air_time": 0.04,
                "min_command": 0.05,
                "contact_force_threshold": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_touchdown_softness": RewardTermCfg(
            func=rewards.gait_touchdown_softness,
            weight=-4.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "allowed_down_vel": 0.12,
                "max_penalty": 4.0,
                "contact_force_threshold": 1.0,
                "modes": _MODE,
            },
        ),
        "gait_touchdown_support_alignment": RewardTermCfg(
            func=rewards.gait_touchdown_support_alignment,
            weight=-3.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "wheel_radius": 0.059,
                "contact_force_threshold": 1.0,
                "height_sensor_name": "base_height_sensor",
                "tolerance": 0.06,
                "max_penalty": 4.0,
                "max_support_offset": 0.30,
                "lateral_weight": 0.35,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "gait_leg_contact_force": RewardTermCfg(
            func=rewards.gait_leg_contact_force,
            weight=-120.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "leg_contact_sensor",
                "force_scale": 5.0,
                "contact_threshold": 0.2,
                "contact_event_scale": 0.5,
                "modes": _MODE,
            },
        ),
        "leg_obstacle_collision": RewardTermCfg(
            func=rewards.leg_obstacle_collision,
            weight=-2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "leg_contact_sensor",
            },
        ),
        "loco_base_height": RewardTermCfg(
            func=rewards.loco_base_height,
            weight=-1.0,
            params={
                "command_name": "velocity_height",
                "height_sensor_name": "base_height_sensor",
                "target_height": 0.32,
            },
        ),
        "loco_orientation": RewardTermCfg(
            func=rewards.loco_orientation,
            weight=-3.0,
            params={"command_name": "velocity_height"},
        ),
        "loco_lin_vel_z": RewardTermCfg(
            func=rewards.loco_lin_vel_z,
            weight=-0.7,
            params={"command_name": "velocity_height"},
        ),
        "loco_ang_vel_xy": RewardTermCfg(
            func=rewards.loco_ang_vel_xy,
            weight=-0.4,
            params={"command_name": "velocity_height"},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-5.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-5.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.08),
        "gait_action_smoothness": RewardTermCfg(
            func=rewards.gait_action_smoothness,
            weight=-0.05,
            params={
                "command_name": "velocity_height",
                "max_penalty": 80.0,
                "modes": _MODE,
            },
        ),
        "loco_dof_pos_limit_cost": RewardTermCfg(
            func=rewards.loco_dof_pos_limit_cost,
            weight=-5.0,
            params={
                "command_name": "velocity_height",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "loco_torque_limit_cost": RewardTermCfg(
            func=rewards.loco_torque_limit_cost,
            weight=-2.0e-4,
            params={
                "command_name": "velocity_height",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "loco_dof_vel_limit_cost": RewardTermCfg(
            func=rewards.loco_dof_vel_limit_cost,
            weight=-1.0e-4,
            params={
                "command_name": "velocity_height",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
    }
