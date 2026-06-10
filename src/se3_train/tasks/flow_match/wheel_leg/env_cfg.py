"""FlowMatch WHEEL_LEG 单标签 task 环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.rewards import is_alive
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_shared import TaskMode
from se3_train.tasks.flow_match.common import single_label_env_cfg

from . import curriculums, rewards

_MODE = (int(TaskMode.WHEEL_LEG),)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 FlowMatch WHEEL_LEG 单标签训练环境。"""
    cfg = single_label_env_cfg(TaskMode.WHEEL_LEG, play=play, use_light_terrain=True)
    cfg.rewards = _wheel_leg_rewards()

    command = cfg.commands["velocity_height"]
    command.lin_vel_x_range = (-1.5, 1.5)
    command.ang_vel_yaw_range = (-1.5, 1.5)

    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=curriculums.commands_vel,
            params={
                "command_name": "velocity_height",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x_range": (-0.4, 0.4),
                        "ang_vel_yaw_range": (-0.4, 0.4),
                    },
                    {
                        "step": 600,
                        "lin_vel_x_range": (-0.8, 0.8),
                        "ang_vel_yaw_range": (-0.8, 0.8),
                    },
                    {
                        "step": 1400,
                        "lin_vel_x_range": (-1.2, 1.2),
                        "ang_vel_yaw_range": (-1.2, 1.2),
                    },
                    {
                        "step": 2400,
                        "lin_vel_x_range": (-1.5, 1.5),
                        "ang_vel_yaw_range": (-1.5, 1.5),
                    },
                ],
            },
        )
    return cfg


def _wheel_leg_rewards() -> dict[str, RewardTermCfg]:
    """WHEEL_LEG 的显式 reward 表：轮主驱，腿负责越障和稳定。"""
    return {
        "is_alive": RewardTermCfg(func=is_alive, weight=1.0),
        "wheel_leg_tracking_lin_vel": RewardTermCfg(
            func=rewards.mode_tracking_lin_vel,
            weight=0.8,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.3,
                "sigma_stand": 0.12,
                "vz_weight": 1.0,
                "modes": _MODE,
            },
        ),
        "wheel_leg_tracking_ang_vel": RewardTermCfg(
            func=rewards.mode_tracking_ang_vel,
            weight=0.6,
            params={"command_name": "velocity_height", "sigma": 0.3, "modes": _MODE},
        ),
        "wheel_swing_clearance": RewardTermCfg(
            func=rewards.wheel_swing_clearance,
            weight=-0.5,
            params={
                "command_name": "velocity_height",
                "target_clearance_m": 0.10,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "wheel_stumble": RewardTermCfg(
            func=rewards.wheel_stumble,
            weight=-2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
            },
        ),
        "leg_obstacle_collision": RewardTermCfg(
            func=rewards.leg_obstacle_collision,
            weight=-2.5,
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
            weight=-4.0,
            params={"command_name": "velocity_height"},
        ),
        "loco_lin_vel_z": RewardTermCfg(
            func=rewards.loco_lin_vel_z,
            weight=-1.0,
            params={"command_name": "velocity_height"},
        ),
        "loco_ang_vel_xy": RewardTermCfg(
            func=rewards.loco_ang_vel_xy,
            weight=-0.5,
            params={"command_name": "velocity_height"},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-2.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.2),
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
