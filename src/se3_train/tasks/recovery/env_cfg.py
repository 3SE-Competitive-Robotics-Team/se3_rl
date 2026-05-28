"""倒地自启任务环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import events, rewards, terminations

_DEFAULT_STANDING_HEIGHT = 0.22


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """平地行走 + 倒地自启混合训练环境。"""

    cfg = flat_env_cfg(play=play)
    recovery_grace_steps = 600
    recovery_stages = (
        (
            {
                "step": 0,
                "prob": 0.50,
                "roll_range": (-0.45, 0.45),
                "pitch_range": (-0.45, 0.45),
                "height_range": (0.22, 0.32),
                "fallen_pose_prob": 0.80,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.35, 1.60),
                "fallen_pitch_abs_range": (1.35, 1.60),
                "fallen_coupled_range": (-0.25, 0.25),
                "fallen_height_range": (0.14, 0.24),
            },
            {
                "step": 2000,
                "prob": 0.55,
                "roll_range": (-0.65, 0.65),
                "pitch_range": (-0.65, 0.65),
                "height_range": (0.20, 0.32),
                "fallen_pose_prob": 0.65,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.45, 1.80),
                "fallen_pitch_abs_range": (1.45, 1.80),
                "fallen_coupled_range": (-0.30, 0.30),
                "fallen_height_range": (0.12, 0.23),
            },
            {
                "step": 5000,
                "prob": 0.75,
                "roll_range": (-0.80, 0.80),
                "pitch_range": (-0.80, 0.80),
                "height_range": (0.18, 0.30),
                "fallen_pose_prob": 0.85,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.50, 2.10),
                "fallen_pitch_abs_range": (1.50, 2.10),
                "fallen_coupled_range": (-0.35, 0.35),
                "fallen_height_range": (0.10, 0.22),
            },
            {
                "step": 9000,
                "prob": 0.85,
                "roll_range": (-1.00, 1.00),
                "pitch_range": (-1.00, 1.00),
                "height_range": (0.16, 0.30),
                "fallen_pose_prob": 0.90,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.45, 2.35),
                "fallen_pitch_abs_range": (1.45, 2.35),
                "fallen_coupled_range": (-0.40, 0.40),
                "fallen_height_range": (0.10, 0.22),
            },
        )
        if not play
        else (
            {
                "step": 0,
                "prob": 1.0,
                "roll_range": (-1.00, 1.00),
                "pitch_range": (-1.00, 1.00),
                "height_range": (0.16, 0.30),
                "fallen_pose_prob": 0.90,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.45, 2.35),
                "fallen_pitch_abs_range": (1.45, 2.35),
                "fallen_coupled_range": (-0.40, 0.40),
                "fallen_height_range": (0.10, 0.22),
            },
        )
    )

    cfg.events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_full,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "recovery_prob": 1.0 if play else 0.15,
            "recovery_stages": list(recovery_stages),
            "recovery_grace_steps": recovery_grace_steps,
            "recovery_command_height": _DEFAULT_STANDING_HEIGHT,
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "recovery_joint_offset_range": 0.25,
            "recovery_joint_vel_range": (-0.8, 0.8),
        },
    )

    cfg.terminations["bad_orientation"] = TerminationTermCfg(
        func=terminations.bad_orientation_delayed,
        time_out=False,
        params={
            "limit_angle": 0.5236,
            "max_steps": 100,
            "recovery_grace_steps": recovery_grace_steps,
        },
    )
    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact,
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
            "recovery_grace_steps": recovery_grace_steps,
            "recovery_terminate": False,
        },
    )

    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.stand_still,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": _DEFAULT_STANDING_HEIGHT,
            "height_tolerance": 40.0,
            "ignore_recovery": True,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot"), "recovery_scale": 0.15},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power,
        weight=-1.03e-4,
        params={"asset_cfg": SceneEntityCfg("robot"), "recovery_scale": 0.15},
    )
    cfg.rewards["action_rate"] = RewardTermCfg(
        func=rewards.action_rate,
        weight=-0.48,
        params={"recovery_scale": 0.15},
    )
    cfg.rewards["recovery_upright"] = RewardTermCfg(func=rewards.recovery_upright, weight=5.0)
    cfg.rewards["recovery_tilt_progress"] = RewardTermCfg(
        func=rewards.recovery_tilt_progress,
        weight=3.0,
        params={"upright_angle_deg": 15.0},
    )
    cfg.rewards["recovery_hard_roll_upright"] = RewardTermCfg(
        func=rewards.recovery_hard_roll_upright,
        weight=4.0,
        params={"min_initial_roll_deg": 75.0, "max_initial_pitch_deg": 35.0},
    )
    cfg.rewards["recovery_hard_pitch_upright"] = RewardTermCfg(
        func=rewards.recovery_hard_pitch_upright,
        weight=4.0,
        params={"min_initial_pitch_deg": 75.0, "max_initial_roll_deg": 35.0},
    )
    cfg.rewards["recovery_height"] = RewardTermCfg(
        func=rewards.recovery_height,
        weight=2.0,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "base_height_sensor",
            "sigma": 0.04,
        },
    )
    cfg.rewards["recovery_wheel_contact"] = RewardTermCfg(
        func=rewards.recovery_wheel_contact,
        weight=1.0,
        params={"sensor_name": "wheel_sensor", "force_threshold": 1.0},
    )
    cfg.rewards["recovery_success"] = RewardTermCfg(
        func=rewards.recovery_success,
        weight=5.0,
        params={
            "sensor_name": "wheel_sensor",
            "height_sensor_name": "base_height_sensor",
            "command_name": "velocity_height",
            "upright_angle_deg": 15.0,
            "height_tolerance": 0.05,
            "ang_vel_threshold": 1.5,
            "force_threshold": 1.0,
        },
    )
    cfg.rewards["recovery_hard_roll_success"] = RewardTermCfg(
        func=rewards.recovery_hard_roll_success,
        weight=8.0,
        params={
            "sensor_name": "wheel_sensor",
            "height_sensor_name": "base_height_sensor",
            "command_name": "velocity_height",
            "upright_angle_deg": 15.0,
            "height_tolerance": 0.05,
            "ang_vel_threshold": 1.5,
            "force_threshold": 1.0,
            "min_initial_roll_deg": 75.0,
            "max_initial_pitch_deg": 35.0,
        },
    )
    cfg.rewards["recovery_hard_pitch_success"] = RewardTermCfg(
        func=rewards.recovery_hard_pitch_success,
        weight=8.0,
        params={
            "sensor_name": "wheel_sensor",
            "height_sensor_name": "base_height_sensor",
            "command_name": "velocity_height",
            "upright_angle_deg": 15.0,
            "height_tolerance": 0.05,
            "ang_vel_threshold": 1.5,
            "force_threshold": 1.0,
            "min_initial_pitch_deg": 75.0,
            "max_initial_roll_deg": 35.0,
        },
    )
    cfg.rewards["recovery_stability"] = RewardTermCfg(
        func=rewards.recovery_stability,
        weight=-0.25,
        params={"ang_vel_weight": 0.25},
    )

    return cfg
