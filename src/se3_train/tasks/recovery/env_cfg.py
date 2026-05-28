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
    recovery_grace_steps = 400
    recovery_stages = (
        (
            {
                "step": 0,
                "prob": 0.15,
                "roll_range": (-0.35, 0.35),
                "pitch_range": (-0.35, 0.35),
                "height_range": (0.24, 0.32),
            },
            {
                "step": 1200,
                "prob": 0.25,
                "roll_range": (-0.65, 0.65),
                "pitch_range": (-0.65, 0.65),
                "height_range": (0.24, 0.34),
            },
            {
                "step": 3000,
                "prob": 0.35,
                "roll_range": (-1.05, 1.05),
                "pitch_range": (-0.90, 0.90),
                "height_range": (0.25, 0.36),
            },
            {
                "step": 7000,
                "prob": 0.50,
                "roll_range": (-1.35, 1.35),
                "pitch_range": (-1.10, 1.10),
                "height_range": (0.26, 0.38),
            },
        )
        if not play
        else (
            {
                "step": 0,
                "prob": 1.0,
                "roll_range": (-1.05, 1.05),
                "pitch_range": (-0.90, 0.90),
                "height_range": (0.25, 0.36),
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
            "recovery_joint_offset_range": 0.16,
            "recovery_joint_vel_range": (-0.4, 0.4),
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
    cfg.rewards["recovery_upright"] = RewardTermCfg(func=rewards.recovery_upright, weight=5.0)
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
        weight=4.0,
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
    cfg.rewards["recovery_stability"] = RewardTermCfg(
        func=rewards.recovery_stability,
        weight=-0.25,
        params={"ang_vel_weight": 0.25},
    )

    return cfg
