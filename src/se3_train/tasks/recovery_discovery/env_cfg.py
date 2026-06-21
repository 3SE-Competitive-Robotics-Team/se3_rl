"""倒地自启 Discovery 阶段环境配置。"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp import events as mdp_events
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """标准姿态 Discovery 环境配置。"""
    cfg = recovery_env_cfg(play=play)

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.height_range = (0.24, 0.30)
    command_cfg.standing_height_range = (0.24, 0.30)

    cfg.rewards["tracking_height"].weight = -1500.0
    cfg.rewards["tracking_height"].params["use_pose_end_gate"] = False
    cfg.rewards["tracking_height"].params["use_inverted_free_upright_height_gate"] = True
    cfg.rewards["upward"].weight = 3.0

    cfg.curriculum = {}
    cfg.events.pop("push_robots", None)
    cfg.events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_recovery_standard_poses,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_xy_range": (-0.15, 0.15),
            "height_offset_range": (0.0, 0.02),
            "yaw_range": (-math.pi, math.pi),
            "roll_jitter_range": (-math.radians(5.0), math.radians(5.0)),
            "pitch_jitter_range": (-math.radians(5.0), math.radians(5.0)),
            "lin_vel_range": (0.0, 0.0),
            "ang_vel_range": (0.0, 0.0),
            "clearance_range": (0.001, 0.005),
            "pose_weights": (0.08, 0.17, 0.17, 0.29, 0.29),
            "recovery_command_height": 0.26,
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "roll_jitter_range": (-math.radians(5.0), math.radians(5.0)),
                    "pitch_jitter_range": (-math.radians(5.0), math.radians(5.0)),
                    "lin_vel_range": (0.0, 0.0),
                    "ang_vel_range": (0.0, 0.0),
                },
                {
                    "iteration": 300,
                    "roll_jitter_range": (-math.radians(10.0), math.radians(10.0)),
                    "pitch_jitter_range": (-math.radians(10.0), math.radians(10.0)),
                    "lin_vel_range": (-0.03, 0.03),
                    "ang_vel_range": (-0.10, 0.10),
                },
                {
                    "iteration": 800,
                    "roll_jitter_range": (-math.radians(15.0), math.radians(15.0)),
                    "pitch_jitter_range": (-math.radians(15.0), math.radians(15.0)),
                    "lin_vel_range": (-0.05, 0.05),
                    "ang_vel_range": (-0.20, 0.20),
                },
                {
                    "iteration": 1500,
                    "roll_jitter_range": (-math.radians(20.0), math.radians(20.0)),
                    "pitch_jitter_range": (-math.radians(20.0), math.radians(20.0)),
                    "lin_vel_range": (-0.08, 0.08),
                    "ang_vel_range": (-0.30, 0.30),
                },
            ],
            "use_iterations": True,
            "steps_per_policy_iter": 64,
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "wheel_joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "align_root_height_to_wheels": True,
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "joint_offset_range": 0.0,
                    "joint_randomization_prob": 0.0,
                },
                {
                    "iteration": 300,
                    "joint_offset_range": 0.10,
                    "joint_vel_range": (-0.20, 0.20),
                    "joint_randomization_prob": 0.25,
                },
                {
                    "iteration": 800,
                    "joint_offset_range": 0.20,
                    "joint_vel_range": (-0.40, 0.40),
                    "joint_randomization_prob": 0.50,
                },
                {
                    "iteration": 1500,
                    "joint_offset_range": 0.25,
                    "joint_vel_range": (-0.50, 0.50),
                    "joint_randomization_prob": 0.75,
                },
            ],
            "use_iterations": True,
            "steps_per_policy_iter": 64,
        },
    )

    return cfg
