"""倒地自启 Discovery 阶段环境配置。"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp import events as mdp_events
from se3_train.tasks.recovery import rewards
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """标准姿态 Discovery 环境配置。"""
    cfg = recovery_env_cfg(play=play)

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.height_range = (0.24, 0.30)
    command_cfg.standing_height_range = (0.24, 0.30)

    phase_horizon_s = 5.0
    cfg.rewards["tracking_height"].weight = -800.0
    cfg.rewards["tracking_height"].params["use_upright_gate"] = True
    cfg.rewards["tracking_height"].params["min_upright_gate"] = 0.0
    cfg.rewards["tracking_height"].params["use_pose_end_gate"] = False
    cfg.rewards["tracking_height"].params["use_inverted_free_upright_height_gate"] = False
    cfg.rewards["tracking_height"].params["phase_horizon_s"] = phase_horizon_s
    cfg.rewards["tracking_height"].params["phase_min_scale"] = 0.15
    cfg.rewards["tracking_height"].params["phase_max_scale"] = 1.0
    cfg.rewards["upward"].func = rewards.recovery_upward
    cfg.rewards["upward"].weight = 3.0
    cfg.rewards["upright_orientation_l2"].params["phase_horizon_s"] = phase_horizon_s
    cfg.rewards["upright_orientation_l2"].params["phase_min_scale"] = 0.15
    cfg.rewards["upright_orientation_l2"].params["phase_max_scale"] = 1.5
    cfg.rewards["upright_zero_velocity"].params["phase_horizon_s"] = phase_horizon_s
    cfg.rewards["upright_zero_velocity"].params["phase_min_scale"] = 0.0
    cfg.rewards["upright_zero_velocity"].params["phase_max_scale"] = 2.0
    cfg.rewards["leg_action_rate"].weight = -0.02
    cfg.rewards["leg_action_rate"].params = {
        "phase_horizon_s": phase_horizon_s,
        "phase_min_scale": 0.3,
        "phase_max_scale": 1.0,
    }
    cfg.rewards["wheel_action_rate"].weight = -0.05
    cfg.rewards["wheel_action_rate"].params = {
        "phase_horizon_s": phase_horizon_s,
        "phase_min_scale": 0.3,
        "phase_max_scale": 1.0,
    }
    cfg.rewards["action_smoothness"].weight = -0.01
    cfg.rewards["action_smoothness"].params["phase_horizon_s"] = phase_horizon_s
    cfg.rewards["action_smoothness"].params["phase_min_scale"] = 0.25
    cfg.rewards["action_smoothness"].params["phase_max_scale"] = 1.0
    cfg.rewards["joint_mirror"].weight = -0.02
    cfg.rewards["joint_mirror"].params["phase_horizon_s"] = phase_horizon_s
    cfg.rewards["joint_mirror"].params["phase_min_scale"] = 0.0
    cfg.rewards["joint_mirror"].params["phase_max_scale"] = 1.0

    # Discovery 只保留自起主目标、安全限位、轻量动作正则，以及近直立后的稳定项。
    keep_rewards = {
        "upward",
        "dof_pos_limits",
        "joint_mirror",
        "leg_action_rate",
        "wheel_action_rate",
        "action_smoothness",
        "tracking_height",
        "upright_orientation_l2",
        "upright_zero_velocity",
        "diagnostics",
    }
    cfg.rewards = {
        name: term for name, term in cfg.rewards.items() if name in keep_rewards
    }

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
                    "iteration": 500,
                    "roll_jitter_range": (-math.radians(8.0), math.radians(8.0)),
                    "pitch_jitter_range": (-math.radians(8.0), math.radians(8.0)),
                    "lin_vel_range": (-0.02, 0.02),
                    "ang_vel_range": (-0.08, 0.08),
                },
                {
                    "iteration": 1200,
                    "roll_jitter_range": (-math.radians(12.0), math.radians(12.0)),
                    "pitch_jitter_range": (-math.radians(12.0), math.radians(12.0)),
                    "lin_vel_range": (-0.03, 0.03),
                    "ang_vel_range": (-0.12, 0.12),
                },
                {
                    "iteration": 2200,
                    "roll_jitter_range": (-math.radians(18.0), math.radians(18.0)),
                    "pitch_jitter_range": (-math.radians(18.0), math.radians(18.0)),
                    "lin_vel_range": (-0.05, 0.05),
                    "ang_vel_range": (-0.20, 0.20),
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
                    "iteration": 500,
                    "joint_offset_range": 0.05,
                    "joint_vel_range": (-0.10, 0.10),
                    "joint_randomization_prob": 0.15,
                },
                {
                    "iteration": 1200,
                    "joint_offset_range": 0.10,
                    "joint_vel_range": (-0.20, 0.20),
                    "joint_randomization_prob": 0.25,
                },
                {
                    "iteration": 2200,
                    "joint_offset_range": 0.18,
                    "joint_vel_range": (-0.35, 0.35),
                    "joint_randomization_prob": 0.50,
                },
            ],
            "use_iterations": True,
            "steps_per_policy_iter": 64,
        },
    )

    return cfg
