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

_DISCOVERY_REWARD_WEIGHTS = {
    "tracking_lin_vel": 3.0,
    "tracking_ang_vel": 1.5,
    "upward": 1.0,
    "tracking_height": -500.0,
    "upright_zero_velocity": -0.05,
    "leg_action_rate": -0.005,
    "wheel_action_rate": -0.005,
    "dof_pos_limits": -5.0,
    "collision": -1.0,
    "contact_forces": -1.5e-4,
    "diagnostics": 1.0,
}


def _configure_discovery_reward_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    """Keep discovery rewards intentionally small and fail loudly on drift."""

    cfg.rewards.clear()
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.tracking_lin_vel,
        weight=3.0,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.25,
            "sigma_stand": 0.05,
            "vz_weight": 0.0,
            "use_upright_gate": True,
            "tracking_upright_full_cos": math.cos(math.radians(15.0)),
        },
    )
    cfg.rewards["tracking_ang_vel"] = RewardTermCfg(
        func=rewards.tracking_ang_vel,
        weight=1.5,
        params={
            "command_name": "velocity_height",
            "sigma": 0.25,
            "sigma_cmd_scale": 0.0,
            "ratio_blend": 0.0,
            "use_upright_gate": True,
            "tracking_upright_full_cos": math.cos(math.radians(15.0)),
        },
    )
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=1.0)
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=-500.0,
        params={
            "command_name": "velocity_height",
            "sigma": 0.0025,
            "height_sensor_name": "base_height_sensor",
            "kernel": "l2",
            "use_upright_gate": True,
            "min_upright_gate": 0.0,
            "use_pose_end_gate": False,
            "upright_gate_angle_deg": 30.0,
            "inverted_gate_angle_deg": 150.0,
        },
    )
    cfg.rewards["upright_zero_velocity"] = RewardTermCfg(
        func=rewards.recovery_upright_zero_velocity_penalty,
        weight=-0.05,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "gate_start_deg": 45.0,
            "gate_full_deg": 15.0,
            "base_speed_scale": 0.15,
            "wheel_speed_scale": 0.12,
            "base_ang_vel_scale": 0.6,
            "max_penalty": 8.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["leg_action_rate"] = RewardTermCfg(
        func=rewards.leg_action_rate,
        weight=-0.005,
    )
    cfg.rewards["wheel_action_rate"] = RewardTermCfg(
        func=rewards.wheel_action_rate,
        weight=-0.005,
    )
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=rewards.dof_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["collision"] = RewardTermCfg(
        func=rewards.collision,
        weight=-1.0,
        params={
            "sensor_name": "collision_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
            "use_recovery_gate": False,
        },
    )
    cfg.rewards["contact_forces"] = RewardTermCfg(
        func=rewards.contact_forces,
        weight=-1.5e-4,
        params={
            "threshold": 20.0,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
            "use_recovery_gate": False,
        },
    )
    cfg.rewards["diagnostics"] = RewardTermCfg(
        func=rewards.recovery_diagnostics,
        weight=1.0,
        params={
            "command_name": "velocity_height",
            "base_height_sensor_name": "base_height_sensor",
            "wheel_sensor_name": "wheel_sensor",
            "leg_contact_sensor_name": "leg_contact_sensor",
            "collision_sensor_name": "collision_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
            "force_threshold": 1.0,
            "contact_force_threshold": 20.0,
            "action_saturation_threshold": 0.95,
            "active_rod_margin_warning": 0.05,
            "log_interval_steps": 256,
            "core_log_interval_steps": 64,
        },
    )
    _assert_discovery_reward_contract(cfg)


def _assert_discovery_reward_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    actual = set(cfg.rewards)
    expected = set(_DISCOVERY_REWARD_WEIGHTS)
    if actual != expected:
        raise RuntimeError(
            "Recovery-Discovery reward contract drifted: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    bad_weights = {
        name: float(cfg.rewards[name].weight)
        for name, expected_weight in _DISCOVERY_REWARD_WEIGHTS.items()
        if abs(float(cfg.rewards[name].weight) - float(expected_weight)) > 1.0e-12
    }
    if bad_weights:
        raise RuntimeError(f"Recovery-Discovery reward weight drifted: {bad_weights}")


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """标准姿态 Discovery 环境配置。"""
    cfg = recovery_env_cfg(play=play)

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.height_range = (0.24, 0.30)
    command_cfg.standing_height_range = (0.24, 0.30)

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

    _configure_discovery_reward_contract(cfg)
    return cfg
