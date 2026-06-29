"""倒地自启 Discovery 阶段环境配置。"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp import events as mdp_events
from se3_train.tasks.recovery import curriculums, rewards
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RECOVERY_STATE_CACHE_PATH = (
    _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_closedchain_stair_v3_40k.npz"
)
_DISCOVERY_MAX_LIN_VEL_X = 1.89
_DISCOVERY_MAX_ANG_VEL_YAW = 9.41

_DISCOVERY_REWARD_WEIGHTS = {
    "tracking_lin_vel": 3.0,
    "tracking_ang_vel": 1.5,
    "upward": 3.0,
    "tracking_height": -1500.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "upright_orientation_l2": -0.5,
    "upright_zero_velocity": -0.05,
    "stand_still": -2.0,
    "joint_pos_penalty": -1.0,
    "leg_action_rate": -0.001,
    "wheel_action_rate": -0.001,
    "action_smoothness": -0.03,
    "leg_torques": -2.0e-4,
    "leg_dof_acc": -2.5e-7,
    "leg_power": -1.0e-4,
    "wheel_torques": -1.0e-4,
    "joint_mirror": -0.05,
    "dof_pos_limits": -5.0,
    "collision": -1.0,
    "contact_forces": -1.5e-4,
    "wheel_air_velocity": -1.0e-3,
    "leg_contact": -1.0,
    "wheel_contact_without_cmd": 0.1,
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
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=3.0)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=rewards.lin_vel_z, weight=-2.0)
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.05)
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=-1500.0,
        params={
            "command_name": "velocity_height",
            "sigma": 0.0025,
            "height_sensor_name": "base_height_sensor",
            "kernel": "l2",
            "use_upright_gate": False,
            "min_upright_gate": 0.0,
            "use_pose_end_gate": False,
            "use_inverted_free_upright_height_gate": True,
            "upright_gate_angle_deg": 30.0,
            "inverted_gate_angle_deg": 150.0,
        },
    )
    cfg.rewards["upright_orientation_l2"] = RewardTermCfg(
        func=rewards.recovery_upright_orientation_l2,
        weight=-0.5,
        params={
            "command_name": "velocity_height",
            "gate_start_deg": 60.0,
            "gate_full_deg": 20.0,
            "roll_scale_rad": 0.14,
            "pitch_scale_rad": 0.20,
            "roll_weight": 1.5,
            "pitch_weight": 1.0,
            "max_penalty": 6.0,
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
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.stand_still,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": 0.26,
            "height_tolerance": 40.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["joint_pos_penalty"] = RewardTermCfg(
        func=rewards.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["leg_action_rate"] = RewardTermCfg(
        func=rewards.leg_action_rate,
        weight=-0.001,
    )
    cfg.rewards["wheel_action_rate"] = RewardTermCfg(
        func=rewards.wheel_action_rate,
        weight=-0.001,
    )
    cfg.rewards["action_smoothness"] = RewardTermCfg(
        func=rewards.action_smoothness,
        weight=-0.03,
        params={
            "command_name": "velocity_height",
            "gate_start_deg": 90.0,
            "gate_full_deg": 30.0,
            "max_penalty": 80.0,
            "leg_scale": 1.0,
            "wheel_scale": 2.0,
        },
    )
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_dof_acc"] = RewardTermCfg(
        func=rewards.leg_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["wheel_torques"] = RewardTermCfg(
        func=rewards.wheel_torques,
        weight=-1.0e-4,
        params={"max_torque": 3.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["joint_mirror"] = RewardTermCfg(
        func=rewards.joint_mirror,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
    cfg.rewards["wheel_air_velocity"] = RewardTermCfg(
        func=rewards.wheel_air_velocity_penalty,
        weight=-1.0e-3,
        params={
            "sensor_name": "wheel_sensor",
            "force_threshold": 1.0,
            "velocity_scale": 1.0,
            "max_penalty": 10000.0,
            "recovery_active_only": False,
            "asset_cfg": SceneEntityCfg("robot"),
            "log_prefix": "Recovery",
        },
    )
    cfg.rewards["leg_contact"] = RewardTermCfg(
        func=rewards.leg_contact_penalty,
        weight=-1.0,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
        },
    )
    cfg.rewards["wheel_contact_without_cmd"] = RewardTermCfg(
        func=rewards.feet_contact_without_cmd,
        weight=0.1,
        params={
            "command_name": "velocity_height",
            "force_threshold": 1.0,
            "cmd_threshold": 0.1,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
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
        func=mdp_events.reset_root_state_recovery_discovery_mixed,
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
            "recovery_command_height": None,
            "recovery_state_cache_path": str(_RECOVERY_STATE_CACHE_PATH),
            "recovery_state_cache_split": "train",
            "source_curriculum_stages": [
                {
                    "iteration": 0,
                    "cache_ratio": 0.0,
                    "near_upright_ratio": 0.0,
                },
                {
                    "iteration": 300,
                    "cache_ratio": 0.0,
                    "near_upright_ratio": 0.0,
                },
                {
                    "iteration": 800,
                    "cache_ratio": 0.0,
                    "near_upright_ratio": 0.0,
                },
                {
                    "iteration": 1500,
                    "cache_ratio": 0.10,
                    "near_upright_ratio": 0.0,
                },
                {
                    "iteration": 2000,
                    "cache_ratio": 0.25,
                    "near_upright_ratio": 0.0,
                },
                {
                    "iteration": 2600,
                    "cache_ratio": 0.45,
                    "near_upright_ratio": 0.0,
                },
                {
                    "iteration": 3400,
                    "cache_ratio": 0.60,
                    "near_upright_ratio": 0.05,
                },
                {
                    "iteration": 4200,
                    "cache_ratio": 0.70,
                    "near_upright_ratio": 0.05,
                },
            ],
            "standard_curriculum_stages": [
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
            "height_conditioned_default": True,
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "joint_offset_range": 0.0,
                    "joint_vel_range": (0.0, 0.0),
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
    if not play:
        cfg.events["push_robots"] = EventTermCfg(
            func=mdp_events.push_robots,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        cfg.curriculum = {
            "commands_vel": CurriculumTermCfg(
                func=curriculums.commands_vel,
                params={
                    "command_name": "velocity_height",
                    "use_iterations": True,
                    "steps_per_policy_iter": 64,
                    "velocity_stages": [
                        {
                            "iteration": 0,
                            "lin_vel_x_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "iteration": 1500,
                            "lin_vel_x_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "iteration": 2000,
                            "lin_vel_x_range": (-0.5, 0.5),
                            "ang_vel_yaw_range": (-1.0, 1.0),
                        },
                        {
                            "iteration": 2600,
                            "lin_vel_x_range": (-1.0, 1.0),
                            "ang_vel_yaw_range": (-2.5, 2.5),
                        },
                        {
                            "iteration": 3400,
                            "lin_vel_x_range": (-1.6, 1.6),
                            "ang_vel_yaw_range": (-5.0, 5.0),
                        },
                        {
                            "iteration": 4200,
                            "lin_vel_x_range": (
                                -_DISCOVERY_MAX_LIN_VEL_X,
                                _DISCOVERY_MAX_LIN_VEL_X,
                            ),
                            "ang_vel_yaw_range": (
                                -_DISCOVERY_MAX_ANG_VEL_YAW,
                                _DISCOVERY_MAX_ANG_VEL_YAW,
                            ),
                        },
                    ],
                },
            ),
            "commands_height": CurriculumTermCfg(
                func=curriculums.commands_height,
                params={
                    "command_name": "velocity_height",
                    "use_iterations": True,
                    "steps_per_policy_iter": 64,
                    "height_stages": [
                        {
                            "iteration": 0,
                            "height_range": (0.24, 0.30),
                        },
                        {
                            "iteration": 300,
                            "height_range": (0.23, 0.32),
                        },
                        {
                            "iteration": 800,
                            "height_range": (0.215, 0.36),
                        },
                        {
                            "iteration": 1500,
                            "height_range": (0.195, 0.390),
                        },
                    ],
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculums.push_disturbance,
                params={
                    "use_iterations": True,
                    "steps_per_policy_iter": 64,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "iteration": 2600,
                            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        },
                        {
                            "iteration": 3400,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "iteration": 4200,
                            "velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)},
                        },
                    ],
                },
            ),
        }

    _configure_discovery_reward_contract(cfg)
    return cfg
