"""纯倒地自起站立任务环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import commands, curriculums, events, rewards, terminations

_ROBOT_DEFAULTS = SharedRobotConfig()
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """纯倒地自起到固定站立态的训练环境。"""

    cfg = flat_env_cfg(play=play)
    cfg.episode_length_s = 9999.0 if play else 10.0

    left_wheel_sensor_cfg = ContactSensorCfg(
        name="left_wheel_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    right_wheel_sensor_cfg = ContactSensorCfg(
        name="right_wheel_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    nonwheel_contact_sensor_cfg = ContactSensorCfg(
        name="nonwheel_contact_sensor",
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
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        left_wheel_sensor_cfg,
        right_wheel_sensor_cfg,
        nonwheel_contact_sensor_cfg,
    )

    cfg.commands["velocity_height"] = commands.RecoveryStandCommandCfg(
        target_height=_DEFAULT_STANDING_HEIGHT,
        resampling_time_range=(5.0, 5.0),
        lin_vel_x_range=(0.0, 0.0),
        ang_vel_yaw_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        roll_range=(0.0, 0.0),
        height_range=(_DEFAULT_STANDING_HEIGHT, _DEFAULT_STANDING_HEIGHT),
        standing_height_range=(_DEFAULT_STANDING_HEIGHT, _DEFAULT_STANDING_HEIGHT),
        standing_ratio=1.0,
        jump_prob=0.0,
        jump_height_range=(0.0, 0.0),
    )

    cfg.events = {
        "reset_scene_to_default": EventTermCfg(
            func=lambda env, env_ids: None,
            mode="reset",
        ),
        "reset_root_state": EventTermCfg(
            func=events.reset_root_state_full_angle_random,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "tilt_range": (0.0, 3.141592653589793),
                "tilt_axis_range": (-3.141592653589793, 3.141592653589793),
                "yaw_range": (-3.141592653589793, 3.141592653589793),
                "height_range": (0.26, 0.36),
                "clearance_range": (0.02, 0.05),
                "pitch_flip_prob": 0.25,
                "pitch_flip_tilt_range": (3.141592653589793, 3.141592653589793),
                "pitch_flip_axis_jitter_range": (0.0, 0.0),
                "pitch_flip_height_range": (0.10, 0.16),
                "pitch_flip_clearance_range": (0.0, 0.02),
                "lin_vel_range": (-0.15, 0.15),
                "ang_vel_range": (-0.8, 0.8),
                "curriculum_stages": [
                    {
                        "iteration": 0,
                        "pitch_flip_prob": 0.25,
                    },
                    {
                        "iteration": 90,
                        "pitch_flip_prob": 0.40,
                    },
                    {
                        "iteration": 140,
                        "pitch_flip_prob": 0.60,
                    },
                    {
                        "iteration": 220,
                        "pitch_flip_prob": 0.80,
                    },
                    {
                        "iteration": 400,
                        "pitch_flip_prob": 1.0,
                    },
                ],
                "use_iterations": True,
                "steps_per_policy_iter": 64,
                "mark_recovery_episode": True,
                "recovery_command_height": _DEFAULT_STANDING_HEIGHT,
            },
        ),
        "reset_joints": EventTermCfg(
            func=events.reset_joints,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_offset_range": 0.0,
                "hip_joint_offset_range": (-0.50, 0.55),
                "knee_joint_offset_range": (-0.45, 0.65),
                "joint_vel_range": (-0.8, 0.8),
                "joint_randomization_prob": 0.25,
                "curriculum_stages": [
                    {
                        "iteration": 0,
                        "joint_randomization_prob": 0.25,
                    },
                    {
                        "iteration": 90,
                        "joint_randomization_prob": 0.40,
                    },
                    {
                        "iteration": 140,
                        "joint_randomization_prob": 0.60,
                    },
                    {
                        "iteration": 220,
                        "joint_randomization_prob": 0.80,
                    },
                    {
                        "iteration": 400,
                        "joint_randomization_prob": 1.0,
                    },
                ],
                "use_iterations": True,
                "steps_per_policy_iter": 64,
            },
        ),
    }

    cfg.curriculum = {}
    if not play:
        cfg.events["push_robots"] = EventTermCfg(
            func=events.push_robots,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        cfg.curriculum["push_disturbance"] = CurriculumTermCfg(
            func=curriculums.push_disturbance,
            params={
                "use_iterations": True,
                "steps_per_policy_iter": 64,
                "push_stages": [
                    {
                        "iteration": 0,
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                    },
                    {
                        "iteration": 500,
                        "velocity_range": {
                            "x": (-0.25, 0.25),
                            "y": (-0.25, 0.25),
                            "yaw": (-0.25, 0.25),
                        },
                    },
                    {
                        "iteration": 1000,
                        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
                    },
                    {
                        "iteration": 1500,
                        "velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-0.8, 0.8)},
                    },
                    {
                        "iteration": 2200,
                        "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-1.0, 1.0)},
                    },
                ],
            },
        )

    success_params = {
        "left_wheel_sensor_name": "left_wheel_sensor",
        "right_wheel_sensor_name": "right_wheel_sensor",
        "nonwheel_sensor_name": "nonwheel_contact_sensor",
        "height_sensor_name": "base_height_sensor",
        "command_name": "velocity_height",
        "upright_angle_deg": 15.0,
        "max_abs_roll_deg": 3.0,
        "max_abs_pitch_deg": 5.0,
        "height_tolerance": 0.02,
        "ang_vel_threshold": 0.5,
        "lin_vel_threshold": 0.05,
        "wheel_speed_threshold": 0.05,
        "wheel_radius": 0.059,
        "force_threshold": 1.0,
        "stable_steps_required": 50,
        "min_episode_steps": 50,
        "min_wheel_lateral_distance": 0.40,
        "max_wheel_lateral_distance": 0.46,
        "max_wheel_fore_aft_offset": 0.03,
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
        "recovery_success": TerminationTermCfg(
            func=terminations.recovery_success,
            time_out=False,
            params=success_params,
        ),
    }

    cfg.rewards = {
        "upward": RewardTermCfg(func=rewards.upward, weight=3.0),
        "upward_progress": RewardTermCfg(
            func=rewards.upward_progress,
            weight=1.0,
            params={"delta_scale": 0.05, "max_reward": 2.0},
        ),
        "recovery_inverted_low_height": RewardTermCfg(
            func=rewards.recovery_inverted_low_height_penalty,
            weight=-1.5,
            params={
                "height_sensor_name": "base_height_sensor",
                "height_floor": 0.24,
                "height_scale": 0.05,
                "max_normalized_low_height": 3.0,
                "tilt_start_deg": 140.0,
                "tilt_full_deg": 170.0,
            },
        ),
        "recovery_height": RewardTermCfg(
            func=rewards.recovery_height,
            weight=3.0,
            params={
                "command_name": "velocity_height",
                "height_sensor_name": "base_height_sensor",
                "sigma": 0.001,
                "gate_start_deg": 60.0,
                "gate_full_deg": 15.0,
                "min_gate": 0.0,
            },
        ),
        "recovery_wheel_contact": RewardTermCfg(
            func=rewards.recovery_stand_wheel_contact,
            weight=2.0,
            params={
                "left_wheel_sensor_name": "left_wheel_sensor",
                "right_wheel_sensor_name": "right_wheel_sensor",
                "force_threshold": 1.0,
                "gate_start_deg": 120.0,
                "gate_full_deg": 45.0,
            },
        ),
        "recovery_nonwheel_clearance": RewardTermCfg(
            func=rewards.recovery_stand_nonwheel_clearance,
            weight=1.5,
            params={
                "nonwheel_sensor_name": "nonwheel_contact_sensor",
                "force_threshold": 1.0,
                "gate_start_deg": 75.0,
                "gate_full_deg": 15.0,
            },
        ),
        "recovery_stillness": RewardTermCfg(
            func=rewards.recovery_stand_stillness,
            weight=2.0,
            params={
                "gate_start_deg": 75.0,
                "gate_full_deg": 15.0,
                "base_speed_scale": 0.12,
                "wheel_speed_scale": 0.12,
            },
        ),
        "recovery_orientation": RewardTermCfg(
            func=rewards.recovery_stand_orientation_penalty,
            weight=-1.0,
            params={
                "command_name": "velocity_height",
                "gate_start_deg": 60.0,
                "gate_full_deg": 15.0,
                "roll_scale_rad": 0.08,
                "pitch_scale_rad": 0.12,
                "roll_weight": 1.5,
                "pitch_weight": 1.0,
                "max_penalty": 6.0,
            },
        ),
        "recovery_zero_velocity": RewardTermCfg(
            func=rewards.recovery_stand_zero_velocity_penalty,
            weight=-0.5,
            params={
                "wheel_radius": 0.059,
                "gate_start_deg": 60.0,
                "gate_full_deg": 15.0,
                "base_speed_scale": 0.05,
                "wheel_speed_scale": 0.05,
                "base_ang_vel_scale": 0.5,
                "action_saturation_threshold": 0.95,
                "max_penalty": 10.0,
            },
        ),
        "recovery_default_joint_pos": RewardTermCfg(
            func=rewards.recovery_stand_default_joint_pos,
            weight=-1.0,
            params={
                "gate_start_deg": 60.0,
                "gate_full_deg": 15.0,
                "max_penalty": 3.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "recovery_joint_mirror": RewardTermCfg(
            func=rewards.recovery_stand_joint_mirror,
            weight=-2.0,
            params={
                "gate_start_deg": 60.0,
                "gate_full_deg": 15.0,
                "hip_weight": 1.5,
                "knee_weight": 1.0,
                "max_penalty": 3.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "recovery_leg_alignment": RewardTermCfg(
            func=rewards.recovery_stand_leg_alignment,
            weight=-2.0,
            params={
                "gate_start_deg": 135.0,
                "gate_full_deg": 15.0,
                "min_lateral_distance": 0.40,
                "max_lateral_distance": 0.46,
                "max_fore_aft_offset": 0.03,
                "lateral_scale": 0.04,
                "fore_aft_scale": 0.03,
                "fore_aft_weight": 3.0,
                "max_penalty": 6.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "recovery_success_bonus": RewardTermCfg(
            func=rewards.recovery_success_bonus,
            weight=1.0,
            params={**success_params, "completion_bonus": 10.0},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.01),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }

    return cfg
