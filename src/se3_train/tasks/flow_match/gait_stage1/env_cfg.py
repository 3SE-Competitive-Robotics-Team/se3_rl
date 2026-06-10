"""GAIT Stage1 平地基础步态环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_shared import TaskMode
from se3_shared.grounded_pose import solve_grounded_pose
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.flow_match.common import apply_task_mode_command, apply_task_mode_observations

from . import curriculums, events, rewards, terminations

GAIT_HEIGHT = 0.35
GAIT_MAX_SPEED = 1.05
GAIT_SWING_CLEARANCE_M = 0.04

_SPEED_END_STEP = 128_000
_PUSH_STAGE_STEPS = (192_000, 256_000)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 Stage1 平地 GAIT 训练环境。"""
    cfg = base_gait_env_cfg(play=play)
    if not play:
        cfg.events["push_robots"] = EventTermCfg(
            func=events.push_robots,
            mode="interval",
            interval_range_s=(4.0, 5.0),
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        cfg.curriculum["command_vel_linear"] = CurriculumTermCfg(
            func=curriculums.commands_vel_linear,
            params={
                "command_name": "velocity_height",
                "start_step": 0,
                "end_step": _SPEED_END_STEP,
                "start_lin_vel_x_range": (0.05, 0.12),
                "end_lin_vel_x_range": (0.05, GAIT_MAX_SPEED),
                "ang_vel_yaw_range": (0.0, 0.0),
            },
        )
        cfg.curriculum["push_disturbance"] = CurriculumTermCfg(
            func=curriculums.push_disturbance,
            params={
                "push_stages": [
                    {
                        "step": 0,
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                    },
                    {
                        "step": _PUSH_STAGE_STEPS[0],
                        "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                    },
                    {
                        "step": _PUSH_STAGE_STEPS[1],
                        "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                    },
                ],
            },
        )
    return cfg


def base_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造三阶段共用的纯 GAIT 机器人基础配置。"""
    cfg = flat_env_cfg(play=play)
    cfg.scene.entities["robot"] = get_serialleg_cfg(lock_wheels=True)

    cfg.events.pop("push_robots", None)
    cfg.curriculum = {}

    apply_task_mode_command(
        cfg,
        mode_probabilities=(0.0, 1.0, 0.0, 0.0, 0.0),
        jump_prob=0.0,
        enable_mode_switch=False,
    )
    apply_task_mode_observations(cfg)

    gait_grounded_pose = solve_grounded_pose(
        GAIT_HEIGHT,
        keep_wheel_x=False,
        align_com_x=True,
    )
    if not gait_grounded_pose.success:
        raise ValueError(
            "无法求解 GAIT 初始触地姿态: "
            f"base_height={GAIT_HEIGHT}, message={gait_grounded_pose.message}"
        )

    cfg.actions["delayed_action"].leg_scale = 0.25
    cfg.actions["delayed_action"].wheel_scale = 0.0
    cfg.actions["delayed_action"].wheel_lock_damping = 0.0
    cfg.actions["delayed_action"].freeze_wheels = True
    cfg.commands["velocity_height"].height_range = (GAIT_HEIGHT, GAIT_HEIGHT)
    cfg.commands["velocity_height"].standing_height_range = (GAIT_HEIGHT, GAIT_HEIGHT)
    cfg.events["reset_root_state"].params["base_height"] = GAIT_HEIGHT
    cfg.events["reset_joints"].params["joint_pos_override"] = gait_grounded_pose.q6
    cfg.events["reset_joints"].params["update_default_joint_pos"] = True
    cfg.commands["velocity_height"].jump_height_range = (0.0, 0.0)
    cfg.commands["velocity_height"].lin_vel_x_range = (
        (0.05, GAIT_MAX_SPEED) if play else (0.05, 0.12)
    )
    cfg.commands["velocity_height"].ang_vel_yaw_range = (0.0, 0.0)
    cfg.commands["velocity_height"].pitch_range = (0.0, 0.0)
    cfg.commands["velocity_height"].roll_range = (0.0, 0.0)
    cfg.commands["velocity_height"].standing_ratio = 0.0
    cfg.commands["velocity_height"].lin_vel_deadband = 0.03
    cfg.commands["velocity_height"].yaw_deadband = 0.1
    if play:
        cfg.commands["velocity_height"].debug_vis = True

    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.BodyContactDelayed(),
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 0.2,
            "delay_steps": 8,
        },
    )
    cfg.terminations["base_link_contact"] = TerminationTermCfg(
        func=terminations.base_link_contact_delayed,
        time_out=False,
        params={
            "sensor_name": "collision_sensor",
            "force_threshold": 1.0,
            "delay_steps": 20,
        },
    )
    cfg.terminations["bad_orientation"] = TerminationTermCfg(
        func=terminations.bad_orientation_delayed,
        time_out=False,
        params={"limit_angle": 0.5236, "max_steps": 16},
    )
    cfg.terminations["low_base_height"] = TerminationTermCfg(
        func=terminations.gait_low_base_height_delayed,
        time_out=False,
        params={
            "sensor_name": "base_height_sensor",
            "min_height": 0.26,
            "max_steps": 25,
        },
    )

    cfg.rewards = gait_rewards()
    cfg.episode_length_s = 20.0
    return cfg


def gait_rewards() -> dict[str, RewardTermCfg]:
    """构造 GAIT 三阶段共用的基础奖励。"""
    return {
        "is_terminated": RewardTermCfg(func=rewards.is_terminated, weight=-100.0),
        "gait_tracking_lin_vel": RewardTermCfg(
            func=rewards.mode_tracking_lin_vel,
            weight=4.0,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.2,
                "sigma_stand": 0.08,
                "vz_weight": 1.0,
                "modes": (int(TaskMode.GAIT),),
            },
        ),
        "gait_tracking_ang_vel": RewardTermCfg(
            func=rewards.mode_tracking_ang_vel,
            weight=0.1,
            params={
                "command_name": "velocity_height",
                "sigma": 0.25,
                "modes": (int(TaskMode.GAIT),),
            },
        ),
        "gait_tracking_ang_vel_l2": RewardTermCfg(
            func=rewards.tracking_ang_vel_l2,
            weight=-1.5,
            params={"command_name": "velocity_height"},
        ),
        "tracking_orientation_l2": RewardTermCfg(
            func=rewards.tracking_orientation_l2,
            weight=-20.0,
            params={"command_name": "velocity_height"},
        ),
        "flat_base_height": RewardTermCfg(
            func=rewards.flat_base_height_penalty_no_jump,
            weight=-12.0,
            params={
                "command_name": "velocity_height",
                "sigma": 0.03,
                "height_sensor_name": "base_height_sensor",
            },
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-11.0,
            params={"soft_limit_deg": 8.0, "hard_limit_deg": 24.0, "max_penalty": 4.0},
        ),
        "gait_low_base_height_barrier": RewardTermCfg(
            func=rewards.gait_low_base_height_barrier,
            weight=-10.0,
            params={
                "command_name": "velocity_height",
                "height_sensor_name": "base_height_sensor",
                "soft_min_height": GAIT_HEIGHT,
                "hard_min_height": 0.26,
                "max_penalty": 4.0,
            },
        ),
        "base_link_contact_penalty": RewardTermCfg(
            func=terminations.BodyContactPenalty(),
            weight=-100.0,
            params={
                "sensor_name": "collision_sensor",
                "force_threshold": 1.0,
            },
        ),
        "leg_contact_penalty": RewardTermCfg(
            func=terminations.BodyContactPenalty(),
            weight=-160.0,
            params={
                "sensor_name": "leg_contact_sensor",
                "force_threshold": 0.2,
            },
        ),
        "lin_vel_z": RewardTermCfg(func=rewards.lin_vel_z, weight=-0.5),
        "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.5),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-8.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-5.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-8.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=rewards.dof_pos_limits,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.06),
        "gait_no_wheel_drive": RewardTermCfg(
            func=rewards.gait_no_wheel_drive,
            weight=-8.0,
            params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "gait_leg_contact_force": RewardTermCfg(
            func=rewards.gait_leg_contact_force,
            weight=-160.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "leg_contact_sensor",
                "force_scale": 5.0,
                "contact_threshold": 0.2,
                "contact_event_scale": 0.5,
            },
        ),
        "gait_natural_swing_clearance": RewardTermCfg(
            func=rewards.gait_natural_swing_clearance,
            weight=2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "leg_sensor_name": "leg_contact_sensor",
                "target_clearance": GAIT_SWING_CLEARANCE_M,
                "balance_window_s": 0.6,
                "contact_force_threshold": 1.0,
                "leg_contact_force_threshold": 0.2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "gait_single_support_contact": RewardTermCfg(
            func=rewards.gait_single_support_contact,
            weight=1.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "contact_force_threshold": 1.0,
            },
        ),
        "gait_single_support_air_time": RewardTermCfg(
            func=rewards.gait_single_support_air_time,
            weight=2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "target_air_time": 0.04,
                "min_command": 0.05,
                "contact_force_threshold": 1.0,
            },
        ),
        "gait_stuck_stance_penalty": RewardTermCfg(
            func=rewards.gait_stuck_stance_penalty,
            weight=-8.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "grace_time_s": 0.22,
                "contact_force_threshold": 1.0,
            },
        ),
        "gait_swing_side_balance_penalty": RewardTermCfg(
            func=rewards.gait_swing_side_balance_penalty,
            weight=-12.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "window_s": 0.6,
                "min_single_support_ratio": 0.1,
                "contact_force_threshold": 1.0,
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
            },
        ),
        "gait_short_air_time_penalty": RewardTermCfg(
            func=rewards.gait_short_air_time_penalty,
            weight=-2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "min_air_time": 0.04,
                "min_command": 0.05,
                "contact_force_threshold": 1.0,
            },
        ),
        "gait_risk_conditioned_stability": RewardTermCfg(
            func=rewards.gait_risk_conditioned_stability,
            weight=-3.0,
            params={
                "command_name": "velocity_height",
                "base_scale": 0.25,
                "speed_start": 0.35,
                "speed_full": 0.90,
                "speed_scale": 0.80,
                "tilt_start_deg": 10.0,
                "tilt_full_deg": 24.0,
                "tilt_scale": 0.35,
                "ang_vel_weight": 0.10,
                "max_penalty": 3.0,
            },
        ),
        "gait_safe_tracking_lin_vel": RewardTermCfg(
            func=rewards.gait_safe_tracking_lin_vel,
            weight=0.8,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.18,
                "sigma_stand": 0.08,
                "vz_weight": 1.0,
                "tilt_safe_deg": 7.0,
                "tilt_unsafe_deg": 18.0,
                "ang_vel_safe": 0.8,
                "ang_vel_unsafe": 2.0,
            },
        ),
        "gait_action_smoothness": RewardTermCfg(
            func=rewards.gait_action_smoothness,
            weight=-0.08,
            params={
                "command_name": "velocity_height",
                "max_penalty": 80.0,
            },
        ),
        "gait_touchdown_softness": RewardTermCfg(
            func=rewards.gait_touchdown_softness,
            weight=-8.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "allowed_down_vel": 0.12,
                "max_penalty": 4.0,
                "contact_force_threshold": 1.0,
            },
        ),
        "gait_touchdown_support_alignment": RewardTermCfg(
            func=rewards.gait_touchdown_support_alignment,
            weight=-4.0,
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
    }


__all__ = ["GAIT_HEIGHT", "GAIT_MAX_SPEED", "base_gait_env_cfg", "env_cfg", "gait_rewards"]
