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
    steps_per_policy_iter = 64
    recovery_state_cache_path = "assets/recovery_states/serialleg_flat_v1.npz"

    def recovery_iter(iteration: int) -> int:
        """把 PPO iter 转成 env.common_step_counter 使用的 policy step。"""
        return int(iteration) * steps_per_policy_iter

    recovery_stages = (
        (
            {
                "step": 0,
                "prob": 0.20,
                "roll_range": (-0.35, 0.35),
                "pitch_range": (-0.35, 0.35),
                "height_range": (0.22, 0.32),
                "fallen_pose_prob": 0.35,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.05, 1.45),
                "fallen_pitch_abs_range": (1.05, 1.45),
                "fallen_coupled_range": (-0.20, 0.20),
                "fallen_height_range": (0.16, 0.25),
                "state_cache_path": recovery_state_cache_path,
                "state_cache_prob": 0.10,
            },
            {
                "step": recovery_iter(500),
                "prob": 0.35,
                "roll_range": (-0.50, 0.50),
                "pitch_range": (-0.50, 0.50),
                "height_range": (0.20, 0.32),
                "fallen_pose_prob": 0.55,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.20, 1.65),
                "fallen_pitch_abs_range": (1.20, 1.65),
                "fallen_coupled_range": (-0.25, 0.25),
                "fallen_height_range": (0.14, 0.24),
                "state_cache_path": recovery_state_cache_path,
                "state_cache_prob": 0.25,
            },
            {
                "step": recovery_iter(1200),
                "prob": 0.55,
                "roll_range": (-0.70, 0.70),
                "pitch_range": (-0.70, 0.70),
                "height_range": (0.18, 0.30),
                "fallen_pose_prob": 0.75,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.35, 2.00),
                "fallen_pitch_abs_range": (1.35, 2.00),
                "fallen_coupled_range": (-0.30, 0.30),
                "fallen_height_range": (0.12, 0.23),
                "state_cache_path": recovery_state_cache_path,
                "state_cache_prob": 0.40,
            },
            {
                "step": recovery_iter(2200),
                "prob": 0.75,
                "roll_range": (-1.00, 1.00),
                "pitch_range": (-1.00, 1.00),
                "height_range": (0.16, 0.30),
                "fallen_pose_prob": 0.85,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.45, 2.35),
                "fallen_pitch_abs_range": (1.45, 2.35),
                "fallen_coupled_range": (-0.40, 0.40),
                "fallen_height_range": (0.10, 0.22),
                "state_cache_path": recovery_state_cache_path,
                "state_cache_prob": 0.55,
            },
            {
                "step": recovery_iter(2600),
                "prob": 0.20,
                "roll_range": (-3.14, 3.14),
                "pitch_range": (-3.14, 3.14),
                "height_range": (0.08, 0.26),
                "fallen_pose_prob": 0.90,
                "fallen_roll_pose_prob": 0.50,
                "fallen_roll_abs_range": (1.45, 3.14),
                "fallen_pitch_abs_range": (1.45, 3.14),
                "fallen_coupled_range": (-0.80, 0.80),
                "fallen_height_range": (0.08, 0.22),
                "state_cache_path": recovery_state_cache_path,
                "state_cache_prob": 0.65,
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
                "state_cache_path": recovery_state_cache_path,
                "state_cache_prob": 0.65,
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
            "recovery_state_cache_path": recovery_state_cache_path,
            "recovery_state_cache_prob": 0.0,
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
            "recovery_terminate": False,
        },
    )
    cfg.terminations["recovery_stagnation"] = TerminationTermCfg(
        func=terminations.recovery_stagnation,
        time_out=False,
        params={"max_steps": 320, "min_delta": 0.015},
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

    cfg.rewards["tracking_orientation_l2"] = RewardTermCfg(
        func=rewards.tracking_orientation_l2,
        weight=-12.0,
        params={"command_name": "velocity_height", "ignore_recovery": True},
    )
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=2.49,
        params={
            "command_name": "velocity_height",
            "sigma": 0.05,
            "height_sensor_name": "base_height_sensor",
            "ignore_recovery": True,
        },
    )
    cfg.rewards["bad_tilt"] = RewardTermCfg(
        func=rewards.bad_tilt,
        weight=-6.0,
        params={
            "soft_limit_deg": 10.0,
            "hard_limit_deg": 30.0,
            "max_penalty": 4.0,
            "ignore_recovery": True,
        },
    )
    cfg.rewards["is_alive"] = RewardTermCfg(
        func=rewards.is_alive,
        weight=1.0,
        params={"recovery_scale": 0.0},
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
    cfg.rewards["recovery_upright"] = RewardTermCfg(
        func=rewards.recovery_upright,
        weight=6.0,
        params={
            "sensor_name": "wheel_sensor",
            "height_sensor_name": "base_height_sensor",
            "command_name": "velocity_height",
            "upright_angle_deg": 15.0,
            "height_tolerance": 0.05,
            "ang_vel_threshold": 1.5,
            "force_threshold": 1.0,
            "power": 1.0,
        },
    )
    cfg.rewards["recovery_height"] = RewardTermCfg(
        func=rewards.recovery_height,
        weight=1.5,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "base_height_sensor",
            "sigma": 0.04,
            "gate_start_deg": 90.0,
            "gate_full_deg": 25.0,
        },
    )
    cfg.rewards["recovery_progress"] = RewardTermCfg(
        func=rewards.recovery_progress,
        weight=3.0,
        params={
            "height_sensor_name": "base_height_sensor",
            "upright_delta_scale": 0.05,
            "height_delta_scale": 0.03,
            "max_reward": 4.0,
        },
    )
    cfg.rewards["recovery_stable_bonus"] = RewardTermCfg(
        func=rewards.recovery_stable_bonus,
        weight=6.0,
        params={
            "sensor_name": "wheel_sensor",
            "height_sensor_name": "base_height_sensor",
            "command_name": "velocity_height",
            "upright_angle_deg": 15.0,
            "height_tolerance": 0.05,
            "ang_vel_threshold": 1.5,
            "force_threshold": 1.0,
            "stable_steps_required": 32,
            "per_step_bonus": 0.1,
            "completion_bonus": 2.0,
        },
    )

    return cfg
