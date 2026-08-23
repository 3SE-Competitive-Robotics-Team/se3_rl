"""倒地自启 Discovery 阶段环境配置。"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_train.mdp import env_groups, terminations
from se3_train.mdp import events as mdp_events
from se3_train.tasks.recovery import curriculums, rewards
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RECOVERY_STATE_CACHE_PATH = (
    _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_closedchain_stair_v3_40k.npz"
)
_DISCOVERY_MAX_LIN_VEL_X = 1.89
_DISCOVERY_MAX_ANG_VEL_YAW = 9.41
_STEPS_PER_POLICY_ITER = 24
_TRAIN_ENV_GROUPS = {"loco": 0.5, "recover": 0.5}
_PLAY_ENV_GROUPS = {"loco": 0.0, "recover": 1.0}
RECOVERY_DISCOVERY_HISTORY_LENGTH = 5

_DISCOVERY_REWARD_WEIGHTS = {
    "tracking_lin_vel": 3.0,
    "tracking_ang_vel": 1.5,
    "upward": 3.0,
    "tracking_height": -1500.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "upright_orientation_l2": -0.5,
    "upright_zero_velocity": -0.20,
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


def _ungrouped_velocity_curriculum_cfg() -> CurriculumTermCfg:
    """保留不分组对照实验原有的固定速度课程。"""
    return CurriculumTermCfg(
        func=curriculums.commands_vel,
        params={
            "command_name": "velocity_height",
            "use_iterations": True,
            "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
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
    )


def _configure_discovery_reward_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    """显式装配 Discovery 奖励表，并在配置漂移时直接失败。"""

    cfg.rewards.clear()
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.tracking_lin_vel,
        weight=3.0,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.25,
            "sigma_stand": 0.02,
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
        weight=-0.20,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "gate_start_deg": 45.0,
            "gate_full_deg": 15.0,
            "base_speed_scale": 0.10,
            "wheel_speed_scale": 0.08,
            "base_ang_vel_scale": 0.30,
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
    inherited_events = dict(cfg.events)
    cfg.events = {
        "assign_env_groups": EventTermCfg(
            func=env_groups.AssignEnvGroups,
            mode="startup",
            params={"env_groups": _PLAY_ENV_GROUPS if play else _TRAIN_ENV_GROUPS},
        )
    }
    reset_scene_event = inherited_events.get("reset_scene_to_default")
    if reset_scene_event is not None:
        cfg.events["reset_scene_to_default"] = reset_scene_event

    root_common_params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "pos_xy_range": (-0.15, 0.15),
        "height_offset_range": (0.0, 0.02),
        "yaw_range": (-math.pi, math.pi),
        "roll_jitter_range": (-math.radians(5.0), math.radians(5.0)),
        "pitch_jitter_range": (-math.radians(5.0), math.radians(5.0)),
        "lin_vel_range": (0.0, 0.0),
        "ang_vel_range": (0.0, 0.0),
        "clearance_range": (0.001, 0.005),
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
        "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
    }
    cfg.events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_recovery_discovery_mixed,
        mode="reset",
        params={
            **root_common_params,
            "pose_weights_by_group": {
                "loco": (1.0, 0.0, 0.0, 0.0, 0.0),
                "recover": (0.0, 0.20, 0.20, 0.30, 0.30),
            },
            "recovery_state_cache_path": str(_RECOVERY_STATE_CACHE_PATH),
            "recovery_state_cache_split": "train",
            "source_curriculum_stages_by_group": {
                "recover": [
                    {"iteration": 0, "cache_ratio": 0.0},
                    {"iteration": 300, "cache_ratio": 0.0},
                    {"iteration": 800, "cache_ratio": 0.0},
                    {"iteration": 1500, "cache_ratio": 0.10},
                    {"iteration": 2000, "cache_ratio": 0.25},
                    {"iteration": 2600, "cache_ratio": 0.45},
                    {"iteration": 3400, "cache_ratio": 0.60},
                    {"iteration": 4200, "cache_ratio": 0.70},
                ],
            },
        },
    )

    joint_common_params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_offset_range": 0.0,
        "joint_vel_range": (0.0, 0.0),
        "wheel_joint_vel_range": (0.0, 0.0),
        "joint_randomization_prob": 0.0,
        "align_root_height_to_wheels": True,
        "height_conditioned_default": True,
        "use_iterations": True,
        "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
    }
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params={
            **joint_common_params,
            "randomization_group_names": ("recover",),
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
        },
    )

    for event_name, event_cfg in inherited_events.items():
        if event_name not in {
            "reset_scene_to_default",
            "reset_root_state",
            "reset_joints",
            "push_robots",
        }:
            cfg.events[event_name] = event_cfg

    critic_cfg = cfg.observations["critic"]
    cfg.observations["critic"] = replace(
        critic_cfg,
        terms={
            **critic_cfg.terms,
            "group_id": ObservationTermCfg(func=env_groups.env_group_id_obs),
        },
    )

    cfg.terminations["loco_bad_orientation"] = TerminationTermCfg(
        func=env_groups.FilteredTerminationWrapper,
        time_out=False,
        params={
            "group_names": ("loco",),
            "wrapped_term": {
                "func": terminations.bad_orientation_delayed,
                "params": {"limit_angle": math.radians(30.0), "max_steps": 25},
            },
        },
    )
    cfg.terminations["loco_base_contact"] = TerminationTermCfg(
        func=env_groups.FilteredTerminationWrapper,
        time_out=False,
        params={
            "group_names": ("loco",),
            "wrapped_term": {
                "func": terminations.base_contact,
                "params": {"sensor_name": "collision_sensor", "force_threshold": 1.0},
            },
        },
    )
    cfg.terminations["recover_stagnation"] = TerminationTermCfg(
        func=env_groups.FilteredTerminationWrapper,
        time_out=False,
        params={
            "group_names": ("recover",),
            "wrapped_term": {
                "func": terminations.recovery_stagnation,
                "params": {"max_steps": 300, "min_delta": 0.02},
            },
        },
    )
    if not play:
        cfg.events["push_robots_loco"] = EventTermCfg(
            func=env_groups.FilteredEventWrapper,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "group_names": ("loco",),
                "wrapped_term": {
                    "func": mdp_events.push_robots,
                    "params": {
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                },
            },
        )
        cfg.curriculum = {
            "commands_vel": CurriculumTermCfg(
                func=curriculums.GroupedRewardVelocityCurriculum,
                params={
                    "command_name": "velocity_height",
                    "loco_group_name": "loco",
                    "recover_group_name": "recover",
                    "loco_init_level": 0.15,
                    "recover_init_level": 0.10,
                    "level_step": 0.05,
                    "max_lin_vel_x": _DISCOVERY_MAX_LIN_VEL_X,
                    "max_ang_vel_yaw": _DISCOVERY_MAX_ANG_VEL_YAW,
                    "evaluation_window_steps": 1000,
                    "required_consecutive_windows": 2,
                    "min_episodes_per_window": 32,
                    "min_tracking_samples": 256,
                    "lin_score_threshold": 0.80,
                    "yaw_score_threshold": 0.70,
                    "recover_ready_threshold": 0.60,
                    "loco_survival_threshold": 0.98,
                    "loco_base_contact_max": 0.02,
                    "action_saturation_max": 0.35,
                    "leg_torque_saturation_max": 0.20,
                    "wheel_torque_saturation_max": 0.20,
                    "base_contact_termination_name": "loco_base_contact",
                },
            ),
            "commands_height": CurriculumTermCfg(
                func=curriculums.commands_height,
                params={
                    "command_name": "velocity_height",
                    "use_iterations": True,
                    "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
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
                    "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
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


def ungrouped_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成与分组实验同超参数、但使用旧混合 reset 的公平基线。"""

    cfg = env_cfg(play=play)
    grouped_events = dict(cfg.events)

    root_params = dict(grouped_events["reset_root_state"].params)
    root_params.pop("pose_weights_by_group", None)
    source_stages_by_group = root_params.pop("source_curriculum_stages_by_group", {})
    root_params.update(
        {
            "pose_weights": (0.08, 0.17, 0.17, 0.29, 0.29),
            "source_curriculum_stages": source_stages_by_group.get("recover"),
        }
    )
    if root_params["source_curriculum_stages"] is not None:
        root_params["source_curriculum_stages"] = [
            {
                **stage,
                "near_upright_ratio": 0.05 if int(stage["iteration"]) >= 3400 else 0.0,
            }
            for stage in root_params["source_curriculum_stages"]
        ]

    joint_params = dict(grouped_events["reset_joints"].params)
    joint_params.pop("randomization_group_names", None)

    cfg.events = {}
    reset_scene = grouped_events.get("reset_scene_to_default")
    if reset_scene is not None:
        cfg.events["reset_scene_to_default"] = reset_scene
    cfg.events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_recovery_discovery_mixed,
        mode="reset",
        params=root_params,
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params=joint_params,
    )
    for event_name in (
        "friction",
        "restitution",
        "base_mass",
        "inertia",
        "com",
        "pd_gains",
        "default_dof_pos",
        "snap_root_to_collision_clearance",
    ):
        event = grouped_events.get(event_name)
        if event is not None:
            cfg.events[event_name] = event
    grouped_push = grouped_events.get("push_robots_loco")
    if grouped_push is not None:
        wrapped_push = grouped_push.params["wrapped_term"]
        cfg.events["push_robots"] = EventTermCfg(
            func=wrapped_push["func"],
            mode="interval",
            interval_range_s=grouped_push.interval_range_s,
            params=dict(wrapped_push["params"]),
        )

    critic_cfg = cfg.observations["critic"]
    cfg.observations["critic"] = replace(
        critic_cfg,
        terms={name: term for name, term in critic_cfg.terms.items() if name != "group_id"},
    )
    cfg.terminations.pop("loco_bad_orientation", None)
    cfg.terminations.pop("loco_base_contact", None)
    cfg.terminations.pop("recover_stagnation", None)
    if not play:
        cfg.curriculum["commands_vel"] = _ungrouped_velocity_curriculum_cfg()
    return cfg


def history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成五帧 actor 历史观测版本，critic 与其余训练契约保持不变。"""

    cfg = env_cfg(play=play)
    actor_cfg = cfg.observations["actor"]
    cfg.observations["actor"] = replace(
        actor_cfg,
        history_length=RECOVERY_DISCOVERY_HISTORY_LENGTH,
        flatten_history_dim=True,
    )
    return cfg
