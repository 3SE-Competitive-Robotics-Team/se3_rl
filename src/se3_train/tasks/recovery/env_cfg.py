"""倒地自启任务环境配置。"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, events, rewards

_ROBOT_DEFAULTS = SharedRobotConfig()
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height
_RECOVERY_STANDING_HEIGHT_RANGE = (0.195, 0.390)
_RECOVERY_INITIAL_HEIGHT_RANGE = (0.24, 0.30)
_RECOVERY_LEG_ACTION_SCALE = 0.25
_RECOVERY_WHEEL_KD = 0.08
_RECOVERY_COMMAND_WHEEL_RADIUS = 0.060
_RECOVERY_COMMAND_HALF_TRACK = 0.200725
_RECOVERY_COMMAND_WHEEL_SPEED_FRACTION = 0.70
_RECOVERY_COMMAND_LIN_VEL_X_MAX = 1.5
_RECOVERY_COMMAND_ANG_VEL_YAW_MAX = 1.0
_RECOVERY_WHEEL_JOINT_VEL_RANGE = (-10.0, 10.0)
_RECOVERY_STANDING_RATIO = 0.05
_TRACKING_UPRIGHT_FULL_COS = math.cos(math.radians(15.0))


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """统一全姿态随机的反倒自起训练环境。"""

    cfg = flat_env_cfg(play=play)
    cfg.scene.entities["robot"] = get_serialleg_cfg(wheel_kd_override=_RECOVERY_WHEEL_KD)
    cfg.sim.nconmax = 64
    cfg.sim.njmax = 256
    action_cfg = cfg.actions["delayed_action"]
    action_cfg.leg_scales = (_RECOVERY_LEG_ACTION_SCALE,) * 4
    action_cfg.action_clip = None
    command_cfg = cfg.commands["velocity_height"]
    command_cfg.resampling_time_range = (10.0, 10.0)
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    initial_height_range = (
        _RECOVERY_STANDING_HEIGHT_RANGE if play else _RECOVERY_INITIAL_HEIGHT_RANGE
    )
    command_cfg.height_range = initial_height_range
    command_cfg.standing_height_range = initial_height_range
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = _RECOVERY_STANDING_RATIO
    command_cfg.constrain_diff_drive_commands = True
    command_cfg.diff_drive_wheel_radius = _RECOVERY_COMMAND_WHEEL_RADIUS
    command_cfg.diff_drive_half_track = _RECOVERY_COMMAND_HALF_TRACK
    command_cfg.diff_drive_max_wheel_speed = _ROBOT_DEFAULTS.action_scale[
        JointGroup.WHEEL_ACTUATORS[0]
    ]
    command_cfg.diff_drive_wheel_speed_fraction = _RECOVERY_COMMAND_WHEEL_SPEED_FRACTION
    command_cfg.jump_prob = 0.0
    command_cfg.enable_jump_lifecycle = False
    command_cfg.enable_jump_metrics = False

    cfg.events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_robotlab_full_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_xy_range": (-0.5, 0.5),
            "height_offset_range": (0.0, 0.02),
            "roll_range": (-3.141592653589793, 3.141592653589793),
            "pitch_range": (-3.141592653589793, 3.141592653589793),
            "yaw_range": (-3.141592653589793, 3.141592653589793),
            "lin_vel_range": (-0.15, 0.15),
            "ang_vel_range": (-0.6, 0.6),
            "clearance_range": (0.0, 0.01),
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "roll_range": (-math.radians(15.0), math.radians(15.0)),
                    "pitch_range": (-math.radians(15.0), math.radians(15.0)),
                    "roll_side_prob": 0.10,
                    "pitch_inverted_prob": 0.0,
                },
                {
                    "iteration": 300,
                    "roll_range": (-math.radians(30.0), math.radians(30.0)),
                    "pitch_range": (-math.radians(30.0), math.radians(30.0)),
                    "roll_side_prob": 0.15,
                    "pitch_inverted_prob": 0.05,
                },
                {
                    "iteration": 650,
                    "roll_range": (-math.radians(60.0), math.radians(60.0)),
                    "pitch_range": (-math.radians(60.0), math.radians(60.0)),
                    "roll_side_prob": 0.20,
                    "pitch_inverted_prob": 0.10,
                },
                {
                    "iteration": 1000,
                    "roll_range": (-math.radians(120.0), math.radians(120.0)),
                    "pitch_range": (-math.radians(120.0), math.radians(120.0)),
                    "roll_side_prob": 0.25,
                    "pitch_inverted_prob": 0.20,
                },
                {
                    "iteration": 1400,
                    "roll_range": (-math.pi, math.pi),
                    "pitch_range": (-math.pi, math.pi),
                    "roll_side_prob": 0.25,
                    "pitch_inverted_prob": 0.25,
                },
            ],
            "use_iterations": True,
            "steps_per_policy_iter": 64,
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.0,
            "joint_vel_range": (-0.8, 0.8),
            "wheel_joint_vel_range": _RECOVERY_WHEEL_JOINT_VEL_RANGE,
            "joint_randomization_prob": 0.25,
            "full_joint_randomization": True,
            "full_front_joint_offset_range": math.pi,
            "full_active_rod_angle_range": _ROBOT_DEFAULTS.active_rod_angle_limits,
            "align_root_height_to_wheels": True,
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "joint_randomization_prob": 0.15,
                },
                {
                    "iteration": 300,
                    "joint_randomization_prob": 0.25,
                },
                {
                    "iteration": 650,
                    "joint_randomization_prob": 0.45,
                },
                {
                    "iteration": 1000,
                    "joint_randomization_prob": 0.70,
                },
                {
                    "iteration": 1400,
                    "joint_randomization_prob": 1.0,
                },
            ],
            "use_iterations": True,
            "steps_per_policy_iter": 64,
        },
    )
    cfg.events["snap_root_to_collision_clearance"] = EventTermCfg(
        func=events.snap_root_to_collision_clearance,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "clearance_range": (0.001, 0.005),
            "max_downward_adjustment": 0.5,
            "max_upward_adjustment": 0.05,
            "command_name": "velocity_height",
        },
    )

    # 自起训练不再区分 normal/fallen/recovery episode，所有样本只受超时等硬错误终止约束。
    cfg.terminations.pop("bad_orientation", None)
    cfg.terminations.pop("leg_contact", None)
    cfg.terminations.pop("recovery_stagnation", None)
    if "catastrophic_state" in cfg.terminations:
        cfg.terminations["catastrophic_state"].params["max_leg_pos_error"] = None
    cfg.curriculum = {}
    if not play:
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
                            "iteration": 300,
                            "lin_vel_x_range": (-0.3, 0.3),
                            "ang_vel_yaw_range": (-0.3, 0.3),
                        },
                        {
                            "iteration": 650,
                            "lin_vel_x_range": (-0.5, 0.5),
                            "ang_vel_yaw_range": (-0.5, 0.5),
                        },
                        {
                            "iteration": 1000,
                            "lin_vel_x_range": (-1.0, 1.0),
                            "ang_vel_yaw_range": (-0.5, 0.5),
                        },
                        {
                            "iteration": 1500,
                            "lin_vel_x_range": (
                                -_RECOVERY_COMMAND_LIN_VEL_X_MAX,
                                _RECOVERY_COMMAND_LIN_VEL_X_MAX,
                            ),
                            "ang_vel_yaw_range": (-0.75, 0.75),
                        },
                        {
                            "iteration": 2200,
                            "lin_vel_x_range": (
                                -_RECOVERY_COMMAND_LIN_VEL_X_MAX,
                                _RECOVERY_COMMAND_LIN_VEL_X_MAX,
                            ),
                            "ang_vel_yaw_range": (
                                -_RECOVERY_COMMAND_ANG_VEL_YAW_MAX,
                                _RECOVERY_COMMAND_ANG_VEL_YAW_MAX,
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
                            "height_range": _RECOVERY_INITIAL_HEIGHT_RANGE,
                        },
                        {
                            "iteration": 300,
                            "height_range": (0.23, 0.32),
                        },
                        {
                            "iteration": 650,
                            "height_range": (0.22, 0.35),
                        },
                        {
                            "iteration": 1000,
                            "height_range": (0.205, 0.37),
                        },
                        {
                            "iteration": 1400,
                            "height_range": _RECOVERY_STANDING_HEIGHT_RANGE,
                        },
                    ],
                },
            ),
        }
    if not play:
        cfg.events["push_robots"] = EventTermCfg(
            func=events.push_robots,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # 去掉轴向姿态惩罚和旧 recovery 专属奖励；高度项沿用 flat 的二次误差惩罚。
    for reward_name in (
        "tracking_orientation_l2",
        "tracking_lin_yaw_joint",
        "bad_tilt",
        "angular_momentum",
        "flat_base_height",
        "flat_base_lin_vel_z",
        "flat_action_smoothness",
        "flat_wheel_contact",
        "same_feet_x_position",
        "flat_leg_contact",
        "flat_wheel_ground_slip",
        "flat_wheel_center_alignment",
        "idle_wheel_motion",
        "is_alive",
    ):
        cfg.rewards.pop(reward_name, None)

    for reward_name in ("tracking_lin_vel", "tracking_ang_vel"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["use_upright_gate"] = True
            cfg.rewards[reward_name].params["tracking_upright_full_cos"] = (
                _TRACKING_UPRIGHT_FULL_COS
            )
    if "tracking_lin_vel" in cfg.rewards:
        cfg.rewards["tracking_lin_vel"].weight = 3.0
        cfg.rewards["tracking_lin_vel"].params["vz_weight"] = 0.0
    if "tracking_ang_vel" in cfg.rewards:
        cfg.rewards["tracking_ang_vel"].weight = 1.5

    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.flat_base_height_penalty_no_jump,
        weight=-4.0,
        params={
            "command_name": "velocity_height",
            "sigma": 0.05,
            "height_sensor_name": "base_height_sensor",
        },
    )

    # 自起训练中 upward 是主目标；高度项只惩罚目标高度误差，动作正则负责压住饱和翻身。
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=2.0)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=rewards.lin_vel_z, weight=-2.0)
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.05)
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
        weight=-0.25,
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
    cfg.rewards.pop("action_rate", None)
    cfg.rewards["leg_action_rate"] = RewardTermCfg(
        func=rewards.leg_action_rate,
        weight=-0.05,
    )
    cfg.rewards["wheel_action_rate"] = RewardTermCfg(
        func=rewards.wheel_action_rate,
        weight=-0.20,
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
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.stand_still,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": _DEFAULT_STANDING_HEIGHT,
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
        weight=-3.0e-4,
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

    return cfg
