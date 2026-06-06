"""倒地自启任务环境配置。"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, events, rewards

_ROBOT_DEFAULTS = SharedRobotConfig()
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height
_RECOVERY_STANDING_HEIGHT_RANGE = (0.20, 0.38)
_RECOVERY_INITIAL_HEIGHT_RANGE = (0.24, 0.30)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """统一全姿态随机的反倒自起训练环境。"""

    cfg = flat_env_cfg(play=play)
    cfg.actions["delayed_action"].height_conditioned_action_default = True
    cfg.actions["delayed_action"].action_default_command_name = "velocity_height"
    command_cfg = cfg.commands["velocity_height"]
    command_cfg.resampling_time_range = (10.0, 10.0)
    command_cfg.lin_vel_x_range = (-1.0, 1.0)
    command_cfg.ang_vel_yaw_range = (-1.0, 1.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    initial_height_range = (
        _RECOVERY_STANDING_HEIGHT_RANGE if play else _RECOVERY_INITIAL_HEIGHT_RANGE
    )
    command_cfg.height_range = initial_height_range
    command_cfg.standing_height_range = initial_height_range
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = 0.02
    command_cfg.jump_prob = 0.0

    cfg.events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_robotlab_full_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_xy_range": (-0.5, 0.5),
            "height_offset_range": (0.0, 0.2),
            "roll_range": (-3.141592653589793, 3.141592653589793),
            "pitch_range": (-3.141592653589793, 3.141592653589793),
            "yaw_range": (-3.141592653589793, 3.141592653589793),
            "lin_vel_range": (-0.5, 0.5),
            "ang_vel_range": (-0.5, 0.5),
            "clearance_range": (0.0, 0.05),
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "roll_range": (-math.radians(15.0), math.radians(15.0)),
                    "pitch_range": (-math.radians(15.0), math.radians(15.0)),
                },
                {
                    "iteration": 300,
                    "roll_range": (-math.radians(30.0), math.radians(30.0)),
                    "pitch_range": (-math.radians(30.0), math.radians(30.0)),
                },
                {
                    "iteration": 650,
                    "roll_range": (-math.radians(60.0), math.radians(60.0)),
                    "pitch_range": (-math.radians(60.0), math.radians(60.0)),
                },
                {
                    "iteration": 1000,
                    "roll_range": (-math.radians(120.0), math.radians(120.0)),
                    "pitch_range": (-math.radians(120.0), math.radians(120.0)),
                },
                {
                    "iteration": 1400,
                    "roll_range": (-math.pi, math.pi),
                    "pitch_range": (-math.pi, math.pi),
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
            "joint_randomization_prob": 0.25,
            "full_joint_randomization": True,
            "full_front_joint_offset_range": 1.0,
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

    # 自起训练不再区分 normal/fallen/recovery episode，所有样本只受超时等硬错误终止约束。
    cfg.terminations.pop("bad_orientation", None)
    cfg.terminations.pop("leg_contact", None)
    cfg.terminations.pop("recovery_stagnation", None)
    cfg.curriculum = {}
    if not play:
        cfg.curriculum = {
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
                            "height_range": (0.23, 0.31),
                        },
                        {
                            "iteration": 650,
                            "height_range": (0.22, 0.34),
                        },
                        {
                            "iteration": 1000,
                            "height_range": (0.21, 0.36),
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

    # 去掉轴向姿态惩罚和旧 recovery 专属奖励；高度奖励保留全姿态梯度。
    for reward_name in (
        "tracking_orientation_l2",
        "tracking_lin_yaw_joint",
        "bad_tilt",
        "angular_momentum",
        "wheel_torques",
        "flat_base_height",
        "flat_base_lin_vel_z",
        "flat_action_smoothness",
        "flat_wheel_contact",
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
    if "tracking_lin_vel" in cfg.rewards:
        cfg.rewards["tracking_lin_vel"].weight = 3.0
        cfg.rewards["tracking_lin_vel"].params["vz_weight"] = 0.0
    if "tracking_ang_vel" in cfg.rewards:
        cfg.rewards["tracking_ang_vel"].weight = 1.5

    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=-800.0,
        params={
            "command_name": "velocity_height",
            "sigma": 0.0025,
            "height_sensor_name": "base_height_sensor",
            "kernel": "l2",
            "use_upright_gate": False,
            "use_pose_end_gate": True,
            "upright_gate_angle_deg": 30.0,
            "inverted_gate_angle_deg": 150.0,
        },
    )

    # 自起训练中 upward 是主目标；高度奖励提供全姿态抬升梯度，动作平滑只保留很轻的正则。
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=2.0)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=rewards.lin_vel_z, weight=-2.0)
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.05)
    cfg.rewards["action_rate"] = RewardTermCfg(func=rewards.action_rate, weight=-0.05)
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_dof_acc"] = RewardTermCfg(
        func=rewards.leg_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
        params={"sensor_name": "collision_sensor", "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["contact_forces"] = RewardTermCfg(
        func=rewards.contact_forces,
        weight=-1.5e-4,
        params={
            "threshold": 35.0,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["upright_leg_contact"] = RewardTermCfg(
        func=rewards.upright_leg_contact_penalty,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
            "min_upright_gate": 0.0,
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
            "contact_force_threshold": 35.0,
            "action_saturation_threshold": 0.95,
            "active_rod_margin_warning": 0.05,
        },
    )

    return cfg
