"""倒地自启任务环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, events, rewards


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """统一全姿态随机的反倒自起训练环境。"""

    cfg = flat_env_cfg(play=play)
    steps_per_policy_iter = 64

    cfg.events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_full_angle_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "tilt_range": (0.0, 3.141592653589793),
            "tilt_axis_range": (-3.141592653589793, 3.141592653589793),
            "yaw_range": (-3.141592653589793, 3.141592653589793),
            "height_range": (0.26, 0.36),
            "clearance_range": (0.02, 0.05),
            "lin_vel_range": (-0.15, 0.15),
            "ang_vel_range": (-0.8, 0.8),
            "use_iterations": True,
            "steps_per_policy_iter": steps_per_policy_iter,
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "tilt_range": (0.0, 1.05),
                    "lin_vel_range": (-0.05, 0.05),
                    "ang_vel_range": (-0.2, 0.2),
                },
                {
                    "iteration": 300,
                    "tilt_range": (0.0, 1.57),
                    "lin_vel_range": (-0.08, 0.08),
                    "ang_vel_range": (-0.35, 0.35),
                },
                {
                    "iteration": 700,
                    "tilt_range": (0.0, 2.36),
                    "lin_vel_range": (-0.12, 0.12),
                    "ang_vel_range": (-0.6, 0.6),
                },
                {
                    "iteration": 1100,
                    "tilt_range": (0.0, 3.141592653589793),
                    "lin_vel_range": (-0.15, 0.15),
                    "ang_vel_range": (-0.8, 0.8),
                },
            ],
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.0,
            "hip_joint_offset_range": (-0.10, 0.10),
            "knee_joint_offset_range": (-0.10, 0.10),
            "joint_vel_range": (-0.2, 0.2),
            "use_iterations": True,
            "steps_per_policy_iter": steps_per_policy_iter,
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "hip_joint_offset_range": (-0.10, 0.10),
                    "knee_joint_offset_range": (-0.10, 0.10),
                    "joint_vel_range": (-0.2, 0.2),
                },
                {
                    "iteration": 500,
                    "hip_joint_offset_range": (-0.20, 0.25),
                    "knee_joint_offset_range": (-0.20, 0.30),
                    "joint_vel_range": (-0.4, 0.4),
                },
                {
                    "iteration": 1000,
                    "hip_joint_offset_range": (-0.35, 0.40),
                    "knee_joint_offset_range": (-0.30, 0.45),
                    "joint_vel_range": (-0.6, 0.6),
                },
                {
                    "iteration": 1500,
                    "hip_joint_offset_range": (-0.50, 0.55),
                    "knee_joint_offset_range": (-0.45, 0.65),
                    "joint_vel_range": (-0.8, 0.8),
                },
            ],
        },
    )

    # 自起训练不再区分 normal/fallen/recovery episode，所有样本只受超时等硬错误终止约束。
    cfg.terminations.pop("bad_orientation", None)
    cfg.terminations.pop("leg_contact", None)
    cfg.terminations.pop("recovery_stagnation", None)
    cfg.events.pop("push_robots", None)
    if cfg.curriculum is not None:
        cfg.curriculum.pop("push_disturbance", None)

    # 去掉轴向姿态惩罚和旧 recovery 专属奖励；自起能力由 upward + 同一套 locomotion reward 学出来。
    for reward_name in (
        "tracking_orientation_l2",
        "bad_tilt",
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

    for reward_name in ("tracking_lin_vel", "tracking_ang_vel", "tracking_lin_yaw_joint"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["use_upright_gate"] = True

    # 对齐 robot_lab 的自起训练比例:upward 是主目标,动作平滑只保留很轻的正则。
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=3.0)
    cfg.rewards["upward_progress"] = RewardTermCfg(
        func=rewards.upward_progress,
        weight=1.0,
        params={"delta_scale": 0.05, "max_reward": 2.0},
    )
    cfg.rewards["action_rate"] = RewardTermCfg(func=rewards.action_rate, weight=-0.01)
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=2.49,
        params={
            "command_name": "velocity_height",
            "sigma": 0.05,
            "height_sensor_name": "base_height_sensor",
            "use_upright_gate": True,
            "min_upright_gate": 0.25,
        },
    )
    cfg.rewards["upright_leg_contact"] = RewardTermCfg(
        func=rewards.upright_leg_contact_penalty,
        weight=-5.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
            "min_upright_gate": 0.5,
        },
    )
    cfg.rewards["upright_wheel_contact"] = RewardTermCfg(
        func=rewards.upright_wheel_contact_penalty,
        weight=-3.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "force_threshold": 1.0,
            "min_upright_gate": 0.5,
        },
    )
    cfg.rewards["upright_wheel_slip"] = RewardTermCfg(
        func=rewards.upright_wheel_slip_penalty,
        weight=-0.8,
        params={
            "command_name": "velocity_height",
            "wheel_radius": 0.059,
            "idle_command_threshold": 0.08,
            "straight_yaw_threshold": 0.20,
            "min_upright_gate": 0.5,
            "idle_wheel_speed_scale": 0.35,
            "slip_speed_scale": 0.45,
            "base_speed_scale": 0.20,
            "max_penalty": 9.0,
        },
    )

    if cfg.curriculum is not None:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=curriculums.commands_vel,
            params={
                "command_name": "velocity_height",
                "use_iterations": True,
                "steps_per_policy_iter": steps_per_policy_iter,
                "velocity_stages": [
                    {
                        "iteration": 0,
                        "lin_vel_x_range": (0.0, 0.0),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 250,
                        "lin_vel_x_range": (-0.3, 0.3),
                        "ang_vel_yaw_range": (-0.3, 0.3),
                    },
                    {
                        "iteration": 600,
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
                        "lin_vel_x_range": (-1.5, 1.5),
                        "ang_vel_yaw_range": (-0.75, 0.75),
                    },
                    {
                        "iteration": 2200,
                        "lin_vel_x_range": (-1.5, 1.5),
                        "ang_vel_yaw_range": (-1.0, 1.0),
                    },
                ],
            },
        )

    return cfg
