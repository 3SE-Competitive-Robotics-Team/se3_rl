# ruff: noqa: F401
from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.rewards import is_alive as mjlab_is_alive
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from se3_train.mdp.actions import SerialLegDelayedActionCfg
from se3_train.mdp.jump_trajectories import DEFAULT_JUMP_TRAJ_HEIGHTS, DEFAULT_JUMP_TRAJ_PATHS
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat import events, observations
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.jump_pretrain import commands, curriculums, rewards, terminations

_DEFAULT_STANDING_HEIGHT = 0.22
_STANDING_HEIGHT_RANGE = (0.20, 0.32)


def _apply_jump_command(
    cfg: ManagerBasedRlEnvCfg,
    jump_prob: float = 0.0,
    jump_height_range: tuple[float, float] = (0.1, 0.3),
) -> None:
    """将 velocity_height 指令替换为 JumpCommandTerm(扩展 8 维)。

    保留所有速度/姿态/高度配置,仅升级指令类型。
    参考阶段和总帧数从共享轨迹库读取。
    """
    cfg.commands = {
        "velocity_height": commands.JumpCommandCfg(
            resampling_time_range=(5.0, 5.0),
            jump_prob=jump_prob,
            jump_height_range=jump_height_range,
            traj_paths=DEFAULT_JUMP_TRAJ_PATHS,
            traj_target_heights=DEFAULT_JUMP_TRAJ_HEIGHTS,
        ),
    }


def _apply_jump_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """在观测中追加 3 维跳跃指令(jump_flag, jump_target_height, jump_phase)。

    actor 从 31 维扩展到 32 维(+jump_phase);critic 同步扩展。
    """
    jump_obs_term = ObservationTermCfg(func=observations.jump_commands_obs)

    actor_terms = dict(cfg.observations["actor"].terms)
    actor_terms["jump_commands"] = jump_obs_term
    cfg.observations["actor"] = replace(cfg.observations["actor"], terms=actor_terms)

    critic_terms = dict(cfg.observations["critic"].terms)
    critic_terms["jump_commands"] = jump_obs_term
    cfg.observations["critic"] = replace(cfg.observations["critic"], terms=critic_terms)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """跳跃预训练环境配置(从行走 checkpoint fine-tune)。

    在行走环境基础上叠加:
    - JumpCommandTerm(8 维指令)
    - 稀疏高度奖励:轮组最大高度和 base 最大高度接近目标
    - 轻量姿态、左右对称和关节限位约束
    - 膝关节冲击终止
    """
    cfg = flat_env_cfg(play=play)

    # 跳跃任务先学习起跳和落地本体能力;通用 push 扰动留到后续鲁棒性阶段。
    cfg.events.pop("push_robots", None)
    cfg.curriculum.pop("push_disturbance", None)

    # 替换指令
    _apply_jump_command(cfg, jump_prob=0.02)

    # 追加跳跃观测
    _apply_jump_observations(cfg)

    # 接地行为统一使用 flat_wheel_contact 惩罚表达,避免正奖励和惩罚同时控制同一行为。
    cfg.rewards.pop("feet_contact_without_cmd", None)

    # PreTrain 平地段固定为 0.22m 蹲姿；跳跃目标高度仍单独采样，避免把站高误当成跳高。
    cfg.commands["velocity_height"].height_range = (
        _DEFAULT_STANDING_HEIGHT,
        _DEFAULT_STANDING_HEIGHT,
    )
    cfg.commands["velocity_height"].standing_height_range = (
        _DEFAULT_STANDING_HEIGHT,
        _DEFAULT_STANDING_HEIGHT,
    )
    cfg.commands["velocity_height"].jump_height_range = (0.1, 0.3)
    cfg.commands["velocity_height"].rsi_takeoff_prob = 0.0
    cfg.commands["velocity_height"].rsi_random_frame = False
    cfg.commands["velocity_height"].jump_cool_down_steps = 100

    # jump_flag=1 时屏蔽与起跳冲突的行走奖励
    # tracking_lin_vel 里有 vz2 惩罚会压制起跳,必须在 jump_flag=1 时清零
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.tracking_lin_vel_no_jump,
        weight=2.73,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.25,
            "sigma_stand": 0.1,
            "vz_weight": 2.0,
        },
    )
    # stand_still 惩罚腿部偏离默认位置,会阻止起跳动作
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.stand_still_no_jump,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": _DEFAULT_STANDING_HEIGHT,
            "height_tolerance": 40.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 力矩上限已放开至 stall_torque(40 N·m),起跳瞬间峰值力矩若被惩罚会压制起跳
    # jump_flag=1 时豁免 leg_torques 和 leg_power 惩罚
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques_no_jump,
        weight=-2.0e-4,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power_no_jump,
        weight=-1.03e-4,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # tracking_height 与跳跃轨迹 pose tracking 梯度相反,jump_flag=1 时必须清零
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height_no_jump,
        weight=2.49,
        params={
            "command_name": "velocity_height",
            "sigma": 0.05,
            "height_sensor_name": "base_height_sensor",
        },
    )

    # 偏航速度跟踪:起跳/飞行期姿态变化正常,不应持续惩罚
    cfg.rewards["tracking_ang_vel"] = RewardTermCfg(
        func=rewards.tracking_ang_vel_no_jump,
        weight=1.73,
        params={"command_name": "velocity_height", "sigma": 0.25},
    )

    cfg.rewards["tracking_orientation_l2"] = RewardTermCfg(
        func=rewards.tracking_orientation_l2_no_jump,
        weight=-6.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
        func=rewards.flat_orientation_l2_no_jump,
        weight=-24.0,
        params={"command_name": "velocity_height"},
    )

    # action_rate:jump 期保持上一轮成功配置,flat/idle 期额外压低动作抖动。
    cfg.rewards["action_rate"] = RewardTermCfg(
        func=rewards.action_rate_no_jump,
        weight=-0.48,
        params={
            "command_name": "velocity_height",
            "idle_command_threshold": 0.08,
            "idle_scale": 1.8,
            "moving_scale": 1.1,
            "max_penalty": 80.0,
        },
    )
    cfg.rewards["idle_wheel_motion"] = RewardTermCfg(
        func=rewards.idle_wheel_motion_penalty_no_jump,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "wheel_radius": 0.059,
            "idle_command_threshold": 0.08,
            "contact_force_threshold": 1.0,
            "base_speed_scale": 0.18,
            "wheel_speed_scale": 0.22,
            "max_penalty": 9.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["flat_wheel_contact"] = RewardTermCfg(
        func=rewards.flat_wheel_contact_penalty_no_jump,
        weight=-3.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "idle_command_threshold": 0.08,
            "contact_force_threshold": 1.0,
        },
    )
    cfg.rewards["flat_action_smoothness"] = RewardTermCfg(
        func=rewards.action_smoothness_no_jump,
        weight=-0.04,
        params={
            "command_name": "velocity_height",
            "idle_command_threshold": 0.08,
            "idle_scale": 1.5,
            "moving_scale": 0.5,
            "max_penalty": 80.0,
        },
    )
    cfg.rewards["jump_action_rate"] = RewardTermCfg(
        func=rewards.action_rate_jump,
        weight=-0.04,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["standing_joint_mirror"] = RewardTermCfg(
        func=rewards.standing_joint_mirror_no_jump,
        weight=-10.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "hip_weight": 4.0,
            "knee_weight": 1.5,
            "low_speed_sigma": 0.35,
            "low_speed_floor": 0.08,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["wheel_distance"] = RewardTermCfg(
        func=rewards.wheel_distance_regularization,
        weight=-4.0,
        params={
            "command_name": "velocity_height",
            "min_lateral_distance": 0.40,
            "max_lateral_distance": 0.46,
            "max_fore_aft_offset": 0.02,
            "lateral_scale": 0.04,
            "fore_aft_scale": 0.025,
            "fore_aft_weight": 2.0,
            "standing_scale": 1.4,
            "grounded_scale": 0.5,
            "takeoff_scale": 1.2,
            "air_scale": 1.0,
            "landing_scale": 1.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_joint_mirror"] = RewardTermCfg(
        func=rewards.jump_joint_mirror,
        weight=-8.0,
        params={
            "command_name": "velocity_height",
            "hip_weight": 4.0,
            "knee_weight": 1.5,
            "grounded_scale": 1.0,
            "takeoff_scale": 4.0,
            "air_scale": 3.0,
            "landing_scale": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_action_mirror"] = RewardTermCfg(
        func=rewards.jump_action_mirror,
        weight=-0.8,
        params={"command_name": "velocity_height"},
    )

    # PreTrain 的高度成功信号很稀疏；稳定惩罚收敛后，策略容易停在"稳但不跳"。
    # 这里保留密集主动起跳信号，但不奖励下蹲本身，避免策略陷入"主动蹲住"的局部最优。
    cfg.rewards["jump_takeoff_impulse"] = RewardTermCfg(
        func=rewards.jump_takeoff_impulse,
        weight=16.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_drive"] = RewardTermCfg(
        func=rewards.jump_takeoff_drive,
        weight=55.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_vz_tracking"] = RewardTermCfg(
        func=rewards.jump_takeoff_vz_tracking,
        weight=75.0,
        params={
            "command_name": "velocity_height",
            "tolerance": 0.45,
            "min_vz_ratio": 0.45,
            "progress_power": 1.0,
            "tracking_mix": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_pretrain_height_success"] = RewardTermCfg(
        func=rewards.jump_pretrain_height_success,
        weight=220.0,
        params={
            "command_name": "velocity_height",
            "base_height_offset": _DEFAULT_STANDING_HEIGHT,
            "wheel_radius": 0.059,
            "relative_tolerance": 0.45,
            "falloff_ratio": 0.25,
            "score_start_s": 1.1,
            "base_weight": 0.0,
            "wheel_weight": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_pretrain_wheel_clearance_progress"] = RewardTermCfg(
        func=rewards.jump_pretrain_wheel_clearance_progress,
        weight=160.0,
        params={
            "command_name": "velocity_height",
            "wheel_radius": 0.059,
            "relative_tolerance": 0.45,
            "falloff_ratio": 0.25,
            "score_start_s": 0.75,
            "progress_power": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_pre_takeoff_wheel_lift"] = RewardTermCfg(
        func=rewards.jump_pre_takeoff_wheel_lift_penalty,
        weight=-6.0,
        params={
            "command_name": "velocity_height",
            "wheel_radius": 0.059,
            "clearance_threshold": 0.015,
            "scale": 0.04,
            "max_penalty": 9.0,
            "max_ref_preload_vz": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_wheel_ground_slip"] = RewardTermCfg(
        func=rewards.jump_wheel_ground_slip,
        weight=-6.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "wheel_radius": 0.059,
            "contact_force_threshold": 1.0,
            "longitudinal_scale": 0.28,
            "lateral_scale": 0.20,
            "max_penalty": 9.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_landing_horizontal_motion"] = RewardTermCfg(
        func=rewards.jump_landing_horizontal_motion_penalty,
        weight=-3.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "wheel_radius": 0.059,
            "contact_force_threshold": 1.0,
            "base_speed_scale": 0.25,
            "wheel_speed_scale": 0.35,
            "max_penalty": 9.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_landing_stability"] = RewardTermCfg(
        func=rewards.jump_landing_stability_penalty,
        weight=-3.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "wheel_radius": 0.059,
            "contact_force_threshold": 1.0,
            "k_gain": 0.03,
            "tolerance": 0.05,
            "max_penalty": 9.0,
            "min_landing_vz": -0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_leg_contact"] = RewardTermCfg(
        func=rewards.jump_leg_contact_penalty,
        weight=-12.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
        },
    )
    cfg.rewards["jump_orientation"] = RewardTermCfg(
        func=rewards.jump_orientation,
        weight=-10.0,
        params={
            "command_name": "velocity_height",
            "pitch_weight": 2.5,
            "roll_weight": 1.0,
            "takeoff_scale": 0.7,
            "air_scale": 1.0,
            "landing_scale": 1.0,
        },
    )
    # PreTrain 必须把"主动蹬地"和"机身直立姿态"一起学掉。
    cfg.rewards["jump_tilt_barrier"] = RewardTermCfg(
        func=rewards.jump_tilt_barrier,
        weight=-4.0,
        params={
            "command_name": "velocity_height",
            "soft_limit_deg": 12.0,
            "hard_limit_deg": 32.0,
            "landing_scale": 1.0,
            "max_penalty": 4.0,
        },
    )
    cfg.rewards["jump_ang_vel_xy"] = RewardTermCfg(
        func=rewards.jump_ang_vel_xy,
        weight=-2.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["jump_ang_vel_z"] = RewardTermCfg(
        func=rewards.jump_ang_vel_z,
        weight=-1.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["jump_wheel_counterspin"] = RewardTermCfg(
        func=rewards.jump_wheel_counterspin,
        weight=-0.08,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_dof_pos_limits_strict"] = RewardTermCfg(
        func=rewards.jump_dof_pos_limits_strict,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 追加膝关节冲击终止
    cfg.terminations["knee_hyperextension"] = TerminationTermCfg(
        func=terminations.knee_hyperextension,
        time_out=False,
        params={
            "command_name": "velocity_height",
            "limit_ratio": 0.98,
            "vel_threshold": 25.0,
            "probability": 0.7,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 跳跃阶段不再用腿部接触传感器终止 episode;真实接触只作为日志留档。
    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact,
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
            "command_name": "velocity_height",
            "terminate": False,
        },
    )

    # 跳跃概率课程(不在 play 模式下运行)
    if not play:
        cfg.curriculum["jump_prob"] = CurriculumTermCfg(
            func=curriculums.jump_prob_curriculum,
            params={
                "command_name": "velocity_height",
                "warmup_iters": 0,
                "rampup_iters": 500,
                "initial_prob": 0.02,
                "final_prob": 0.10,
            },
        )
        cfg.curriculum["jump_pretrain_constraints"] = CurriculumTermCfg(
            func=curriculums.jump_pretrain_constraint_weight_curriculum,
            params={
                "start_iter": 150,
                "ramp_iters": 650,
            },
        )

    return cfg
