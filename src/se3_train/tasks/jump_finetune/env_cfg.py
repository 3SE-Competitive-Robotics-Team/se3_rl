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
from se3_train.tasks.jump_finetune import commands, curriculums, rewards, terminations
from se3_train.tasks.jump_pretrain.env_cfg import env_cfg as pretrain_env_cfg

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
    """跳跃精细训练环境配置(从 PreTrain checkpoint fine-tune)。

    在 PreTrain 基础上替换为精细奖励集:
    - 降权保留 PreTrain 起跳信号
    - 新增参考轨迹 tracking、真实离地高度、姿态/对称/着陆专项约束
    - 新增目标高度课程([0.1,0.3] → [0.1,0.6])
    """
    cfg = pretrain_env_cfg(play=play)

    cfg.rewards.pop("jump_pretrain_height_success", None)
    cfg.rewards.pop("jump_pretrain_wheel_clearance_progress", None)

    # fine-tune 保持中等跳跃占比,避免完整 tracking 重新压过行走梯度。
    _apply_jump_command(cfg, jump_prob=0.12)
    _apply_jump_observations(cfg)
    cfg.commands["velocity_height"].rsi_takeoff_prob = 1.0  # fine-tune 全量使用轨迹起点初始化

    # -------------------------------------------------------------------------
    # 跳跃奖励体系(Fine-tune 阶段完整配置)
    # 设计原则:
    #   轨迹 tracking(3项)负责时序引导,覆盖高度/速度/关节角三个维度
    #   真实离地高度只在 apex 附近评价,避免惩罚合法上升过程
    #   补充约束负责 tracking 未覆盖的维度:姿态、对称性、轮速、落地力
    #   地面期:只保留低权重主动起跳信号,避免遗忘,同时不压过轨迹目标
    # -------------------------------------------------------------------------

    # --- 地面期:保留主动起跳信号 ---
    #
    # 2026-05-21 试验记录:
    # 1. 只保留低权重 takeoff_drive/vz_tracking 时,PostTrain 会很快把
    #    max_airborne_vz 从 1.4m/s 压到 0.6m/s,得到"稳定但跳不高"的策略。
    # 2. 只恢复中等起跳权重仍不够;base pose tracking 的 xy/rot 分项会让
    #    "稳定低跳"拿到较高分数。这里把起跳奖励恢复到 PreTrain 量级,
    #    同时让 pose tracking 由 z 高度主导,再由对称/姿态项收动作质量。

    cfg.rewards["jump_takeoff_drive"] = RewardTermCfg(
        func=rewards.jump_takeoff_drive,
        weight=30.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_impulse"] = RewardTermCfg(
        func=rewards.jump_takeoff_impulse,
        weight=5.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_wheel_ground_slip"] = RewardTermCfg(
        func=rewards.jump_wheel_ground_slip,
        weight=-18.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "wheel_radius": 0.059,
            "contact_force_threshold": 1.0,
            "longitudinal_scale": 0.35,
            "lateral_scale": 0.20,
            "max_penalty": 9.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_vz_tracking"] = RewardTermCfg(
        func=rewards.jump_takeoff_vz_tracking,
        weight=120.0,
        params={
            "command_name": "velocity_height",
            "tolerance": 0.45,
            "min_vz_ratio": 0.55,
            "progress_power": 1.5,
            "tracking_mix": 0.6,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # jump_vel_encourage 极小权重保留,仅防止 airborne 期完全无信号。
    # 主要垂直速度目标由 traj_vz 负责。
    cfg.rewards["jump_vel_encourage"] = RewardTermCfg(
        func=rewards.jump_vel_encourage,
        weight=60.0,
        params={"command_name": "velocity_height", "weight_scale": 1.0},
    )

    # PreTrain 继承的膝关节过伸惩罚,tracking 不覆盖关节限位
    cfg.rewards["jump_dof_pos_limits_strict"] = RewardTermCfg(
        func=rewards.jump_dof_pos_limits_strict,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Fine-tune 专用动作平滑:起跳段保留蹬地自由度,空中/落地加重惩罚。
    # PreTrain 仍使用默认弱 action_rate,避免早期主动起跳被过早压制。
    cfg.rewards["jump_action_rate"] = RewardTermCfg(
        func=rewards.action_rate_jump,
        weight=-0.04,
        params={
            "command_name": "velocity_height",
            "grounded_scale": 0.35,
            "takeoff_scale": 0.20,
            "air_scale": 1.0,
            "landing_scale": 1.2,
            "max_penalty": 80.0,
        },
    )

    # -----------------------------------------------------------------------
    # 轨迹 tracking 奖励(核心,覆盖完整跳跃时序)
    # 构型维度:base_link 6DoF pose + 4 个腿部关节角。
    # 速度维度:保留 vz tracking 作为起跳/落地节奏约束。
    # 轨迹库使用 0.1-0.6m 六条轨迹;每条 ±0.10m 匹配窗口覆盖低到中高度课程。
    # 奖励函数按 jump_target_height 选择最近轨迹,并对高度/速度做小幅缩放。
    # -----------------------------------------------------------------------
    _TRAJ_PATHS = DEFAULT_JUMP_TRAJ_PATHS
    _TRAJ_HEIGHTS = DEFAULT_JUMP_TRAJ_HEIGHTS
    _HEIGHT_TOL = 0.10  # 单条轨迹高度匹配容忍带(±0.10m)

    # base_link 6DoF pose tracking:xyz + roll/pitch/yaw。
    # 轨迹文件不含 base_quat,因此姿态参考定义为 reset 后 yaw 不变、roll/pitch 直立。
    # reset 随机 xy/yaw 会在 reset 事件里缓存为每个 env 的参考零点。
    cfg.rewards["traj_base_pose_6d"] = RewardTermCfg(
        func=rewards.traj_base_pose_6d_tracking,
        weight=12.0,
        params={
            "command_name": "velocity_height",
            "traj_path": _TRAJ_PATHS,
            "sigma_xy": 0.08,
            "sigma_z": 0.10,
            "sigma_rot": 0.25,
            "xy_weight": 0.05,
            "z_weight": 0.85,
            "rot_weight": 0.10,
            "traj_target_heights": _TRAJ_HEIGHTS,
            "height_match_tol": _HEIGHT_TOL,
            "grounded_weight": 0.05,  # 接地期极小权重,只给准备姿态弱引导,防止站着不动拿满分
        },
    )

    # 垂直速度 tracking:阶段相关 std 的 exp 正奖励
    # 保证起跳加速度节奏、飞行抛物线速度和着陆缓冲速度都符合参考。
    # 这里使用正奖励而不是 L1 扣分,减少早期"低速贴轨迹"的保守解。
    cfg.rewards["traj_vz"] = RewardTermCfg(
        func=rewards.traj_vz_tracking,
        weight=3.0,
        params={
            "command_name": "velocity_height",
            "traj_path": _TRAJ_PATHS,
            "std_grounded": 0.45,
            "std_takeoff": 0.60,
            "std_air": 0.55,
            "std_landing": 0.45,
            "traj_target_heights": _TRAJ_HEIGHTS,
            "height_match_tol": _HEIGHT_TOL,
        },
    )

    # 关节角 tracking:exp 形状,覆盖蹲/展/收腿/缓冲时序
    # sigma=0.15rad(约 8.6°),精度适中,不过分约束策略微调自由度
    cfg.rewards["traj_joint_pos"] = RewardTermCfg(
        func=rewards.traj_joint_pos_tracking,
        weight=5.0,
        params={
            "command_name": "velocity_height",
            "traj_path": _TRAJ_PATHS,
            "sigma": 0.15,
            "sigma_grounded": 0.26,
            "sigma_takeoff": 0.24,
            "sigma_air": 0.17,
            "sigma_landing": 0.18,
            "traj_target_heights": _TRAJ_HEIGHTS,
            "height_match_tol": _HEIGHT_TOL,
            "asset_cfg": SceneEntityCfg("robot"),
            "grounded_weight": 0.05,  # 接地期极小权重,防止站着不动拿满分
        },
    )

    # -----------------------------------------------------------------------
    # 补充约束(tracking 未覆盖的维度)
    # -----------------------------------------------------------------------

    # 真实轮子离地高度:只在参考 apex 附近对齐 jump_target_height。
    # 完整上升/下降时序由 traj_base_pose_6d 和 traj_vz 负责。
    cfg.rewards["jump_wheel_clr_tracking"] = RewardTermCfg(
        func=rewards.jump_wheel_clr_tracking,
        weight=-3.0,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "wheel_height_sensor",
            "relative_tolerance": 0.15,
            "falloff_ratio": 0.10,
            "apex_ref_vz_window": 0.35,
        },
    )

    # 姿态:pitch+roll L2 惩罚。严格 pitch 回正只放在 jump_flag=0 的平地段;
    # 一旦进入跳跃窗口,grounded/preload、takeoff、air、landing 都使用跳跃自己的
    # 温和阶段约束,避免把主动伸腿速度压掉。
    cfg.rewards["jump_orientation"] = RewardTermCfg(
        func=rewards.jump_orientation,
        weight=-6.0,
        params={
            "command_name": "velocity_height",
            "pitch_weight": 1.5,
            "roll_weight": 1.0,
            "takeoff_scale": 0.35,
            "air_scale": 1.0,
            "landing_scale": 0.8,
        },
    )
    # tilt barrier:sim2sim sweep 显示 pitch 在 20° 左右已经影响观感和落地,
    # 但上一轮训练在 quality_weight_progress=0 时 vz 已经跌破 1.2m/s,说明早期
    # barrier 不能太早抢起跳梯度;从 18° 开始介入,把更强姿态修正留给后期课程。
    # 这比直接降低 bad_orientation 终止阈值更平滑,不会截断早期探索。
    cfg.rewards["jump_tilt_barrier"] = RewardTermCfg(
        func=rewards.jump_tilt_barrier,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "soft_limit_deg": 18.0,
            "hard_limit_deg": 38.0,
            "landing_scale": 0.7,
            "max_penalty": 4.0,
        },
    )

    # 角速度:与姿态惩罚配合抑制空中翻滚
    cfg.rewards["jump_ang_vel_xy"] = RewardTermCfg(
        func=rewards.jump_ang_vel_xy,
        weight=-1.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["jump_ang_vel_z"] = RewardTermCfg(
        func=rewards.jump_ang_vel_z,
        weight=-3.0,
        params={"command_name": "velocity_height"},
    )

    # 左右关节对称性:tracking 参考角虽然对称,但显式惩罚更直接
    cfg.rewards["jump_joint_mirror"] = RewardTermCfg(
        func=rewards.jump_joint_mirror,
        weight=-12.0,
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
    # 显式轮距:学习 Tron1 的 pen_feet_distance 设计,用几何距离直接约束双轮。
    # SerialLeg 默认站立左右轮距约 0.433m,因此这里约束 0.40~0.46m,同时惩罚
    # 前后错位,解决"两个轮一前一后"这类 joint_mirror 看不准的问题。
    cfg.rewards["wheel_distance"] = RewardTermCfg(
        func=rewards.wheel_distance_regularization,
        weight=-5.0,
        params={
            "command_name": "velocity_height",
            "min_lateral_distance": 0.40,
            "max_lateral_distance": 0.46,
            "max_fore_aft_offset": 0.03,
            "lateral_scale": 0.04,
            "fore_aft_scale": 0.03,
            "fore_aft_weight": 1.5,
            "standing_scale": 1.0,
            "grounded_scale": 0.4,
            "takeoff_scale": 1.2,
            "air_scale": 1.0,
            "landing_scale": 1.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_action_mirror"] = RewardTermCfg(
        func=rewards.jump_action_mirror,
        weight=-1.2,
        params={"command_name": "velocity_height"},
    )

    # 空中轮速:抑制陀螺效应,tracking 不约束轮子
    cfg.rewards["jump_wheel_vel"] = RewardTermCfg(
        func=rewards.jump_wheel_vel,
        weight=-0.02,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_wheel_counterspin"] = RewardTermCfg(
        func=rewards.jump_wheel_counterspin,
        weight=-0.04,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 着陆力对称性:tracking 无力的维度
    cfg.rewards["landing_symmetry"] = RewardTermCfg(
        func=rewards.landing_symmetry,
        weight=-1.5,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
        },
    )

    # 落地恢复:landing 阶段奖励水平静止和高度回归站立位
    cfg.rewards["jump_landing_recovery"] = RewardTermCfg(
        func=rewards.jump_landing_recovery,
        weight=6.0,
        params={
            "command_name": "velocity_height",
            "sigma_vxy": 0.4,
            "sigma_h": 0.08,
            "target_height": _DEFAULT_STANDING_HEIGHT,
            "height_sensor_name": "base_height_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_landing_base_height"] = RewardTermCfg(
        func=rewards.jump_landing_base_height_penalty,
        weight=-4.0,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "base_height_sensor",
            "target_height": _DEFAULT_STANDING_HEIGHT,
            "tolerance": 0.035,
            "scale": 0.06,
        },
    )
    # PostTrain 起跳前静止 + 起跳时水平惩罚(权重比 PreTrain 更大,要求更严格)
    cfg.rewards["jump_pre_takeoff_stillness"] = RewardTermCfg(
        func=rewards.jump_pre_takeoff_stillness,
        weight=8.0,
        params={
            "command_name": "velocity_height",
            "sigma_vxy": 0.35,
            "stillness_window_steps": 40,  # 只在 jump_flag 后前 40 步(0.4s)奖励静止,超过必须起跳
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_horizontal_penalty"] = RewardTermCfg(
        func=rewards.jump_takeoff_horizontal_penalty,
        weight=-12.0,
        params={
            "command_name": "velocity_height",
            "sigma_vxy": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 目标高度课程(fine-tune 阶段扩大范围)
    # 同时覆盖继承自 pretrain 的 jump_prob 课程:fine-tune 从 iter=0 使用 0.12。
    if not play:
        cfg.curriculum["jump_prob"] = CurriculumTermCfg(
            func=curriculums.jump_prob_curriculum,
            params={
                "command_name": "velocity_height",
                "warmup_iters": 0,
                "rampup_iters": 0,
                "final_prob": 0.12,
            },
        )
        cfg.curriculum["jump_height"] = CurriculumTermCfg(
            func=curriculums.jump_height_curriculum,
            params={
                "command_name": "velocity_height",
                "expand_iter": 1000,
                "ramp_iters": 1500,
                "initial_range": (0.1, 0.3),
                "final_range": (0.1, 0.6),
            },
        )
        cfg.curriculum["jump_quality_weights"] = CurriculumTermCfg(
            func=curriculums.jump_quality_weight_curriculum,
            params={
                "start_iter": 1000,
                "ramp_iters": 2000,
            },
        )

    return cfg
