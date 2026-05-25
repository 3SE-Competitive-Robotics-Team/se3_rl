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

from se3_train.mdp import events, observations, rewards, terminations
from se3_train.mdp.actions import SerialLegDelayedActionCfg
from se3_train.mdp.curriculums import commands_vel as curriculum_commands_vel
from se3_train.mdp.curriculums import push_disturbance as curriculum_push_disturbance
from se3_train.mdp.jump_commands import JumpCommandCfg
from se3_train.mdp.jump_curriculums import (
    jump_height_curriculum,
    jump_pretrain_constraint_weight_curriculum,
    jump_prob_curriculum,
    jump_quality_weight_curriculum,
)
from se3_train.mdp.jump_rewards import (
    action_rate_jump,
    action_rate_no_jump,
    action_smoothness_no_jump,
    feet_contact_without_cmd_no_jump,
    flat_orientation_l2_no_jump,
    idle_wheel_motion_penalty_no_jump,
    jump_action_mirror,
    jump_ang_vel_xy,
    jump_ang_vel_z,
    jump_dof_pos_limits_strict,
    jump_joint_mirror,
    jump_landing_base_height_penalty,
    jump_landing_horizontal_motion_penalty,
    jump_landing_recovery,
    jump_leg_contact_penalty,
    jump_orientation,
    jump_pre_takeoff_stillness,
    jump_pre_takeoff_wheel_lift_penalty,
    jump_pretrain_height_success,
    jump_pretrain_wheel_clearance_progress,
    jump_takeoff_drive,
    jump_takeoff_horizontal_penalty,
    jump_takeoff_impulse,
    jump_takeoff_vz_tracking,
    jump_tilt_barrier,
    jump_vel_encourage,
    jump_wheel_clr_tracking,
    jump_wheel_counterspin,
    jump_wheel_ground_slip,
    jump_wheel_vel,
    landing_symmetry,
    leg_power_no_jump,
    leg_torques_no_jump,
    stand_still_no_jump,
    standing_joint_mirror_no_jump,
    tracking_ang_vel_no_jump,
    tracking_height_no_jump,
    tracking_lin_vel_no_jump,
    tracking_orientation_l2_no_jump,
    wheel_distance_regularization,
)
from se3_train.mdp.jump_terminations import knee_hyperextension
from se3_train.mdp.jump_traj_tracking import (
    traj_base_pose_6d_tracking,
    traj_joint_pos_tracking,
    traj_vz_tracking,
)
from se3_train.mdp.jump_trajectories import DEFAULT_JUMP_TRAJ_HEIGHTS, DEFAULT_JUMP_TRAJ_PATHS
from se3_train.robot_cfg import get_serialleg_cfg


def se3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """SerialLeg 轮腿机器人的平地环境配置。"""

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_serialleg_cfg()},
        num_envs=1024,
        env_spacing=3.0,
    )

    cfg = ManagerBasedRlEnvCfg(
        decimation=5,
        scene=scene,
    )

    collision_sensor_cfg = ContactSensorCfg(
        name="collision_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(base_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    leg_contact_sensor_cfg = ContactSensorCfg(
        name="leg_contact_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(lf0_Link|lf1_Link|rf0_Link|rf1_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    wheel_sensor_cfg = ContactSensorCfg(
        name="wheel_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link|r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    base_height_sensor_cfg = TerrainHeightSensorCfg(
        name="base_height_sensor",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=4),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )

    # 轮子离地高度传感器:从左轮 body 向下打射线,测量真实轮子离地距离
    # 用于 jump_wheel_clr_tracking,防止策略通过收腿套利 base_link 高度
    wheel_height_sensor_cfg = TerrainHeightSensorCfg(
        name="wheel_height_sensor",
        frame=ObjRef(type="body", name="l_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )

    critic_height_sensor_cfg = TerrainHeightSensorCfg(
        name="critic_height_sensor",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.15, num_samples=8),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="mean",
    )

    cfg.scene.sensors = (
        collision_sensor_cfg,
        leg_contact_sensor_cfg,
        wheel_sensor_cfg,
        base_height_sensor_cfg,
        critic_height_sensor_cfg,
        wheel_height_sensor_cfg,
    )

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=observations.base_ang_vel_obs,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=observations.projected_gravity_obs,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "commands": ObservationTermCfg(func=observations.commands_obs),
        "leg_joint_pos": ObservationTermCfg(
            func=observations.leg_joint_pos_obs,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "leg_joint_vel": ObservationTermCfg(
            func=observations.leg_joint_vel_obs,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "wheel_pos": ObservationTermCfg(func=observations.wheel_pos_obs),
        "wheel_vel": ObservationTermCfg(func=observations.wheel_vel_obs),
        "last_actions": ObservationTermCfg(func=observations.last_actions_obs),
        "jump_commands": ObservationTermCfg(func=observations.jump_commands_obs),
    }

    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=observations.base_lin_vel_obs),
        "wheel_contact_forces": ObservationTermCfg(
            func=observations.wheel_contact_force_obs,
            params={"sensor_name": "wheel_sensor"},
        ),
        "base_height": ObservationTermCfg(
            func=observations.base_height_obs,
            params={"sensor_name": "critic_height_sensor"},
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.actions = {
        "delayed_action": SerialLegDelayedActionCfg(entity_name="robot"),
    }

    cfg.commands = {
        "velocity_height": JumpCommandCfg(
            resampling_time_range=(5.0, 5.0),
            jump_prob=0.0,  # 行走任务不触发跳跃
        ),
    }

    cfg.rewards = {
        "tracking_lin_vel": RewardTermCfg(
            func=rewards.tracking_lin_vel,
            weight=2.73,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.25,
                "sigma_stand": 0.1,
                "vz_weight": 2.0,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=rewards.tracking_ang_vel,
            weight=1.73,
            params={"command_name": "velocity_height", "sigma": 0.25},
        ),
        # 姿态相关项只使用惩罚语义:偏离目标姿态扣分,明显倾斜加重扣分。
        "tracking_orientation_l2": RewardTermCfg(
            func=rewards.tracking_orientation_l2,
            weight=-12.0,
            params={"command_name": "velocity_height"},
        ),
        "tracking_height": RewardTermCfg(
            func=rewards.tracking_height,
            weight=2.49,
            params={
                "command_name": "velocity_height",
                "sigma": 0.05,
                "height_sensor_name": "base_height_sensor",
            },
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-4.0,
            params={"soft_limit_deg": 12.0, "hard_limit_deg": 35.0, "max_penalty": 4.0},
        ),
        "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.146),
        "angular_momentum": RewardTermCfg(
            func=rewards.angular_momentum,
            weight=-5.0e-5,
        ),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-2.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "wheel_torques": RewardTermCfg(
            func=rewards.wheel_torques,
            weight=-1.0e-4,
            params={"max_torque": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-1.0,
            params={
                "command_name": "velocity_height",
                "command_threshold": 0.1,
                "default_height": 0.26,
                "height_tolerance": 40.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-2.17e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-1.03e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.48),
        "joint_mirror": RewardTermCfg(
            func=rewards.joint_mirror,
            weight=-0.179,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=rewards.dof_pos_limits,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "collision": RewardTermCfg(
            func=rewards.collision,
            weight=-2.51,
            params={"sensor_name": "collision_sensor", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "contact_forces": RewardTermCfg(
            func=rewards.contact_forces,
            weight=-1.07e-3,
            params={
                "threshold": 35.0,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "feet_contact_without_cmd": RewardTermCfg(
            func=rewards.feet_contact_without_cmd,
            weight=0.386,
            params={
                "command_name": "velocity_height",
                "force_threshold": 1.0,
                "cmd_threshold": 0.1,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # 业界标准:移除显式 termination 惩罚,改用 alive reward 隐式机制
        # 摔倒 → episode 结束 → 损失后续所有 alive 累积(隐式 penalty ~= alive * remaining_steps)
        # ETH/Unitree/CMU 所有框架 termination weight = 0,这是行业共识
        "is_alive": RewardTermCfg(func=mjlab_is_alive, weight=1.0),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=terminations.bad_orientation_delayed,
            time_out=False,
            params={"limit_angle": 0.5236, "max_steps": 100},
        ),
        "leg_contact": TerminationTermCfg(
            func=terminations.leg_contact,
            time_out=False,
            params={"sensor_name": "leg_contact_sensor", "force_threshold": 1.0},
        ),
    }

    if not play:
        cfg.curriculum = {
            "command_vel": CurriculumTermCfg(
                func=curriculum_commands_vel,
                params={
                    "command_name": "velocity_height",
                    "velocity_stages": [
                        {
                            "step": 0,
                            "lin_vel_x_range": (0.0, 0.0),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "step": 500,
                            "lin_vel_x_range": (-0.5, 0.5),
                            "ang_vel_yaw_range": (-0.5, 0.5),
                        },
                        {
                            "step": 1500,
                            "lin_vel_x_range": (-1.0, 1.0),
                            "ang_vel_yaw_range": (-1.0, 1.0),
                        },
                        {
                            "step": 2500,
                            "lin_vel_x_range": (-1.5, 1.5),
                            "ang_vel_yaw_range": (-2.0, 2.0),
                        },
                        {
                            "step": 3500,
                            "lin_vel_x_range": (-2.0, 2.0),
                            "ang_vel_yaw_range": (-2.5, 2.5),
                        },
                        {
                            "step": 4500,
                            "lin_vel_x_range": (-2.5, 2.5),
                            "ang_vel_yaw_range": (-3.0, 3.0),
                        },
                    ],
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculum_push_disturbance,
                params={
                    "push_stages": [
                        {
                            "step": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "step": 2000,
                            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                        },
                        {
                            "step": 5000,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "step": 10000,
                            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                        },
                        {
                            "step": 20000,
                            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                        },
                        {
                            "step": 40000,
                            "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
                        },
                    ],
                },
            ),
        }

    if play:
        cfg.events = {
            "reset_scene_to_default": EventTermCfg(
                func=lambda env, env_ids: None,
                mode="reset",
            ),
            "reset_root_state": EventTermCfg(
                func=events.reset_root_state_full,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "reset_joints": EventTermCfg(
                func=events.reset_joints,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
        }
        cfg.episode_length_s = 9999.0
    else:
        cfg.events = {
            "reset_scene_to_default": EventTermCfg(
                func=lambda env, env_ids: None,
                mode="reset",
            ),
            "reset_root_state": EventTermCfg(
                func=events.reset_root_state_full,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "reset_joints": EventTermCfg(
                func=events.reset_joints,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "friction": EventTermCfg(
                func=events.randomize_friction,
                mode="startup",
                params={"friction_range": (0.2, 1.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "restitution": EventTermCfg(
                func=events.randomize_restitution,
                mode="startup",
                params={"restitution_range": (0.0, 0.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "base_mass": EventTermCfg(
                func=events.randomize_base_mass,
                mode="startup",
                params={"mass_range": (-0.5, 1.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "inertia": EventTermCfg(
                func=events.randomize_inertia,
                mode="startup",
                params={"inertia_range": (0.8, 1.2), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "com": EventTermCfg(
                func=events.randomize_com,
                mode="startup",
                params={"com_range": 0.05, "asset_cfg": SceneEntityCfg("robot")},
            ),
            "pd_gains": EventTermCfg(
                func=events.randomize_pd_gains,
                mode="startup",
                params={
                    "kp_range": (0.9, 1.1),  # 收窄:配合 stall_torque 上限,避免 kp 偏软时振荡跪地
                    "kd_range": (0.9, 1.1),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "default_dof_pos": EventTermCfg(
                func=events.randomize_default_dof_pos,
                mode="startup",
                params={
                    "offset_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "push_robots": EventTermCfg(
                func=events.push_robots,
                mode="interval",
                interval_range_s=(5.0, 6.0),
                params={
                    "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        }
        cfg.episode_length_s = 20.0

    cfg.scale_rewards_by_dt = True
    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1040,
        mujoco=MujocoCfg(timestep=0.002),
    )
    cfg.viewer = ViewerConfig()

    return cfg


def se3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """带地形课程的崎岖地形环境配置。"""

    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    cfg = se3_flat_env_cfg(play=play)

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
    )

    return cfg


# ---------------------------------------------------------------------------
# 跳跃环境:公共辅助函数
# ---------------------------------------------------------------------------


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
        "velocity_height": JumpCommandCfg(
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


# ---------------------------------------------------------------------------
# 阶段1:SE3-WheelLegged-Jump-PreTrain-GRU
# ---------------------------------------------------------------------------


def se3_jump_pretrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """跳跃预训练环境配置(从行走 checkpoint fine-tune)。

    在行走环境基础上叠加:
    - JumpCommandTerm(8 维指令)
    - 稀疏高度奖励:轮组最大高度和 base 最大高度接近目标
    - 轻量姿态、左右对称和关节限位约束
    - 膝关节冲击终止
    """
    cfg = se3_flat_env_cfg(play=play)

    # 跳跃任务先学习起跳和落地本体能力;通用 push 扰动留到后续鲁棒性阶段。
    cfg.events.pop("push_robots", None)
    cfg.curriculum.pop("push_disturbance", None)

    # 替换指令
    _apply_jump_command(cfg, jump_prob=0.02)

    # 追加跳跃观测
    _apply_jump_observations(cfg)

    # PreTrain 不使用参考轨迹 RSI;只保留 jump_flag、目标高度和固定长度单次动作窗口。
    cfg.commands["velocity_height"].jump_height_range = (0.1, 0.3)
    cfg.commands["velocity_height"].rsi_takeoff_prob = 0.0
    cfg.commands["velocity_height"].rsi_random_frame = False
    cfg.commands["velocity_height"].jump_cool_down_steps = 100

    # jump_flag=1 时屏蔽与起跳冲突的行走奖励
    # tracking_lin_vel 里有 vz2 惩罚会压制起跳,必须在 jump_flag=1 时清零
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=tracking_lin_vel_no_jump,
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
        func=stand_still_no_jump,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": 0.26,
            "height_tolerance": 40.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 力矩上限已放开至 stall_torque(40 N·m),起跳瞬间峰值力矩若被惩罚会压制起跳
    # jump_flag=1 时豁免 leg_torques 和 leg_power 惩罚
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=leg_torques_no_jump,
        weight=-2.0e-4,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=leg_power_no_jump,
        weight=-1.03e-4,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # tracking_height 与跳跃轨迹 pose tracking 梯度相反,jump_flag=1 时必须清零
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=tracking_height_no_jump,
        weight=2.49,
        params={
            "command_name": "velocity_height",
            "sigma": 0.05,
            "height_sensor_name": "base_height_sensor",
        },
    )

    # 偏航速度跟踪:起跳/飞行期姿态变化正常,不应持续惩罚
    cfg.rewards["tracking_ang_vel"] = RewardTermCfg(
        func=tracking_ang_vel_no_jump,
        weight=1.73,
        params={"command_name": "velocity_height", "sigma": 0.25},
    )

    cfg.rewards["tracking_orientation_l2"] = RewardTermCfg(
        func=tracking_orientation_l2_no_jump,
        weight=-6.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
        func=flat_orientation_l2_no_jump,
        weight=-24.0,
        params={"command_name": "velocity_height"},
    )

    # action_rate:jump 期保持上一轮成功配置,flat/idle 期额外压低动作抖动。
    cfg.rewards["action_rate"] = RewardTermCfg(
        func=action_rate_no_jump,
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
        func=idle_wheel_motion_penalty_no_jump,
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
    cfg.rewards["flat_action_smoothness"] = RewardTermCfg(
        func=action_smoothness_no_jump,
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
        func=action_rate_jump,
        weight=-0.04,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["standing_joint_mirror"] = RewardTermCfg(
        func=standing_joint_mirror_no_jump,
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
        func=wheel_distance_regularization,
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
        func=jump_joint_mirror,
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
        func=jump_action_mirror,
        weight=-0.8,
        params={"command_name": "velocity_height"},
    )

    # PreTrain 的高度成功信号很稀疏；稳定惩罚收敛后，策略容易停在"稳但不跳"。
    # 这里保留密集主动起跳信号，但不奖励下蹲本身，避免策略陷入"主动蹲住"的局部最优。
    cfg.rewards["jump_takeoff_impulse"] = RewardTermCfg(
        func=jump_takeoff_impulse,
        weight=16.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_drive"] = RewardTermCfg(
        func=jump_takeoff_drive,
        weight=55.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_vz_tracking"] = RewardTermCfg(
        func=jump_takeoff_vz_tracking,
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
        func=jump_pretrain_height_success,
        weight=220.0,
        params={
            "command_name": "velocity_height",
            "base_height_offset": 0.26,
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
        func=jump_pretrain_wheel_clearance_progress,
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
        func=jump_pre_takeoff_wheel_lift_penalty,
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
        func=jump_wheel_ground_slip,
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
        func=jump_landing_horizontal_motion_penalty,
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
    cfg.rewards["jump_leg_contact"] = RewardTermCfg(
        func=jump_leg_contact_penalty,
        weight=-12.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
        },
    )
    cfg.rewards["jump_orientation"] = RewardTermCfg(
        func=jump_orientation,
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
        func=jump_tilt_barrier,
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
        func=jump_ang_vel_xy,
        weight=-2.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["jump_ang_vel_z"] = RewardTermCfg(
        func=jump_ang_vel_z,
        weight=-1.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["jump_wheel_counterspin"] = RewardTermCfg(
        func=jump_wheel_counterspin,
        weight=-0.08,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_dof_pos_limits_strict"] = RewardTermCfg(
        func=jump_dof_pos_limits_strict,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 追加膝关节冲击终止
    cfg.terminations["knee_hyperextension"] = TerminationTermCfg(
        func=knee_hyperextension,
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
            func=jump_prob_curriculum,
            params={
                "command_name": "velocity_height",
                "warmup_iters": 0,
                "rampup_iters": 500,
                "initial_prob": 0.02,
                "final_prob": 0.10,
            },
        )
        cfg.curriculum["jump_pretrain_constraints"] = CurriculumTermCfg(
            func=jump_pretrain_constraint_weight_curriculum,
            params={
                "start_iter": 150,
                "ramp_iters": 650,
            },
        )

    return cfg


# ---------------------------------------------------------------------------
# 阶段2:SE3-WheelLegged-Jump-GRU
# ---------------------------------------------------------------------------


def se3_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """跳跃精细训练环境配置(从 PreTrain checkpoint fine-tune)。

    在 PreTrain 基础上替换为精细奖励集:
    - 降权保留 PreTrain 起跳信号
    - 新增参考轨迹 tracking、真实离地高度、姿态/对称/着陆专项约束
    - 新增目标高度课程([0.1,0.3] → [0.1,0.6])
    """
    cfg = se3_jump_pretrain_env_cfg(play=play)

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
        func=jump_takeoff_drive,
        weight=30.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_impulse"] = RewardTermCfg(
        func=jump_takeoff_impulse,
        weight=5.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_wheel_ground_slip"] = RewardTermCfg(
        func=jump_wheel_ground_slip,
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
        func=jump_takeoff_vz_tracking,
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
        func=jump_vel_encourage,
        weight=60.0,
        params={"command_name": "velocity_height", "weight_scale": 1.0},
    )

    # PreTrain 继承的膝关节过伸惩罚,tracking 不覆盖关节限位
    cfg.rewards["jump_dof_pos_limits_strict"] = RewardTermCfg(
        func=jump_dof_pos_limits_strict,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Fine-tune 专用动作平滑:起跳段保留蹬地自由度,空中/落地加重惩罚。
    # PreTrain 仍使用默认弱 action_rate,避免早期主动起跳被过早压制。
    cfg.rewards["jump_action_rate"] = RewardTermCfg(
        func=action_rate_jump,
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
        func=traj_base_pose_6d_tracking,
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
        func=traj_vz_tracking,
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
        func=traj_joint_pos_tracking,
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
        func=jump_wheel_clr_tracking,
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
        func=jump_orientation,
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
        func=jump_tilt_barrier,
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
        func=jump_ang_vel_xy,
        weight=-1.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["jump_ang_vel_z"] = RewardTermCfg(
        func=jump_ang_vel_z,
        weight=-3.0,
        params={"command_name": "velocity_height"},
    )

    # 左右关节对称性:tracking 参考角虽然对称,但显式惩罚更直接
    cfg.rewards["jump_joint_mirror"] = RewardTermCfg(
        func=jump_joint_mirror,
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
        func=wheel_distance_regularization,
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
        func=jump_action_mirror,
        weight=-1.2,
        params={"command_name": "velocity_height"},
    )

    # 空中轮速:抑制陀螺效应,tracking 不约束轮子
    cfg.rewards["jump_wheel_vel"] = RewardTermCfg(
        func=jump_wheel_vel,
        weight=-0.02,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_wheel_counterspin"] = RewardTermCfg(
        func=jump_wheel_counterspin,
        weight=-0.04,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 着陆力对称性:tracking 无力的维度
    cfg.rewards["landing_symmetry"] = RewardTermCfg(
        func=landing_symmetry,
        weight=-1.5,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
        },
    )

    # jump_flag=1 时关闭 feet_contact_without_cmd,防止蹲着不跳也能拿分
    cfg.rewards["feet_contact_without_cmd"] = RewardTermCfg(
        func=feet_contact_without_cmd_no_jump,
        weight=0.386,
        params={
            "command_name": "velocity_height",
            "force_threshold": 1.0,
            "cmd_threshold": 0.1,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 落地恢复:landing 阶段奖励水平静止和高度回归站立位
    cfg.rewards["jump_landing_recovery"] = RewardTermCfg(
        func=jump_landing_recovery,
        weight=6.0,
        params={
            "command_name": "velocity_height",
            "sigma_vxy": 0.4,
            "sigma_h": 0.08,
            "target_height": 0.26,
            "height_sensor_name": "base_height_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_landing_base_height"] = RewardTermCfg(
        func=jump_landing_base_height_penalty,
        weight=-4.0,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "base_height_sensor",
            "target_height": 0.26,
            "tolerance": 0.035,
            "scale": 0.06,
        },
    )
    # PostTrain 起跳前静止 + 起跳时水平惩罚(权重比 PreTrain 更大,要求更严格)
    cfg.rewards["jump_pre_takeoff_stillness"] = RewardTermCfg(
        func=jump_pre_takeoff_stillness,
        weight=8.0,
        params={
            "command_name": "velocity_height",
            "sigma_vxy": 0.35,
            "stillness_window_steps": 40,  # 只在 jump_flag 后前 40 步(0.4s)奖励静止,超过必须起跳
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_horizontal_penalty"] = RewardTermCfg(
        func=jump_takeoff_horizontal_penalty,
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
            func=jump_prob_curriculum,
            params={
                "command_name": "velocity_height",
                "warmup_iters": 0,
                "rampup_iters": 0,
                "final_prob": 0.12,
            },
        )
        cfg.curriculum["jump_height"] = CurriculumTermCfg(
            func=jump_height_curriculum,
            params={
                "command_name": "velocity_height",
                "expand_iter": 1000,
                "ramp_iters": 1500,
                "initial_range": (0.1, 0.3),
                "final_range": (0.1, 0.6),
            },
        )
        cfg.curriculum["jump_quality_weights"] = CurriculumTermCfg(
            func=jump_quality_weight_curriculum,
            params={
                "start_iter": 1000,
                "ramp_iters": 2000,
            },
        )

    return cfg
