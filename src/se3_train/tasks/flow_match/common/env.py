"""FlowMatch task 组共享环境配置。"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxOpenStairsTerrainCfg,
    BoxRandomSpreadTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_shared import TaskMode
from se3_train.mdp.jump_trajectories import DEFAULT_JUMP_TRAJ_HEIGHTS, DEFAULT_JUMP_TRAJ_PATHS
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import commands, curriculums, observations, rewards, terminations

_DEFAULT_STANDING_HEIGHT = 0.22
_STANDING_HEIGHT_RANGE = (0.20, 0.32)


def apply_task_mode_command(
    cfg: ManagerBasedRlEnvCfg,
    mode_probabilities: tuple[float, ...] = (0.35, 0.15, 0.20, 0.15, 0.15),
    jump_prob: float = 0.15,
    enable_mode_switch: bool = True,
) -> None:
    """替换为统一 Task Mode 指令。"""
    cfg.commands = {
        "velocity_height": commands.TaskModeCommandCfg(
            resampling_time_range=(5.0, 5.0),
            jump_prob=jump_prob,
            mode_probabilities=mode_probabilities,
            enable_mode_switch=enable_mode_switch,
            height_range=_STANDING_HEIGHT_RANGE,
            standing_height_range=_STANDING_HEIGHT_RANGE,
            jump_height_range=(0.1, 0.3),
            traj_paths=DEFAULT_JUMP_TRAJ_PATHS,
            traj_target_heights=DEFAULT_JUMP_TRAJ_HEIGHTS,
        ),
    }


def apply_task_mode_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """把旧 3D jump command 观测替换为 13D Task Mode 语义观测。"""
    task_mode_obs_term = ObservationTermCfg(func=observations.task_mode_obs)

    actor_terms = dict(cfg.observations["actor"].terms)
    actor_terms.pop("jump_commands", None)
    actor_terms["task_mode"] = task_mode_obs_term
    cfg.observations["actor"] = replace(cfg.observations["actor"], terms=actor_terms)

    critic_terms = dict(cfg.observations["critic"].terms)
    critic_terms.pop("jump_commands", None)
    critic_terms["task_mode"] = task_mode_obs_term
    cfg.observations["critic"] = replace(cfg.observations["critic"], terms=critic_terms)


def apply_task_mode_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """挂载包含 jump 的统一 Task Mode 专属奖励。"""
    apply_loco_task_mode_rewards(cfg)

    cfg.rewards["jump_takeoff_vz_tracking"] = RewardTermCfg(
        func=rewards.jump_takeoff_vz_tracking,
        weight=40.0,
        params={
            "command_name": "velocity_height",
            "tolerance": 0.45,
            "min_vz_ratio": 0.45,
            "progress_power": 1.0,
            "tracking_mix": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_takeoff_drive"] = RewardTermCfg(
        func=rewards.jump_takeoff_drive,
        weight=25.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_vel_encourage"] = RewardTermCfg(
        func=rewards.jump_vel_encourage,
        weight=30.0,
        params={"command_name": "velocity_height", "weight_scale": 1.0},
    )
    cfg.rewards["jump_orientation"] = RewardTermCfg(
        func=rewards.jump_orientation,
        weight=-6.0,
        params={
            "command_name": "velocity_height",
            "pitch_weight": 2.5,
            "roll_weight": 1.0,
            "takeoff_scale": 0.7,
            "air_scale": 1.0,
            "landing_scale": 1.0,
        },
    )
    cfg.rewards["jump_landing_recovery"] = RewardTermCfg(
        func=rewards.jump_landing_recovery,
        weight=8.0,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "base_height_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["jump_ang_vel_xy"] = RewardTermCfg(
        func=rewards.jump_ang_vel_xy,
        weight=-1.5,
        params={"command_name": "velocity_height"},
    )


def apply_loco_task_mode_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """挂载四种 loco mode 的奖励和软约束。"""
    cfg.rewards.pop("flat_wheel_center_alignment", None)
    cfg.rewards.pop("flat_wheel_ground_slip", None)
    cfg.rewards["joint_mirror"].weight = -0.179

    wheel_modes = (int(TaskMode.WHEEL),)
    gait_modes = (int(TaskMode.GAIT),)
    wheel_leg_modes = (int(TaskMode.WHEEL_LEG),)
    gait_wheel_modes = (int(TaskMode.GAIT_WHEEL),)
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.mode_tracking_lin_vel,
        weight=2.73,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.25,
            "sigma_stand": 0.1,
            "vz_weight": 2.0,
            "modes": wheel_modes,
        },
    )
    cfg.rewards["tracking_ang_vel"] = RewardTermCfg(
        func=rewards.mode_tracking_ang_vel,
        weight=1.73,
        params={"command_name": "velocity_height", "sigma": 0.25, "modes": wheel_modes},
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.mode_stand_still,
        weight=-0.8,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": _DEFAULT_STANDING_HEIGHT,
            "height_tolerance": 40.0,
            "modes": wheel_modes,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["wheel_feet_distance"] = RewardTermCfg(
        func=rewards.wheel_feet_distance,
        weight=-25.0,
        params={
            "command_name": "velocity_height",
            "min_feet_distance": 0.43,
            "max_feet_distance": 0.46,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["wheel_idle_action_rate"] = RewardTermCfg(
        func=rewards.wheel_idle_action_rate,
        weight=-0.4,
        params={
            "command_name": "velocity_height",
            "idle_command_threshold": 0.08,
            "max_penalty": 80.0,
        },
    )
    cfg.rewards["wheel_idle_motion"] = RewardTermCfg(
        func=rewards.wheel_idle_motion_penalty,
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
    cfg.rewards["wheel_straight_yaw_drift"] = RewardTermCfg(
        func=rewards.wheel_straight_yaw_drift,
        weight=0.0,
        params={
            "command_name": "velocity_height",
            "min_speed": 0.2,
            "max_yaw_command": 0.05,
        },
    )
    cfg.rewards["wheel_straight_lateral_vel"] = RewardTermCfg(
        func=rewards.wheel_straight_lateral_vel,
        weight=0.0,
        params={
            "command_name": "velocity_height",
            "min_speed": 0.2,
            "max_yaw_command": 0.05,
        },
    )
    cfg.rewards["wheel_in_place_linear_vel"] = RewardTermCfg(
        func=rewards.wheel_in_place_linear_vel,
        weight=0.0,
        params={
            "command_name": "velocity_height",
            "max_linear_command": 0.05,
            "min_yaw_command": 0.2,
        },
    )
    cfg.rewards["gait_no_wheel_drive"] = RewardTermCfg(
        func=rewards.gait_no_wheel_drive,
        weight=-4.0,
        params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["gait_leg_idle_penalty"] = RewardTermCfg(
        func=rewards.gait_leg_idle_penalty,
        weight=-1.5,
        params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["gait_wheel_velocity_assist"] = RewardTermCfg(
        func=rewards.gait_wheel_velocity_assist,
        weight=0.3,
        params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["gait_tracking_lin_vel"] = RewardTermCfg(
        func=rewards.mode_tracking_lin_vel,
        weight=3.0,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.35,
            "sigma_stand": 0.12,
            "vz_weight": 1.0,
            "modes": gait_modes,
        },
    )
    cfg.rewards["wheel_leg_tracking_lin_vel"] = RewardTermCfg(
        func=rewards.mode_tracking_lin_vel,
        weight=0.6,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.3,
            "sigma_stand": 0.12,
            "vz_weight": 1.0,
            "modes": wheel_leg_modes,
        },
    )
    cfg.rewards["gait_wheel_tracking_lin_vel"] = RewardTermCfg(
        func=rewards.mode_tracking_lin_vel,
        weight=0.6,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.3,
            "sigma_stand": 0.12,
            "vz_weight": 1.0,
            "modes": gait_wheel_modes,
        },
    )
    cfg.rewards["wheel_swing_clearance"] = RewardTermCfg(
        func=rewards.wheel_swing_clearance,
        weight=-0.5,
        params={
            "command_name": "velocity_height",
            "target_clearance_m": 0.10,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["wheel_stumble"] = RewardTermCfg(
        func=rewards.wheel_stumble,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
        },
    )
    cfg.rewards["leg_obstacle_collision"] = RewardTermCfg(
        func=rewards.leg_obstacle_collision,
        weight=-2.5,
        params={
            "command_name": "velocity_height",
            "sensor_name": "leg_contact_sensor",
        },
    )
    cfg.rewards["loco_base_height"] = RewardTermCfg(
        func=rewards.loco_base_height,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "height_sensor_name": "base_height_sensor",
            "target_height": 0.32,
        },
    )
    cfg.rewards["loco_orientation"] = RewardTermCfg(
        func=rewards.loco_orientation,
        weight=-4.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["loco_lin_vel_z"] = RewardTermCfg(
        func=rewards.loco_lin_vel_z,
        weight=-1.0,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["loco_ang_vel_xy"] = RewardTermCfg(
        func=rewards.loco_ang_vel_xy,
        weight=-0.5,
        params={"command_name": "velocity_height"},
    )
    cfg.rewards["loco_dof_pos_limit_cost"] = RewardTermCfg(
        func=rewards.loco_dof_pos_limit_cost,
        weight=-5.0,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["loco_torque_limit_cost"] = RewardTermCfg(
        func=rewards.loco_torque_limit_cost,
        weight=-2.0e-4,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["loco_dof_vel_limit_cost"] = RewardTermCfg(
        func=rewards.loco_dof_vel_limit_cost,
        weight=-1.0e-4,
        params={
            "command_name": "velocity_height",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


def task_mode_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """统一 Task Mode 训练环境。"""
    cfg = flat_env_cfg(play=play)
    apply_task_mode_command(cfg)
    apply_task_mode_observations(cfg)
    apply_task_mode_rewards(cfg)
    return cfg


def loco_light_terrain_cfg() -> TerrainGeneratorCfg:
    """FlowMatch 四模式后期课程使用的低强度地形子集。"""
    return TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=1.0,
        num_rows=5,
        num_cols=6,
        color_scheme="height",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.25),
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=0.20,
                noise_range=(0.01, 0.04),
                noise_step=0.01,
                border_width=0.25,
            ),
            "wave_terrain": HfWaveTerrainCfg(
                proportion=0.15,
                amplitude_range=(0.0, 0.06),
                num_waves=2,
                border_width=0.25,
            ),
            "open_stairs": BoxOpenStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.03, 0.08),
                step_width_range=(0.45, 0.80),
                platform_width=1.5,
                border_width=0.25,
                inverted=False,
            ),
            "random_spread_boxes": BoxRandomSpreadTerrainCfg(
                proportion=0.15,
                num_boxes=30,
                box_width_range=(0.10, 0.35),
                box_length_range=(0.10, 0.60),
                box_height_range=(0.03, 0.08),
                platform_width=1.5,
                border_width=0.25,
            ),
            "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                obstacle_width_range=(0.20, 0.50),
                obstacle_height_range=(0.03, 0.08),
                num_obstacles=20,
                platform_width=1.5,
                border_width=0.25,
            ),
        },
        difficulty_range=(0.0, 0.6),
        add_lights=True,
    )


def loco_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """FlowMatch 四模式训练环境，不采样 jump。"""
    cfg = flat_env_cfg(play=play)
    apply_task_mode_command(
        cfg,
        mode_probabilities=(0.45, 0.20, 0.20, 0.15, 0.0),
        jump_prob=0.0,
        enable_mode_switch=True,
    )
    apply_task_mode_observations(cfg)
    apply_loco_task_mode_rewards(cfg)
    cfg.commands["velocity_height"].height_range = _STANDING_HEIGHT_RANGE
    cfg.commands["velocity_height"].standing_height_range = _STANDING_HEIGHT_RANGE
    cfg.commands["velocity_height"].jump_height_range = (0.0, 0.0)

    if not play:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {
                "step": 0,
                "lin_vel_x_range": (0.0, 0.0),
                "ang_vel_yaw_range": (0.0, 0.0),
            },
            {
                "step": 100,
                "lin_vel_x_range": (-0.5, 0.5),
                "ang_vel_yaw_range": (-0.5, 0.5),
            },
            {
                "step": 200,
                "lin_vel_x_range": (-1.0, 1.0),
                "ang_vel_yaw_range": (-1.0, 1.0),
            },
            {
                "step": 300,
                "lin_vel_x_range": (-1.5, 1.5),
                "ang_vel_yaw_range": (-1.5, 1.5),
            },
            {
                "step": 400,
                "lin_vel_x_range": (-2.0, 2.0),
                "ang_vel_yaw_range": (-2.0, 2.0),
            },
            {
                "step": 500,
                "lin_vel_x_range": (-2.5, 2.5),
                "ang_vel_yaw_range": (-2.5, 2.5),
            },
        ]

    return cfg


def loco_script_play_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flow 脚本切换 play 环境，由外部脚本显式写入 mode。"""
    cfg = loco_env_cfg(play=play)
    command = cfg.commands["velocity_height"]
    command.mode_probabilities = (1.0, 0.0, 0.0, 0.0, 0.0)
    command.enable_mode_switch = False
    command.jump_prob = 0.0
    command.lin_vel_x_range = (-2.5, 2.5)
    command.ang_vel_yaw_range = (-3.0, 3.0)
    command.height_range = _STANDING_HEIGHT_RANGE
    command.standing_height_range = _STANDING_HEIGHT_RANGE
    command.jump_height_range = (0.0, 0.0)
    if play:
        command.debug_vis = True
    cfg.terminations["base_link_contact"] = TerminationTermCfg(
        func=terminations.base_link_contact_delayed,
        time_out=False,
        params={
            "sensor_name": "collision_sensor",
            "force_threshold": 1.0,
            "delay_steps": 20,
        },
    )
    cfg.terminations["low_base_height"] = TerminationTermCfg(
        func=terminations.gait_low_base_height_delayed,
        time_out=False,
        params={
            "sensor_name": "base_height_sensor",
            "min_height": 0.12,
            "max_steps": 25,
        },
    )
    return cfg


def single_label_env_cfg(
    mode: TaskMode,
    play: bool = False,
    use_light_terrain: bool = False,
) -> ManagerBasedRlEnvCfg:
    """构造 FlowMatch 单语义标签训练环境。"""
    cfg = task_mode_env_cfg(play=play) if mode == TaskMode.JUMP else loco_env_cfg(play=play)

    probabilities = [0.0] * len(TaskMode)
    probabilities[int(mode)] = 1.0
    command = cfg.commands["velocity_height"]
    command.mode_probabilities = tuple(probabilities)
    command.jump_prob = 1.0 if mode == TaskMode.JUMP else 0.0
    command.enable_mode_switch = False

    if mode == TaskMode.JUMP:
        cfg.events.pop("push_robots", None)
        cfg.curriculum.pop("push_disturbance", None)
        command.height_range = (_DEFAULT_STANDING_HEIGHT, _DEFAULT_STANDING_HEIGHT)
        command.standing_height_range = (_DEFAULT_STANDING_HEIGHT, _DEFAULT_STANDING_HEIGHT)
        command.jump_height_range = (0.1, 0.3)
        command.rsi_takeoff_prob = 0.0
        command.rsi_random_frame = False
        command.jump_cool_down_steps = 100
    else:
        command.jump_height_range = (0.0, 0.0)

    if use_light_terrain:
        cfg.scene.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=loco_light_terrain_cfg(),
            max_init_terrain_level=1,
        )
    return cfg


def wheel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造纯平地 WHEEL 专家训练环境。"""
    cfg = single_label_env_cfg(TaskMode.WHEEL, play=play)

    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=curriculums.wheel_expert_motion_curriculum,
            params={
                "command_name": "velocity_height",
                "velocity_stages": [
                    {
                        "iter": 0,
                        "lin_vel_x_range": (0.0, 0.0),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iter": 300,
                        "lin_vel_x_range": (-0.4, 0.4),
                        "ang_vel_yaw_range": (-0.4, 0.4),
                    },
                    {
                        "iter": 600,
                        "lin_vel_x_range": (-0.8, 0.8),
                        "ang_vel_yaw_range": (-0.8, 0.8),
                    },
                    {
                        "iter": 1000,
                        "lin_vel_x_range": (-1.2, 1.2),
                        "ang_vel_yaw_range": (-1.5, 1.5),
                    },
                    {
                        "iter": 1400,
                        "lin_vel_x_range": (-1.8, 1.8),
                        "ang_vel_yaw_range": (-2.2, 2.2),
                    },
                    {
                        "iter": 1800,
                        "lin_vel_x_range": (-2.5, 2.5),
                        "ang_vel_yaw_range": (-3.0, 3.0),
                    },
                ],
                "profile_stages": [
                    {
                        "iter": 0,
                        "wheel_profile_probabilities": (1.0, 0.0, 0.0),
                    },
                    {
                        "iter": 600,
                        "wheel_profile_probabilities": (0.6, 0.25, 0.15),
                    },
                    {
                        "iter": 1000,
                        "wheel_profile_probabilities": (0.5, 0.3, 0.2),
                    },
                    {
                        "iter": 1400,
                        "wheel_profile_probabilities": (0.4, 0.35, 0.25),
                    },
                ],
                "reward_weight_stages": [
                    {
                        "term_name": "wheel_straight_yaw_drift",
                        "start_iter": 800,
                        "ramp_iters": 1200,
                        "initial_weight": 0.0,
                        "final_weight": -2.0,
                    },
                    {
                        "term_name": "wheel_straight_lateral_vel",
                        "start_iter": 800,
                        "ramp_iters": 1200,
                        "initial_weight": 0.0,
                        "final_weight": -1.0,
                    },
                    {
                        "term_name": "wheel_in_place_linear_vel",
                        "start_iter": 800,
                        "ramp_iters": 1200,
                        "initial_weight": 0.0,
                        "final_weight": -2.0,
                    },
                ],
            },
        )

    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.BodyContactDelayed(),
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 0.2,
            "delay_steps": 20,
        },
    )
    cfg.terminations["base_link_contact"] = TerminationTermCfg(
        func=terminations.BodyContactDelayed(),
        time_out=False,
        params={
            "sensor_name": "collision_sensor",
            "force_threshold": 1.0,
            "delay_steps": 20,
        },
    )
    cfg.rewards["is_terminated"] = RewardTermCfg(func=rewards.is_terminated, weight=-100.0)
    cfg.rewards["leg_contact_penalty"] = RewardTermCfg(
        func=terminations.BodyContactGracePenalty(),
        weight=-100.0,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 0.2,
            "delay_steps": 20,
        },
    )
    cfg.rewards["base_link_contact_penalty"] = RewardTermCfg(
        func=terminations.BodyContactGracePenalty(),
        weight=-100.0,
        params={
            "sensor_name": "collision_sensor",
            "force_threshold": 1.0,
            "delay_steps": 20,
        },
    )

    return cfg


def loco_light_terrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """FlowMatch 四模式的轻地形/低障碍课程环境。"""
    cfg = loco_env_cfg(play=play)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=loco_light_terrain_cfg(),
        max_init_terrain_level=1,
    )
    return cfg
