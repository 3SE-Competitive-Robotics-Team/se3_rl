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
    BoxRandomGridTerrainCfg,
    BoxRandomSpreadTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_shared import TaskMode
from se3_shared.grounded_pose import solve_grounded_pose
from se3_train.mdp.jump_trajectories import DEFAULT_JUMP_TRAJ_HEIGHTS, DEFAULT_JUMP_TRAJ_PATHS
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import commands, curriculums, observations, rewards, terminations

_DEFAULT_STANDING_HEIGHT = 0.22
_STANDING_HEIGHT_RANGE = (0.20, 0.32)
_GAIT_HEIGHT = 0.35
_GAIT_SWING_CLEARANCE_M = 0.04


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
    cfg.rewards["wheel_default_pose"] = RewardTermCfg(
        func=rewards.wheel_default_pose,
        weight=-1.0,
        params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
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


def gait_finetune_light_terrain_cfg() -> TerrainGeneratorCfg:
    """GAIT fine-tune 使用的低强度崎岖和低矮障碍地形。"""
    return TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=1.0,
        num_rows=4,
        num_cols=5,
        color_scheme="height",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.79),
            "random_grid": BoxRandomGridTerrainCfg(
                proportion=0.10,
                grid_width=0.55,
                grid_height_range=(0.020, 0.090),
                platform_width=1.4,
                merge_similar_heights=True,
                height_merge_threshold=0.006,
                max_merge_distance=3,
                border_width=0.25,
            ),
            "random_spread_boxes": BoxRandomSpreadTerrainCfg(
                proportion=0.03,
                num_boxes=14,
                box_width_range=(0.08, 0.22),
                box_length_range=(0.08, 0.35),
                box_height_range=(0.020, 0.10),
                platform_width=1.2,
                border_width=0.25,
            ),
            "open_stairs_up": BoxOpenStairsTerrainCfg(
                proportion=0.06,
                step_height_range=(0.020, 0.090),
                step_width_range=(0.65, 1.00),
                platform_width=1.4,
                border_width=0.25,
                step_thickness=0.05,
                inverted=True,
            ),
            "open_stairs_down": BoxOpenStairsTerrainCfg(
                proportion=0.02,
                step_height_range=(0.020, 0.090),
                step_width_range=(0.65, 1.00),
                platform_width=1.4,
                border_width=0.25,
                step_thickness=0.05,
                inverted=False,
            ),
        },
        difficulty_range=(0.0, 0.8),
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
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
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
        ]

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


def gait_pretrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """纯 GAIT 模式预训练。"""
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
        _GAIT_HEIGHT,
        keep_wheel_x=False,
        align_com_x=True,
    )
    if not gait_grounded_pose.success:
        raise ValueError(
            "无法求解 GAIT 初始触地姿态: "
            f"base_height={_GAIT_HEIGHT}, message={gait_grounded_pose.message}"
        )

    cfg.actions["delayed_action"].leg_scale = 0.25
    cfg.actions["delayed_action"].wheel_scale = 0.0
    cfg.actions["delayed_action"].wheel_lock_damping = 0.0
    cfg.actions["delayed_action"].freeze_wheels = True
    cfg.commands["velocity_height"].height_range = (_GAIT_HEIGHT, _GAIT_HEIGHT)
    cfg.commands["velocity_height"].standing_height_range = (_GAIT_HEIGHT, _GAIT_HEIGHT)
    cfg.events["reset_root_state"].params["base_height"] = _GAIT_HEIGHT
    cfg.events["reset_joints"].params["joint_pos_override"] = gait_grounded_pose.q6
    cfg.events["reset_joints"].params["update_default_joint_pos"] = True
    cfg.commands["velocity_height"].jump_height_range = (0.0, 0.0)
    cfg.commands["velocity_height"].lin_vel_x_range = (-0.35, 0.35)
    cfg.commands["velocity_height"].ang_vel_yaw_range = (0.0, 0.0)
    cfg.commands["velocity_height"].pitch_range = (0.0, 0.0)
    cfg.commands["velocity_height"].roll_range = (0.0, 0.0)
    cfg.commands["velocity_height"].standing_ratio = 0.0
    cfg.commands["velocity_height"].lin_vel_deadband = 0.1
    cfg.commands["velocity_height"].yaw_deadband = 0.1
    if play:
        cfg.commands["velocity_height"].debug_vis = True

    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact,
        time_out=False,
        params={"sensor_name": "leg_contact_sensor", "force_threshold": 0.2},
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
        params={"limit_angle": 0.5236, "max_steps": 8},
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

    cfg.rewards = _gait_pretrain_rewards()
    cfg.episode_length_s = 20.0
    return cfg


def _gait_pretrain_rewards() -> dict[str, RewardTermCfg]:
    """构造纯 GAIT 预训练奖励。"""
    return {
        "is_terminated": RewardTermCfg(func=rewards.is_terminated, weight=-100.0),
        "gait_tracking_lin_vel": RewardTermCfg(
            func=rewards.mode_tracking_lin_vel,
            weight=1.0,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.25,
                "sigma_stand": 0.12,
                "vz_weight": 1.0,
                "modes": (int(TaskMode.GAIT),),
            },
        ),
        "gait_tracking_ang_vel": RewardTermCfg(
            func=rewards.mode_tracking_ang_vel,
            weight=0.2,
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
            weight=-18.0,
            params={"command_name": "velocity_height"},
        ),
        "flat_base_height": RewardTermCfg(
            func=rewards.flat_base_height_penalty_no_jump,
            weight=-18.0,
            params={
                "command_name": "velocity_height",
                "sigma": 0.03,
                "height_sensor_name": "base_height_sensor",
            },
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-10.0,
            params={"soft_limit_deg": 8.0, "hard_limit_deg": 24.0, "max_penalty": 4.0},
        ),
        "gait_low_base_height_barrier": RewardTermCfg(
            func=rewards.gait_low_base_height_barrier,
            weight=-10.0,
            params={
                "command_name": "velocity_height",
                "height_sensor_name": "base_height_sensor",
                "soft_min_height": 0.35,
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
            weight=-100.0,
            params={
                "sensor_name": "leg_contact_sensor",
                "force_threshold": 0.2,
            },
        ),
        "lin_vel_z": RewardTermCfg(func=rewards.lin_vel_z, weight=-0.5),
        "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.25),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-8.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-5.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=rewards.dof_pos_limits,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(func=rewards.action_rate, weight=-0.03),
        "gait_no_wheel_drive": RewardTermCfg(
            func=rewards.gait_no_wheel_drive,
            weight=-8.0,
            params={"command_name": "velocity_height", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "gait_leg_contact_force": RewardTermCfg(
            func=rewards.gait_leg_contact_force,
            weight=-120.0,
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
            weight=6.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "leg_sensor_name": "leg_contact_sensor",
                "target_clearance": _GAIT_SWING_CLEARANCE_M,
                "balance_window_s": 0.6,
                "contact_force_threshold": 1.0,
                "leg_contact_force_threshold": 0.2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "gait_single_support_contact": RewardTermCfg(
            func=rewards.gait_single_support_contact,
            weight=2.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "contact_force_threshold": 1.0,
            },
        ),
        "gait_single_support_air_time": RewardTermCfg(
            func=rewards.gait_single_support_air_time,
            weight=6.0,
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
            weight=-28.0,
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
            weight=18.0,
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
            weight=26.0,
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
    }


def gait_finetune_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """纯 GAIT fine-tune 环境。"""
    cfg = gait_pretrain_env_cfg(play=play)

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=gait_finetune_light_terrain_cfg(),
        max_init_terrain_level=0,
    )

    cfg.commands["velocity_height"].lin_vel_x_range = (0.05, 1.5) if play else (0.05, 0.12)
    cfg.commands["velocity_height"].lin_vel_deadband = 0.03

    cfg.rewards["gait_tracking_lin_vel"].weight = 4.0
    cfg.rewards["gait_tracking_lin_vel"].params["sigma_move"] = 0.2
    cfg.rewards["gait_tracking_lin_vel"].params["sigma_stand"] = 0.08
    cfg.rewards["gait_tracking_lin_vel"].params["vz_weight"] = 1.0
    cfg.rewards["gait_tracking_ang_vel"].weight = 0.1
    cfg.rewards["gait_natural_swing_clearance"].weight = 2.0
    cfg.rewards["gait_single_support_contact"].weight = 1.0
    cfg.rewards["gait_single_support_air_time"].weight = 2.0
    cfg.rewards["gait_air_time"].weight = 5.0
    cfg.rewards["gait_alternating_air_time"].weight = 8.0
    cfg.rewards["gait_swing_side_balance_penalty"].weight = -12.0
    cfg.rewards["action_rate"].weight = -0.06
    cfg.rewards["leg_dof_acc"].weight = -5.0e-7
    cfg.rewards["leg_power"].weight = -8.0e-4
    cfg.rewards["ang_vel_xy"].weight = -0.4
    cfg.rewards["flat_base_height"].weight = -12.0
    cfg.rewards["gait_leg_contact_force"].weight = -160.0
    cfg.rewards["leg_contact_penalty"].weight = -160.0

    cfg.rewards["gait_action_smoothness"] = RewardTermCfg(
        func=rewards.gait_action_smoothness,
        weight=-0.08,
        params={
            "command_name": "velocity_height",
            "max_penalty": 80.0,
        },
    )
    cfg.rewards["gait_touchdown_softness"] = RewardTermCfg(
        func=rewards.gait_touchdown_softness,
        weight=-8.0,
        params={
            "command_name": "velocity_height",
            "sensor_name": "wheel_sensor",
            "allowed_down_vel": 0.12,
            "max_penalty": 4.0,
            "contact_force_threshold": 1.0,
        },
    )
    cfg.rewards["gait_touchdown_support_alignment"] = RewardTermCfg(
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
    )
    if not play:
        cfg.curriculum["command_vel_linear"] = CurriculumTermCfg(
            func=curriculums.commands_vel_linear,
            params={
                "command_name": "velocity_height",
                "start_step": 0,
                "end_step": 19200,
                "start_lin_vel_x_range": (0.05, 0.12),
                "end_lin_vel_x_range": (0.05, 1.5),
                "ang_vel_yaw_range": (0.0, 0.0),
            },
        )
        cfg.curriculum["gait_terrain_distribution"] = CurriculumTermCfg(
            func=curriculums.gait_terrain_distribution_linear,
            params={
                "start_step": 0,
                "end_step": 19200,
                "start_proportions": (0.89, 0.06, 0.01, 0.03, 0.01),
                "end_proportions": (0.30, 0.27, 0.19, 0.18, 0.06),
                "start_max_level": 0,
                "end_max_level": 3,
            },
        )
    return cfg
