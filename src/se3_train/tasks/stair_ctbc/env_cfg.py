"""CTBC 台阶训练环境配置。"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_shared import (
    REFERENCE_CTBC_CONTACT_WINDOW,
    REFERENCE_CTBC_FF_AMPLITUDE,
    REFERENCE_CTBC_FF_PERIOD_S,
    REFERENCE_CTBC_FORCE_THRESHOLD_N,
    REFERENCE_CTBC_HIP_RATIO,
    REFERENCE_CTBC_KNEE_RATIO,
    REFERENCE_CTBC_LEG_LENGTH_AMPLITUDE_M,
    REFERENCE_CTBC_LEG_SCALE,
    REFERENCE_CTBC_SWING_ANGLE_AMPLITUDE_RAD,
)
from se3_train.mdp import curriculums, events, stair_rewards, terminations
from se3_train.robot_cfg import STAIR_FOURBAR_SURROGATE_MJCF_PATH, get_serialleg_cfg
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg
from se3_train.tasks.stair_ctbc.terrains import BoxRampTerrainCfg, BoxStageStairsTerrainCfg

_REFERENCE_CTBC_WHEEL_SCALE = 45.0
_REFERENCE_CTBC_WHEEL_AMP = 1.5
_STAIR_REWARD_TERRAIN_TYPES = ("stage_stairs", "inv_pyramid_stairs")
_TERRAIN_CURRICULUM_TYPES = (
    "stage_stairs",
    "inv_pyramid_stairs",
    "ramp_43deg_400mm",
    "ramp_17deg_350mm",
)
_PLAY_NUM_ENVS = len(_TERRAIN_CURRICULUM_TYPES)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成台阶 CTBC 环境，继承 recovery 能力并覆盖回台阶训练分布。"""
    cfg = recovery_env_cfg(play=play)
    cfg.scene.entities["robot"] = get_serialleg_cfg(mjcf_path=STAIR_FOURBAR_SURROGATE_MJCF_PATH)
    cfg.sim.nconmax = 96
    cfg.sim.njmax = 640

    stair_sub_terrains = (
        {
            "stage_stairs": BoxStageStairsTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
            ),
            "inv_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
                step_height_range=(0.04, 0.20),
                step_width=0.5,
                platform_width=2.0,
                border_width=1.0,
            ),
            "ramp_43deg_400mm": BoxRampTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
                slope_deg=43.0,
                height=0.40,
            ),
            "ramp_17deg_350mm": BoxRampTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
                slope_deg=17.0,
                height=0.35,
                top_platform_length=0.0,
            ),
        }
        if play
        else {
            "stage_stairs": BoxStageStairsTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
            ),
            "inv_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                size=(8.0, 8.0),
                step_height_range=(0.04, 0.20),
                step_width=0.5,
                platform_width=2.0,
                border_width=1.0,
            ),
            "ramp_43deg_400mm": BoxRampTerrainCfg(
                proportion=0.20,
                size=(8.0, 8.0),
                slope_deg=43.0,
                height=0.40,
            ),
            "ramp_17deg_350mm": BoxRampTerrainCfg(
                proportion=0.20,
                size=(8.0, 8.0),
                slope_deg=17.0,
                height=0.35,
                top_platform_length=0.0,
            ),
            "flat": BoxFlatTerrainCfg(proportion=0.10, size=(8.0, 8.0)),
        }
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=True,
            size=(8.0, 8.0),
            border_width=20.0,
            border_height=1.0,
            num_rows=1 if play else 20,
            num_cols=_PLAY_NUM_ENVS if play else 20,
            difficulty_range=(1.0, 1.0) if play else (0.0, 1.0),
            add_lights=True,
            sub_terrains=stair_sub_terrains,
        ),
        max_init_terrain_level=0,
    )
    if play:
        # curriculum=True 时 MJLab 每种地形固定一列；play 默认四个 env 正好覆盖四类地形。
        cfg.scene.num_envs = _PLAY_NUM_ENVS
    cfg.scene.env_spacing = 4.0

    wheel_riser_sensor = ContactSensorCfg(
        name="wheel_riser_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link|r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "normal", "tangent"),
        reduce="maxforce",
        num_slots=4,
        global_frame=True,
    )
    left_wheel_center_height_sensor = TerrainHeightSensorCfg(
        name="left_wheel_center_height_sensor",
        frame=ObjRef(type="body", name="l_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg(rings=(), include_center=True),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )
    right_wheel_center_height_sensor = TerrainHeightSensorCfg(
        name="right_wheel_center_height_sensor",
        frame=ObjRef(type="body", name="r_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg(rings=(), include_center=True),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )
    cfg.scene.sensors = (
        *tuple(cfg.scene.sensors or ()),
        left_wheel_center_height_sensor,
        right_wheel_center_height_sensor,
        wheel_riser_sensor,
    )

    cfg.actions["delayed_action"].height_conditioned_action_default = True
    cfg.actions["delayed_action"].action_default_command_name = "velocity_height"

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.20, 1.00)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    command_cfg.height_range = (0.38, 0.39)
    command_cfg.standing_height_range = (0.38, 0.39)
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = 0.0
    command_cfg.jump_prob = 0.0
    command_cfg.enable_jump_lifecycle = False
    command_cfg.enable_jump_metrics = False
    _disable_recovery_command_schedules(command_cfg)
    if play:
        command_cfg.lin_vel_x_range = (0.25, 0.25)
        command_cfg.height_range = (0.39, 0.39)
        command_cfg.standing_height_range = (0.39, 0.39)

    cfg.terminations["bad_orientation"] = TerminationTermCfg(
        func=terminations.bad_orientation_delayed,
        time_out=False,
        params={"limit_angle": 0.698, "max_steps": 100},
    )
    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact,
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 80.0,
            "command_name": "velocity_height",
            "terminate": False,
        },
    )

    new_events = dict(cfg.events)
    new_events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_robotlab_full_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_xy_range": (0.0, 0.0) if play else (-0.03, 0.03),
            "pos_xy_offset": (0.0, 0.0),
            "height_offset_range": (0.0, 0.0) if play else (0.0, 0.2),
            "roll_range": (0.0, 0.0),
            "pitch_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
            "lin_vel_range": (0.0, 0.0),
            "ang_vel_range": (0.0, 0.0),
            "clearance_range": (0.0, 0.02) if play else (0.0, 0.05),
        },
    )
    new_events["reset_joints"] = EventTermCfg(
        func=events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "full_joint_randomization": False,
            "full_front_joint_offset_range": math.pi,
            "full_active_rod_angle_range": (0.0, 0.0),
            "align_root_height_to_wheels": True,
            "height_conditioned_default": True,
            "command_name": "velocity_height",
            "terrain_height_sensor_names": (
                "left_wheel_center_height_sensor",
                "right_wheel_center_height_sensor",
            ),
            "allow_wheel_clearance_lowering": True,
            "max_wheel_clearance_adjustment": 0.25,
        },
    )
    new_events["init_stair_climb_state"] = EventTermCfg(
        func=events.init_stair_climb_state,
        mode="startup",
        params={
            "contact_window": REFERENCE_CTBC_CONTACT_WINDOW,
            "force_threshold_n": REFERENCE_CTBC_FORCE_THRESHOLD_N,
            "ff_amplitude_rad": REFERENCE_CTBC_FF_AMPLITUDE,
            "ff_period_s": REFERENCE_CTBC_FF_PERIOD_S,
            "cooldown_s": 0.4,
            "reference_leg_scale": REFERENCE_CTBC_LEG_SCALE,
            "hip_ratio": REFERENCE_CTBC_HIP_RATIO,
            "knee_ratio": REFERENCE_CTBC_KNEE_RATIO,
            "ff_style": "leg_length",
            "leg_length_amplitude_m": REFERENCE_CTBC_LEG_LENGTH_AMPLITUDE_M,
            "swing_angle_amplitude_rad": REFERENCE_CTBC_SWING_ANGLE_AMPLITUDE_RAD,
            "ann_start_iter": 800,
            "ann_end_iter": 1800,
            "phantom_trigger_iter": 0,
        },
    )
    new_events["step_stair_climb_state"] = EventTermCfg(
        func=events.step_stair_climb_state,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "sensor_name": "wheel_sensor",
            "riser_sensor_name": "wheel_riser_sensor",
            "riser_normal_z_max": 0.5,
            "num_steps_per_env": 64,
            "stair_terrain_type_names": ("stage_stairs", "inv_pyramid_stairs"),
            "disable_during_recovery": True,
        },
    )
    new_events["reset_stair_climb_state"] = EventTermCfg(
        func=events.reset_stair_climb_state,
        mode="reset",
    )
    new_events.pop("push_robots", None)
    cfg.events = new_events

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
                            "lin_vel_x_range": (0.20, 1.00),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "iteration": 400,
                            "lin_vel_x_range": (0.20, 1.50),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                        {
                            "iteration": 900,
                            "lin_vel_x_range": (0.20, 2.00),
                            "ang_vel_yaw_range": (0.0, 0.0),
                        },
                    ],
                },
            ),
            "terrain_levels": CurriculumTermCfg(
                func=curriculums.terrain_levels,
                params={
                    "use_iterations": True,
                    "steps_per_policy_iter": 64,
                    "level_stages": [
                        {"iteration": 0, "max_difficulty": 0.0},
                        {"iteration": 300, "max_difficulty": 0.30},
                        {"iteration": 700, "max_difficulty": 0.50},
                        {"iteration": 1100, "max_difficulty": 0.70},
                        {"iteration": 1500, "max_difficulty": 1.0},
                    ],
                    "terrain_type_names": _TERRAIN_CURRICULUM_TYPES,
                },
            ),
        }
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=stair_rewards.stair_forward_progress,
        weight=3.0,
        params={"command_name": "velocity_height", "sigma": 0.25},
    )
    if "flat_leg_contact" in cfg.rewards:
        cfg.rewards["flat_leg_contact"].weight = -2.0
    if "collision" in cfg.rewards:
        cfg.rewards["collision"].weight = -2.0
    if "action_rate" in cfg.rewards:
        cfg.rewards["action_rate"].func = stair_rewards.action_rate_no_ctbc
    if "joint_pos_penalty" in cfg.rewards:
        cfg.rewards["joint_pos_penalty"].func = stair_rewards.joint_pos_penalty_no_ctbc
    if "leg_torques" in cfg.rewards:
        cfg.rewards["leg_torques"].func = stair_rewards.leg_torques_no_ctbc
    if "leg_power" in cfg.rewards:
        cfg.rewards["leg_power"].func = stair_rewards.leg_power_no_ctbc
    if "stand_still" in cfg.rewards:
        cfg.rewards["stand_still"].func = stair_rewards.stand_still_no_ctbc
    if "contact_forces" in cfg.rewards:
        cfg.rewards["contact_forces"].func = stair_rewards.contact_forces_no_ctbc
        cfg.rewards["contact_forces"].weight = -2.0e-4
    if "tracking_height" in cfg.rewards:
        cfg.rewards["tracking_height"].weight = 3.0
        cfg.rewards["tracking_height"].params.update(
            {
                "sigma": 0.04,
                "kernel": "exp",
                "use_upright_gate": False,
                "use_pose_end_gate": False,
            }
        )

    cfg.rewards["stair_climb_height"] = RewardTermCfg(
        func=stair_rewards.stair_climb_height,
        weight=3.0,
        params={
            "command_name": "velocity_height",
            "max_gain": 0.35,
            "forward_gate_start": 0.10,
            "forward_gate_width": 0.25,
            "terrain_type_names": _STAIR_REWARD_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_forward_distance"] = RewardTermCfg(
        func=stair_rewards.stair_forward_distance,
        weight=1.0,
        params={
            "max_progress": 1.0,
            "terrain_type_names": _STAIR_REWARD_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_feet_clearance"] = RewardTermCfg(
        func=stair_rewards.stair_feet_clearance,
        weight=1.0,
        params={"sensor_name": "wheel_sensor", "h_min": 0.08, "h_max": 0.35},
    )
    cfg.rewards["stair_feet_air_time"] = RewardTermCfg(
        func=stair_rewards.stair_feet_air_time,
        weight=1.0,
        params={"sensor_name": "wheel_sensor"},
    )
    cfg.rewards["stair_contact_number"] = RewardTermCfg(
        func=stair_rewards.stair_contact_number,
        weight=1.0,
        params={"sensor_name": "wheel_sensor"},
    )
    cfg.rewards["stair_wheel_swing_zero_vel"] = RewardTermCfg(
        func=stair_rewards.stair_wheel_swing_zero_vel,
        weight=0.25,
        params={"sensor_name": "wheel_sensor"},
    )
    cfg.rewards["stair_contact_diagnostics"] = RewardTermCfg(
        func=stair_rewards.stair_contact_diagnostics,
        weight=0.001,
        params={
            "wheel_sensor_name": "wheel_sensor",
            "leg_sensor_name": "leg_contact_sensor",
            "collision_sensor_name": "collision_sensor",
            "force_threshold": 1.0,
        },
    )

    cfg.episode_length_s = 8.0 if not play else 9999.0
    return cfg


def _disable_recovery_command_schedules(command_cfg: object) -> None:
    """关闭 recovery 专属指令调度，避免台阶任务先蹲低或随机转向。"""
    overrides = {
        "height_balance_schedule_enabled": False,
        "exclusive_linear_yaw_commands": False,
        "linear_command_ratio": 1.0,
    }
    for name, value in overrides.items():
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, value)


def _ctbc_wheel_amp(action_cfg: object) -> float:
    """按当前 wheel action scale 折算 CTBC 轮速前馈幅值。"""
    wheel_scale = float(getattr(action_cfg, "wheel_scale", _REFERENCE_CTBC_WHEEL_SCALE))
    if wheel_scale <= 1.0e-6:
        return _REFERENCE_CTBC_WHEEL_AMP
    return _REFERENCE_CTBC_WHEEL_AMP * _REFERENCE_CTBC_WHEEL_SCALE / wheel_scale
