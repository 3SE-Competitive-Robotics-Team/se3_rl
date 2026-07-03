"""倒金字塔 CTBC 台阶任务环境配置。"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import terminations
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.recovery_discovery.env_cfg import (
    _configure_discovery_reward_contract as _configure_recovery_discovery_reward_contract,
)

from . import curriculums, events, observations, rewards
from .forward_stairs import BoxForwardStairsTerrainCfg

_ROBOT_DEFAULTS = SharedRobotConfig()
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_STAIR_RECOVERY_STATE_CACHE_PATH = (
    _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_closedchain_stair_v3_40k.npz"
)
_STAIR_WHEEL_KD = 0.08
_STAIR_COMMAND_WHEEL_RADIUS = 0.060
_STAIR_COMMAND_HALF_TRACK = 0.200725
_STAIR_COMMAND_WHEEL_SPEED_FRACTION = 0.70
_STAIR_COMMAND_INITIAL_VEL_X_RANGE = (1.30, 1.40)
_STAIR_SUPPORT_FORCE_THRESHOLD_N = 5.0
_STAIR_SUPPORT_CLEARANCE_TOL_M = 0.025
_STAIR_SUPPORT_DURATION_S = 0.30
_STAIR_BASE_COLLISION_THRESHOLD_N = 10.0
_STAIR_RECOVERY_GRACE_STEPS = 400
_STAIR_TERRAIN_TYPES = ("forward_stairs",)
_RECOVERY_TERRAIN_TYPES = ("flat",)
_TASK_MIXTURE_STAIR_PROB = 0.85
_TASK_MIXTURE_SHARED_PROB = 0.15
_DISCOVERY_MAX_LIN_VEL_X = 1.89
_DISCOVERY_MAX_ANG_VEL_YAW = 9.41
_DISCOVERY_HEIGHT_RANGE = (0.195, 0.390)
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height
_STAIR_STEP_HEIGHT_RANGE = (0.05, 0.20)
_STAIR_COMMAND_HEIGHT_CLEARANCE_M = 0.11
_STAIR_COMMAND_INITIAL_HEIGHT_M = 0.34
_STAIR_COMMAND_HEIGHT_MIN = 0.195
_STAIR_COMMAND_HEIGHT_MAX = 0.39
_INITIAL_STAIR_HEIGHT_RANGE = (
    _STAIR_COMMAND_INITIAL_HEIGHT_M,
    _STAIR_COMMAND_INITIAL_HEIGHT_M,
)
_WALKING_PHASE_ITERATIONS = 0
_STEPS_PER_POLICY_ITER = 64
_WATCH_ITER_ENV = "SE3_WATCH_ITER"
_WATCH_TERRAIN_LEVEL_ENV = "SE3_WATCH_TERRAIN_LEVEL"
_WATCH_COMMAND_HEIGHT_ENV = "SE3_WATCH_COMMAND_HEIGHT"
_TRAIN_VIEW_ITER_ENV = "SE3_TRAIN_VIEW_ITER"
_TRAIN_VIEW_TERRAIN_LEVEL_ENV = "SE3_TRAIN_VIEW_TERRAIN_LEVEL"
_TRAIN_VIEW_COMMAND_HEIGHT_ENV = "SE3_TRAIN_VIEW_COMMAND_HEIGHT"
_STAIR_LEVEL_MAX_STAGES = (
    (0, 2),
    (900, 4),
    (1400, 6),
    (2000, 7),
    (2600, 9),
)
_STAIR_LEVEL_BUCKETS = (
    (0, 2),
    (3, 6),
    (7, 9),
)
_STAIR_BUCKET_WEIGHT_STAGES = (
    (0, (1.00, 0.00, 0.00)),
    (900, (0.80, 0.20, 0.00)),
    (1400, (0.55, 0.40, 0.05)),
    (2000, (0.35, 0.50, 0.15)),
    (2600, (0.20, 0.45, 0.35)),
    (3200, (0.15, 0.35, 0.50)),
)
_REQUIRED_STAIR_REWARD_NAMES = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "tracking_height",
    "diagnostics",
    "stair_forward_velocity",
    "stair_velocity_error",
    "stair_climb_progress",
    "stair_base_collision_force",
    "stair_lateral_offset",
    "stair_heading_error",
    "stair_commanded_stall",
    "stair_diagnostics",
)
_TASK_MIXTURE_STAGES = (
    {"iteration": 0, "stair_prob": 0.85, "shared_prob": 0.15},
    {"iteration": 900, "stair_prob": 0.82, "shared_prob": 0.18},
    {"iteration": 1400, "stair_prob": 0.80, "shared_prob": 0.20},
    {"iteration": 2000, "stair_prob": 0.78, "shared_prob": 0.22},
    {"iteration": 2600, "stair_prob": 0.75, "shared_prob": 0.25},
)


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _first_int_env(*names: str) -> int | None:
    for name in names:
        value = _int_env(name)
        if value is not None:
            return value
    return None


def _first_float_env(*names: str) -> float | None:
    for name in names:
        value = _float_env(name)
        if value is not None:
            return value
    return None


def _assert_stair_reward_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    """确认台阶任务没有被其他 reward contract 清空或覆盖。"""
    missing = [name for name in _REQUIRED_STAIR_REWARD_NAMES if name not in cfg.rewards]
    if missing:
        raise RuntimeError(
            "台阶任务 reward contract 缺少关键项: "
            + ", ".join(missing)
            + "；不要调用会清空 cfg.rewards 的 recovery/discovery contract。"
        )


def _stair_terrain_cfg() -> TerrainGeneratorCfg:
    """构造倒金字塔台阶地形；机器人出生在坑底，向外移动即上台阶。"""
    return TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=1.0,
        num_rows=10,
        num_cols=20,
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains={
            "forward_stairs": BoxForwardStairsTerrainCfg(
                proportion=0.80,
                size=(8.0, 8.0),
                step_height_range=_STAIR_STEP_HEIGHT_RANGE,
                step_depth=0.80,
                step_count=6,
                stair_start_x=1.0,
                spawn_x=1.0,
                half_width=6.0,
            ),
            "flat": BoxFlatTerrainCfg(
                proportion=0.20,
                size=(8.0, 8.0),
            ),
        },
    )


def _replace_sensor(
    sensors: tuple[object, ...] | None,
    sensor_cfg: object,
) -> tuple[object, ...]:
    """按 name 替换已有 sensor；不存在时追加。"""
    name = getattr(sensor_cfg, "name", None)
    result = []
    replaced = False
    for sensor in tuple(sensors or ()):
        if getattr(sensor, "name", None) == name:
            result.append(sensor_cfg)
            replaced = True
        else:
            result.append(sensor)
    if not replaced:
        result.append(sensor_cfg)
    return tuple(result)


def _sanitize_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    """台阶 box 接触偶发 NaN 时，将观测清洗为有限值。"""
    for group_name in ("actor", "critic"):
        group_cfg = cfg.observations.get(group_name)
        if group_cfg is not None:
            cfg.observations[group_name] = replace(group_cfg, nan_policy="sanitize")


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 CTBC teacher-forcing 的倒金字塔台阶爬升环境。"""
    cfg = flat_env_cfg(play=play)

    cfg.scene.entities["robot"] = get_serialleg_cfg(
        wheel_kd_override=_STAIR_WHEEL_KD,
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=_stair_terrain_cfg(),
        max_init_terrain_level=0,
    )
    cfg.scene.env_spacing = 4.0

    wheel_riser_sensor_cfg = ContactSensorCfg(
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
    base_collision_sensor_cfg = ContactSensorCfg(
        name="collision_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(base_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=8,
    )
    wheel_height_sensor_cfg = TerrainHeightSensorCfg(
        name="wheel_height_sensor",
        frame=(
            ObjRef(type="body", name="l_wheel_Link", entity="robot"),
            ObjRef(type="body", name="r_wheel_Link", entity="robot"),
        ),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )
    cfg.scene.sensors = _replace_sensor(cfg.scene.sensors, base_collision_sensor_cfg)
    cfg.scene.sensors = _replace_sensor(cfg.scene.sensors, wheel_height_sensor_cfg)
    cfg.scene.sensors = (*tuple(cfg.scene.sensors or ()), wheel_riser_sensor_cfg)

    cfg.actions["delayed_action"].height_conditioned_action_default = True
    cfg.actions["delayed_action"].action_default_command_name = "velocity_height"

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.resampling_time_range = (5.0, 5.0)
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    command_cfg.height_range = _INITIAL_STAIR_HEIGHT_RANGE
    command_cfg.standing_height_range = _INITIAL_STAIR_HEIGHT_RANGE
    command_cfg.height_resample_on_reset_only = True
    command_cfg.standing_ratio = 0.0
    command_cfg.lin_vel_deadband = 0.05
    command_cfg.yaw_deadband = 0.05
    command_cfg.terrain_aware_height = True
    command_cfg.terrain_height_clearance = 0.02
    command_cfg.body_collision_bottom_offset = -0.12
    command_cfg.constrain_diff_drive_commands = True
    command_cfg.diff_drive_wheel_radius = _STAIR_COMMAND_WHEEL_RADIUS
    command_cfg.diff_drive_half_track = _STAIR_COMMAND_HALF_TRACK
    command_cfg.diff_drive_max_wheel_speed = _ROBOT_DEFAULTS.action_scale[
        JointGroup.WHEEL_ACTUATORS[0]
    ]
    command_cfg.diff_drive_wheel_speed_fraction = _STAIR_COMMAND_WHEEL_SPEED_FRACTION
    command_cfg.jump_prob = 0.0
    command_cfg.enable_jump_lifecycle = False
    command_cfg.enable_jump_metrics = False
    watch_command_height = _first_float_env(
        _WATCH_COMMAND_HEIGHT_ENV,
        _TRAIN_VIEW_COMMAND_HEIGHT_ENV,
    )
    fixed_watch_command_height = watch_command_height is not None
    if fixed_watch_command_height:
        command_height = float(watch_command_height)
        command_cfg.height_range = (command_height, command_height)
        command_cfg.standing_height_range = (command_height, command_height)

    last_action_term = ObservationTermCfg(func=observations.last_actions_obs)
    ctbc_phase_term = ObservationTermCfg(func=observations.ctbc_phase_obs)
    for group_name in ("actor", "critic"):
        group_cfg = cfg.observations.get(group_name)
        if group_cfg is None:
            continue
        terms = dict(group_cfg.terms)
        terms["last_actions"] = last_action_term
        terms["jump_commands"] = ctbc_phase_term
        cfg.observations[group_name] = replace(group_cfg, terms=terms)

    _configure_recovery_discovery_reward_contract(cfg)
    cfg.rewards["upward"] = replace(
        cfg.rewards["upward"],
        func=rewards.stair_scaled_upward,
        params={
            "stair_scale": 0.1,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    tracking_height_params = dict(cfg.rewards["tracking_height"].params or {})
    tracking_height_params.update(
        {
            "ctbc_active_scale": 0.2,
            "stair_mode_scale": 0.3,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        }
    )
    cfg.rewards["tracking_height"] = replace(
        cfg.rewards["tracking_height"],
        func=rewards.ctbc_scaled_tracking_height,
        params=tracking_height_params,
    )
    cfg.rewards["stair_forward_velocity"] = RewardTermCfg(
        func=rewards.stair_phase_forward_progress,
        weight=2.2,
        params={
            "command_name": "velocity_height",
            "sigma": 0.25,
            "radial_velocity_blend": 1.0,
            "ratio_blend": 0.6,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
            "walking_phase_iterations": _WALKING_PHASE_ITERATIONS,
            "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
        },
    )
    cfg.rewards["stair_velocity_error"] = RewardTermCfg(
        func=rewards.stair_velocity_error_penalty,
        weight=-5.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.5,
            "deadband_mps": 0.15,
            "error_scale_mps": 0.8,
            "max_penalty": 1.0,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_climb_progress"] = RewardTermCfg(
        func=rewards.stair_climb_progress,
        weight=3.0,
        params={
            "max_height_gain": 1.0,
            "height_sensor_name": "wheel_height_sensor",
            "contact_sensor_name": "wheel_sensor",
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
            "contact_force_threshold_n": _STAIR_SUPPORT_FORCE_THRESHOLD_N,
            "wheel_radius_m": _STAIR_COMMAND_WHEEL_RADIUS,
            "wheel_clearance_tol_m": _STAIR_SUPPORT_CLEARANCE_TOL_M,
            "riser_sensor_name": "wheel_riser_sensor",
            "riser_contact_force_threshold_n": 1.0,
            "riser_normal_z_max": 0.5,
        },
    )
    cfg.rewards["stair_base_collision_force"] = RewardTermCfg(
        func=rewards.stair_base_collision_force_penalty,
        weight=-8.0,
        params={
            "sensor_name": "collision_sensor",
            "force_threshold_n": _STAIR_BASE_COLLISION_THRESHOLD_N,
            "force_scale_n": 80.0,
            "max_penalty": 5.0,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_lateral_offset"] = RewardTermCfg(
        func=rewards.stair_lateral_offset_penalty,
        weight=-4.0,
        params={
            "deadband_m": 0.25,
            "full_penalty_m": 0.55,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_heading_error"] = RewardTermCfg(
        func=rewards.stair_heading_error_penalty,
        weight=-3.0,
        params={
            "deadband_deg": 12.0,
            "full_penalty_deg": 60.0,
            "yaw_reference": 0.0,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_commanded_stall"] = RewardTermCfg(
        func=rewards.stair_commanded_stall,
        weight=-10.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.5,
            "forward_speed_threshold": 0.4,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_diagnostics"] = RewardTermCfg(
        func=rewards.stair_diagnostics,
        weight=1.0,
        params={
            "command_name": "velocity_height",
            "step_height_range": _STAIR_STEP_HEIGHT_RANGE,
            "min_success_steps": 1.0,
            "success_height_tolerance_m": 0.015,
            "step_depth_m": 0.80,
            "forward_progress_step_fraction": 0.85,
            "hold_duration_s": _STAIR_SUPPORT_DURATION_S,
            "upright_threshold": -0.93,
            "max_vertical_speed_mps": 0.60,
            "min_signed_x_velocity_mps": 0.35,
            "max_ang_vel_radps": 2.5,
            "max_lateral_offset_m": 0.55,
            "max_support_drop_steps": 0.20,
            "height_sensor_name": "wheel_height_sensor",
            "contact_sensor_name": "wheel_sensor",
            "leg_contact_sensor_name": "leg_contact_sensor",
            "base_contact_sensor_name": "collision_sensor",
            "riser_contact_sensor_name": "wheel_riser_sensor",
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
            "contact_force_threshold_n": _STAIR_SUPPORT_FORCE_THRESHOLD_N,
            "illegal_contact_force_threshold_n": _STAIR_BASE_COLLISION_THRESHOLD_N,
            "riser_contact_force_threshold_n": 1.0,
            "wheel_radius_m": _STAIR_COMMAND_WHEEL_RADIUS,
            "wheel_clearance_tol_m": _STAIR_SUPPORT_CLEARANCE_TOL_M,
            "riser_normal_z_max": 0.5,
            "riser_stall_duration_s": 0.15,
        },
    )
    _assert_stair_reward_contract(cfg)
    if "bad_orientation" in cfg.terminations:
        cfg.terminations["bad_orientation"].params = {
            "limit_angle": 0.698,
            "max_steps": 100,
            "recovery_grace_steps": _STAIR_RECOVERY_GRACE_STEPS,
            "recovery_terminate": False,
        }
    cfg.terminations["stair_sanity"] = TerminationTermCfg(
        func=terminations.stair_sanity,
        time_out=False,
        params={
            "task_mode_attr": "_stair_task_mode",
            "stair_task_mode_id": 0,
            "min_steps": 1,
            "max_steps": 0,
            "immediate_steps": 3,
            "delayed_steps": 12,
            "tilt_limit": 0.5236,
            "backward_vx_limit": -0.10,
            "yaw_rate_limit": 2.0,
            "yaw_error_limit": math.radians(75.0),
            "yaw_reference": 0.0,
        },
    )
    if "catastrophic_state" in cfg.terminations:
        catastrophic_params = dict(cfg.terminations["catastrophic_state"].params or {})
        catastrophic_params["ignore_recovery_leg_pos_error"] = True
        cfg.terminations["catastrophic_state"] = replace(
            cfg.terminations["catastrophic_state"],
            params=catastrophic_params,
        )
    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact_delayed,
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 80.0,
            "max_steps": 50,
            "recovery_grace_steps": _STAIR_RECOVERY_GRACE_STEPS,
            "recovery_terminate": False,
        },
    )

    cfg.events = dict(cfg.events)
    if "push_robots" in cfg.events:
        push_event_params = dict(cfg.events["push_robots"].params or {})
        push_event_params["skip_recovery_active"] = True
        cfg.events["push_robots"] = replace(
            cfg.events["push_robots"],
            params=push_event_params,
        )
    if not play:
        cfg.events["reset_root_state"] = replace(
            cfg.events["reset_root_state"],
            func=events.reset_root_state_stair_shared,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "shared_state_cache_path": str(_STAIR_RECOVERY_STATE_CACHE_PATH),
                "shared_state_cache_split": "train",
                "shared_recovery_grace_steps": _STAIR_RECOVERY_GRACE_STEPS,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
            },
        )
    reset_joint_params = dict(cfg.events["reset_joints"].params or {})
    reset_joint_params.update(
        {
            "height_conditioned_default": True,
            "command_name": "velocity_height",
            "shared_joint_offset_range": 0.25,
            "shared_joint_vel_range": (-0.50, 0.50),
            "shared_joint_randomization_prob": 0.75,
        }
    )
    cfg.events["reset_joints"] = replace(
        cfg.events["reset_joints"],
        func=events.reset_joints_stair_shared,
        params=reset_joint_params,
    )
    if not play:
        reset_events: dict[str, EventTermCfg] = {}
        task_mode_sample = EventTermCfg(
            func=events.sample_stair_shared_task_mode,
            mode="reset",
            params={
                "stair_prob": _TASK_MIXTURE_STAIR_PROB,
                "shared_prob": _TASK_MIXTURE_SHARED_PROB,
                "mixture_stages": _TASK_MIXTURE_STAGES,
                "stair_terrain_type_name": _STAIR_TERRAIN_TYPES[0],
                "shared_terrain_type_name": _RECOVERY_TERRAIN_TYPES[0],
                "max_level_stages": _STAIR_LEVEL_MAX_STAGES,
                "level_buckets": _STAIR_LEVEL_BUCKETS,
                "bucket_weight_stages": _STAIR_BUCKET_WEIGHT_STAGES,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                "balance_occupancy": True,
            },
        )
        task_mode_commands = EventTermCfg(
            func=events.apply_stair_shared_rehearsal_commands,
            mode="reset",
            params={
                "command_name": "velocity_height",
                "lin_vel_x_range": (-_DISCOVERY_MAX_LIN_VEL_X, _DISCOVERY_MAX_LIN_VEL_X),
                "ang_vel_yaw_range": (
                    -_DISCOVERY_MAX_ANG_VEL_YAW,
                    _DISCOVERY_MAX_ANG_VEL_YAW,
                ),
                "height_range": _DISCOVERY_HEIGHT_RANGE,
            },
        )
        for event_name, event_term in cfg.events.items():
            if event_name == "reset_root_state":
                reset_events["sample_stair_shared_task_mode"] = task_mode_sample
                reset_events[event_name] = event_term
                reset_events["apply_stair_shared_rehearsal_commands"] = task_mode_commands
            else:
                reset_events[event_name] = event_term
        cfg.events = reset_events
    cfg.events["init_stair_climb_state"] = EventTermCfg(
        func=events.init_stair_climb_state,
        mode="startup",
        params={
            "trigger_mode": "force",
            "contact_window": 1,
            "force_threshold_n": 10.0,
            "pitch_threshold_deg": 10.0,
            "pitch_window": 5,
            "ff_amplitude_rad": 0.0,
            "coordinate_mode": "body_polar",
            "leg_length_m": 0.18,
            "swing_angle_deg": 14.0,
            "ff_duration_s": 0.60,
            "ff_wheel_action": 0.20,
            "ff_start_iter": _WALKING_PHASE_ITERATIONS,
            "ann_start_iter": 200,
            "ann_end_iter": 500,
            "phantom_trigger_iter": 0,
            "allow_bilateral_trigger": False,
            "profile_path": None,
        },
    )
    cfg.events["step_stair_climb_state"] = EventTermCfg(
        func=events.step_stair_climb_state,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "sensor_name": "wheel_sensor",
            "riser_sensor_name": "wheel_riser_sensor",
            "riser_normal_z_max": 0.5,
            "num_steps_per_env": _STEPS_PER_POLICY_ITER,
        },
    )
    cfg.events["enforce_shared_rehearsal_commands"] = EventTermCfg(
        func=events.enforce_shared_rehearsal_commands,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "command_name": "velocity_height",
            "lin_vel_x_range": (-_DISCOVERY_MAX_LIN_VEL_X, _DISCOVERY_MAX_LIN_VEL_X),
            "ang_vel_yaw_range": (-_DISCOVERY_MAX_ANG_VEL_YAW, _DISCOVERY_MAX_ANG_VEL_YAW),
            "height_range": _DISCOVERY_HEIGHT_RANGE,
        },
    )
    cfg.events["reset_stair_climb_state"] = EventTermCfg(
        func=events.reset_stair_climb_state,
        mode="reset",
    )
    watch_iter = _first_int_env(_WATCH_ITER_ENV, _TRAIN_VIEW_ITER_ENV)
    if watch_iter is not None:
        cfg.events["set_train_view_iteration"] = EventTermCfg(
            func=events.set_train_view_iteration,
            mode="startup",
            params={
                "iteration": watch_iter,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
            },
        )

    watch_terrain_level = _first_int_env(_WATCH_TERRAIN_LEVEL_ENV, _TRAIN_VIEW_TERRAIN_LEVEL_ENV)
    force_watch_stair_terrain = watch_terrain_level is not None and (
        watch_iter is None or watch_iter >= _WALKING_PHASE_ITERATIONS
    )
    if force_watch_stair_terrain:
        cfg.events["set_fixed_stair_terrain"] = EventTermCfg(
            func=events.set_fixed_stair_terrain,
            mode="startup",
            params={
                "terrain_level": watch_terrain_level,
                "terrain_type_name": _STAIR_TERRAIN_TYPES[0],
            },
        )

    if not play:
        cfg.curriculum = dict(cfg.curriculum)
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=curriculums.commands_vel,
            params={
                "command_name": "velocity_height",
                "use_iterations": True,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                "fixed_iteration": watch_iter,
                "velocity_stages": [
                    {
                        "iteration": 0,
                        "lin_vel_x_range": _STAIR_COMMAND_INITIAL_VEL_X_RANGE,
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 200,
                        "lin_vel_x_range": _STAIR_COMMAND_INITIAL_VEL_X_RANGE,
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 500,
                        "lin_vel_x_range": _STAIR_COMMAND_INITIAL_VEL_X_RANGE,
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 900,
                        "lin_vel_x_range": (1.20, 1.90),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 1400,
                        "lin_vel_x_range": (1.00, 1.90),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 2000,
                        "lin_vel_x_range": (0.80, 1.90),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 2600,
                        "lin_vel_x_range": (0.70, 1.90),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 3200,
                        "lin_vel_x_range": (0.60, 1.90),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                ],
            },
        )
        cfg.curriculum["command_height"] = CurriculumTermCfg(
            func=curriculums.commands_height,
            params={
                "command_name": "velocity_height",
                "use_iterations": True,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                "interpolate": True,
                "fixed_iteration": watch_iter,
                "height_stages": [
                    {
                        "iteration": 0,
                        "height_range": _INITIAL_STAIR_HEIGHT_RANGE,
                    },
                    {
                        "iteration": 900,
                        "height_range": _INITIAL_STAIR_HEIGHT_RANGE,
                    },
                    {
                        "iteration": 1400,
                        "height_range": _INITIAL_STAIR_HEIGHT_RANGE,
                    },
                    {
                        "iteration": 2000,
                        "height_range": _INITIAL_STAIR_HEIGHT_RANGE,
                    },
                    {
                        "iteration": 2600,
                        "height_range": _INITIAL_STAIR_HEIGHT_RANGE,
                    },
                ],
            },
        )
        cfg.curriculum["level_aware_height_floor"] = CurriculumTermCfg(
            func=curriculums.stair_level_aware_command_height_floor,
            params={
                "command_name": "velocity_height",
                "level_height_floors": curriculums.DEFAULT_LEVEL_AWARE_HEIGHT_FLOORS,
                "step_height_range": _STAIR_STEP_HEIGHT_RANGE,
                "terrain_type_names": _STAIR_TERRAIN_TYPES,
            },
        )
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=curriculums.stair_terrain_levels,
            params={
                "asset_name": "robot",
                "standing_height": _DEFAULT_STANDING_HEIGHT,
                "move_up_distance_ratio": 0.10,
                "move_down_distance_ratio": 0.06,
                "move_up_min_steps": 2.0,
                "hold_height_tolerance_m": 0.02,
                "step_height_range": _STAIR_STEP_HEIGHT_RANGE,
                "height_sensor_name": "wheel_height_sensor",
                "contact_sensor_name": "wheel_sensor",
                "contact_force_threshold_n": _STAIR_SUPPORT_FORCE_THRESHOLD_N,
                "wheel_radius_m": _STAIR_COMMAND_WHEEL_RADIUS,
                "wheel_clearance_tol_m": _STAIR_SUPPORT_CLEARANCE_TOL_M,
                "riser_contact_sensor_name": "wheel_riser_sensor",
                "riser_contact_force_threshold_n": 1.0,
                "riser_normal_z_max": 0.5,
                "support_duration_s": _STAIR_SUPPORT_DURATION_S,
                "upright_threshold": -0.85,
                "terrain_type_names": _STAIR_TERRAIN_TYPES,
                "walking_phase_iterations": _WALKING_PHASE_ITERATIONS,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                "fixed_iteration": watch_iter,
                "max_level_stages": _STAIR_LEVEL_MAX_STAGES,
                "level_buckets": _STAIR_LEVEL_BUCKETS,
                "bucket_weight_stages": _STAIR_BUCKET_WEIGHT_STAGES,
                "gate_success_rate_threshold": 0.50,
                "gate_support_rate_threshold": 0.65,
                "gate_no_drop_rate_threshold": 0.85,
                "gate_drop_tolerance_steps": 0.20,
                "gate_min_eval_envs": 64,
                "gate_consecutive_passes": 2,
            },
        )
        if fixed_watch_command_height:
            cfg.curriculum.pop("command_height", None)
            cfg.curriculum.pop("level_aware_height_floor", None)
        if force_watch_stair_terrain:
            cfg.curriculum.pop("terrain_levels", None)
        if "push_disturbance" in cfg.curriculum:
            push_params = dict(cfg.curriculum["push_disturbance"].params or {})
            push_params.update(
                {
                    "use_iterations": True,
                    "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                    "fixed_iteration": watch_iter,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "iteration": 1200,
                            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        },
                        {
                            "iteration": 1800,
                            "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
                        },
                        {
                            "iteration": 2400,
                            "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)},
                        },
                        {
                            "iteration": 2800,
                            "velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)},
                        },
                    ],
                }
            )
            cfg.curriculum["push_disturbance"] = replace(
                cfg.curriculum["push_disturbance"],
                params=push_params,
            )

    _sanitize_observations(cfg)
    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1040,
        mujoco=MujocoCfg(
            timestep=_ROBOT_DEFAULTS.sim_dt,
            iterations=12,
            ls_iterations=8,
            ccd_iterations=15,
            tolerance=1e-6,
        ),
    )
    cfg.clip_observations = 100.0
    return cfg
