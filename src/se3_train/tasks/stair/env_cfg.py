"""倒金字塔 CTBC 台阶任务环境配置。"""

from __future__ import annotations

import os
from dataclasses import replace

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
    BoxInvertedPyramidStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import terminations
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, events, observations, rewards

_ROBOT_DEFAULTS = SharedRobotConfig()
_STAIR_WHEEL_KD = 0.08
_STAIR_COMMAND_WHEEL_RADIUS = 0.060
_STAIR_COMMAND_HALF_TRACK = 0.200725
_STAIR_COMMAND_WHEEL_SPEED_FRACTION = 0.70
_STAIR_TERRAIN_TYPES = ("inv_pyramid_stairs",)
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height
_WALKING_HEIGHT_RANGE = (_DEFAULT_STANDING_HEIGHT, _DEFAULT_STANDING_HEIGHT)
_WALKING_PHASE_ITERATIONS = 800
_STEPS_PER_POLICY_ITER = 64
_WATCH_ITER_ENV = "SE3_WATCH_ITER"
_WATCH_TERRAIN_LEVEL_ENV = "SE3_WATCH_TERRAIN_LEVEL"
_WATCH_COMMAND_HEIGHT_ENV = "SE3_WATCH_COMMAND_HEIGHT"
_TRAIN_VIEW_ITER_ENV = "SE3_TRAIN_VIEW_ITER"
_TRAIN_VIEW_TERRAIN_LEVEL_ENV = "SE3_TRAIN_VIEW_TERRAIN_LEVEL"
_TRAIN_VIEW_COMMAND_HEIGHT_ENV = "SE3_TRAIN_VIEW_COMMAND_HEIGHT"


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
            "inv_pyramid_stairs": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.8,
                size=(8.0, 8.0),
                step_height_range=(0.05, 0.20),
                step_width=0.5,
                platform_width=2.0,
                border_width=1.0,
            ),
            "flat": BoxFlatTerrainCfg(proportion=0.2, size=(8.0, 8.0)),
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

    cfg.scene.entities["robot"] = get_serialleg_cfg(wheel_kd_override=_STAIR_WHEEL_KD)
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
    command_cfg.height_range = _WALKING_HEIGHT_RANGE
    command_cfg.standing_height_range = _WALKING_HEIGHT_RANGE
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
    for group_name in ("actor", "critic"):
        group_cfg = cfg.observations.get(group_name)
        if group_cfg is None:
            continue
        terms = dict(group_cfg.terms)
        terms["last_actions"] = last_action_term
        cfg.observations[group_name] = replace(group_cfg, terms=terms)

    flat_tracking_lin_vel_cfg = cfg.rewards["tracking_lin_vel"]
    flat_tracking_lin_vel_params = dict(flat_tracking_lin_vel_cfg.params or {})
    flat_tracking_lin_vel_params.update(
        {
            "walking_phase_iterations": _WALKING_PHASE_ITERATIONS,
            "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
        }
    )
    cfg.rewards["flat_tracking_lin_vel"] = replace(
        flat_tracking_lin_vel_cfg,
        func=rewards.flat_phase_tracking_lin_vel,
        params=flat_tracking_lin_vel_params,
    )
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.stair_phase_forward_progress,
        weight=2.73,
        params={
            "command_name": "velocity_height",
            "sigma": 0.25,
            "walking_phase_iterations": _WALKING_PHASE_ITERATIONS,
            "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
        },
    )
    flat_contact_rewards = {
        "flat_wheel_contact": rewards.flat_phase_wheel_contact_penalty,
        "flat_leg_contact": rewards.flat_phase_leg_contact_penalty,
    }
    for reward_name, reward_func in flat_contact_rewards.items():
        if reward_name in cfg.rewards:
            contact_params = dict(cfg.rewards[reward_name].params or {})
            contact_params.update(
                {
                    "walking_phase_iterations": _WALKING_PHASE_ITERATIONS,
                    "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                }
            )
            cfg.rewards[reward_name] = replace(
                cfg.rewards[reward_name],
                func=reward_func,
                params=contact_params,
            )
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques_no_ctbc,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power_no_ctbc,
        weight=-1.03e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.stand_still_no_ctbc,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": _ROBOT_DEFAULTS.default_base_height,
            "height_tolerance": 40.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["action_rate"] = RewardTermCfg(func=rewards.action_rate_no_ctbc, weight=-0.48)
    cfg.rewards["contact_forces"] = RewardTermCfg(
        func=rewards.contact_forces_no_ctbc,
        weight=-1.07e-3,
        params={
            "threshold": 35.0,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stair_feet_clearance"] = RewardTermCfg(
        func=rewards.stair_feet_clearance,
        weight=2.0,
        params={"sensor_name": "wheel_height_sensor", "h_min": 0.03, "h_max": 0.30},
    )
    cfg.rewards["stair_climb_progress"] = RewardTermCfg(
        func=rewards.stair_climb_progress,
        weight=3.0,
        params={
            "max_height_gain": 1.0,
            "max_radial_progress": 4.0,
            "radial_weight": 0.25,
            "standing_height": _DEFAULT_STANDING_HEIGHT,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_riser_stall"] = RewardTermCfg(
        func=rewards.stair_riser_stall,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "min_duration_s": 0.25,
            "command_threshold": 0.2,
            "speed_threshold": 0.15,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_commanded_stall"] = RewardTermCfg(
        func=rewards.stair_commanded_stall,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.2,
            "forward_speed_threshold": 0.15,
            "vertical_speed_threshold": 0.04,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["stair_feet_air_time"] = RewardTermCfg(
        func=rewards.stair_feet_air_time,
        weight=2.0,
        params={"sensor_name": "wheel_sensor"},
    )
    cfg.rewards["stair_contact_number"] = RewardTermCfg(
        func=rewards.stair_contact_number,
        weight=2.0,
        params={"sensor_name": "wheel_sensor"},
    )
    cfg.rewards["stair_wheel_swing_zero_vel"] = RewardTermCfg(
        func=rewards.stair_wheel_swing_zero_vel,
        weight=0.5,
        params={"sensor_name": "wheel_sensor"},
    )
    cfg.rewards["obs_steps_climbed"] = RewardTermCfg(
        func=rewards.stair_steps_climbed,
        weight=0.001,
        params={
            "step_height": None,
            "step_height_range": (0.05, 0.20),
            "step_depth": 0.30,
            "start_x_offset": 0.0,
            "standing_height": _DEFAULT_STANDING_HEIGHT,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["obs_x_progress"] = RewardTermCfg(
        func=rewards.stair_max_x_progress,
        weight=0.001,
    )
    cfg.rewards["obs_height_gain"] = RewardTermCfg(
        func=rewards.stair_height_gain,
        weight=0.001,
        params={
            "command_name": "velocity_height",
            "standing_height": _DEFAULT_STANDING_HEIGHT,
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )
    cfg.rewards["obs_terrain_level"] = RewardTermCfg(
        func=rewards.stair_terrain_level,
        weight=0.001,
    )
    cfg.rewards["stair_diagnostics"] = RewardTermCfg(
        func=rewards.stair_diagnostics,
        weight=1.0,
        params={
            "command_name": "velocity_height",
            "terrain_type_names": _STAIR_TERRAIN_TYPES,
        },
    )

    if "bad_orientation" in cfg.terminations:
        cfg.terminations["bad_orientation"].params = {"limit_angle": 0.698, "max_steps": 100}
    cfg.terminations["leg_contact"] = TerminationTermCfg(
        func=terminations.leg_contact_delayed,
        time_out=False,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 80.0,
            "max_steps": 50,
        },
    )

    cfg.events = dict(cfg.events)
    reset_joint_params = dict(cfg.events["reset_joints"].params or {})
    reset_joint_params.update(
        {
            "height_conditioned_default": True,
            "command_name": "velocity_height",
        }
    )
    cfg.events["reset_joints"] = replace(
        cfg.events["reset_joints"],
        params=reset_joint_params,
    )
    cfg.events["init_stair_climb_state"] = EventTermCfg(
        func=events.init_stair_climb_state,
        mode="startup",
        params={
            "contact_window": 3,
            "force_threshold_n": 30.0,
            "ff_amplitude_rad": 1.2,
            "ff_period_s": 0.6,
            "ff_start_iter": _WALKING_PHASE_ITERATIONS,
            "ann_start_iter": _WALKING_PHASE_ITERATIONS + 200,
            "ann_end_iter": _WALKING_PHASE_ITERATIONS + 800,
            "phantom_trigger_iter": 0,
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
                        "lin_vel_x_range": (0.0, 0.0),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": 100,
                        "lin_vel_x_range": (-0.5, 0.5),
                        "ang_vel_yaw_range": (-0.5, 0.5),
                    },
                    {
                        "iteration": 250,
                        "lin_vel_x_range": (-1.0, 1.0),
                        "ang_vel_yaw_range": (-1.0, 1.0),
                    },
                    {
                        "iteration": 400,
                        "lin_vel_x_range": (-1.5, 1.5),
                        "ang_vel_yaw_range": (-1.8, 1.8),
                    },
                    {
                        "iteration": 600,
                        "lin_vel_x_range": (-2.0, 2.0),
                        "ang_vel_yaw_range": (-2.5, 2.5),
                    },
                    {
                        "iteration": 700,
                        "lin_vel_x_range": (-2.5, 2.5),
                        "ang_vel_yaw_range": (-3.0, 3.0),
                    },
                    {
                        "iteration": _WALKING_PHASE_ITERATIONS,
                        "lin_vel_x_range": (0.6, 1.0),
                        "ang_vel_yaw_range": (0.0, 0.0),
                    },
                    {
                        "iteration": _WALKING_PHASE_ITERATIONS + 250,
                        "lin_vel_x_range": (0.6, 1.6),
                        "ang_vel_yaw_range": (-0.25, 0.25),
                    },
                    {
                        "iteration": _WALKING_PHASE_ITERATIONS + 500,
                        "lin_vel_x_range": (0.6, 2.5),
                        "ang_vel_yaw_range": (-0.50, 0.50),
                    },
                    {
                        "iteration": _WALKING_PHASE_ITERATIONS + 800,
                        "lin_vel_x_range": (0.6, 2.5),
                        "ang_vel_yaw_range": (-0.80, 0.80),
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
                        "height_range": _WALKING_HEIGHT_RANGE,
                    },
                    {
                        "iteration": 150,
                        "height_range": (0.22, 0.28),
                    },
                    {
                        "iteration": 350,
                        "height_range": (0.23, 0.32),
                    },
                    {
                        "iteration": 600,
                        "height_range": (0.24, 0.37),
                    },
                    {
                        "iteration": _WALKING_PHASE_ITERATIONS,
                        "height_range": (0.24, 0.37),
                    },
                ],
            },
        )
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=curriculums.stair_terrain_levels,
            params={
                "asset_name": "robot",
                "standing_height": _DEFAULT_STANDING_HEIGHT,
                "move_up_distance_ratio": 0.35,
                "move_down_distance_ratio": 0.12,
                "upright_threshold": -0.5,
                "terrain_type_names": _STAIR_TERRAIN_TYPES,
                "walking_phase_iterations": _WALKING_PHASE_ITERATIONS,
                "flat_terrain_type_name": "flat",
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                "fixed_iteration": watch_iter,
            },
        )
        if fixed_watch_command_height:
            cfg.curriculum.pop("command_height", None)
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
                            "iteration": _WALKING_PHASE_ITERATIONS + 600,
                            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                        },
                        {
                            "iteration": _WALKING_PHASE_ITERATIONS + 1000,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "iteration": _WALKING_PHASE_ITERATIONS + 1500,
                            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
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
