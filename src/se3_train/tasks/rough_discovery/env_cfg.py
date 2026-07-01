"""Rough terrain discovery GRU 任务环境配置。"""

from __future__ import annotations

import os
import re
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import TerrainEntityCfg

from se3_train.mdp import curriculums as mdp_curriculums
from se3_train.mdp import events as mdp_events
from se3_train.mdp import terminations as mdp_terminations
from se3_train.tasks.recovery_discovery.env_cfg import env_cfg as discovery_env_cfg

_DISCOVERY_MAX_LIN_VEL_X = 1.89
_DISCOVERY_MAX_ANG_VEL_YAW = 9.41
_ROUGH_DISCOVERY_STANDING_RATIO = 0.10
_ROUGH_DISCOVERY_MOVING_MIN_COMMAND_NORM = 0.15
_ROUGH_DISCOVERY_STAIR_HEIGHT_MAX = 0.20
_ROUGH_DISCOVERY_HEIGHT_MIN_CLAMPS = ((0.10, 0.28), (0.15, 0.35))
_ROUGH_DISCOVERY_HEIGHT_TERRAIN_TYPES = ("pyramid_stairs", "pyramid_stairs_inv")
_STEPS_PER_POLICY_ITER = 64
_CHECKPOINT_ITER_RE = re.compile(r"model_(\d+)")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _terrain_level_stages() -> list[dict[str, int]]:
    return [
        {"iteration": 0, "max_level": 0},
        {"iteration": 400, "max_level": 1},
        {"iteration": 800, "max_level": 2},
        {"iteration": 1300, "max_level": 3},
        {"iteration": 1900, "max_level": 4},
        {"iteration": 2600, "max_level": 5},
        {"iteration": 3400, "max_level": 6},
        {"iteration": 4300, "max_level": 7},
        {"iteration": 5100, "max_level": 8},
        {"iteration": 5800, "max_level": _int_env("SE3_ROUGH_DISCOVERY_MAX_LEVEL", 9)},
    ]


def _stair_approach_stages() -> list[dict[str, float | int]]:
    final_level = _int_env("SE3_ROUGH_DISCOVERY_MAX_LEVEL", 9)
    return [
        {"iteration": 0, "ratio": 0.10, "min_level": 0, "max_level": 0},
        {"iteration": 400, "ratio": 0.15, "min_level": 0, "max_level": 1},
        {"iteration": 800, "ratio": 0.20, "min_level": 0, "max_level": 2},
        {"iteration": 1300, "ratio": 0.25, "min_level": 0, "max_level": 3},
        {"iteration": 1900, "ratio": 0.30, "min_level": 0, "max_level": 4},
        {"iteration": 2600, "ratio": 0.35, "min_level": 1, "max_level": 5},
        {"iteration": 3400, "ratio": 0.40, "min_level": 1, "max_level": 6},
        {"iteration": 4300, "ratio": 0.45, "min_level": 2, "max_level": 7},
        {"iteration": 5100, "ratio": 0.45, "min_level": 2, "max_level": 8},
        {"iteration": 5800, "ratio": 0.50, "min_level": 2, "max_level": final_level},
    ]


def _stair_contact_stages() -> list[dict[str, float | int]]:
    final_level = _int_env("SE3_ROUGH_DISCOVERY_MAX_LEVEL", 9)
    return [
        {"iteration": 0, "ratio": 0.0, "min_level": 0, "max_level": 0},
        {"iteration": 800, "ratio": 0.05, "min_level": 0, "max_level": 2},
        {"iteration": 1300, "ratio": 0.10, "min_level": 0, "max_level": 3},
        {"iteration": 1900, "ratio": 0.15, "min_level": 0, "max_level": 4},
        {"iteration": 2600, "ratio": 0.20, "min_level": 1, "max_level": 5},
        {"iteration": 3400, "ratio": 0.25, "min_level": 1, "max_level": 6},
        {"iteration": 4300, "ratio": 0.30, "min_level": 2, "max_level": 7},
        {"iteration": 5100, "ratio": 0.30, "min_level": 2, "max_level": 8},
        {"iteration": 5800, "ratio": 0.30, "min_level": 2, "max_level": final_level},
    ]


def _velocity_stages() -> list[dict[str, object]]:
    return [
        {
            "iteration": 0,
            "lin_vel_x_range": (-0.4, 0.4),
            "ang_vel_yaw_range": (-0.8, 0.8),
        },
        {
            "iteration": 400,
            "lin_vel_x_range": (-0.8, 0.8),
            "ang_vel_yaw_range": (-1.5, 1.5),
        },
        {
            "iteration": 1000,
            "lin_vel_x_range": (-1.2, 1.2),
            "ang_vel_yaw_range": (-3.0, 3.0),
        },
        {
            "iteration": 1800,
            "lin_vel_x_range": (-_DISCOVERY_MAX_LIN_VEL_X, _DISCOVERY_MAX_LIN_VEL_X),
            "ang_vel_yaw_range": (-6.0, 6.0),
        },
        {
            "iteration": 2600,
            "lin_vel_x_range": (-_DISCOVERY_MAX_LIN_VEL_X, _DISCOVERY_MAX_LIN_VEL_X),
            "ang_vel_yaw_range": (-_DISCOVERY_MAX_ANG_VEL_YAW, _DISCOVERY_MAX_ANG_VEL_YAW),
        },
    ]


def _rough_discovery_terrain_cfg():
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
    terrain_cfg.curriculum = True
    terrain_cfg.num_cols = len(terrain_cfg.sub_terrains)
    stair_height_max = _float_env(
        "SE3_ROUGH_DISCOVERY_STAIR_HEIGHT_MAX",
        _ROUGH_DISCOVERY_STAIR_HEIGHT_MAX,
    )
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        sub_terrain = terrain_cfg.sub_terrains.get(name)
        if sub_terrain is not None and hasattr(sub_terrain, "step_height_range"):
            sub_terrain.step_height_range = (0.0, stair_height_max)
    horizontal_scale = _float_env("SE3_ROUGH_DISCOVERY_HFIELD_HORIZONTAL_SCALE", 0.2)
    for sub_terrain in terrain_cfg.sub_terrains.values():
        if hasattr(sub_terrain, "horizontal_scale"):
            sub_terrain.horizontal_scale = horizontal_scale
    return terrain_cfg


def _checkpoint_iteration_from_env() -> int | None:
    raw_iteration = os.environ.get("SE3_WATCH_ITER", "")
    if raw_iteration.isdigit():
        return int(raw_iteration)

    selected = os.environ.get("SE3_VISER_SELECTED_CHECKPOINT", "")
    match = _CHECKPOINT_ITER_RE.search(selected)
    if match is None:
        return None
    return int(match.group(1))


def _terrain_level_for_iteration(iteration: int) -> int:
    max_level = 0
    for stage in _terrain_level_stages():
        if iteration >= int(stage["iteration"]):
            max_level = int(stage["max_level"])
    return max(0, max_level)


def _play_max_init_terrain_level() -> int:
    explicit = os.environ.get("SE3_ROUGH_DISCOVERY_PLAY_TERRAIN_LEVEL")
    if explicit not in (None, ""):
        return max(0, _int_env("SE3_ROUGH_DISCOVERY_PLAY_TERRAIN_LEVEL", 0))

    iteration = _checkpoint_iteration_from_env()
    if iteration is None:
        return _int_env("SE3_ROUGH_DISCOVERY_MAX_INIT_LEVEL", 2)
    return _terrain_level_for_iteration(iteration)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """从 recovery-discovery 迁移到 rough terrain 的环境配置。"""
    cfg = discovery_env_cfg(play=play)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=_rough_discovery_terrain_cfg(),
        max_init_terrain_level=(
            _play_max_init_terrain_level()
            if play
            else _int_env("SE3_ROUGH_DISCOVERY_MAX_INIT_LEVEL", 0)
        ),
    )
    cfg.sim.nconmax = _int_env("SE3_ROUGH_DISCOVERY_NCONMAX", 256)
    cfg.sim.njmax = _int_env("SE3_ROUGH_DISCOVERY_NJMAX", 1040)
    cfg.sim.contact_sensor_maxmatch = _int_env("SE3_ROUGH_DISCOVERY_CONTACT_SENSOR_MAXMATCH", 128)
    if not _bool_env("SE3_ROUGH_DISCOVERY_KEEP_WHEEL_HEIGHT_SENSOR", True):
        cfg.scene.sensors = tuple(
            sensor
            for sensor in cfg.scene.sensors
            if getattr(sensor, "name", None) != "wheel_height_sensor"
        )

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (-0.4, 0.4)
    command_cfg.ang_vel_yaw_range = (-0.8, 0.8)
    command_cfg.standing_ratio = _ROUGH_DISCOVERY_STANDING_RATIO
    command_cfg.moving_command_min_norm = _ROUGH_DISCOVERY_MOVING_MIN_COMMAND_NORM
    command_cfg.terrain_aware_height = True
    command_cfg.terrain_step_height_min_clamps = _ROUGH_DISCOVERY_HEIGHT_MIN_CLAMPS
    command_cfg.terrain_step_height_type_names = _ROUGH_DISCOVERY_HEIGHT_TERRAIN_TYPES

    reset_params = cfg.events["reset_root_state"].params
    reset_params["recovery_state_cache_path"] = None
    reset_params["recovery_state_cache_split"] = "train"
    reset_params["standard_recovery_zero_velocity_command"] = False
    reset_params["pose_weights"] = (0.80, 0.05, 0.05, 0.05, 0.05)
    reset_params["stair_approach_curriculum_stages"] = _stair_approach_stages()
    reset_params["stair_approach_terrain_type_names"] = (
        "pyramid_stairs",
        "pyramid_stairs_inv",
    )
    reset_params["stair_approach_distance_range"] = (0.3, 1.2)
    reset_params["stair_approach_lateral_range"] = (-1.0, 1.0)
    reset_params["stair_approach_yaw_jitter_range"] = (-0.15, 0.15)
    reset_params["stair_approach_command_vx_range"] = (0.35, 0.90)
    reset_params["stair_approach_root_lin_vel_x_range"] = (0.0, 0.20)
    reset_params["stair_contact_curriculum_stages"] = _stair_contact_stages()
    reset_params["stair_contact_terrain_type_names"] = (
        "pyramid_stairs",
        "pyramid_stairs_inv",
    )
    reset_params["stair_contact_offset_range"] = (-0.08, 0.12)
    reset_params["stair_contact_lateral_range"] = (-1.0, 1.0)
    reset_params["stair_contact_yaw_jitter_range"] = (-0.35, 0.35)
    reset_params["stair_contact_command_vx_range"] = (0.25, 0.90)
    reset_params["stair_contact_root_lin_vel_x_range"] = (-0.35, 0.15)
    reset_params["stair_contact_roll_jitter_range"] = (-0.12, 0.12)
    reset_params["stair_contact_pitch_jitter_range"] = (-0.18, 0.18)
    reset_params["stair_contact_height_offset_range"] = (-0.025, 0.025)
    cfg.events["reset_joints"].params["terrain_height_sensor_names"] = ("wheel_height_sensor",)
    reset_params["source_curriculum_stages"] = [
        {"iteration": 0, "cache_ratio": 0.0, "near_upright_ratio": 0.10},
        {"iteration": 600, "cache_ratio": 0.0, "near_upright_ratio": 0.15},
        {"iteration": 1200, "cache_ratio": 0.0, "near_upright_ratio": 0.20},
    ]
    reset_params["standard_curriculum_stages"] = [
        {
            "iteration": 0,
            "roll_jitter_range": (-0.08726646259971647, 0.08726646259971647),
            "pitch_jitter_range": (-0.08726646259971647, 0.08726646259971647),
            "lin_vel_range": (-0.03, 0.03),
            "ang_vel_range": (-0.10, 0.10),
            "pose_weights": (0.90, 0.025, 0.025, 0.025, 0.025),
        },
        {
            "iteration": 800,
            "roll_jitter_range": (-0.17453292519943295, 0.17453292519943295),
            "pitch_jitter_range": (-0.17453292519943295, 0.17453292519943295),
            "lin_vel_range": (-0.05, 0.05),
            "ang_vel_range": (-0.20, 0.20),
            "pose_weights": (0.85, 0.04, 0.04, 0.035, 0.035),
        },
        {
            "iteration": 1600,
            "roll_jitter_range": (-0.2617993877991494, 0.2617993877991494),
            "pitch_jitter_range": (-0.2617993877991494, 0.2617993877991494),
            "lin_vel_range": (-0.08, 0.08),
            "ang_vel_range": (-0.30, 0.30),
            "pose_weights": (0.80, 0.05, 0.05, 0.05, 0.05),
        },
    ]

    cfg.events["mark_online_settle"] = EventTermCfg(
        func=mdp_events.mark_online_settle,
        mode="reset",
        params={
            "settle_steps": _int_env("SE3_ROUGH_DISCOVERY_SETTLE_STEPS", 12),
            "settle_attr": "_online_settle_remaining",
        },
    )
    cfg.terminations["rough_stair_progress_failure"] = TerminationTermCfg(
        func=mdp_terminations.rough_stair_progress_failure,
        time_out=False,
        params={
            "command_name": "velocity_height",
            "min_command_vx": 0.15,
            "grace_steps": 80,
            "max_stuck_steps": 120,
            "min_progress_delta": 0.08,
            "max_backward_progress": -0.35,
            "max_backward_steps": 40,
        },
    )
    cfg.terminations["rough_stair_progress_success"] = TerminationTermCfg(
        func=mdp_terminations.rough_stair_progress_success,
        time_out=True,
        params={
            "command_name": "velocity_height",
            "min_command_vx": 0.15,
            "min_episode_steps": 20,
        },
    )

    if not play:
        if "commands_vel" in cfg.curriculum:
            cfg.curriculum["commands_vel"].params["velocity_stages"] = _velocity_stages()
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=mdp_curriculums.terrain_levels,
            params={
                "use_iterations": True,
                "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                "level_stages": _terrain_level_stages(),
            },
        )
        if "push_disturbance" in cfg.curriculum:
            cfg.curriculum["push_disturbance"].params["push_stages"] = [
                {"iteration": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"iteration": 1000, "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
                {"iteration": 1800, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
                {"iteration": 2400, "velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
            ]

    return cfg
