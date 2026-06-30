"""Rough terrain discovery GRU 任务环境配置。"""

from __future__ import annotations

import os
import re
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.terrains import TerrainEntityCfg

from se3_train.mdp import curriculums as mdp_curriculums
from se3_train.mdp import events as mdp_events
from se3_train.tasks.recovery_discovery.env_cfg import env_cfg as discovery_env_cfg

_DISCOVERY_MAX_LIN_VEL_X = 1.89
_DISCOVERY_MAX_ANG_VEL_YAW = 9.41
_ROUGH_DISCOVERY_STANDING_RATIO = 0.10
_ROUGH_DISCOVERY_MOVING_MIN_COMMAND_NORM = 0.15
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


def _terrain_level_stages() -> list[dict[str, int]]:
    return [
        {"iteration": 0, "max_level": 1},
        {"iteration": 600, "max_level": 2},
        {"iteration": 1200, "max_level": 3},
        {"iteration": 2000, "max_level": _int_env("SE3_ROUGH_DISCOVERY_MAX_LEVEL", 5)},
    ]


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
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    cfg = discovery_env_cfg(play=play)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=(
            _play_max_init_terrain_level()
            if play
            else _int_env("SE3_ROUGH_DISCOVERY_MAX_INIT_LEVEL", 2)
        ),
    )
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 1040

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (-_DISCOVERY_MAX_LIN_VEL_X, _DISCOVERY_MAX_LIN_VEL_X)
    command_cfg.ang_vel_yaw_range = (
        -_DISCOVERY_MAX_ANG_VEL_YAW,
        _DISCOVERY_MAX_ANG_VEL_YAW,
    )
    command_cfg.standing_ratio = _ROUGH_DISCOVERY_STANDING_RATIO
    command_cfg.moving_command_min_norm = _ROUGH_DISCOVERY_MOVING_MIN_COMMAND_NORM

    reset_params = cfg.events["reset_root_state"].params
    reset_params["recovery_state_cache_path"] = None
    reset_params["recovery_state_cache_split"] = "train"
    reset_params["standard_recovery_zero_velocity_command"] = False
    reset_params["pose_weights"] = (0.80, 0.05, 0.05, 0.05, 0.05)
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

    if not play:
        if "commands_vel" in cfg.curriculum:
            cfg.curriculum["commands_vel"].params["velocity_stages"] = [
                {
                    "iteration": 0,
                    "lin_vel_x_range": (
                        -_DISCOVERY_MAX_LIN_VEL_X,
                        _DISCOVERY_MAX_LIN_VEL_X,
                    ),
                    "ang_vel_yaw_range": (
                        -_DISCOVERY_MAX_ANG_VEL_YAW,
                        _DISCOVERY_MAX_ANG_VEL_YAW,
                    ),
                },
            ]
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
