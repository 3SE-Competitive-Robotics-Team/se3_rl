"""Run a short stair checkpoint rollout and print stability/CTBC diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from se3_shared import RobotConfig as SharedRobotConfig

TASK_NAME = "SE3-WheelLegged-Stair-GRU"
WATCH_USE_TRAIN_ENV_ENV = "SE3_WATCH_USE_TRAIN_ENV"
WATCH_ITER_ENV = "SE3_WATCH_ITER"
WATCH_TERRAIN_LEVEL_ENV = "SE3_WATCH_TERRAIN_LEVEL"
WATCH_COMMAND_HEIGHT_ENV = "SE3_WATCH_COMMAND_HEIGHT"
_ROBOT_DEFAULTS = SharedRobotConfig()
_STAIR_STEP_HEIGHT_RANGE = (0.05, 0.20)
_WHEEL_RADIUS_M = 0.060


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose stair walking and CTBC trigger behavior from a checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--terrain-level", type=int, default=None)
    parser.add_argument("--command-vx", type=float, default=None)
    parser.add_argument("--command-yaw", type=float, default=None)
    parser.add_argument("--command-height", type=float, default=None)
    parser.add_argument("--ff-x-m", type=float, default=None)
    parser.add_argument("--ff-lift-m", type=float, default=None)
    parser.add_argument("--ff-period-s", type=float, default=None)
    parser.add_argument("--ff-rise-ratio", type=float, default=None)
    parser.add_argument("--ff-hold-ratio", type=float, default=None)
    parser.add_argument("--ff-wheel-action", type=float, default=None)
    parser.add_argument("--no-bilateral-trigger", action="store_true")
    parser.add_argument("--contact-window", type=int, default=None)
    parser.add_argument("--force-threshold-n", type=float, default=None)
    parser.add_argument(
        "--manual-trigger-time",
        type=float,
        default=None,
        help="在指定秒数强制触发 CTBC，便于隔离前馈本身的效果。",
    )
    parser.add_argument(
        "--manual-trigger-side",
        choices=("left", "right", "both"),
        default="left",
        help="手动触发时激活的侧别。",
    )
    parser.add_argument(
        "--fixed-scene",
        action="store_true",
        help="关闭课程和域随机化，用固定场景诊断前馈。",
    )
    parser.add_argument("--no-terminations", action="store_true")
    parser.add_argument("--pass-step-ratio", type=float, default=0.7)
    parser.add_argument("--pass-radial-m", type=float, default=0.45)
    parser.add_argument("--pass-final-radial-m", type=float, default=0.30)
    parser.add_argument("--pass-tilt-deg", type=float, default=45.0)
    parser.add_argument("--pass-support-ratio", type=float, default=0.7)
    parser.add_argument("--pass-support-duration-s", type=float, default=0.10)
    parser.add_argument("--pass-wheel-contact-n", type=float, default=1.0)
    parser.add_argument("--pass-wheel-clearance-tol-m", type=float, default=0.035)
    parser.add_argument("--strict-min-success-steps", type=float, default=1.0)
    parser.add_argument("--strict-height-tolerance-m", type=float, default=0.015)
    parser.add_argument("--strict-forward-progress-m", type=float, default=None)
    parser.add_argument("--strict-step-depth-m", type=float, default=0.50)
    parser.add_argument("--strict-forward-progress-step-fraction", type=float, default=0.75)
    parser.add_argument("--strict-hold-duration-s", type=float, default=0.20)
    parser.add_argument("--strict-upright-threshold", type=float, default=-0.90)
    parser.add_argument("--strict-max-vertical-speed-mps", type=float, default=1.0)
    parser.add_argument("--strict-illegal-contact-force-n", type=float, default=5.0)
    parser.add_argument("--strict-riser-stall-duration-s", type=float, default=0.15)
    parser.add_argument(
        "--start-x-offset-m",
        type=float,
        default=None,
        help="诊断用：reset 后把 base 放到 env origin + x offset，便于从第一阶前方起步。",
    )
    parser.add_argument(
        "--start-y-offset-m",
        type=float,
        default=None,
        help="诊断用：reset 后把 base 放到 env origin + y offset。",
    )
    parser.add_argument(
        "--start-yaw-deg",
        type=float,
        default=None,
        help="诊断用：reset 后固定 yaw，0 度表示朝世界 +x。",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _set_watch_env(args: argparse.Namespace) -> None:
    os.environ[WATCH_USE_TRAIN_ENV_ENV] = "1"
    if args.iteration is not None:
        os.environ[WATCH_ITER_ENV] = str(args.iteration)
    if args.terrain_level is not None:
        os.environ[WATCH_TERRAIN_LEVEL_ENV] = str(args.terrain_level)
    if args.command_height is not None:
        os.environ[WATCH_COMMAND_HEIGHT_ENV] = str(args.command_height)


def _load_policy(env, checkpoint: Path, task: str, device: str):
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    agent_cfg = load_rl_cfg(task)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()
    return policy


def _mean(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.nanmean(value.float()).item())


def _max(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.nan_to_num(value.float()).max().item())


def _finite(value: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(value, nan=fill, posinf=fill, neginf=fill)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    numerator = _finite(numerator)
    denominator = _finite(denominator)
    return torch.where(
        torch.abs(denominator) > 1.0e-6,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def _command_stats(base_env) -> dict[str, float]:
    try:
        command = base_env.command_manager.get_command("velocity_height")
    except Exception:
        return {}
    if not isinstance(command, torch.Tensor) or command.numel() == 0:
        return {}
    return {
        "cmd_vx_mean": _mean(command[:, 0]),
        "cmd_vx_min": float(command[:, 0].min().item()),
        "cmd_vx_max": float(command[:, 0].max().item()),
        "cmd_yaw_mean": _mean(command[:, 1]) if command.shape[1] > 1 else 0.0,
        "cmd_yaw_min": float(command[:, 1].min().item()) if command.shape[1] > 1 else 0.0,
        "cmd_yaw_max": float(command[:, 1].max().item()) if command.shape[1] > 1 else 0.0,
        "cmd_height_mean": _mean(command[:, 4]) if command.shape[1] > 4 else 0.0,
        "cmd_height_min": float(command[:, 4].min().item()) if command.shape[1] > 4 else 0.0,
        "cmd_height_max": float(command[:, 4].max().item()) if command.shape[1] > 4 else 0.0,
    }


def _terrain_stats(base_env) -> dict[str, float]:
    terrain = getattr(base_env.scene, "terrain", None)
    if terrain is None:
        return {}
    levels = getattr(terrain, "terrain_levels", None)
    types = getattr(terrain, "terrain_types", None)
    stats: dict[str, float] = {}
    if isinstance(levels, torch.Tensor):
        stats["terrain_level_mean"] = _mean(levels)
        stats["terrain_level_min"] = float(levels.min().item())
        stats["terrain_level_max"] = float(levels.max().item())
    if isinstance(types, torch.Tensor):
        stats["terrain_type_mean"] = _mean(types)
    return stats


def _step_height_for_envs(base_env) -> torch.Tensor:
    terrain = getattr(base_env.scene, "terrain", None)
    if terrain is None:
        return torch.full(
            (base_env.num_envs,),
            float(_STAIR_STEP_HEIGHT_RANGE[0]),
            device=base_env.device,
        )
    levels = getattr(terrain, "terrain_levels", None)
    if not isinstance(levels, torch.Tensor):
        levels = torch.zeros(base_env.num_envs, device=base_env.device)
    levels = levels.to(device=base_env.device).float()
    terrain_generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    num_rows = max(1, int(getattr(terrain_generator, "num_rows", 10)) - 1)
    min_height, max_height = _STAIR_STEP_HEIGHT_RANGE
    alpha = torch.clamp(levels, min=0.0, max=float(num_rows)) / float(num_rows)
    return float(min_height) + alpha * (float(max_height) - float(min_height))


def _stair_height_gain(base_env) -> torch.Tensor:
    robot = base_env.scene["robot"]
    origin_z = (
        base_env.scene.env_origins[:, 2]
        if base_env.scene.env_origins is not None
        else torch.zeros(base_env.num_envs, device=base_env.device)
    )
    return _finite(
        robot.data.root_link_pos_w[:, 2] - origin_z - _ROBOT_DEFAULTS.default_base_height
    )


def _wheel_body_ids(base_env) -> list[int]:
    attr_name = "_diagnose_stair_wheel_body_ids"
    cached = getattr(base_env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = base_env.scene["robot"]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    setattr(base_env, attr_name, body_ids)
    return body_ids


def _wheel_terrain_measurements(
    base_env,
    body_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = base_env.scene["robot"]
    sensor = base_env.scene["wheel_height_sensor"]
    heights = _finite(sensor.data.heights)
    if heights.ndim == 1:
        heights = heights.unsqueeze(-1)
    if heights.shape[1] < len(body_ids):
        heights = heights.expand(-1, len(body_ids))
    heights = heights[:, : len(body_ids)]
    wheel_pos_w = _finite(robot.data.body_link_pos_w[:, body_ids, :])
    wheel_center_z = wheel_pos_w[:, :, 2]
    terrain_z = wheel_center_z - heights
    wheel_bottom_clearance = heights - float(_WHEEL_RADIUS_M)
    return terrain_z, heights, wheel_bottom_clearance, wheel_pos_w


def _wheel_contact_force(base_env, sensor_name: str = "wheel_sensor") -> torch.Tensor:
    sensor = base_env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(base_env.num_envs, 2, device=base_env.device)
    force = _finite(data.force)
    force_mag = torch.linalg.vector_norm(force, dim=-1)
    if force_mag.ndim == 3:
        force_mag = force_mag.amax(dim=-1)
    if force_mag.ndim == 1:
        force_mag = force_mag.unsqueeze(-1)
    if force_mag.shape[1] < 2:
        force_mag = force_mag.expand(-1, 2)
    return force_mag[:, :2]


def _override_ctbc_cfg(cfg, args: argparse.Namespace) -> None:
    if not hasattr(cfg, "events") or "init_stair_climb_state" not in cfg.events:
        return
    term = cfg.events["init_stair_climb_state"]
    params = dict(term.params or {})
    overrides = {
        "ff_x_m": args.ff_x_m,
        "ff_lift_m": args.ff_lift_m,
        "ff_period_s": args.ff_period_s,
        "ff_rise_ratio": args.ff_rise_ratio,
        "ff_hold_ratio": args.ff_hold_ratio,
        "ff_wheel_action": args.ff_wheel_action,
        "force_threshold_n": args.force_threshold_n,
        "contact_window": args.contact_window,
        "allow_bilateral_trigger": False if args.no_bilateral_trigger else None,
    }
    for name, value in overrides.items():
        if value is not None:
            if name == "contact_window":
                params[name] = int(value)
            elif name == "allow_bilateral_trigger":
                params[name] = bool(value)
            else:
                params[name] = float(value)
    cfg.events["init_stair_climb_state"] = replace(term, params=params)


def _fix_scene_for_feedforward(cfg, args: argparse.Namespace) -> None:
    """固定诊断场景，避免随机化掩盖前馈效果。"""
    cfg.curriculum = dict(getattr(cfg, "curriculum", {}) or {})
    for name in ("command_vel", "command_height", "terrain_levels", "push_disturbance"):
        cfg.curriculum.pop(name, None)

    cfg.events = dict(getattr(cfg, "events", {}) or {})
    for name in (
        "friction",
        "restitution",
        "base_mass",
        "inertia",
        "com",
        "pd_gains",
        "default_dof_pos",
        "push_robots",
    ):
        cfg.events.pop(name, None)

    if "reset_root_state" in cfg.events:
        params = dict(cfg.events["reset_root_state"].params or {})
        params["recovery_prob"] = 0.0
        params["recovery_state_cache_prob"] = 0.0
        cfg.events["reset_root_state"] = replace(cfg.events["reset_root_state"], params=params)
    if "sample_stair_task_mode" in cfg.events:
        params = dict(cfg.events["sample_stair_task_mode"].params or {})
        params["stair_prob"] = 1.0
        params["recovery_prob"] = 0.0
        if args.terrain_level is not None:
            level = max(0, int(args.terrain_level))
            params["max_level_stages"] = ((0, level),)
            params["level_buckets"] = ((level, level),)
            params["bucket_weight_stages"] = ((0, (1.0,)),)
        cfg.events["sample_stair_task_mode"] = replace(
            cfg.events["sample_stair_task_mode"],
            params=params,
        )
    cfg.events.pop("enforce_recovery_active_commands", None)


def _maybe_manual_trigger(
    base_env,
    time_s: float,
    trigger_time: float | None,
    triggered: bool,
    side: str,
) -> bool:
    if triggered or trigger_time is None or time_s < float(trigger_time):
        return triggered
    state = getattr(base_env, "stair_climb_state", None)
    if state is None:
        return triggered
    side_ids = (0, 1) if side == "both" else (0,) if side == "left" else (1,)
    # 诊断专用入口：直接置相位，隔离前馈轨迹本身。
    state._ff_phase[:] = -1
    for side_id in side_ids:
        state._ff_phase[:, side_id] = 0
    state._cooldown[:] = 0
    return True


def _maybe_set_start_pose(base_env, args: argparse.Namespace) -> None:
    if (
        args.start_x_offset_m is None
        and args.start_y_offset_m is None
        and args.start_yaw_deg is None
    ):
        return
    robot = base_env.scene["robot"]
    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    pos = _finite(robot.data.root_link_pos_w).clone()
    quat = _finite(robot.data.root_link_quat_w).clone()
    origins = base_env.scene.env_origins
    if origins is not None:
        if args.start_x_offset_m is not None:
            pos[:, 0] = origins[:, 0] + float(args.start_x_offset_m)
        if args.start_y_offset_m is not None:
            pos[:, 1] = origins[:, 1] + float(args.start_y_offset_m)
    else:
        if args.start_x_offset_m is not None:
            pos[:, 0] += float(args.start_x_offset_m)
        if args.start_y_offset_m is not None:
            pos[:, 1] += float(args.start_y_offset_m)
    if args.start_yaw_deg is not None:
        yaw = torch.full(
            (base_env.num_envs,),
            math.radians(float(args.start_yaw_deg)),
            device=base_env.device,
        )
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
    vel = torch.zeros(base_env.num_envs, 6, device=base_env.device)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    base_env.sim.forward()


def _state_snapshot(base_env) -> dict[str, Any]:
    state = getattr(base_env, "stair_climb_state", None)
    if state is None:
        return {}
    return {
        "ctbc_local_iter": int(state.local_iteration),
        "ctbc_kff": float(state.kff),
        "ctbc_active_rate_now": _mean(state.contact_triggered().float()),
        "ctbc_stable_rate_now": _mean(state.stable_contact.any(dim=-1).float()),
        "ctbc_force_mean_now": _mean(state.latest_contact_force),
        "ctbc_force_max_now": _max(state.latest_contact_force),
        "ctbc_cycles_mean": _mean(state.complete_ff_cycle_count.float()),
        "ctbc_ff_x_m": float(state.ff_x_m),
        "ctbc_ff_lift_m": float(state.ff_lift_m),
        "ctbc_ff_period_steps": float(state.ff_period_steps),
        "ctbc_ff_rise_steps": float(state.ff_rise_steps),
        "ctbc_ff_hold_steps": float(state.ff_hold_steps),
        "ctbc_ff_return_steps": float(state.ff_return_steps),
        "ctbc_ff_wheel_action": float(state.ff_wheel_action),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _set_watch_env(args)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    import se3_train  # noqa: F401
    from se3_train.tasks.stair.rewards import stair_success_components

    configure_torch_backends()
    torch.manual_seed(int(args.seed))

    cfg = load_env_cfg(args.task, play=False)
    cfg.seed = int(args.seed)
    cfg.scene.num_envs = int(args.num_envs)
    if args.fixed_scene:
        _fix_scene_for_feedforward(cfg, args)
    _override_ctbc_cfg(cfg, args)
    command_cfg = cfg.commands.get("velocity_height")
    if command_cfg is not None:
        if args.command_vx is not None:
            command_cfg.lin_vel_x_range = (float(args.command_vx), float(args.command_vx))
        if args.command_yaw is not None:
            command_cfg.ang_vel_yaw_range = (float(args.command_yaw), float(args.command_yaw))
        if args.command_height is not None:
            command_cfg.height_range = (float(args.command_height), float(args.command_height))
            command_cfg.standing_height_range = (
                float(args.command_height),
                float(args.command_height),
            )
    if (args.command_vx is not None or args.command_yaw is not None) and hasattr(cfg, "curriculum"):
        cfg.curriculum.pop("command_vel", None)
    if args.no_terminations:
        cfg.terminations = {}

    base_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env)
    policy = _load_policy(env, args.checkpoint, args.task, args.device)
    env.reset()
    _maybe_set_start_pose(base_env, args)
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()

    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("delayed_action")
    steps = max(1, math.ceil(float(args.seconds) / float(base_env.step_dt)))
    initial_base_pos = _finite(robot.data.root_link_pos_w).clone()
    step_height = _step_height_for_envs(base_env)
    wheel_body_ids = _wheel_body_ids(base_env)
    (
        initial_wheel_terrain_z,
        initial_wheel_heights,
        initial_wheel_bottom_clearance,
        initial_wheel_pos_w,
    ) = _wheel_terrain_measurements(base_env, wheel_body_ids)
    if base_env.scene.env_origins is not None:
        initial_wheel_x_offset = initial_wheel_pos_w[:, :, 0] - base_env.scene.env_origins[
            :, 0
        ].unsqueeze(1)
    else:
        initial_wheel_x_offset = initial_wheel_pos_w[:, :, 0] - initial_wheel_pos_w[:, :, 0].mean(
            dim=1, keepdim=True
        )

    active_sum = 0.0
    stable_sum = 0.0
    force_sum = 0.0
    force_max = 0.0
    raw_sat_sum = 0.0
    unclipped_sat_sum = 0.0
    active_rod_clamp_sum = 0.0
    delta_abs_sum = 0.0
    delta_abs_peak = 0.0
    vx_sum = 0.0
    vx_abs_max = 0.0
    tilt_sum = 0.0
    tilt_max = 0.0
    tilt_gt_30_sum = 0.0
    tilt_gt_45_sum = 0.0
    tilt_gt_60_sum = 0.0
    pitch_abs_sum = 0.0
    roll_abs_sum = 0.0
    done_sum = 0.0
    reward_sum = 0.0
    base_height_sum = 0.0
    base_height_max = -float("inf")
    base_height_min = float("inf")
    x_progress_sum = 0.0
    x_progress_max = -float("inf")
    ctbc_requested_dx_min = 0.0
    ctbc_requested_dz_max = 0.0
    ctbc_requested_norm_max = 0.0
    wheel_target_delta_norm_max = 0.0
    manual_triggered = False
    max_stair_height_gain = torch.zeros(base_env.num_envs, device=base_env.device)
    max_steps_climbed = torch.zeros(base_env.num_envs, device=base_env.device)
    max_radial_progress = torch.zeros(base_env.num_envs, device=base_env.device)
    max_tilt_per_env = torch.zeros(base_env.num_envs, device=base_env.device)
    max_any_wheel_terrain_rise = torch.zeros(base_env.num_envs, device=base_env.device)
    max_both_wheel_terrain_rise = torch.zeros(base_env.num_envs, device=base_env.device)
    max_wheel_contact_force = torch.zeros(base_env.num_envs, device=base_env.device)
    max_x_progress = torch.zeros(base_env.num_envs, device=base_env.device)
    max_any_wheel_x_offset = torch.max(initial_wheel_x_offset, dim=1).values.clone()
    max_both_wheel_x_offset = torch.min(initial_wheel_x_offset, dim=1).values.clone()
    max_consecutive_both_support_steps = torch.zeros(
        base_env.num_envs,
        device=base_env.device,
        dtype=torch.long,
    )
    consecutive_both_support_steps = torch.zeros(
        base_env.num_envs,
        device=base_env.device,
        dtype=torch.long,
    )
    upper_support_any_sum = 0.0
    upper_support_both_sum = 0.0
    strict_success_sum = 0.0
    strict_candidate_sum = 0.0
    strict_height_sum = 0.0
    strict_forward_sum = 0.0
    strict_upright_sum = 0.0
    strict_vertical_speed_sum = 0.0
    strict_legal_contact_sum = 0.0
    strict_riser_clear_sum = 0.0
    strict_wheel_contact_sum = 0.0
    strict_near_support_height_sum = 0.0
    strict_valid_sum = 0.0
    strict_duration_sum = 0.0
    strict_duration_max = 0.0
    strict_current_both_rise_sum = 0.0
    strict_current_both_rise_max = 0.0
    strict_radial_sum = 0.0
    strict_radial_max = 0.0
    strict_height_target_sum = 0.0
    strict_forward_target_sum = 0.0
    contact_any_sum = 0.0
    contact_both_sum = 0.0
    wheel_height_sum = 0.0
    wheel_bottom_clearance_abs_sum = 0.0
    samples = 0

    for step in range(steps):
        time_s = step * float(base_env.step_dt)
        manual_triggered = _maybe_manual_trigger(
            base_env,
            time_s,
            args.manual_trigger_time,
            manual_triggered,
            args.manual_trigger_side,
        )
        with torch.no_grad():
            obs = env.get_observations()
            action = policy(obs)
            _, reward, dones, _ = env.step(action)

        state = getattr(base_env, "stair_climb_state", None)
        if state is not None:
            active_sum += _mean(state.contact_triggered().float())
            stable_sum += _mean(state.stable_contact.any(dim=-1).float())
            force = state.latest_contact_force
            force_sum += _mean(force)
            force_max = max(force_max, _max(force))

        raw_action = getattr(action_term, "raw_action", base_env.action_manager.action)
        unclipped_action = getattr(action_term, "unclipped_action", raw_action)
        delta = getattr(action_term, "ctbc_action_delta", torch.zeros_like(raw_action))
        raw_sat_sum += _mean((torch.abs(raw_action) >= 0.999).any(dim=-1).float())
        unclipped_sat_sum += _mean((torch.abs(unclipped_action) >= 0.999).any(dim=-1).float())
        delta_abs = torch.abs(delta[:, :4]).amax(dim=-1)
        delta_abs_sum += _mean(delta_abs)
        delta_abs_peak = max(delta_abs_peak, _max(delta_abs))
        active_rod_clamped = getattr(action_term, "active_rod_angle_target_clamped", None)
        if isinstance(active_rod_clamped, torch.Tensor):
            active_rod_clamp_sum += _mean(active_rod_clamped.any(dim=-1).float())

        pg = _finite(robot.data.projected_gravity_b)
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
        pitch = torch.rad2deg(torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0)))
        roll = torch.rad2deg(torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0)))
        vx = _finite(robot.data.root_link_lin_vel_b[:, 0])
        tilt_sum += _mean(tilt)
        tilt_max = max(tilt_max, _max(tilt))
        max_tilt_per_env = torch.maximum(max_tilt_per_env, _finite(tilt))
        tilt_gt_30_sum += _mean((tilt > 30.0).float())
        tilt_gt_45_sum += _mean((tilt > 45.0).float())
        tilt_gt_60_sum += _mean((tilt > 60.0).float())
        pitch_abs_sum += _mean(torch.abs(pitch))
        roll_abs_sum += _mean(torch.abs(roll))
        vx_sum += _mean(vx)
        vx_abs_max = max(vx_abs_max, _max(torch.abs(vx)))
        base_pos = _finite(robot.data.root_link_pos_w)
        if base_env.scene.env_origins is not None:
            radial_progress_now = torch.linalg.vector_norm(
                base_pos[:, :2] - base_env.scene.env_origins[:, :2],
                dim=1,
            )
        else:
            radial_progress_now = torch.linalg.vector_norm(
                base_pos[:, :2] - initial_base_pos[:, :2],
                dim=1,
            )
        radial_progress_now = _finite(radial_progress_now)
        max_radial_progress = torch.maximum(max_radial_progress, radial_progress_now)
        stair_height_gain_now = torch.clamp(_stair_height_gain(base_env), min=0.0)
        max_stair_height_gain = torch.maximum(max_stair_height_gain, stair_height_gain_now)
        max_steps_climbed = torch.maximum(
            max_steps_climbed,
            stair_height_gain_now / torch.clamp(step_height, min=1.0e-6),
        )
        wheel_terrain_z, wheel_heights, wheel_bottom_clearance, wheel_pos_w = (
            _wheel_terrain_measurements(
                base_env,
                wheel_body_ids,
            )
        )
        if base_env.scene.env_origins is not None:
            wheel_x_offset = wheel_pos_w[:, :, 0] - base_env.scene.env_origins[:, 0].unsqueeze(1)
        else:
            wheel_x_offset = wheel_pos_w[:, :, 0] - initial_wheel_pos_w[:, :, 0]
        max_any_wheel_x_offset = torch.maximum(
            max_any_wheel_x_offset,
            torch.max(wheel_x_offset, dim=1).values,
        )
        max_both_wheel_x_offset = torch.maximum(
            max_both_wheel_x_offset,
            torch.min(wheel_x_offset, dim=1).values,
        )
        wheel_terrain_rise = _finite(wheel_terrain_z - initial_wheel_terrain_z)
        any_wheel_terrain_rise = torch.max(wheel_terrain_rise, dim=1).values
        both_wheel_terrain_rise = torch.min(wheel_terrain_rise, dim=1).values
        max_any_wheel_terrain_rise = torch.maximum(
            max_any_wheel_terrain_rise,
            any_wheel_terrain_rise,
        )
        max_both_wheel_terrain_rise = torch.maximum(
            max_both_wheel_terrain_rise,
            both_wheel_terrain_rise,
        )
        wheel_force = _wheel_contact_force(base_env)
        max_wheel_contact_force = torch.maximum(
            max_wheel_contact_force,
            torch.max(wheel_force, dim=1).values,
        )
        wheel_contact = wheel_force >= float(args.pass_wheel_contact_n)
        near_support_height = wheel_heights <= (
            float(_WHEEL_RADIUS_M) + float(args.pass_wheel_clearance_tol_m)
        )
        required_support_rise = step_height.unsqueeze(-1) * float(args.pass_support_ratio)
        upper_supported = (
            (wheel_terrain_rise >= required_support_rise) & wheel_contact & near_support_height
        )
        upper_support_any = upper_supported.any(dim=1)
        upper_support_both = upper_supported.all(dim=1)
        consecutive_both_support_steps = torch.where(
            upper_support_both,
            consecutive_both_support_steps + 1,
            torch.zeros_like(consecutive_both_support_steps),
        )
        max_consecutive_both_support_steps = torch.maximum(
            max_consecutive_both_support_steps,
            consecutive_both_support_steps,
        )
        upper_support_any_sum += _mean(upper_support_any.float())
        upper_support_both_sum += _mean(upper_support_both.float())
        strict = stair_success_components(
            base_env,
            step_height_range=_STAIR_STEP_HEIGHT_RANGE,
            min_success_steps=float(args.strict_min_success_steps),
            success_height_tolerance_m=float(args.strict_height_tolerance_m),
            forward_progress_m=args.strict_forward_progress_m,
            step_depth_m=float(args.strict_step_depth_m),
            forward_progress_step_fraction=float(args.strict_forward_progress_step_fraction),
            hold_duration_s=float(args.strict_hold_duration_s),
            upright_threshold=float(args.strict_upright_threshold),
            max_vertical_speed_mps=float(args.strict_max_vertical_speed_mps),
            illegal_contact_force_threshold_n=float(args.strict_illegal_contact_force_n),
            wheel_radius_m=float(_WHEEL_RADIUS_M),
            wheel_clearance_tol_m=float(args.pass_wheel_clearance_tol_m),
            riser_stall_duration_s=float(args.strict_riser_stall_duration_s),
            record=True,
        )
        strict_success_sum += _mean(strict["success"].float())
        strict_candidate_sum += _mean(strict["candidate"].float())
        strict_height_sum += _mean(strict["height_ok"].float())
        strict_forward_sum += _mean(strict["forward_ok"].float())
        strict_upright_sum += _mean(strict["upright_ok"].float())
        strict_vertical_speed_sum += _mean(strict["vertical_speed_ok"].float())
        strict_legal_contact_sum += _mean(strict["legal_contact_ok"].float())
        strict_riser_clear_sum += _mean(strict["riser_clear"].float())
        strict_wheel_contact_sum += _mean(strict["wheel_contact"].float())
        strict_near_support_height_sum += _mean(strict["near_support_height"].float())
        strict_valid_sum += _mean(strict["valid"].float())
        strict_duration_sum += _mean(strict["duration"])
        strict_duration_max = max(strict_duration_max, _max(strict["duration"]))
        strict_current_both_rise_sum += _mean(strict["current_both_rise"])
        strict_current_both_rise_max = max(
            strict_current_both_rise_max,
            _max(strict["current_both_rise"]),
        )
        strict_radial_sum += _mean(strict["radial_distance"])
        strict_radial_max = max(strict_radial_max, _max(strict["radial_distance"]))
        strict_height_target_sum += _mean(strict["height_target"])
        strict_forward_target_sum += _mean(strict["forward_target"])
        contact_any_sum += _mean(wheel_contact.any(dim=1).float())
        contact_both_sum += _mean(wheel_contact.all(dim=1).float())
        wheel_height_sum += _mean(wheel_heights)
        wheel_bottom_clearance_abs_sum += _mean(torch.abs(wheel_bottom_clearance))
        base_height = base_pos[:, 2]
        x_progress = base_pos[:, 0] - initial_base_pos[:, 0]
        base_height_sum += _mean(base_height)
        base_height_max = max(base_height_max, _max(base_height))
        base_height_min = min(base_height_min, float(base_height.min().item()))
        x_progress_sum += _mean(x_progress)
        x_progress_max = max(x_progress_max, _max(x_progress))
        max_x_progress = torch.maximum(max_x_progress, _finite(x_progress))
        requested = getattr(action_term, "ctbc_wheel_delta_xz", None)
        if isinstance(requested, torch.Tensor):
            requested = _finite(requested)
            ctbc_requested_dx_min = min(
                ctbc_requested_dx_min,
                float(requested[..., 0].min().item()),
            )
            ctbc_requested_dz_max = max(
                ctbc_requested_dz_max,
                float(requested[..., 1].max().item()),
            )
            ctbc_requested_norm_max = max(
                ctbc_requested_norm_max,
                _max(torch.linalg.vector_norm(requested, dim=-1)),
            )
        actual_wheel = getattr(action_term, "actual_wheel_xz", None)
        target_wheel = getattr(action_term, "target_wheel_xz", None)
        if isinstance(actual_wheel, torch.Tensor) and isinstance(target_wheel, torch.Tensor):
            target_delta = _finite(target_wheel - actual_wheel)
            wheel_target_delta_norm_max = max(
                wheel_target_delta_norm_max,
                _max(torch.linalg.vector_norm(target_delta, dim=-1)),
            )
        done_sum += _mean(dones.float())
        reward_sum += _mean(reward)
        samples += 1

    denom = max(1, samples)
    final_base_pos = _finite(robot.data.root_link_pos_w)
    final_height = final_base_pos[:, 2]
    final_x_progress = final_base_pos[:, 0] - initial_base_pos[:, 0]
    if base_env.scene.env_origins is not None:
        final_radial_progress = torch.linalg.vector_norm(
            final_base_pos[:, :2] - base_env.scene.env_origins[:, :2],
            dim=1,
        )
    else:
        final_radial_progress = torch.linalg.vector_norm(
            final_base_pos[:, :2] - initial_base_pos[:, :2],
            dim=1,
        )
    final_radial_progress = _finite(final_radial_progress)
    final_stair_height_gain = torch.clamp(_stair_height_gain(base_env), min=0.0)
    final_steps_climbed = final_stair_height_gain / torch.clamp(step_height, min=1.0e-6)
    (
        final_wheel_terrain_z,
        final_wheel_heights,
        final_wheel_bottom_clearance,
        final_wheel_pos_w,
    ) = _wheel_terrain_measurements(base_env, wheel_body_ids)
    if base_env.scene.env_origins is not None:
        final_wheel_x_offset = final_wheel_pos_w[:, :, 0] - base_env.scene.env_origins[
            :, 0
        ].unsqueeze(1)
    else:
        final_wheel_x_offset = final_wheel_pos_w[:, :, 0] - initial_wheel_pos_w[:, :, 0]
    final_wheel_terrain_rise = _finite(final_wheel_terrain_z - initial_wheel_terrain_z)
    final_any_wheel_terrain_rise = torch.max(final_wheel_terrain_rise, dim=1).values
    final_both_wheel_terrain_rise = torch.min(final_wheel_terrain_rise, dim=1).values
    stair_height_gain_drop = torch.clamp(
        max_stair_height_gain - final_stair_height_gain,
        min=0.0,
    )
    stair_steps_climbed_drop = torch.clamp(max_steps_climbed - final_steps_climbed, min=0.0)
    stair_steps_climbed_final_to_max = _safe_ratio(final_steps_climbed, max_steps_climbed)
    stair_height_gain_final_to_max = _safe_ratio(
        final_stair_height_gain,
        max_stair_height_gain,
    )
    any_wheel_terrain_drop = torch.clamp(
        max_any_wheel_terrain_rise - final_any_wheel_terrain_rise,
        min=0.0,
    )
    both_wheel_terrain_drop = torch.clamp(
        max_both_wheel_terrain_rise - final_both_wheel_terrain_rise,
        min=0.0,
    )
    any_wheel_terrain_final_to_max = _safe_ratio(
        final_any_wheel_terrain_rise,
        max_any_wheel_terrain_rise,
    )
    both_wheel_terrain_final_to_max = _safe_ratio(
        final_both_wheel_terrain_rise,
        max_both_wheel_terrain_rise,
    )
    x_progress_drop = torch.clamp(max_x_progress - final_x_progress, min=0.0)
    negative_final_x_progress = final_x_progress < -1.0e-6
    stair_height_gain_dropped = stair_height_gain_drop > 1.0e-6
    stair_steps_climbed_dropped = stair_steps_climbed_drop > 1.0e-6
    any_wheel_terrain_dropped = any_wheel_terrain_drop > 1.0e-6
    both_wheel_terrain_dropped = both_wheel_terrain_drop > 1.0e-6
    x_progress_dropped = x_progress_drop > 1.0e-6
    pass_by_height = max_steps_climbed >= float(args.pass_step_ratio)
    pass_by_radial = max_radial_progress >= float(args.pass_radial_m)
    pass_by_final_radial = final_radial_progress >= float(args.pass_final_radial_m)
    pass_by_tilt = max_tilt_per_env <= float(args.pass_tilt_deg)
    legacy_passed = pass_by_height & pass_by_radial & pass_by_final_radial & pass_by_tilt
    required_support_steps = max(
        1,
        math.ceil(float(args.pass_support_duration_s) / float(base_env.step_dt)),
    )
    pass_by_upper_support = max_consecutive_both_support_steps >= int(required_support_steps)
    passed = pass_by_upper_support & pass_by_tilt
    legacy_only_passed = legacy_passed & ~passed
    strict_only_passed = passed & ~legacy_passed
    strict_legacy_disagree = legacy_passed != passed
    up_then_dropped = pass_by_height & (final_steps_climbed < float(args.pass_step_ratio))
    wheel_both_up_then_dropped = (
        max_both_wheel_terrain_rise >= step_height * float(args.pass_support_ratio)
    ) & (final_both_wheel_terrain_rise < step_height * float(args.pass_support_ratio))
    passed_env_ids = torch.nonzero(passed, as_tuple=False).flatten().detach().cpu().tolist()
    legacy_only_env_ids = (
        torch.nonzero(legacy_only_passed, as_tuple=False).flatten().detach().cpu().tolist()
    )
    negative_final_x_env_ids = (
        torch.nonzero(negative_final_x_progress, as_tuple=False).flatten().detach().cpu().tolist()
    )
    upper_support_env_ids = (
        torch.nonzero(pass_by_upper_support, as_tuple=False).flatten().detach().cpu().tolist()
    )
    tilt_pass_env_ids = (
        torch.nonzero(pass_by_tilt, as_tuple=False).flatten().detach().cpu().tolist()
    )
    state = getattr(base_env, "stair_climb_state", None)
    if state is not None:
        strict_max_duration = _finite(state.max_stair_success_duration())
    else:
        strict_max_duration = torch.zeros(base_env.num_envs, device=base_env.device)
    strict_episode_success = strict_max_duration >= float(args.strict_hold_duration_s)
    strict_episode_success_env_ids = (
        torch.nonzero(strict_episode_success, as_tuple=False).flatten().detach().cpu().tolist()
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "device": args.device,
        "num_envs": int(args.num_envs),
        "seconds": float(args.seconds),
        "steps": int(steps),
        "step_dt": float(base_env.step_dt),
        "mean_reward_per_step": reward_sum / denom,
        "done_rate_per_step": done_sum / denom,
        "ctbc_active_rate": active_sum / denom,
        "ctbc_stable_contact_rate": stable_sum / denom,
        "ctbc_force_mean_n": force_sum / denom,
        "ctbc_force_max_n": force_max,
        "ctbc_action_delta_abs_max_mean": delta_abs_sum / denom,
        "ctbc_action_delta_abs_peak": delta_abs_peak,
        "raw_action_saturation_rate": raw_sat_sum / denom,
        "unclipped_action_saturation_rate": unclipped_sat_sum / denom,
        "active_rod_clamp_rate": active_rod_clamp_sum / denom,
        "base_vx_mean_mps": vx_sum / denom,
        "base_vx_abs_max_mps": vx_abs_max,
        "base_height_initial_mean_m": _mean(initial_base_pos[:, 2]),
        "base_height_mean_m": base_height_sum / denom,
        "base_height_final_mean_m": _mean(final_height),
        "base_height_min_m": base_height_min if math.isfinite(base_height_min) else 0.0,
        "base_height_max_m": base_height_max if math.isfinite(base_height_max) else 0.0,
        "base_height_gain_max_m": (
            base_height_max - _mean(initial_base_pos[:, 2])
            if math.isfinite(base_height_max)
            else 0.0
        ),
        "stair_step_height_mean_m": _mean(step_height),
        "stair_height_gain_max_mean_m": _mean(max_stair_height_gain),
        "stair_height_gain_max_max_m": _max(max_stair_height_gain),
        "stair_height_gain_final_mean_m": _mean(final_stair_height_gain),
        "stair_height_gain_drop_final_from_max_mean_m": _mean(stair_height_gain_drop),
        "stair_height_gain_drop_final_from_max_max_m": _max(stair_height_gain_drop),
        "stair_height_gain_drop_final_from_max_rate": _mean(stair_height_gain_dropped.float()),
        "stair_height_gain_final_to_max_mean": _mean(stair_height_gain_final_to_max),
        "stair_height_gain_final_to_max_min": float(stair_height_gain_final_to_max.min().item()),
        "stair_steps_climbed_max_mean": _mean(max_steps_climbed),
        "stair_steps_climbed_max_max": _max(max_steps_climbed),
        "stair_steps_climbed_final_mean": _mean(final_steps_climbed),
        "stair_steps_climbed_drop_final_from_max_mean": _mean(stair_steps_climbed_drop),
        "stair_steps_climbed_drop_final_from_max_max": _max(stair_steps_climbed_drop),
        "stair_steps_climbed_drop_final_from_max_rate": _mean(stair_steps_climbed_dropped.float()),
        "stair_steps_climbed_final_to_max_mean": _mean(stair_steps_climbed_final_to_max),
        "stair_steps_climbed_final_to_max_min": float(
            stair_steps_climbed_final_to_max.min().item()
        ),
        "stair_up_then_dropped_rate": _mean(up_then_dropped.float()),
        "pass_step_ratio": float(args.pass_step_ratio),
        "pass_radial_m": float(args.pass_radial_m),
        "pass_final_radial_m": float(args.pass_final_radial_m),
        "pass_tilt_deg": float(args.pass_tilt_deg),
        "pass_support_ratio": float(args.pass_support_ratio),
        "pass_support_duration_s": float(args.pass_support_duration_s),
        "pass_support_required_steps": int(required_support_steps),
        "pass_wheel_contact_n": float(args.pass_wheel_contact_n),
        "pass_wheel_clearance_tol_m": float(args.pass_wheel_clearance_tol_m),
        "strict_min_success_steps": float(args.strict_min_success_steps),
        "strict_height_tolerance_m": float(args.strict_height_tolerance_m),
        "strict_forward_progress_m": args.strict_forward_progress_m,
        "strict_step_depth_m": float(args.strict_step_depth_m),
        "strict_forward_progress_step_fraction": float(args.strict_forward_progress_step_fraction),
        "strict_hold_duration_s": float(args.strict_hold_duration_s),
        "strict_upright_threshold": float(args.strict_upright_threshold),
        "strict_max_vertical_speed_mps": float(args.strict_max_vertical_speed_mps),
        "strict_illegal_contact_force_n": float(args.strict_illegal_contact_force_n),
        "strict_riser_stall_duration_s": float(args.strict_riser_stall_duration_s),
        "strict_pass_rate": _mean(strict_episode_success.float()),
        "strict_episode_success_rate": _mean(strict_episode_success.float()),
        "strict_episode_success_env_ids": [
            int(env_id) for env_id in strict_episode_success_env_ids
        ],
        "strict_success_rate_per_step": strict_success_sum / denom,
        "strict_candidate_rate_per_step": strict_candidate_sum / denom,
        "strict_height_cond_rate_per_step": strict_height_sum / denom,
        "strict_forward_cond_rate_per_step": strict_forward_sum / denom,
        "strict_upright_cond_rate_per_step": strict_upright_sum / denom,
        "strict_vertical_speed_cond_rate_per_step": strict_vertical_speed_sum / denom,
        "strict_legal_contact_cond_rate_per_step": strict_legal_contact_sum / denom,
        "strict_riser_clear_rate_per_step": strict_riser_clear_sum / denom,
        "strict_wheel_contact_rate_per_step": strict_wheel_contact_sum / denom,
        "strict_near_support_height_rate_per_step": strict_near_support_height_sum / denom,
        "strict_valid_rate_per_step": strict_valid_sum / denom,
        "strict_success_duration_mean_s": strict_duration_sum / denom,
        "strict_success_duration_max_s": strict_duration_max,
        "strict_success_max_duration_mean_s": _mean(strict_max_duration),
        "strict_success_max_duration_max_s": _max(strict_max_duration),
        "strict_current_both_rise_mean_m": strict_current_both_rise_sum / denom,
        "strict_current_both_rise_max_m": strict_current_both_rise_max,
        "strict_radial_distance_mean_m": strict_radial_sum / denom,
        "strict_radial_distance_max_m": strict_radial_max,
        "strict_height_target_mean_m": strict_height_target_sum / denom,
        "strict_forward_target_mean_m": strict_forward_target_sum / denom,
        "pass_rate": _mean(passed.float()),
        "legacy_pass_rate": _mean(legacy_passed.float()),
        "legacy_minus_strict_pass_rate": _mean(legacy_passed.float()) - _mean(passed.float()),
        "legacy_only_pass_rate": _mean(legacy_only_passed.float()),
        "strict_only_pass_rate": _mean(strict_only_passed.float()),
        "strict_legacy_disagree_rate": _mean(strict_legacy_disagree.float()),
        "pass_height_rate": _mean(pass_by_height.float()),
        "pass_radial_rate": _mean(pass_by_radial.float()),
        "pass_final_radial_rate": _mean(pass_by_final_radial.float()),
        "pass_tilt_rate": _mean(pass_by_tilt.float()),
        "pass_upper_support_rate": _mean(pass_by_upper_support.float()),
        "passed_env_ids": [int(env_id) for env_id in passed_env_ids],
        "legacy_only_env_ids": [int(env_id) for env_id in legacy_only_env_ids],
        "upper_support_env_ids": [int(env_id) for env_id in upper_support_env_ids],
        "tilt_pass_env_ids": [int(env_id) for env_id in tilt_pass_env_ids],
        "initial_wheel_terrain_z_mean_m": _mean(initial_wheel_terrain_z),
        "initial_wheel_height_mean_m": _mean(initial_wheel_heights),
        "initial_wheel_bottom_clearance_mean_m": _mean(initial_wheel_bottom_clearance),
        "initial_wheel_x_offset_any_mean_m": _mean(torch.max(initial_wheel_x_offset, dim=1).values),
        "initial_wheel_x_offset_both_mean_m": _mean(
            torch.min(initial_wheel_x_offset, dim=1).values
        ),
        "wheel_x_offset_any_max_mean_m": _mean(max_any_wheel_x_offset),
        "wheel_x_offset_both_max_mean_m": _mean(max_both_wheel_x_offset),
        "wheel_x_offset_final_any_mean_m": _mean(torch.max(final_wheel_x_offset, dim=1).values),
        "wheel_x_offset_final_both_mean_m": _mean(torch.min(final_wheel_x_offset, dim=1).values),
        "wheel_terrain_rise_any_max_mean_m": _mean(max_any_wheel_terrain_rise),
        "wheel_terrain_rise_any_max_max_m": _max(max_any_wheel_terrain_rise),
        "wheel_terrain_rise_both_max_mean_m": _mean(max_both_wheel_terrain_rise),
        "wheel_terrain_rise_both_max_max_m": _max(max_both_wheel_terrain_rise),
        "wheel_terrain_rise_final_any_mean_m": _mean(final_any_wheel_terrain_rise),
        "wheel_terrain_rise_final_both_mean_m": _mean(final_both_wheel_terrain_rise),
        "wheel_terrain_rise_any_drop_final_from_max_mean_m": _mean(any_wheel_terrain_drop),
        "wheel_terrain_rise_any_drop_final_from_max_max_m": _max(any_wheel_terrain_drop),
        "wheel_terrain_rise_any_drop_final_from_max_rate": _mean(any_wheel_terrain_dropped.float()),
        "wheel_terrain_rise_any_final_to_max_mean": _mean(any_wheel_terrain_final_to_max),
        "wheel_terrain_rise_any_final_to_max_min": float(
            any_wheel_terrain_final_to_max.min().item()
        ),
        "wheel_terrain_rise_both_drop_final_from_max_mean_m": _mean(both_wheel_terrain_drop),
        "wheel_terrain_rise_both_drop_final_from_max_max_m": _max(both_wheel_terrain_drop),
        "wheel_terrain_rise_both_drop_final_from_max_rate": _mean(
            both_wheel_terrain_dropped.float()
        ),
        "wheel_terrain_rise_both_final_to_max_mean": _mean(both_wheel_terrain_final_to_max),
        "wheel_terrain_rise_both_final_to_max_min": float(
            both_wheel_terrain_final_to_max.min().item()
        ),
        "wheel_both_up_then_dropped_rate": _mean(wheel_both_up_then_dropped.float()),
        "wheel_height_mean_m": wheel_height_sum / denom,
        "wheel_height_final_mean_m": _mean(final_wheel_heights),
        "wheel_bottom_clearance_abs_mean_m": wheel_bottom_clearance_abs_sum / denom,
        "wheel_bottom_clearance_final_abs_mean_m": _mean(torch.abs(final_wheel_bottom_clearance)),
        "wheel_contact_any_rate": contact_any_sum / denom,
        "wheel_contact_both_rate": contact_both_sum / denom,
        "wheel_contact_force_max_mean_n": _mean(max_wheel_contact_force),
        "wheel_contact_force_max_max_n": _max(max_wheel_contact_force),
        "upper_support_any_rate_per_step": upper_support_any_sum / denom,
        "upper_support_both_rate_per_step": upper_support_both_sum / denom,
        "upper_support_both_max_duration_s_mean": _mean(
            max_consecutive_both_support_steps.float() * float(base_env.step_dt)
        ),
        "upper_support_both_max_duration_s_max": _max(
            max_consecutive_both_support_steps.float() * float(base_env.step_dt)
        ),
        "radial_progress_max_mean_m": _mean(max_radial_progress),
        "radial_progress_max_max_m": _max(max_radial_progress),
        "radial_progress_final_mean_m": _mean(final_radial_progress),
        "x_progress_mean_m": x_progress_sum / denom,
        "x_progress_max_mean_m": _mean(max_x_progress),
        "x_progress_final_mean_m": _mean(final_x_progress),
        "x_progress_max_m": x_progress_max if math.isfinite(x_progress_max) else 0.0,
        "x_progress_drop_final_from_max_mean_m": _mean(x_progress_drop),
        "x_progress_drop_final_from_max_max_m": _max(x_progress_drop),
        "x_progress_drop_final_from_max_rate": _mean(x_progress_dropped.float()),
        "x_progress_final_negative": bool(torch.any(negative_final_x_progress).item()),
        "x_progress_final_negative_rate": _mean(negative_final_x_progress.float()),
        "x_progress_final_negative_env_ids": [int(env_id) for env_id in negative_final_x_env_ids],
        "tilt_mean_deg": tilt_sum / denom,
        "tilt_max_deg": tilt_max,
        "tilt_gt_30_rate": tilt_gt_30_sum / denom,
        "tilt_gt_45_rate": tilt_gt_45_sum / denom,
        "tilt_gt_60_rate": tilt_gt_60_sum / denom,
        "tilt_final_mean_deg": _mean(tilt),
        "tilt_final_max_deg": _max(tilt),
        "pitch_abs_mean_deg": pitch_abs_sum / denom,
        "roll_abs_mean_deg": roll_abs_sum / denom,
        "ctbc_requested_dx_min_m": ctbc_requested_dx_min,
        "ctbc_requested_dz_max_m": ctbc_requested_dz_max,
        "ctbc_requested_norm_max_m": ctbc_requested_norm_max,
        "wheel_target_delta_norm_max_m": wheel_target_delta_norm_max,
        "manual_triggered": bool(manual_triggered),
        "manual_trigger_side": args.manual_trigger_side,
        "start_x_offset_m": args.start_x_offset_m,
        "start_y_offset_m": args.start_y_offset_m,
        "start_yaw_deg": args.start_yaw_deg,
        "ff_x_m": None if args.ff_x_m is None else float(args.ff_x_m),
        "ff_lift_m": None if args.ff_lift_m is None else float(args.ff_lift_m),
        "ff_period_s": None if args.ff_period_s is None else float(args.ff_period_s),
        "ff_rise_ratio": None if args.ff_rise_ratio is None else float(args.ff_rise_ratio),
        "ff_hold_ratio": None if args.ff_hold_ratio is None else float(args.ff_hold_ratio),
        "ff_wheel_action": None if args.ff_wheel_action is None else float(args.ff_wheel_action),
    }
    result.update(_command_stats(base_env))
    result.update(_terrain_stats(base_env))
    result.update(_state_snapshot(base_env))
    env.close()
    return result


def main() -> None:
    args = _parse_args()
    result = _run(args)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
