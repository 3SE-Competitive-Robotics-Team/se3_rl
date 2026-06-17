"""Run a short stair checkpoint rollout and print stability/CTBC diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

TASK_NAME = "SE3-WheelLegged-Stair-GRU"
WATCH_USE_TRAIN_ENV_ENV = "SE3_WATCH_USE_TRAIN_ENV"
WATCH_ITER_ENV = "SE3_WATCH_ITER"
WATCH_TERRAIN_LEVEL_ENV = "SE3_WATCH_TERRAIN_LEVEL"
WATCH_COMMAND_HEIGHT_ENV = "SE3_WATCH_COMMAND_HEIGHT"
WARM_START_ITER_ENV = "SE3_WARM_START_ITERATION"
WARM_START_STEPS_PER_ITER_ENV = "SE3_WARM_START_STEPS_PER_ITER"


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
    parser.add_argument("--no-terminations", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _set_watch_env(args: argparse.Namespace) -> None:
    os.environ[WATCH_USE_TRAIN_ENV_ENV] = "1"
    if args.iteration is not None:
        os.environ[WATCH_ITER_ENV] = str(args.iteration)
        os.environ[WARM_START_ITER_ENV] = str(args.iteration)
        os.environ[WARM_START_STEPS_PER_ITER_ENV] = "64"
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
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _set_watch_env(args)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    import se3_train  # noqa: F401

    configure_torch_backends()
    torch.manual_seed(int(args.seed))

    cfg = load_env_cfg(args.task, play=False)
    cfg.scene.num_envs = int(args.num_envs)
    command_cfg = cfg.commands.get("velocity_height")
    if command_cfg is not None:
        if args.command_vx is not None:
            command_cfg.lin_vel_x_range = (float(args.command_vx), float(args.command_vx))
        if args.command_yaw is not None:
            command_cfg.ang_vel_yaw_range = (float(args.command_yaw), float(args.command_yaw))
    if (args.command_vx is not None or args.command_yaw is not None) and hasattr(cfg, "curriculum"):
        cfg.curriculum.pop("command_vel", None)
    if args.no_terminations:
        cfg.terminations = {}

    base_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env)
    policy = _load_policy(env, args.checkpoint, args.task, args.device)
    env.reset()
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()

    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("delayed_action")
    steps = max(1, math.ceil(float(args.seconds) / float(base_env.step_dt)))

    active_sum = 0.0
    stable_sum = 0.0
    force_sum = 0.0
    force_max = 0.0
    raw_sat_sum = 0.0
    unclipped_sat_sum = 0.0
    delta_abs_sum = 0.0
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
    samples = 0

    for _ in range(steps):
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
        raw_sat_sum += _mean((torch.abs(raw_action) > 1.0).any(dim=-1).float())
        unclipped_sat_sum += _mean((torch.abs(unclipped_action) > 1.0).any(dim=-1).float())
        delta_abs_sum += _mean(torch.abs(delta[:, :4]).amax(dim=-1))

        pg = _finite(robot.data.projected_gravity_b)
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
        pitch = torch.rad2deg(torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0)))
        roll = torch.rad2deg(torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0)))
        vx = _finite(robot.data.root_link_lin_vel_b[:, 0])
        tilt_sum += _mean(tilt)
        tilt_max = max(tilt_max, _max(tilt))
        tilt_gt_30_sum += _mean((tilt > 30.0).float())
        tilt_gt_45_sum += _mean((tilt > 45.0).float())
        tilt_gt_60_sum += _mean((tilt > 60.0).float())
        pitch_abs_sum += _mean(torch.abs(pitch))
        roll_abs_sum += _mean(torch.abs(roll))
        vx_sum += _mean(vx)
        vx_abs_max = max(vx_abs_max, _max(torch.abs(vx)))
        done_sum += _mean(dones.float())
        reward_sum += _mean(reward)
        samples += 1

    denom = max(1, samples)
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
        "raw_action_saturation_rate": raw_sat_sum / denom,
        "unclipped_action_saturation_rate": unclipped_sat_sum / denom,
        "base_vx_mean_mps": vx_sum / denom,
        "base_vx_abs_max_mps": vx_abs_max,
        "tilt_mean_deg": tilt_sum / denom,
        "tilt_max_deg": tilt_max,
        "tilt_gt_30_rate": tilt_gt_30_sum / denom,
        "tilt_gt_45_rate": tilt_gt_45_sum / denom,
        "tilt_gt_60_rate": tilt_gt_60_sum / denom,
        "tilt_final_mean_deg": _mean(tilt),
        "tilt_final_max_deg": _max(tilt),
        "pitch_abs_mean_deg": pitch_abs_sum / denom,
        "roll_abs_mean_deg": roll_abs_sum / denom,
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
