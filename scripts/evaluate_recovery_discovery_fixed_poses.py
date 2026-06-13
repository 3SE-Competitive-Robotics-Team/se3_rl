"""固定姿态批量评估 Discovery recovery checkpoint。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import se3_train  # noqa: F401
from se3_train.mdp.rewards import _contact_diagnostic_stats

TASK_NAME = "SE3-WheelLegged-Recovery-Discovery-GRU"
POSE_WEIGHTS: dict[str, tuple[float, float, float, float, float]] = {
    "standing": (1.0, 0.0, 0.0, 0.0, 0.0),
    "left_side": (0.0, 1.0, 0.0, 0.0, 0.0),
    "right_side": (0.0, 0.0, 1.0, 0.0, 0.0),
    "prone": (0.0, 0.0, 0.0, 1.0, 0.0),
    "supine": (0.0, 0.0, 0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class PoseResult:
    checkpoint: str
    pose: str
    episodes: int
    success_rate: float
    mean_standup_time_s: float | None
    final_height_error_m: float
    action_saturation_rate: float
    leg_contact_rate: float
    early_done_rate: float
    final_tilt_deg: float


def _set_if_present(params: dict, key: str, value) -> None:
    if key in params:
        params[key] = value


def _build_env_cfg(pose: str, num_envs: int, episode_s: float, command_height: float):
    cfg = load_env_cfg(TASK_NAME, play=True)
    cfg.scene.num_envs = int(num_envs)
    cfg.episode_length_s = float(episode_s)
    cfg.auto_reset = False

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.height_range = (command_height, command_height)
    command_cfg.standing_height_range = (command_height, command_height)
    command_cfg.standing_ratio = 1.0

    root_params = cfg.events["reset_root_state"].params
    root_params.update(
        {
            "pos_xy_range": (0.0, 0.0),
            "height_offset_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
            "roll_jitter_range": (0.0, 0.0),
            "pitch_jitter_range": (0.0, 0.0),
            "lin_vel_range": (0.0, 0.0),
            "ang_vel_range": (0.0, 0.0),
            "clearance_range": (0.003, 0.003),
            "pose_weights": POSE_WEIGHTS[pose],
            "recovery_command_height": command_height,
            "curriculum_stages": [],
            "use_iterations": False,
            "offset_iter": 0,
        }
    )

    joint_params = cfg.events["reset_joints"].params
    joint_params.update(
        {
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "align_root_height_to_wheels": True,
            "curriculum_stages": [],
            "use_iterations": False,
            "offset_iter": 0,
        }
    )
    _set_if_present(joint_params, "wheel_joint_vel_range", (0.0, 0.0))
    return cfg


def _load_policy(env: RslRlVecEnvWrapper, checkpoint: Path, device: str):
    agent_cfg = load_rl_cfg(TASK_NAME)
    runner_cls = load_runner_cls(TASK_NAME) or MjlabOnPolicyRunner
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


def _evaluate_pose(
    checkpoint: Path,
    pose: str,
    num_envs: int,
    episode_s: float,
    command_height: float,
    success_height_tolerance: float,
    success_tilt_deg: float,
    hold_s: float,
    action_saturation_threshold: float,
    device: str,
) -> PoseResult:
    env_cfg = _build_env_cfg(pose, num_envs, episode_s, command_height)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env)
    policy = _load_policy(env, checkpoint, device)
    env.reset()
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()

    robot = base_env.scene["robot"]
    height_sensor = base_env.scene["base_height_sensor"]
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    step_dt = float(base_env.step_dt)
    max_steps = math.ceil(episode_s / step_dt)
    hold_steps = max(1, math.ceil(hold_s / step_dt))

    done = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.bool)
    success = torch.zeros_like(done)
    success_streak = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
    success_time = torch.full((base_env.num_envs,), torch.nan, device=base_env.device)
    sample_count = torch.zeros(base_env.num_envs, device=base_env.device)
    saturation_sum = torch.zeros_like(sample_count)
    leg_contact_sum = torch.zeros_like(sample_count)
    final_height_error = torch.full_like(sample_count, torch.nan)
    final_tilt_deg = torch.full_like(sample_count, torch.nan)

    for step_idx in range(1, max_steps + 1):
        active = ~done
        if not torch.any(active):
            break

        with torch.no_grad():
            obs = env.get_observations()
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)

        active_ids = active.nonzero(as_tuple=False).squeeze(-1)
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt_deg = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
        base_height = torch.nan_to_num(
            height_sensor.data.heights[:, 0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        height_error = torch.abs(base_height - command_height)
        action = base_env.action_manager.action
        action_saturated = torch.max(torch.abs(action), dim=1).values > action_saturation_threshold
        leg_contact_ratio, _, _ = _contact_diagnostic_stats(base_env, "leg_contact_sensor", 1.0)
        leg_contact = leg_contact_ratio > 0.0

        sample_count[active_ids] += 1.0
        saturation_sum[active_ids] += action_saturated[active_ids].float()
        leg_contact_sum[active_ids] += leg_contact[active_ids].float()
        final_height_error[active_ids] = height_error[active_ids]
        final_tilt_deg[active_ids] = tilt_deg[active_ids]

        success_now = (tilt_deg < success_tilt_deg) & (height_error < success_height_tolerance)
        success_streak[active_ids] = torch.where(
            success_now[active_ids],
            success_streak[active_ids] + 1,
            torch.zeros_like(success_streak[active_ids]),
        )
        newly_success = active & ~success & (success_streak >= hold_steps)
        if torch.any(newly_success):
            first_hold_step = step_idx - hold_steps + 1
            success_time[newly_success] = max(0, first_hold_step) * step_dt
            success[newly_success] = True

        step_done = dones.to(device=base_env.device, dtype=torch.bool) & active
        if torch.any(step_done):
            done |= step_done
            if reset_fn is not None:
                reset_fn(step_done.to(dtype=torch.long))
            base_env.reset(env_ids=env_ids[step_done])

    valid_samples = torch.clamp(sample_count, min=1.0)
    successful_times = success_time[success]
    mean_time = (
        float(torch.nanmean(successful_times).item()) if successful_times.numel() > 0 else None
    )
    early_done = done & (sample_count < max_steps)
    result = PoseResult(
        checkpoint=checkpoint.name,
        pose=pose,
        episodes=base_env.num_envs,
        success_rate=float(success.float().mean().item()),
        mean_standup_time_s=mean_time,
        final_height_error_m=float(torch.nanmean(final_height_error).item()),
        action_saturation_rate=float((saturation_sum / valid_samples).mean().item()),
        leg_contact_rate=float((leg_contact_sum / valid_samples).mean().item()),
        early_done_rate=float(early_done.float().mean().item()),
        final_tilt_deg=float(torch.nanmean(final_tilt_deg).item()),
    )
    env.close()
    return result


def _format_result_table(results: list[PoseResult]) -> str:
    lines = [
        "| checkpoint | pose | success | standup_time_s | height_err_m | action_sat | leg_contact | early_done | final_tilt_deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        time_text = "n/a" if item.mean_standup_time_s is None else f"{item.mean_standup_time_s:.3f}"
        lines.append(
            "| "
            f"{item.checkpoint} | {item.pose} | {item.success_rate:.3f} | {time_text} | "
            f"{item.final_height_error_m:.4f} | {item.action_saturation_rate:.3f} | "
            f"{item.leg_contact_rate:.3f} | {item.early_done_rate:.3f} | "
            f"{item.final_tilt_deg:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--episode-s", type=float, default=5.0)
    parser.add_argument("--command-height", type=float, default=0.26)
    parser.add_argument("--success-height-tolerance", type=float, default=0.02)
    parser.add_argument("--success-tilt-deg", type=float, default=15.0)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoints = [Path(value).expanduser().resolve() for value in args.checkpoint]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    results: list[PoseResult] = []
    for checkpoint in checkpoints:
        for pose in POSE_WEIGHTS:
            print(f"[INFO] evaluating {checkpoint.name} pose={pose}", flush=True)
            results.append(
                _evaluate_pose(
                    checkpoint=checkpoint,
                    pose=pose,
                    num_envs=args.num_envs,
                    episode_s=args.episode_s,
                    command_height=args.command_height,
                    success_height_tolerance=args.success_height_tolerance,
                    success_tilt_deg=args.success_tilt_deg,
                    hold_s=args.hold_s,
                    action_saturation_threshold=args.action_saturation_threshold,
                    device=device,
                )
            )

    print(_format_result_table(results))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] wrote {args.output_json}")


if __name__ == "__main__":
    main()
