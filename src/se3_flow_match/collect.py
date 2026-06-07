"""Flow Matching teacher 长时闭环 rollout 采集。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import se3_train  # noqa: F401
from se3_shared import TASK_MODE_LOCOMOTION_CONTRACT, ObservationConfig

from .registry import DistillTaskSpec, parse_task_names, task_spec
from .task_mode import overwrite_task_mode_obs
from .teachers import TeacherPolicy

_COMMAND_NAME = "velocity_height"
_OBS_CFG = ObservationConfig()


def collect_teacher_rollouts(
    *,
    tasks: list[str],
    output: Path,
    num_envs: int,
    command_hold_s: float = 15.0,
    command_batches: int = 4,
    coverage: str = "stratified",
    device: str,
    steps: int | None = None,
    sequence_length: int | None = None,
) -> None:
    """采集多个 teacher 的固定 command 长时闭环 rollout 并保存。"""
    del steps, sequence_length  # 兼容旧调用；当前统一使用 command hold window。
    if num_envs <= 0:
        raise ValueError(f"num_envs 必须为正数，实际为 {num_envs}")
    if command_hold_s <= 0.0:
        raise ValueError(f"command_hold_s 必须为正数，实际为 {command_hold_s}")
    if command_batches <= 0:
        raise ValueError(f"command_batches 必须为正数，实际为 {command_batches}")
    if coverage != "stratified":
        raise ValueError(f"当前只支持 stratified coverage，实际为 {coverage}")
    configure_torch_backends()

    all_obs: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    all_dones: list[torch.Tensor] = []
    all_modes: list[torch.Tensor] = []
    all_commands: list[torch.Tensor] = []
    teacher_names: list[str] = []
    used_specs: list[dict[str, object]] = []
    task_rollouts: list[dict[str, object]] = []

    for name in tasks:
        spec = task_spec(name)
        obs, actions, dones, modes, commands, names, rollout_metadata = _collect_one_task(
            spec=spec,
            num_envs=num_envs,
            command_hold_s=command_hold_s,
            command_batches=command_batches,
            coverage=coverage,
            device=device,
        )
        all_obs.append(obs)
        all_actions.append(actions)
        all_dones.append(dones)
        all_modes.append(modes)
        all_commands.append(commands)
        teacher_names.extend(names)
        used_specs.append(_spec_metadata(spec))
        task_rollouts.append(rollout_metadata)

    payload = {
        "obs": torch.cat(all_obs, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "dones": torch.cat(all_dones, dim=0),
        "modes": torch.cat(all_modes, dim=0),
        "commands": torch.cat(all_commands, dim=0),
        "teacher_names": teacher_names,
        "metadata": {
            "tasks": tasks,
            "num_envs": num_envs,
            "command_hold_s": float(command_hold_s),
            "command_batches": int(command_batches),
            "coverage": coverage,
            "teacher_specs": used_specs,
            "task_rollouts": task_rollouts,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"[flow-collect] saved {output} obs={tuple(payload['obs'].shape)} "
        f"actions={tuple(payload['actions'].shape)} commands={tuple(payload['commands'].shape)}",
        flush=True,
    )


def _collect_one_task(
    *,
    spec: DistillTaskSpec,
    num_envs: int,
    command_hold_s: float,
    command_batches: int,
    coverage: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], dict]:
    """采集单个 teacher 的固定 command 长时闭环 rollout。"""
    if spec.teacher_path is None:
        raise ValueError(f"{spec.name} 未配置 teacher checkpoint")
    teacher = TeacherPolicy(spec.teacher_path, device=device)
    env_cfg = load_env_cfg(spec.task_id, play=True)
    _apply_final_command_ranges(env_cfg, spec.task_id)
    env_cfg.scene.num_envs = int(num_envs)
    _extend_command_resampling(env_cfg, command_hold_s)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        command_term = env.command_manager.get_term(_COMMAND_NAME)
        hold_steps = max(1, round(float(command_hold_s) / float(env.step_dt)))
        obs_batches: list[torch.Tensor] = []
        action_batches: list[torch.Tensor] = []
        done_batches: list[torch.Tensor] = []
        mode_batches: list[torch.Tensor] = []
        command_batches_cpu: list[torch.Tensor] = []

        for batch_idx in range(command_batches):
            commands = _sample_commands(
                command_term.cfg,
                num_envs=num_envs,
                device=torch.device(device),
                coverage=coverage,
            )
            obs_dict, _ = env.reset()
            teacher.reset(batch_size=num_envs)
            _set_fixed_command(env, spec, commands, command_hold_s=command_hold_s)
            actor_obs = _prepare_actor_obs(_actor_obs(obs_dict), spec, commands)

            obs_steps: list[torch.Tensor] = []
            action_steps: list[torch.Tensor] = []
            done_steps: list[torch.Tensor] = []
            mode_steps: list[torch.Tensor] = []
            for _step in range(hold_steps):
                _set_fixed_command(env, spec, commands, command_hold_s=command_hold_s)
                actor_obs = _prepare_actor_obs(actor_obs, spec, commands)
                raw_action = teacher.act(actor_obs)
                target_action = _apply_action_policy(raw_action, spec.action_policy)
                next_obs, _rew, terminated, truncated, _extras = env.step(raw_action)
                done = (terminated | truncated).to(dtype=torch.bool)

                obs_steps.append(actor_obs.detach().cpu())
                action_steps.append(target_action.detach().cpu())
                done_steps.append(done.detach().cpu())
                mode_steps.append(torch.full((num_envs,), int(spec.mode), dtype=torch.long))

                teacher.reset_done(done)
                actor_obs = _actor_obs(next_obs)

            obs_batches.append(torch.stack(obs_steps, dim=1).contiguous())
            action_batches.append(torch.stack(action_steps, dim=1).contiguous())
            done_batches.append(torch.stack(done_steps, dim=1).contiguous())
            mode_batches.append(torch.stack(mode_steps, dim=1).contiguous())
            command_batches_cpu.append(commands.detach().cpu().contiguous())
            print(
                f"[flow-collect] task={spec.name} batch={batch_idx + 1}/{command_batches} "
                f"hold_steps={hold_steps}",
                flush=True,
            )

        obs = torch.cat(obs_batches, dim=0)
        actions = torch.cat(action_batches, dim=0)
        dones = torch.cat(done_batches, dim=0)
        modes = torch.cat(mode_batches, dim=0)
        commands = torch.cat(command_batches_cpu, dim=0)
        names = [spec.name] * int(obs.shape[0])
        metadata = {
            "task": spec.name,
            "task_id": spec.task_id,
            "num_trajectories": int(obs.shape[0]),
            "command_hold_steps": int(hold_steps),
            "command_hold_s": float(command_hold_s),
            "step_dt": float(env.step_dt),
            "command_ranges": _command_ranges(command_term.cfg),
        }
        return obs, actions, dones, modes, commands, names, metadata
    finally:
        env.close()


def _extend_command_resampling(env_cfg: object, command_hold_s: float) -> None:
    """把 env 内部 command resample 时间拉长，避免采集窗口中途换指令。"""
    command_cfg = env_cfg.commands[_COMMAND_NAME]
    hold = max(float(command_hold_s) + 1.0, 1.0)
    command_cfg.resampling_time_range = (hold, hold)


def _apply_final_command_ranges(env_cfg: object, task_id: str) -> None:
    """用 teacher 训练 cfg 的最终课程范围覆盖 play env 的采集范围。"""
    train_cfg = load_env_cfg(task_id, play=False)
    command_cfg = env_cfg.commands[_COMMAND_NAME]
    train_command_cfg = train_cfg.commands[_COMMAND_NAME]
    for name in (
        "lin_vel_x_range",
        "ang_vel_yaw_range",
        "pitch_range",
        "roll_range",
        "height_range",
    ):
        if hasattr(train_command_cfg, name):
            setattr(command_cfg, name, getattr(train_command_cfg, name))
    for term_cfg in getattr(train_cfg, "curriculum", {}).values():
        params = getattr(term_cfg, "params", {})
        if params.get("command_name") != _COMMAND_NAME:
            continue
        _apply_velocity_stage_range(command_cfg, params)
        _apply_linear_final_range(command_cfg, params)


def _apply_velocity_stage_range(command_cfg: object, params: dict[str, object]) -> None:
    """读取阶梯课程的最后一个速度范围。"""
    stages = params.get("velocity_stages")
    if not isinstance(stages, list) or not stages:
        return
    last_stage = max(stages, key=_stage_progress)
    if "lin_vel_x_range" in last_stage:
        command_cfg.lin_vel_x_range = last_stage["lin_vel_x_range"]
    if "ang_vel_yaw_range" in last_stage:
        command_cfg.ang_vel_yaw_range = last_stage["ang_vel_yaw_range"]


def _apply_linear_final_range(command_cfg: object, params: dict[str, object]) -> None:
    """读取线性课程的最终速度范围。"""
    if "end_lin_vel_x_range" in params:
        command_cfg.lin_vel_x_range = params["end_lin_vel_x_range"]
    if "ang_vel_yaw_range" in params:
        command_cfg.ang_vel_yaw_range = params["ang_vel_yaw_range"]


def _stage_progress(stage: object) -> int:
    """读取课程阶段阈值，兼容 iter 和 step。"""
    if not isinstance(stage, dict):
        return 0
    return int(stage.get("iter", stage.get("step", 0)))


def _sample_commands(
    command_cfg: object,
    *,
    num_envs: int,
    device: torch.device,
    coverage: str,
) -> torch.Tensor:
    """按 env cfg 范围采样固定 command。"""
    if coverage != "stratified":
        raise ValueError(f"当前只支持 stratified coverage，实际为 {coverage}")
    ranges = _command_ranges(command_cfg)
    commands = torch.empty(num_envs, 5, device=device, dtype=torch.float32)
    for dim, (_name, value_range) in enumerate(ranges.items()):
        commands[:, dim] = _stratified_values(value_range, count=num_envs, device=device)
    _overwrite_low_speed_commands(commands, command_cfg)
    _apply_deadband(commands, command_cfg)
    return commands


def _command_ranges(command_cfg: object) -> dict[str, tuple[float, float]]:
    """读取当前 teacher env cfg 的 5D command 范围。"""
    names = (
        "lin_vel_x_range",
        "ang_vel_yaw_range",
        "pitch_range",
        "roll_range",
        "height_range",
    )
    result: dict[str, tuple[float, float]] = {}
    for name in names:
        raw = getattr(command_cfg, name)
        lo, hi = float(raw[0]), float(raw[1])
        if hi < lo:
            raise ValueError(f"{name} 上界小于下界：{raw}")
        result[name] = (lo, hi)
    return result


def _stratified_values(
    value_range: tuple[float, float], *, count: int, device: torch.device
) -> torch.Tensor:
    """为单个连续维度生成分层随机样本。"""
    lo, hi = value_range
    if abs(hi - lo) <= 1.0e-8:
        return torch.full((count,), lo, device=device, dtype=torch.float32)
    bins = torch.arange(count, device=device, dtype=torch.float32) + torch.rand(
        count, device=device
    )
    values = lo + (hi - lo) * (bins / float(count))
    return values[torch.randperm(count, device=device)]


def _overwrite_low_speed_commands(commands: torch.Tensor, command_cfg: object) -> None:
    """保留少量站立/低速 command，提升低速稳定覆盖。"""
    count = max(1, int(commands.shape[0]) // 16)
    if count <= 0:
        return
    ranges = _command_ranges(command_cfg)
    commands[:count, 0] = _nearest_zero_in_range(ranges["lin_vel_x_range"])
    commands[:count, 1] = _nearest_zero_in_range(ranges["ang_vel_yaw_range"])
    commands[:count, 2] = _nearest_zero_in_range(ranges["pitch_range"])
    commands[:count, 3] = _nearest_zero_in_range(ranges["roll_range"])
    height_range = ranges["height_range"]
    standing_height = getattr(command_cfg, "standing_height_range", height_range)
    standing_mid = 0.5 * (float(standing_height[0]) + float(standing_height[1]))
    commands[:count, 4] = min(max(standing_mid, height_range[0]), height_range[1])


def _nearest_zero_in_range(value_range: tuple[float, float]) -> float:
    """返回范围内最接近 0 的值。"""
    lo, hi = value_range
    if lo <= 0.0 <= hi:
        return 0.0
    return lo if abs(lo) < abs(hi) else hi


def _apply_deadband(commands: torch.Tensor, command_cfg: object) -> None:
    """复现 BasicCommandTerm 对速度/yaw 的死区处理。"""
    lin_deadband = float(getattr(command_cfg, "lin_vel_deadband", 0.0))
    yaw_deadband = float(getattr(command_cfg, "yaw_deadband", 0.0))
    commands[:, 0] = torch.where(
        torch.abs(commands[:, 0]) < lin_deadband,
        torch.zeros_like(commands[:, 0]),
        commands[:, 0],
    )
    commands[:, 1] = torch.where(
        torch.abs(commands[:, 1]) < yaw_deadband,
        torch.zeros_like(commands[:, 1]),
        commands[:, 1],
    )


def _set_fixed_command(
    env: ManagerBasedRlEnv,
    spec: DistillTaskSpec,
    commands: torch.Tensor,
    *,
    command_hold_s: float,
) -> None:
    """把固定 command 和固定 task mode 写回 env command term。"""
    term = env.command_manager.get_term(_COMMAND_NAME)
    term.command[:, :5] = commands.to(device=term.command.device, dtype=term.command.dtype)
    if term.command.shape[1] >= 11:
        term.command[:, 5:8] = 0.0
        term.command[:, 8] = float(int(spec.mode))
        term.command[:, 9] = 1.0
        term.command[:, 10] = float(int(spec.mode))
    if hasattr(term, "time_left"):
        term.time_left[:] = max(float(command_hold_s), float(env.step_dt) * 2.0)
    if hasattr(term, "_standing_mask"):
        standing = (commands[:, 0].abs() <= 1.0e-8) & (commands[:, 1].abs() <= 1.0e-8)
        term._standing_mask[:] = standing.to(device=term.command.device)
    _set_optional_long_tensor(term, "_mode", int(spec.mode))
    _set_optional_long_tensor(term, "_prev_mode", int(spec.mode))
    _set_optional_long_tensor(term, "_mode_elapsed_steps", 1)
    _set_optional_long_tensor(term, "_mode_switch_steps", 10**9)
    _set_optional_long_tensor(term, "_mode_blend_steps", 1)
    _set_optional_long_tensor(term, "_jump_stage", 0)
    _set_optional_long_tensor(term, "_traj_step", 0)
    _set_optional_long_tensor(term, "_jump_cool_down", 0)


def _set_optional_long_tensor(term: object, name: str, value: int) -> None:
    """如果 term 有指定内部 long tensor，就整体写入固定值。"""
    tensor = getattr(term, name, None)
    if isinstance(tensor, torch.Tensor):
        tensor[:] = int(value)


def _prepare_actor_obs(
    obs: torch.Tensor,
    spec: DistillTaskSpec,
    commands: torch.Tensor,
) -> torch.Tensor:
    """覆盖 actor obs 中的 command 和 task_mode 条件。"""
    obs = _overwrite_command_obs(obs, commands)
    return overwrite_task_mode_obs(obs, spec.mode)


def _overwrite_command_obs(obs: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
    """覆盖 42D obs 中的 5D scaled command 切片。"""
    expected = TASK_MODE_LOCOMOTION_CONTRACT.num_obs
    if obs.shape[-1] != expected:
        raise ValueError(f"obs 末维必须为 {expected}，实际为 {obs.shape[-1]}")
    out = obs.clone()
    flat = out.reshape(-1, out.shape[-1])
    commands = commands.to(device=flat.device, dtype=flat.dtype)
    if commands.ndim != 2 or commands.shape[1] != 5:
        raise ValueError(f"commands 必须是 [B, 5]，实际为 {tuple(commands.shape)}")
    if commands.shape[0] != flat.shape[0]:
        raise ValueError(f"commands batch 必须是 {flat.shape[0]}，实际为 {commands.shape[0]}")
    scale = torch.tensor(_OBS_CFG.command_scale, device=flat.device, dtype=flat.dtype)
    flat[:, TASK_MODE_LOCOMOTION_CONTRACT.observation.require_slice("commands")] = commands * scale
    return out


def _actor_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """读取 actor 观测并转成 float32。"""
    obs = obs_dict["actor"]
    if not isinstance(obs, torch.Tensor):
        raise TypeError("actor obs 必须是 torch.Tensor")
    expected = TASK_MODE_LOCOMOTION_CONTRACT.num_obs
    if obs.ndim != 2 or obs.shape[1] != expected:
        raise ValueError(f"actor obs 必须是 [B, {expected}]，实际为 {tuple(obs.shape)}")
    return obs.to(dtype=torch.float32)


def _apply_action_policy(action: torch.Tensor, policy: str) -> torch.Tensor:
    """按 teacher 语义修正 action target。"""
    if policy == "default":
        return action
    if policy == "zero_wheels":
        out = action.clone()
        out[:, 4:6] = 0.0
        return out
    raise ValueError(f"未知 action_policy：{policy}")


def _spec_metadata(spec: DistillTaskSpec) -> dict[str, object]:
    """转成可序列化 metadata。"""
    raw = asdict(spec)
    raw["mode"] = int(spec.mode)
    raw["teacher_path"] = None if spec.teacher_path is None else str(spec.teacher_path)
    return raw


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(description="Collect Flow Matching teacher rollouts")
    parser.add_argument("--tasks", default="wheel,gait")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--command-hold-s", type=float, default=15.0)
    parser.add_argument("--command-batches", type=int, default=4)
    parser.add_argument("--coverage", choices=("stratified",), default="stratified")
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path("data/flow_match/wheel_gait.pt"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    collect_teacher_rollouts(
        tasks=parse_task_names(args.tasks),
        output=args.output,
        num_envs=int(args.num_envs),
        command_hold_s=float(args.command_hold_s),
        command_batches=int(args.command_batches),
        coverage=str(args.coverage),
        device=str(args.device),
        steps=args.steps,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
