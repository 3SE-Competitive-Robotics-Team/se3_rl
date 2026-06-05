"""Flow Matching teacher rollout 采集。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import se3_train  # noqa: F401

from .registry import DistillTaskSpec, parse_task_names, task_spec
from .task_mode import overwrite_task_mode_obs
from .teachers import TeacherPolicy


def collect_teacher_rollouts(
    *,
    tasks: list[str],
    output: Path,
    num_envs: int,
    steps: int,
    sequence_length: int,
    device: str,
) -> None:
    """采集多个 teacher 的 rollout 并保存。"""
    if steps <= 0:
        raise ValueError(f"steps 必须为正数，实际为 {steps}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length 必须为正数，实际为 {sequence_length}")
    configure_torch_backends()

    all_obs: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    all_dones: list[torch.Tensor] = []
    all_modes: list[torch.Tensor] = []
    teacher_names: list[str] = []
    used_specs: list[dict[str, object]] = []

    for name in tasks:
        spec = task_spec(name)
        obs, actions, dones, modes, names = _collect_one_task(
            spec=spec,
            num_envs=num_envs,
            steps=steps,
            sequence_length=sequence_length,
            device=device,
        )
        all_obs.append(obs)
        all_actions.append(actions)
        all_dones.append(dones)
        all_modes.append(modes)
        teacher_names.extend(names)
        used_specs.append(_spec_metadata(spec))

    payload = {
        "obs": torch.cat(all_obs, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "dones": torch.cat(all_dones, dim=0),
        "modes": torch.cat(all_modes, dim=0),
        "teacher_names": teacher_names,
        "metadata": {
            "tasks": tasks,
            "num_envs": num_envs,
            "steps": steps,
            "sequence_length": sequence_length,
            "teacher_specs": used_specs,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"[flow-collect] saved {output} obs={tuple(payload['obs'].shape)} "
        f"actions={tuple(payload['actions'].shape)}"
    )


def _collect_one_task(
    *,
    spec: DistillTaskSpec,
    num_envs: int,
    steps: int,
    sequence_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """采集单个 teacher rollout。"""
    if spec.teacher_path is None:
        raise ValueError(f"{spec.name} 未配置 teacher checkpoint")
    teacher = TeacherPolicy(spec.teacher_path, device=device)
    env_cfg = load_env_cfg(spec.task_id, play=True)
    env_cfg.scene.num_envs = int(num_envs)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        obs_steps: list[torch.Tensor] = []
        action_steps: list[torch.Tensor] = []
        done_steps: list[torch.Tensor] = []
        mode_steps: list[torch.Tensor] = []

        sequence_count = _sequence_count(steps, sequence_length)
        for _ in range(sequence_count):
            obs_dict, _ = env.reset()
            actor_obs = _actor_obs(obs_dict)
            teacher.reset(batch_size=num_envs)

            for _step in range(sequence_length):
                actor_obs = overwrite_task_mode_obs(actor_obs, spec.mode)
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

        obs = torch.stack(obs_steps, dim=1)
        actions = torch.stack(action_steps, dim=1)
        dones = torch.stack(done_steps, dim=1)
        modes = torch.stack(mode_steps, dim=1)
        obs, actions, dones, modes = _chunk_sequences(
            obs, actions, dones, modes, sequence_length=sequence_length
        )
        names = [spec.name] * int(obs.shape[0])
        return obs, actions, dones, modes, names
    finally:
        env.close()


def _sequence_count(steps: int, sequence_length: int) -> int:
    """把 steps 解释为每个 env 采集的总步数，并按序列长度向上取整。"""
    return max(1, (int(steps) + int(sequence_length) - 1) // int(sequence_length))


def _actor_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """读取 actor 观测并转成 float32。"""
    obs = obs_dict["actor"]
    if not isinstance(obs, torch.Tensor):
        raise TypeError("actor obs 必须是 torch.Tensor")
    if obs.ndim != 2 or obs.shape[1] != 42:
        raise ValueError(f"actor obs 必须是 [B, 42]，实际为 {tuple(obs.shape)}")
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


def _chunk_sequences(
    obs: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    modes: torch.Tensor,
    *,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 [E, S, D] rollout 切成 [N, T, D] 序列。"""
    usable = (obs.shape[1] // sequence_length) * sequence_length
    if usable == 0:
        raise ValueError(f"steps={obs.shape[1]} 小于 sequence_length={sequence_length}，无法切序列")
    obs = obs[:, :usable]
    actions = actions[:, :usable]
    dones = dones[:, :usable]
    modes = modes[:, :usable]
    envs = obs.shape[0]
    chunks = usable // sequence_length
    obs = obs.reshape(envs * chunks, sequence_length, obs.shape[-1])
    actions = actions.reshape(envs * chunks, sequence_length, actions.shape[-1])
    dones = dones.reshape(envs * chunks, sequence_length)
    modes = modes.reshape(envs * chunks, sequence_length)
    return obs.contiguous(), actions.contiguous(), dones.contiguous(), modes.contiguous()


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
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=64)
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
        steps=int(args.steps),
        sequence_length=int(args.sequence_length),
        device=str(args.device),
    )


if __name__ == "__main__":
    main()
