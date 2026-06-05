"""Flow 蒸馏模型和 expert 的 play 闭环对照诊断。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

import se3_train  # noqa: F401
from se3_flow_match.registry import DistillTaskSpec, task_spec
from se3_flow_match.runtime import FlowPolicyRuntime
from se3_flow_match.task_mode import overwrite_task_mode_obs
from se3_flow_match.teachers import TeacherPolicy
from se3_shared import TASK_MODE_NAMES, TaskMode

_COMMAND_NAME = "velocity_height"
_ROBOT_NAME = "robot"
_ACTION_DIM = 6
_OBS_DIM = 42
_FIRST_ACTION_STEPS = 20
_EARLY_TILT_STEPS = 100


@dataclass
class RolloutResult:
    """单条闭环 rollout 的统计结果。"""

    summary: dict[str, Any]
    action_first20: torch.Tensor
    tilt_first100: torch.Tensor


class _RolloutStats:
    """在线累计 rollout 统计，避免保存整段大轨迹。"""

    def __init__(self, *, num_envs: int, steps: int) -> None:
        self.num_envs = int(num_envs)
        self.steps = int(steps)
        self.done_by_step: list[int] = []
        self.action_abs_by_step_first20: list[float] = []
        self.leg_action_abs_by_step_first20: list[float] = []
        self.wheel_action_abs_by_step_first20: list[float] = []
        self.tilt_deg_by_step_first100: list[float] = []
        self.clip_fraction_by_step_first20: list[float] = []
        self.done_total = 0
        self.tilt_sum = 0.0
        self.tilt_count = 0
        self.tilt_max = 0.0
        self.base_height_sum = 0.0
        self.base_height_count = 0
        self.base_height_min = float("inf")
        self.action_abs_sum = 0.0
        self.leg_action_abs_sum = 0.0
        self.wheel_action_abs_sum = 0.0
        self.action_count = 0
        self.leg_action_count = 0
        self.wheel_action_count = 0
        self.clip_count = 0
        self.command_samples: list[torch.Tensor] = []
        self.task_mode_tail_samples: list[torch.Tensor] = []

    def record_state(self, *, env: ManagerBasedRlEnv, obs: torch.Tensor, step_idx: int) -> None:
        """记录 step 前的状态和观测契约。"""
        tilt_deg = _tilt_deg(env)
        base_height = _base_height(env)
        self.tilt_sum += float(tilt_deg.sum().item())
        self.tilt_count += int(tilt_deg.numel())
        self.tilt_max = max(self.tilt_max, float(tilt_deg.max().item()))
        self.base_height_sum += float(base_height.sum().item())
        self.base_height_count += int(base_height.numel())
        self.base_height_min = min(self.base_height_min, float(base_height.min().item()))
        if step_idx < _EARLY_TILT_STEPS:
            self.tilt_deg_by_step_first100.append(float(tilt_deg.mean().item()))
        self.command_samples.append(_command(env).detach().cpu())
        if step_idx == 0:
            self.task_mode_tail_samples.append(obs[:, 29:42].detach().cpu())

    def record_action(
        self,
        *,
        action: torch.Tensor,
        action_for_step: torch.Tensor,
        clip_actions: float | None,
        step_idx: int,
    ) -> None:
        """记录 policy 原始 action 和实际送入 env 的裁剪信息。"""
        action_cpu = action.detach().float().cpu()
        stepped_cpu = action_for_step.detach().float().cpu()
        if action_cpu.ndim != 2 or action_cpu.shape[1] != _ACTION_DIM:
            raise ValueError(f"action 必须是 [B, 6]，实际为 {tuple(action_cpu.shape)}")
        action_abs = action_cpu.abs()
        leg_abs = action_abs[:, :4]
        wheel_abs = action_abs[:, 4:6]
        self.action_abs_sum += float(action_abs.sum().item())
        self.leg_action_abs_sum += float(leg_abs.sum().item())
        self.wheel_action_abs_sum += float(wheel_abs.sum().item())
        self.action_count += int(action_abs.numel())
        self.leg_action_count += int(leg_abs.numel())
        self.wheel_action_count += int(wheel_abs.numel())
        clipped = (stepped_cpu != action_cpu).to(dtype=torch.float32)
        self.clip_count += int(clipped.sum().item())
        if step_idx < _FIRST_ACTION_STEPS:
            self.action_abs_by_step_first20.append(float(action_abs.mean().item()))
            self.leg_action_abs_by_step_first20.append(float(leg_abs.mean().item()))
            self.wheel_action_abs_by_step_first20.append(float(wheel_abs.mean().item()))
            if clip_actions is None:
                self.clip_fraction_by_step_first20.append(0.0)
            else:
                self.clip_fraction_by_step_first20.append(float(clipped.mean().item()))

    def record_done(self, done: torch.Tensor) -> None:
        """记录本步 done 数量。"""
        count = int(done.to(dtype=torch.bool).sum().item())
        self.done_by_step.append(count)
        self.done_total += count

    def summary(self) -> dict[str, Any]:
        """输出可 JSON 序列化的 summary。"""
        denom = max(1, self.steps * self.num_envs)
        command_summary = _summarize_commands(torch.cat(self.command_samples, dim=0))
        task_mode_summary = _summarize_task_mode_tail(torch.cat(self.task_mode_tail_samples, dim=0))
        return {
            "num_envs": self.num_envs,
            "steps": self.steps,
            "done_total": int(self.done_total),
            "done_rate": float(self.done_total / denom),
            "done_by_step": self.done_by_step,
            "tilt_deg_mean": _safe_div(self.tilt_sum, self.tilt_count),
            "tilt_deg_max": float(self.tilt_max),
            "tilt_deg_mean_by_step_first100": self.tilt_deg_by_step_first100,
            "base_height_mean": _safe_div(self.base_height_sum, self.base_height_count),
            "base_height_min": float(self.base_height_min),
            "action_abs_mean": _safe_div(self.action_abs_sum, self.action_count),
            "leg_action_abs_mean": _safe_div(self.leg_action_abs_sum, self.leg_action_count),
            "wheel_action_abs_mean": _safe_div(self.wheel_action_abs_sum, self.wheel_action_count),
            "action_abs_mean_by_step_first20": self.action_abs_by_step_first20,
            "leg_action_abs_mean_by_step_first20": self.leg_action_abs_by_step_first20,
            "wheel_action_abs_mean_by_step_first20": self.wheel_action_abs_by_step_first20,
            "action_clip_fraction": _safe_div(self.clip_count, self.action_count),
            "action_clip_fraction_by_step_first20": self.clip_fraction_by_step_first20,
            "command_summary": command_summary,
            "task_mode_obs_summary": task_mode_summary,
        }


def run_compare(
    *,
    checkpoint: Path,
    task_mode: TaskMode,
    num_envs: int,
    steps: int,
    device: str,
    output: Path,
    sample_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    """运行 expert 和 Flow 两条闭环 rollout，并写出诊断 JSON。"""
    configure_torch_backends()
    spec = _spec_for_mode(task_mode)
    if spec.teacher_path is None:
        raise ValueError(f"{spec.name} 暂未配置 teacher checkpoint")
    expert = _run_expert_rollout(
        spec=spec,
        num_envs=num_envs,
        steps=steps,
        device=device,
        seed=seed,
    )
    flow = _run_flow_rollout(
        checkpoint=checkpoint,
        spec=spec,
        num_envs=num_envs,
        steps=steps,
        device=device,
        sample_steps=sample_steps,
        seed=seed,
    )
    diagnostics = _diagnose(task_mode=task_mode, expert=expert, flow=flow)
    payload = {
        "task_mode": task_mode.name.lower(),
        "task_id": spec.task_id,
        "checkpoint": str(checkpoint),
        "teacher_path": str(spec.teacher_path),
        "device": device,
        "seed": int(seed),
        "expert": expert.summary,
        "flow": flow.summary,
        "diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(payload)
    return payload


def _run_expert_rollout(
    *,
    spec: DistillTaskSpec,
    num_envs: int,
    steps: int,
    device: str,
    seed: int,
) -> RolloutResult:
    """运行 teacher 闭环。"""
    assert spec.teacher_path is not None
    teacher = TeacherPolicy(spec.teacher_path, device=device)
    env, clip_actions = _make_env(spec, num_envs=num_envs, device=device, seed=seed)
    try:
        obs_dict, _extras = env.reset(seed=seed)
        teacher.reset(batch_size=num_envs)
        stats = _RolloutStats(num_envs=num_envs, steps=steps)
        obs = _actor_obs(obs_dict)
        for step_idx in range(steps):
            policy_obs = overwrite_task_mode_obs(obs, spec.mode)
            stats.record_state(env=env, obs=policy_obs, step_idx=step_idx)
            with torch.no_grad():
                raw_action = teacher.act(policy_obs)
            action = _apply_action_policy(raw_action, spec.action_policy)
            action_for_step = _clip_action(raw_action, clip_actions)
            stats.record_action(
                action=action,
                action_for_step=_clip_action(action, clip_actions),
                clip_actions=clip_actions,
                step_idx=step_idx,
            )
            obs_dict, _rew, terminated, truncated, _extras = env.step(action_for_step)
            done = (terminated | truncated).to(dtype=torch.bool)
            stats.record_done(done)
            teacher.reset_done(done)
            obs = _actor_obs(obs_dict)
        return RolloutResult(
            summary=stats.summary(),
            action_first20=torch.tensor(stats.action_abs_by_step_first20),
            tilt_first100=torch.tensor(stats.tilt_deg_by_step_first100),
        )
    finally:
        env.close()


def _run_flow_rollout(
    *,
    checkpoint: Path,
    spec: DistillTaskSpec,
    num_envs: int,
    steps: int,
    device: str,
    sample_steps: int | None,
    seed: int,
) -> RolloutResult:
    """运行 Flow policy 闭环。"""
    runtime = FlowPolicyRuntime(
        checkpoint,
        device=device,
        sample_steps=sample_steps,
        task_mode=spec.mode,
    )
    env, clip_actions = _make_env(spec, num_envs=num_envs, device=device, seed=seed)
    try:
        obs_dict, _extras = env.reset(seed=seed)
        runtime.reset()
        stats = _RolloutStats(num_envs=num_envs, steps=steps)
        obs = _actor_obs(obs_dict)
        for step_idx in range(steps):
            policy_obs = overwrite_task_mode_obs(obs, spec.mode)
            stats.record_state(env=env, obs=policy_obs, step_idx=step_idx)
            with torch.no_grad():
                action = runtime({"actor": obs})
            action_for_step = _clip_action(action, clip_actions)
            stats.record_action(
                action=action,
                action_for_step=action_for_step,
                clip_actions=clip_actions,
                step_idx=step_idx,
            )
            obs_dict, _rew, terminated, truncated, _extras = env.step(action_for_step)
            done = (terminated | truncated).to(dtype=torch.bool)
            stats.record_done(done)
            runtime.model.reset_done(done)
            obs = _actor_obs(obs_dict)
        return RolloutResult(
            summary=stats.summary(),
            action_first20=torch.tensor(stats.action_abs_by_step_first20),
            tilt_first100=torch.tensor(stats.tilt_deg_by_step_first100),
        )
    finally:
        env.close()


def _make_env(
    spec: DistillTaskSpec,
    *,
    num_envs: int,
    device: str,
    seed: int,
) -> tuple[ManagerBasedRlEnv, float | None]:
    """创建对应 task 的 MJLab env。"""
    torch.manual_seed(int(seed))
    env_cfg = load_env_cfg(spec.task_id, play=True)
    env_cfg.scene.num_envs = int(num_envs)
    agent_cfg = load_rl_cfg(spec.task_id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    clip_actions = getattr(agent_cfg, "clip_actions", None)
    return env, None if clip_actions is None else float(clip_actions)


def _spec_for_mode(task_mode: TaskMode) -> DistillTaskSpec:
    """从 task mode 找 registry spec。"""
    for name in ("wheel", "gait"):
        spec = task_spec(name)
        if spec.mode == task_mode:
            return spec
    raise ValueError(f"当前只支持 wheel/gait，实际为 {task_mode.name.lower()}")


def _clip_action(action: torch.Tensor, clip_actions: float | None) -> torch.Tensor:
    """复现 RslRlVecEnvWrapper 的动作裁剪。"""
    if clip_actions is None:
        return action
    return torch.clamp(action, -float(clip_actions), float(clip_actions))


def _apply_action_policy(action: torch.Tensor, policy: str) -> torch.Tensor:
    """按蒸馏标签语义修正用于对比的 teacher action。"""
    if policy == "default":
        return action
    if policy == "zero_wheels":
        out = action.clone()
        out[:, 4:6] = 0.0
        return out
    raise ValueError(f"未知 action_policy：{policy}")


def _actor_obs(obs_dict: dict[str, object]) -> torch.Tensor:
    """读取 actor obs。"""
    obs = obs_dict["actor"]
    if not isinstance(obs, torch.Tensor):
        raise TypeError("actor obs 必须是 torch.Tensor")
    if obs.ndim != 2 or obs.shape[1] != _OBS_DIM:
        raise ValueError(f"actor obs 必须是 [B, 42]，实际为 {tuple(obs.shape)}")
    return obs.to(dtype=torch.float32)


def _robot(env: ManagerBasedRlEnv) -> object:
    """读取 robot entity。"""
    return env.scene[_ROBOT_NAME]


def _tilt_deg(env: ManagerBasedRlEnv) -> torch.Tensor:
    """计算机身相对直立方向倾角，单位 degree。"""
    pg_z = _robot(env).data.projected_gravity_b[:, 2]
    return torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))


def _base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """计算相对 env origin 的 base 高度。"""
    robot = _robot(env)
    height = robot.data.root_link_pos_w[:, 2]
    origins = getattr(env.scene, "env_origins", None)
    if isinstance(origins, torch.Tensor):
        height = height - origins[:, 2]
    return height


def _command(env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取 5D raw command。"""
    cmd = env.command_manager.get_command(_COMMAND_NAME)
    if not isinstance(cmd, torch.Tensor) or cmd.ndim != 2 or cmd.shape[1] < 5:
        raise ValueError(f"{_COMMAND_NAME} command 必须至少是 [B, 5]，实际为 {cmd!r}")
    return cmd[:, :5].to(dtype=torch.float32)


def _summarize_commands(commands: torch.Tensor) -> dict[str, dict[str, float]]:
    """汇总 command 分布。"""
    names = ("vx", "wz", "pitch", "roll", "height")
    return {name: _summarize_1d(commands[:, idx]) for idx, name in enumerate(names)}


def _summarize_task_mode_tail(tail: torch.Tensor) -> dict[str, Any]:
    """汇总 13D task_mode 观测尾部。"""
    current = tail[:, 0:4]
    prev = tail[:, 4:8]
    blend = tail[:, 8]
    return {
        "current_semantic_mean": _round_list(current.mean(dim=0)),
        "prev_semantic_mean": _round_list(prev.mean(dim=0)),
        "blend": _summarize_1d(blend),
        "jump_tail_abs_mean": float(tail[:, 9:13].abs().mean().item()),
    }


def _summarize_1d(values: torch.Tensor) -> dict[str, float]:
    """汇总一维 tensor。"""
    values = values.detach().float().cpu().reshape(-1)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _diagnose(
    *,
    task_mode: TaskMode,
    expert: RolloutResult,
    flow: RolloutResult,
) -> dict[str, Any]:
    """根据闭环统计生成启发式诊断。"""
    action_delta = _relative_delta(
        _mean_tensor(flow.action_first20),
        _mean_tensor(expert.action_first20),
    )
    early_action_mismatch = abs(action_delta) > 0.35
    early_tilt_delta = _mean_tensor(flow.tilt_first100) - _mean_tensor(expert.tilt_first100)
    early_tilt_mismatch = early_tilt_delta > 8.0
    expert_done_rate = float(expert.summary["done_rate"])
    flow_done_rate = float(flow.summary["done_rate"])
    done_rate_mismatch = flow_done_rate > max(
        expert_done_rate * 2.0 + 0.01, expert_done_rate + 0.05
    )
    wheel_contract = _wheel_action_contract(task_mode, expert.summary, flow.summary)
    messages = _diagnosis_messages(
        early_action_mismatch=early_action_mismatch,
        action_delta=action_delta,
        early_tilt_mismatch=early_tilt_mismatch,
        early_tilt_delta=early_tilt_delta,
        done_rate_mismatch=done_rate_mismatch,
        expert_done_rate=expert_done_rate,
        flow_done_rate=flow_done_rate,
        wheel_contract=wheel_contract,
    )
    return {
        "early_action_mismatch": early_action_mismatch,
        "early_action_relative_delta": action_delta,
        "early_tilt_mismatch": early_tilt_mismatch,
        "early_tilt_delta_deg": early_tilt_delta,
        "done_rate_mismatch": done_rate_mismatch,
        "wheel_action_contract_mismatch": bool(wheel_contract["mismatch"]),
        "wheel_action_contract": wheel_contract,
        "human_readable": messages,
    }


def _wheel_action_contract(
    task_mode: TaskMode,
    expert_summary: dict[str, Any],
    flow_summary: dict[str, Any],
) -> dict[str, Any]:
    """检查 wheel action 语义是否明显漂移。"""
    expert_wheel = float(expert_summary["wheel_action_abs_mean"])
    flow_wheel = float(flow_summary["wheel_action_abs_mean"])
    if task_mode == TaskMode.GAIT:
        ratio = None if expert_wheel < 1.0e-6 else flow_wheel / expert_wheel
        mismatch = flow_wheel > 0.05
        reason = "GAIT 下 Flow wheel action 应接近 0，避免锁轮语义在 wheel env 中变成真实轮速"
    else:
        ratio = flow_wheel / max(expert_wheel, 1.0e-6)
        mismatch = flow_wheel < max(0.05, expert_wheel * 0.35)
        reason = "WHEEL 下 Flow wheel action 明显低于 expert，轮驱动可能没学到"
    return {
        "mismatch": bool(mismatch),
        "expert_wheel_action_abs_mean": expert_wheel,
        "flow_wheel_action_abs_mean": flow_wheel,
        "flow_to_expert_ratio": ratio,
        "reason": reason,
    }


def _diagnosis_messages(
    *,
    early_action_mismatch: bool,
    action_delta: float,
    early_tilt_mismatch: bool,
    early_tilt_delta: float,
    done_rate_mismatch: bool,
    expert_done_rate: float,
    flow_done_rate: float,
    wheel_contract: dict[str, Any],
) -> list[str]:
    """生成面向人的诊断结论。"""
    messages: list[str] = []
    if early_action_mismatch:
        direction = "低于" if action_delta < 0.0 else "高于"
        messages.append(
            f"Flow 前 20 步 action 幅值已经{direction} expert，优先查冷启动监督、采样和 action target 语义。"
        )
    else:
        messages.append(
            "Flow 前 20 步 action 幅值和 expert 接近，后续差异更可能来自闭环状态分布漂移。"
        )
    if early_tilt_mismatch:
        messages.append(
            f"Flow 前 100 步 tilt 均值比 expert 高 {early_tilt_delta:.2f} deg，优先查姿态稳定和 reset 后早期状态覆盖。"
        )
    if done_rate_mismatch:
        messages.append(
            f"Flow done rate={flow_done_rate:.4f} 明显高于 expert={expert_done_rate:.4f}，优先查摔倒终止对应的姿态/高度统计。"
        )
    if wheel_contract["mismatch"]:
        messages.append(str(wheel_contract["reason"]))
    if len(messages) == 1 and not early_action_mismatch:
        messages.append(
            "当前闭环统计没有直接暴露首步动作问题，下一步可以加 shadow action 对比同一状态下的 expert/Flow 输出。"
        )
    return messages


def _relative_delta(value: float, reference: float) -> float:
    """计算相对差异。"""
    return (value - reference) / max(abs(reference), 1.0e-6)


def _mean_tensor(values: torch.Tensor) -> float:
    """安全计算 tensor 均值。"""
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _safe_div(total: float, count: int) -> float:
    """避免除零。"""
    return float(total / max(1, count))


def _round_list(values: torch.Tensor) -> list[float]:
    """把 tensor 转成普通 list。"""
    return [float(v) for v in values.detach().float().cpu().tolist()]


def _print_report(payload: dict[str, Any]) -> None:
    """打印控制台摘要。"""
    expert = payload["expert"]
    flow = payload["flow"]
    diagnostics = payload["diagnostics"]
    print(f"[flow-play-compare] task={payload['task_mode']} output saved")
    print(
        "[flow-play-compare] done_rate "
        f"expert={expert['done_rate']:.4f} flow={flow['done_rate']:.4f}"
    )
    print(
        "[flow-play-compare] tilt_deg_mean/max "
        f"expert={expert['tilt_deg_mean']:.2f}/{expert['tilt_deg_max']:.2f} "
        f"flow={flow['tilt_deg_mean']:.2f}/{flow['tilt_deg_max']:.2f}"
    )
    print(
        "[flow-play-compare] base_height_mean/min "
        f"expert={expert['base_height_mean']:.3f}/{expert['base_height_min']:.3f} "
        f"flow={flow['base_height_mean']:.3f}/{flow['base_height_min']:.3f}"
    )
    print(
        "[flow-play-compare] action_abs_mean leg/wheel "
        f"expert={expert['action_abs_mean']:.3f}/{expert['leg_action_abs_mean']:.3f}/"
        f"{expert['wheel_action_abs_mean']:.3f} "
        f"flow={flow['action_abs_mean']:.3f}/{flow['leg_action_abs_mean']:.3f}/"
        f"{flow['wheel_action_abs_mean']:.3f}"
    )
    print("[flow-play-compare] diagnosis:")
    for message in diagnostics["human_readable"]:
        print(f"  - {message}")


def _parse_mode(raw: str) -> TaskMode:
    """解析 task mode 名称。"""
    normalized = raw.lower()
    if normalized not in TASK_MODE_NAMES:
        raise argparse.ArgumentTypeError(f"未知 task mode：{raw}")
    mode = TaskMode[normalized.upper()]
    if mode not in (TaskMode.WHEEL, TaskMode.GAIT):
        raise argparse.ArgumentTypeError("当前诊断脚本只支持 wheel/gait")
    return mode


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(description="Compare expert and Flow closed-loop play stats")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-mode", type=_parse_mode, default=TaskMode.GAIT)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    run_compare(
        checkpoint=args.checkpoint,
        task_mode=args.task_mode,
        num_envs=int(args.num_envs),
        steps=int(args.steps),
        device=str(args.device),
        output=args.output,
        sample_steps=args.sample_steps,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
