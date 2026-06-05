"""Flow policy 本地 play。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import se3_train  # noqa: F401
from se3_shared import TASK_MODE_NAMES, TaskMode

from .registry import TASK_SPECS
from .runtime import FlowPolicyRuntime


@dataclass(frozen=True)
class ModeEvent:
    """按时间触发的 task mode 切换事件。"""

    time_s: float
    mode: TaskMode


class ScriptedFlowPolicy:
    """给 Flow policy 注入按时间变化的 task_mode 条件。"""

    def __init__(
        self,
        runtime: FlowPolicyRuntime,
        *,
        env,
        default_mode: TaskMode,
        events: list[ModeEvent],
        blend_s: float,
    ) -> None:
        """初始化脚本状态。"""
        self.runtime = runtime
        self.env = env
        self.events = sorted(events, key=lambda event: event.time_s)
        self.blend_s = max(float(blend_s), 1.0e-6)
        self.default_mode = default_mode
        self.current_mode = default_mode
        self.prev_mode = default_mode
        self.switch_time_s = 0.0
        self.next_event_idx = 0
        self.runtime.set_task_mode(default_mode)

    def reset(self) -> None:
        """重置 policy 和脚本状态。"""
        self.runtime.reset()
        self.current_mode = self.default_mode
        if self.events and self.events[0].time_s <= 0.0:
            self.current_mode = self.events[0].mode
        self.prev_mode = self.current_mode
        self.switch_time_s = 0.0
        self.next_event_idx = 1 if self.events and self.events[0].time_s <= 0.0 else 0
        self.runtime.set_task_mode(self.current_mode)

    def __call__(self, obs_dict: object) -> torch.Tensor:
        """按 env 时间更新 mode 条件后调用 Flow policy。"""
        t = float(self.env.unwrapped.common_step_counter) * float(self.env.unwrapped.step_dt)
        while (
            self.next_event_idx < len(self.events) and t >= self.events[self.next_event_idx].time_s
        ):
            event = self.events[self.next_event_idx]
            self.prev_mode = self.current_mode
            self.current_mode = event.mode
            self.switch_time_s = event.time_s
            self.next_event_idx += 1
        blend = min(max((t - self.switch_time_s) / self.blend_s, 0.0), 1.0)
        self.runtime.set_task_mode(self.current_mode, prev_mode=self.prev_mode, blend=blend)
        return self.runtime(obs_dict)


def run_flow_play(
    *,
    checkpoint: Path,
    task_mode: TaskMode,
    task_mode_script: list[ModeEvent],
    blend_s: float,
    num_envs: int,
    device: str,
    viewer: str,
    sample_steps: int | None,
    max_steps: int | None,
) -> None:
    """运行 Flow policy play。"""
    configure_torch_backends()
    task_id = _task_id_for_play(task_mode, has_script=bool(task_mode_script))
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = int(num_envs)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(task_id)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runtime = FlowPolicyRuntime(
        checkpoint,
        device=device,
        sample_steps=sample_steps,
        task_mode=task_mode,
    )
    policy = ScriptedFlowPolicy(
        runtime,
        env=wrapped,
        default_mode=task_mode,
        events=task_mode_script,
        blend_s=blend_s,
    )
    try:
        if viewer == "none":
            _run_headless(wrapped, policy, max_steps=max_steps or 200)
        else:
            resolved = _resolve_viewer(viewer)
            if resolved == "native":
                NativeMujocoViewer(wrapped, policy).run(num_steps=max_steps)
            elif resolved == "viser":
                ViserPlayViewer(wrapped, policy).run(num_steps=max_steps)
            else:
                raise ValueError(f"不支持的 viewer：{viewer}")
    finally:
        wrapped.close()


def _run_headless(env: RslRlVecEnvWrapper, policy: ScriptedFlowPolicy, *, max_steps: int) -> None:
    """无 viewer 跑固定步数，用于 smoke。"""
    policy.reset()
    obs, _ = env.reset()
    for _ in range(max_steps):
        with torch.no_grad():
            action = policy(obs)
            obs, _rew, dones, _extras = env.step(action)
            if hasattr(policy.runtime.model, "reset_done"):
                policy.runtime.model.reset_done(dones)
    print(f"[flow-play] headless steps={max_steps}")


def _task_id_for_play(mode: TaskMode, *, has_script: bool) -> str:
    """选择 play 环境。"""
    if has_script:
        return "SE3-WheelLegged-FlowMatch-Wheel-GRU"
    for spec in TASK_SPECS.values():
        if spec.mode == mode:
            return spec.task_id
    raise ValueError(f"找不到 TaskMode {mode.name} 对应 play task")


def _resolve_viewer(viewer: str) -> str:
    """解析 viewer backend。"""
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _parse_mode(raw: str) -> TaskMode:
    """解析 task mode 名称。"""
    normalized = raw.lower()
    if normalized not in TASK_MODE_NAMES:
        raise argparse.ArgumentTypeError(f"未知 task mode：{raw}")
    return TaskMode[normalized.upper()]


def _parse_script(raw: str) -> list[ModeEvent]:
    """解析 0s:wheel,5s:gait 形式的切换脚本。"""
    if not raw:
        return []
    events: list[ModeEvent] = []
    for part in raw.split(","):
        time_raw, mode_raw = part.split(":", 1)
        time_s = float(time_raw.rstrip("s"))
        events.append(ModeEvent(time_s=time_s, mode=_parse_mode(mode_raw)))
    events.sort(key=lambda event: event.time_s)
    return events


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(description="Play Flow Matching policy in MJLab")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-mode", type=_parse_mode, default=TaskMode.WHEEL)
    parser.add_argument("--task-mode-script", type=_parse_script, default=[])
    parser.add_argument("--blend-s", type=float, default=0.5)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--viewer", choices=("auto", "native", "viser", "none"), default="auto")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    run_flow_play(
        checkpoint=args.checkpoint,
        task_mode=args.task_mode,
        task_mode_script=args.task_mode_script,
        blend_s=float(args.blend_s),
        num_envs=int(args.num_envs),
        device=str(args.device),
        viewer=str(args.viewer),
        sample_steps=args.sample_steps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
