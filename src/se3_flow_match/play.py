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
from se3_shared import TASK_MODE_NAMES, JointGroup, TaskMode
from se3_shared.grounded_pose import solve_grounded_pose

from .registry import TASK_SPECS
from .runtime import FlowPolicyRuntime
from .task_mode import overwrite_task_mode_obs

_SCRIPT_TASK_ID = "SE3-WheelLegged-FlowMatch-Loco-Script-GRU"
_COMMAND_NAME = "velocity_height"
_ACTION_NAME = "delayed_action"
_WHEEL_COMMAND = {
    "lin_vel_x_range": (-2.5, 2.5),
    "ang_vel_yaw_range": (-3.0, 3.0),
    "pitch_range": (-0.2, 0.2),
    "roll_range": (-0.1, 0.1),
    "height_range": (0.20, 0.32),
    "standing_height_range": (0.20, 0.32),
    "lin_vel_deadband": 0.1,
    "yaw_deadband": 0.1,
}
_GAIT_COMMAND = {
    "lin_vel_x_range": (0.05, 1.5),
    "ang_vel_yaw_range": (0.0, 0.0),
    "pitch_range": (0.0, 0.0),
    "roll_range": (0.0, 0.0),
    "height_range": (0.35, 0.35),
    "standing_height_range": (0.35, 0.35),
    "lin_vel_deadband": 0.03,
    "yaw_deadband": 0.1,
}
_WHEEL_ACTION_SCALE = 20.0
_GAIT_HEIGHT = 0.35


@dataclass(frozen=True)
class ModeEvent:
    """按时间触发的 task mode 切换事件。"""

    time_s: float
    mode: TaskMode


class ScriptModeContract:
    """脚本 play 里随 mode 切换的环境物理契约。"""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        """缓存 wheel/gait 两套默认关节姿态。"""
        robot = env.scene["robot"]
        self._wheel_default_joint_pos = robot.data.default_joint_pos.clone()
        self._gait_default_joint_pos = self._wheel_default_joint_pos.clone()
        gait_pose = solve_grounded_pose(
            _GAIT_HEIGHT,
            keep_wheel_x=False,
            align_com_x=True,
        )
        if not gait_pose.success:
            raise ValueError(f"无法求解 GAIT play 默认姿态：{gait_pose.message}")
        joint_ids = torch.tensor(
            JointGroup.ALL,
            device=robot.data.default_joint_pos.device,
            dtype=torch.long,
        )
        q6 = torch.tensor(
            gait_pose.q6,
            device=robot.data.default_joint_pos.device,
            dtype=robot.data.default_joint_pos.dtype,
        )
        self._gait_default_joint_pos[:, joint_ids] = q6

    def apply(self, env: ManagerBasedRlEnv, mode: TaskMode) -> None:
        """应用当前 mode 的 default_joint_pos。"""
        robot = env.scene["robot"]
        if mode == TaskMode.GAIT:
            robot.data.default_joint_pos[:] = self._gait_default_joint_pos
        else:
            robot.data.default_joint_pos[:] = self._wheel_default_joint_pos


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
        self.contract = ScriptModeContract(env.unwrapped)
        self.runtime.set_task_mode(default_mode)
        self._sync_env_contract(default_mode, default_mode, 1.0)

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
        self._sync_env_contract(self.current_mode, self.prev_mode, 1.0)

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
        self._sync_env_contract(self.current_mode, self.prev_mode, blend)
        return self.runtime(
            _overwrite_script_actor_obs(
                obs_dict,
                self.env.unwrapped,
                self.current_mode,
                self.prev_mode,
                blend,
            )
        )

    def _sync_env_contract(self, current: TaskMode, prev: TaskMode, blend: float) -> None:
        """同步脚本 mode 到 env 的 command 和动作物理语义。"""
        env = self.env.unwrapped
        self.contract.apply(env, current)
        _set_command_mode(env, current, prev, blend)
        _set_command_shape(env, current)
        _set_action_mode(env, current)


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
        return _SCRIPT_TASK_ID
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


def _set_command_mode(
    env: ManagerBasedRlEnv, current: TaskMode, prev: TaskMode, blend: float
) -> None:
    """把脚本 mode 写入 env command term，保证观测和 viewer 一致。"""
    term = env.command_manager.get_term(_COMMAND_NAME)
    command = term.command
    if command.shape[1] < 11:
        return
    command[:, 5:8] = 0.0
    command[:, 8] = float(int(current))
    command[:, 9] = float(blend)
    command[:, 10] = float(int(prev))
    _fill_long_tensor(term, "_mode", int(current))
    _fill_long_tensor(term, "_prev_mode", int(prev))
    _fill_long_tensor(term, "_jump_stage", 0)
    _fill_long_tensor(term, "_traj_step", 0)
    _fill_long_tensor(term, "_jump_cool_down", 0)
    mode_blend_steps = getattr(term, "_mode_blend_steps", None)
    if isinstance(mode_blend_steps, torch.Tensor):
        mode_blend_steps[:] = 1
    mode_elapsed_steps = getattr(term, "_mode_elapsed_steps", None)
    if isinstance(mode_elapsed_steps, torch.Tensor):
        mode_elapsed_steps[:] = max(round(float(blend) * 1000.0), 0)


def _set_command_shape(env: ManagerBasedRlEnv, mode: TaskMode) -> None:
    """按当前 mode 限制 command 范围，并修正已经采样的 command。"""
    term = env.command_manager.get_term(_COMMAND_NAME)
    cfg = term.cfg
    shape = _GAIT_COMMAND if mode == TaskMode.GAIT else _WHEEL_COMMAND
    for name, value in shape.items():
        setattr(cfg, name, value)
    command = term.command
    if command.shape[1] < 5:
        return
    command[:, 0] = command[:, 0].clamp(*shape["lin_vel_x_range"])
    command[:, 1] = command[:, 1].clamp(*shape["ang_vel_yaw_range"])
    command[:, 2] = command[:, 2].clamp(*shape["pitch_range"])
    command[:, 3] = command[:, 3].clamp(*shape["roll_range"])
    command[:, 4] = command[:, 4].clamp(*shape["height_range"])
    _apply_deadband(command, shape)
    if mode == TaskMode.GAIT:
        command[:, 4] = float(shape["height_range"][0])
        standing_mask = getattr(term, "_standing_mask", None)
        if isinstance(standing_mask, torch.Tensor):
            standing_mask[:] = False


def _overwrite_script_actor_obs(
    obs_dict: object,
    env: ManagerBasedRlEnv,
    current: TaskMode,
    prev: TaskMode,
    blend: float,
) -> object:
    """覆盖本步 policy 输入里的 command 和 task_mode，避免 viewer 取 obs 慢一拍。"""
    if not isinstance(obs_dict, dict) or "actor" not in obs_dict:
        return obs_dict
    actor = obs_dict["actor"]
    if not isinstance(actor, torch.Tensor) or actor.ndim != 2 or actor.shape[1] != 42:
        return obs_dict
    out_actor = actor.clone()
    command = env.command_manager.get_command(_COMMAND_NAME)
    scale = torch.tensor((2.0, 0.25, 5.0, 5.0, 5.0), device=out_actor.device, dtype=out_actor.dtype)
    out_actor[:, 6:11] = command[:, :5].to(device=out_actor.device, dtype=out_actor.dtype) * scale
    out_actor = overwrite_task_mode_obs(out_actor, current, prev=prev, blend=blend)
    out = dict(obs_dict)
    out["actor"] = out_actor
    return out


def _set_action_mode(env: ManagerBasedRlEnv, mode: TaskMode) -> None:
    """按 mode 切换轮子动作物理语义。"""
    action_term = env.action_manager.get_term(_ACTION_NAME)
    if mode == TaskMode.GAIT:
        action_term.cfg.wheel_scale = 0.0
        action_term.cfg.wheel_lock_damping = 0.0
        action_term.cfg.freeze_wheels = True
    else:
        action_term.cfg.wheel_scale = _WHEEL_ACTION_SCALE
        action_term.cfg.wheel_lock_damping = None
        action_term.cfg.freeze_wheels = False


def _apply_deadband(command: torch.Tensor, shape: dict[str, object]) -> None:
    """对当前 command 应用 mode 对应死区。"""
    lin_deadband = float(shape["lin_vel_deadband"])
    yaw_deadband = float(shape["yaw_deadband"])
    command[:, 0] = torch.where(
        command[:, 0].abs() < lin_deadband,
        torch.zeros_like(command[:, 0]),
        command[:, 0],
    )
    command[:, 1] = torch.where(
        command[:, 1].abs() < yaw_deadband,
        torch.zeros_like(command[:, 1]),
        command[:, 1],
    )


def _fill_long_tensor(term: object, name: str, value: int) -> None:
    """如果 term 有指定 long tensor，就整体写入固定值。"""
    tensor = getattr(term, name, None)
    if isinstance(tensor, torch.Tensor):
        tensor[:] = int(value)


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
