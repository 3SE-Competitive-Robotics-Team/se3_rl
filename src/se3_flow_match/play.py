"""Flow policy 本地 play。"""

from __future__ import annotations

import argparse
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, VerbosityLevel, ViserPlayViewer

import se3_train  # noqa: F401
from se3_shared import TASK_MODE_NAMES, JointGroup, ObservationConfig, RobotConfig, TaskMode
from se3_shared.grounded_pose import solve_grounded_pose

from .registry import TASK_SPECS
from .runtime import FlowPolicyRuntime
from .task_mode import overwrite_task_mode_obs

_SCRIPT_TASK_ID = "SE3-WheelLegged-FlowMatch-Loco-Script-GRU"
_COMMAND_NAME = "velocity_height"
_ACTION_NAME = "delayed_action"
_WHEEL_HEIGHT = float(RobotConfig().default_base_height)
_WHEEL_COMMAND = {
    "lin_vel_x_range": (-2.5, 2.5),
    "ang_vel_yaw_range": (-3.0, 3.0),
    "pitch_range": (-0.2, 0.2),
    "roll_range": (-0.1, 0.1),
    "height_range": (_WHEEL_HEIGHT, _WHEEL_HEIGHT),
    "standing_height_range": (_WHEEL_HEIGHT, _WHEEL_HEIGHT),
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
_WHEEL_LOCK_RANGE = (-1.0e-4, 1.0e-4)
_WHEEL_LOCK_DAMPING = 10.0
_WHEEL_LOCK_FRICTIONLOSS = 1.0
_WHEEL_JOINT_NAMES = ("l_wheel_Joint", "r_wheel_Joint")
_WHEEL_LOCK_MODEL_FIELDS = ("jnt_limited", "jnt_range", "dof_damping", "dof_frictionloss")
_SCRIPT_COMMAND_HOLD_S = 1.0e6
_WHEEL_TO_GAIT_PREP_S = 3.0
_OBS_CFG = ObservationConfig()
_WHEEL_TO_GAIT_PREP_COMMAND = {
    "lin_vel_x_range": (0.0, 0.0),
    "ang_vel_yaw_range": (0.0, 0.0),
    "pitch_range": (0.0, 0.0),
    "roll_range": (0.0, 0.0),
    "height_range": (_GAIT_HEIGHT, _GAIT_HEIGHT),
    "standing_height_range": (_GAIT_HEIGHT, _GAIT_HEIGHT),
    "lin_vel_deadband": 0.0,
    "yaw_deadband": 0.0,
}


@dataclass(frozen=True)
class ModeEvent:
    """按时间触发的 task mode 切换事件。"""

    time_s: float
    mode: TaskMode


class ScriptModeContract:
    """脚本 play 里随 mode 切换的环境物理契约。"""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        """缓存 wheel/gait 两套 reset 和默认关节姿态。"""
        robot = env.scene["robot"]
        self._wheel_default_joint_pos = robot.data.default_joint_pos.clone()
        self._gait_default_joint_pos = self._wheel_default_joint_pos.clone()
        self._wheel_base_height = _read_reset_base_height(env)
        reset_joint_params = _read_reset_joint_params(env)
        self._wheel_joint_pos_override = reset_joint_params["joint_pos_override"]
        self._wheel_update_default_joint_pos = bool(reset_joint_params["update_default_joint_pos"])
        self._wheel_lock_contract = WheelJointLockContract(env)
        gait_pose = solve_grounded_pose(
            _GAIT_HEIGHT,
            keep_wheel_x=False,
            align_com_x=True,
        )
        if not gait_pose.success:
            raise ValueError(f"无法求解 GAIT play 默认姿态：{gait_pose.message}")
        self._gait_joint_pos_override = tuple(float(value) for value in gait_pose.q6)
        joint_ids = torch.tensor(
            JointGroup.ALL,
            device=robot.data.default_joint_pos.device,
            dtype=torch.long,
        )
        q6 = torch.tensor(
            self._gait_joint_pos_override,
            device=robot.data.default_joint_pos.device,
            dtype=robot.data.default_joint_pos.dtype,
        )
        self._gait_default_joint_pos[:, joint_ids] = q6

    def apply(self, env: ManagerBasedRlEnv, mode: TaskMode) -> None:
        """应用当前 mode 的 reset 高度、reset 关节姿态和 default_joint_pos。"""
        robot = env.scene["robot"]
        if mode == TaskMode.GAIT:
            robot.data.default_joint_pos[:] = self._gait_default_joint_pos
            _set_reset_base_height(env, _GAIT_HEIGHT)
            _set_reset_joint_params(
                env,
                joint_pos_override=self._gait_joint_pos_override,
                update_default_joint_pos=True,
            )
            self._wheel_lock_contract.apply(env, locked=True)
            _zero_wheel_joint_state(env)
        else:
            robot.data.default_joint_pos[:] = self._wheel_default_joint_pos
            _set_reset_base_height(env, self._wheel_base_height)
            _set_reset_joint_params(
                env,
                joint_pos_override=self._wheel_joint_pos_override,
                update_default_joint_pos=self._wheel_update_default_joint_pos,
            )
            self._wheel_lock_contract.apply(env, locked=False)

    def apply_wheel_to_gait_prep(self, env: ManagerBasedRlEnv) -> None:
        """应用 WHEEL->GAIT 准备段契约：腿按 gait 默认位姿，轮子仍可主动刹停。"""
        robot = env.scene["robot"]
        robot.data.default_joint_pos[:] = self._gait_default_joint_pos
        _set_reset_base_height(env, _GAIT_HEIGHT)
        _set_reset_joint_params(
            env,
            joint_pos_override=self._gait_joint_pos_override,
            update_default_joint_pos=True,
        )
        self._wheel_lock_contract.apply(env, locked=False)


class WheelJointLockContract:
    """脚本 GAIT 模式使用的轮轴物理锁。"""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        """缓存 wheel 模式下的轮轴模型字段。"""
        _ensure_model_fields(env, _WHEEL_LOCK_MODEL_FIELDS)
        mj_model = env.sim.mj_model
        self._joint_ids = tuple(_find_mj_joint_id(mj_model, name) for name in _WHEEL_JOINT_NAMES)
        self._dof_ids = tuple(int(mj_model.jnt_dofadr[joint_id]) for joint_id in self._joint_ids)
        model = env.sim.model
        self._wheel_jnt_limited = model.jnt_limited[list(self._joint_ids)].clone()
        self._wheel_jnt_range = model.jnt_range[:, list(self._joint_ids), :].clone()
        self._wheel_dof_damping = model.dof_damping[:, list(self._dof_ids)].clone()
        self._wheel_dof_frictionloss = model.dof_frictionloss[:, list(self._dof_ids)].clone()

    def apply(self, env: ManagerBasedRlEnv, *, locked: bool) -> None:
        """按当前模式写入轮轴模型字段。"""
        model = env.sim.model
        joint_ids = list(self._joint_ids)
        dof_ids = list(self._dof_ids)
        if locked:
            model.jnt_limited[joint_ids] = 1
            model.jnt_range[:, joint_ids, 0] = _WHEEL_LOCK_RANGE[0]
            model.jnt_range[:, joint_ids, 1] = _WHEEL_LOCK_RANGE[1]
            model.dof_damping[:, dof_ids] = _WHEEL_LOCK_DAMPING
            model.dof_frictionloss[:, dof_ids] = _WHEEL_LOCK_FRICTIONLOSS
        else:
            model.jnt_limited[joint_ids] = self._wheel_jnt_limited
            model.jnt_range[:, joint_ids, :] = self._wheel_jnt_range
            model.dof_damping[:, dof_ids] = self._wheel_dof_damping
            model.dof_frictionloss[:, dof_ids] = self._wheel_dof_frictionloss
        self._sync_host_model(env, locked=locked)

    def _sync_host_model(self, env: ManagerBasedRlEnv, *, locked: bool) -> None:
        """同步 host MjModel，避免 viewer 侧模型字段长期滞后。"""
        mj_model = env.sim.mj_model
        joint_ids = list(self._joint_ids)
        dof_ids = list(self._dof_ids)
        if locked:
            mj_model.jnt_limited[joint_ids] = 1
            mj_model.jnt_range[joint_ids, 0] = _WHEEL_LOCK_RANGE[0]
            mj_model.jnt_range[joint_ids, 1] = _WHEEL_LOCK_RANGE[1]
            mj_model.dof_damping[dof_ids] = _WHEEL_LOCK_DAMPING
            mj_model.dof_frictionloss[dof_ids] = _WHEEL_LOCK_FRICTIONLOSS
        else:
            mj_model.jnt_limited[joint_ids] = self._wheel_jnt_limited.detach().cpu().numpy()
            mj_model.jnt_range[joint_ids, :] = self._wheel_jnt_range[0].detach().cpu().numpy()
            mj_model.dof_damping[dof_ids] = self._wheel_dof_damping[0].detach().cpu().numpy()
            mj_model.dof_frictionloss[dof_ids] = (
                self._wheel_dof_frictionloss[0].detach().cpu().numpy()
            )


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
        self._script_start_step = int(self.env.unwrapped.common_step_counter)
        self._pending_mode: TaskMode | None = None
        self._pending_prev_mode: TaskMode | None = None
        self._pending_switch_time_s: float | None = None
        self._pending_script_command: torch.Tensor | None = None
        (
            self.current_mode,
            self.prev_mode,
            self.switch_time_s,
            self.next_event_idx,
        ) = self._initial_script_state()
        self.contract = ScriptModeContract(env.unwrapped)
        self.script_command: torch.Tensor | None = None
        self.runtime.set_task_mode(self.current_mode)
        self._sync_env_contract(self.current_mode, self.prev_mode, 1.0)

    def reset(self) -> None:
        """重置 policy 和脚本状态。"""
        self.runtime.reset()
        self._reset_script_clock()
        self._clear_pending_switch()
        (
            self.current_mode,
            self.prev_mode,
            self.switch_time_s,
            self.next_event_idx,
        ) = self._initial_script_state()
        self.script_command = None
        self.runtime.set_task_mode(self.current_mode)
        self._sync_env_contract(self.current_mode, self.prev_mode, 1.0)

    def pre_env_reset(self) -> None:
        """在 viewer 触发 env.reset 前恢复脚本初始 reset 契约。"""
        current, prev, _switch_time_s, _next_event_idx = self._initial_script_state()
        self.contract.apply(self.env.unwrapped, current)
        _set_command_mode(self.env.unwrapped, current, prev, 1.0)
        _set_action_mode(self.env.unwrapped, current)
        _set_termination_mode(self.env.unwrapped, current)

    def post_env_step(self, dones: torch.Tensor) -> None:
        """auto-reset 后把当前脚本 mode 重新写回刚 reset 的 env。"""
        if dones.any():
            self.reset()

    def _initial_script_state(self) -> tuple[TaskMode, TaskMode, float, int]:
        """计算脚本初始 mode，支持多个 0s 事件时以后出现的为准。"""
        current = self.default_mode
        next_event_idx = 0
        while next_event_idx < len(self.events) and self.events[next_event_idx].time_s <= 0.0:
            current = self.events[next_event_idx].mode
            next_event_idx += 1
        switch_time_s = -self.blend_s if next_event_idx > 0 else 0.0
        return current, current, switch_time_s, next_event_idx

    def _reset_script_clock(self) -> None:
        """把脚本时间线锚定到当前 env step。"""
        self._script_start_step = int(self.env.unwrapped.common_step_counter)

    def _script_time_s(self) -> float:
        """返回从最近一次 play reset 开始的脚本时间。"""
        elapsed_steps = int(self.env.unwrapped.common_step_counter) - self._script_start_step
        return max(elapsed_steps, 0) * float(self.env.unwrapped.step_dt)

    def __call__(self, obs_dict: object) -> torch.Tensor:
        """按 env 时间更新 mode 条件后调用 Flow policy。"""
        t = self._script_time_s()
        self._complete_pending_switch_if_ready(t)
        while (
            self.next_event_idx < len(self.events) and t >= self.events[self.next_event_idx].time_s
        ):
            event = self.events[self.next_event_idx]
            self.next_event_idx += 1
            if self._start_wheel_to_gait_prep(event):
                break
            self._apply_event(event)
            self._complete_pending_switch_if_ready(t)
            if self._pending_mode is not None:
                break
        blend = min(max((t - self.switch_time_s) / self.blend_s, 0.0), 1.0)
        wheel_to_gait_prep = self._pending_mode == TaskMode.GAIT
        if wheel_to_gait_prep:
            blend = 1.0
        self.runtime.set_task_mode(self.current_mode, prev_mode=self.prev_mode, blend=blend)
        self._sync_env_contract(
            self.current_mode,
            self.prev_mode,
            blend,
            wheel_to_gait_prep=wheel_to_gait_prep,
        )
        if wheel_to_gait_prep:
            return _zero_policy_action(obs_dict, self.env.unwrapped)
        return self.runtime(
            _overwrite_script_actor_obs(
                obs_dict,
                self.env.unwrapped,
                self.current_mode,
                self.prev_mode,
                blend,
            )
        )

    def _apply_event(self, event: ModeEvent) -> None:
        """应用一个普通 mode 事件。"""
        self.prev_mode = self.current_mode
        self.current_mode = event.mode
        self.switch_time_s = event.time_s
        self.script_command = _shape_command(
            self.script_command,
            self.current_mode,
        )

    def _start_wheel_to_gait_prep(self, event: ModeEvent) -> bool:
        """WHEEL 切 GAIT 前插入原地抬高准备段。"""
        if self.current_mode != TaskMode.WHEEL or event.mode != TaskMode.GAIT:
            return False
        if _WHEEL_TO_GAIT_PREP_S <= 0.0:
            return False
        self._pending_mode = TaskMode.GAIT
        self._pending_prev_mode = self.current_mode
        self._pending_switch_time_s = event.time_s + _WHEEL_TO_GAIT_PREP_S
        self._pending_script_command = (
            None
            if self.script_command is None
            else _wheel_to_gait_prep_command(self.script_command)
        )
        self.prev_mode = self.current_mode
        self.switch_time_s = event.time_s
        return True

    def _complete_pending_switch_if_ready(self, t: float) -> None:
        """准备段结束后真正切到 pending mode。"""
        if self._pending_mode is None or self._pending_switch_time_s is None:
            return
        if t < self._pending_switch_time_s:
            return
        self.current_mode = self._pending_mode
        self.prev_mode = self.current_mode
        self.switch_time_s = t - self.blend_s
        self.script_command = self._pending_script_command
        self.runtime.reset()
        self.env.unwrapped.action_manager.reset()
        self.env.unwrapped.termination_manager.reset()
        self._clear_pending_switch()

    def _clear_pending_switch(self) -> None:
        """清理未完成的延迟切换状态。"""
        self._pending_mode = None
        self._pending_prev_mode = None
        self._pending_switch_time_s = None
        self._pending_script_command = None

    def _sync_env_contract(
        self,
        current: TaskMode,
        prev: TaskMode,
        blend: float,
        *,
        wheel_to_gait_prep: bool = False,
    ) -> None:
        """同步脚本 mode 到 env 的 command 和动作物理语义。"""
        env = self.env.unwrapped
        if wheel_to_gait_prep:
            self.contract.apply_wheel_to_gait_prep(env)
        else:
            self.contract.apply(env, current)
        _set_command_mode(env, current, prev, blend)
        self.script_command = _set_command_shape(
            env,
            current,
            self.script_command,
            wheel_to_gait_prep=wheel_to_gait_prep,
        )
        action_mode = TaskMode.WHEEL if wheel_to_gait_prep else current
        _set_action_mode(env, action_mode)
        if wheel_to_gait_prep:
            _set_wheel_to_gait_prep_termination_mode(env)
        else:
            _set_termination_mode(env, current)
        if wheel_to_gait_prep:
            env.action_manager.reset()


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
            _reset_env_with_policy(wrapped, policy)
            resolved = _resolve_viewer(viewer)
            if resolved == "native":
                ScriptedNativeMujocoViewer(wrapped, policy).run(num_steps=max_steps)
            elif resolved == "viser":
                ScriptedViserPlayViewer(wrapped, policy).run(num_steps=max_steps)
            else:
                raise ValueError(f"不支持的 viewer：{viewer}")
    finally:
        wrapped.close()


def _run_headless(env: RslRlVecEnvWrapper, policy: ScriptedFlowPolicy, *, max_steps: int) -> None:
    """无 viewer 跑固定步数，用于 smoke。"""
    obs = _reset_env_with_policy(env, policy)
    for _ in range(max_steps):
        with torch.no_grad():
            action = policy(obs)
            obs, _rew, dones, _extras = env.step(action)
            policy.post_env_step(dones)
    print(f"[flow-play] headless steps={max_steps}")


def _reset_env_with_policy(env: RslRlVecEnvWrapper, policy: ScriptedFlowPolicy) -> object:
    """先同步脚本 reset 契约，再重置 env。"""
    policy.reset()
    obs, _ = env.reset()
    policy.reset()
    obs = env.get_observations()
    return obs


class ScriptedNativeMujocoViewer(NativeMujocoViewer):
    """支持脚本 reset 契约的 Native viewer。"""

    def _execute_step(self) -> bool:
        """执行一步 viewer 仿真，并在 auto-reset 后恢复脚本契约。"""
        return _execute_scripted_viewer_step(self)

    def reset_environment(self) -> None:
        """先写入脚本初始 reset 契约，再执行 viewer reset。"""
        self.policy.pre_env_reset()
        super().reset_environment()


class ScriptedViserPlayViewer(ViserPlayViewer):
    """支持脚本 reset 契约的 Viser viewer。"""

    def _execute_step(self) -> bool:
        """执行一步 viewer 仿真，并在 auto-reset 后恢复脚本契约。"""
        return _execute_scripted_viewer_step(self)

    def reset_environment(self) -> None:
        """先写入脚本初始 reset 契约，再执行 viewer reset。"""
        self.policy.pre_env_reset()
        super().reset_environment()


def _execute_scripted_viewer_step(viewer: object) -> bool:
    """复刻 MJLab viewer step，并补上脚本契约 post-step hook。"""
    try:
        with torch.no_grad():
            obs = viewer.env.get_observations()
            actions = viewer.policy(obs)
            _obs, _rew, dones, _extras = viewer.env.step(actions)
            viewer.policy.post_env_step(dones)
            viewer._step_count += 1
            viewer._stats_steps += 1
            return True
    except Exception:
        viewer._last_error = traceback.format_exc()
        viewer.log(
            f"[ERROR] Exception during step:\n{viewer._last_error}",
            VerbosityLevel.SILENT,
        )
        viewer.pause()
        return False


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


def _set_command_shape(
    env: ManagerBasedRlEnv,
    mode: TaskMode,
    script_command: torch.Tensor | None,
    *,
    wheel_to_gait_prep: bool = False,
) -> torch.Tensor:
    """按当前 mode 限制 command 范围，并修正已经采样的 command。"""
    term = env.command_manager.get_term(_COMMAND_NAME)
    cfg = term.cfg
    _hold_script_command(term)
    shape = _command_shape_for_mode(mode, wheel_to_gait_prep=wheel_to_gait_prep)
    for name, value in shape.items():
        setattr(cfg, name, value)
    command = term.command
    if command.shape[1] < 5:
        raise ValueError(f"{_COMMAND_NAME} command 必须至少 5D，实际为 {tuple(command.shape)}")
    if script_command is None or script_command.shape != command[:, :5].shape:
        script_command = command[:, :5].clone()
    script_command = _shape_command(script_command, mode)
    command_source = script_command
    if wheel_to_gait_prep:
        command_source = _wheel_to_gait_prep_command(script_command)
    command[:, :5] = command_source.to(device=command.device, dtype=command.dtype)
    _apply_deadband(command, shape)
    if not wheel_to_gait_prep:
        script_command = command[:, :5].clone()
    height = _GAIT_HEIGHT if wheel_to_gait_prep else _mode_height(mode)
    command[:, 4] = height
    if not wheel_to_gait_prep:
        script_command[:, 4] = height
    if mode == TaskMode.GAIT or wheel_to_gait_prep:
        standing_mask = getattr(term, "_standing_mask", None)
        if isinstance(standing_mask, torch.Tensor):
            standing_mask[:] = False
    return script_command


def _shape_command(command: torch.Tensor | None, mode: TaskMode) -> torch.Tensor | None:
    """把脚本保存的 raw command 修整到当前 mode 合法范围。"""
    if command is None:
        return None
    out = command.clone()
    shape = _command_shape_for_mode(mode)
    out[:, 0] = out[:, 0].clamp(*shape["lin_vel_x_range"])
    out[:, 1] = out[:, 1].clamp(*shape["ang_vel_yaw_range"])
    out[:, 2] = out[:, 2].clamp(*shape["pitch_range"])
    out[:, 3] = out[:, 3].clamp(*shape["roll_range"])
    out[:, 4] = out[:, 4].clamp(*shape["height_range"])
    out[:, 4] = _mode_height(mode)
    return out


def _command_shape_for_mode(
    mode: TaskMode,
    *,
    wheel_to_gait_prep: bool = False,
) -> dict[str, object]:
    """返回当前命令约束。"""
    if wheel_to_gait_prep:
        return _WHEEL_TO_GAIT_PREP_COMMAND
    return _GAIT_COMMAND if mode == TaskMode.GAIT else _WHEEL_COMMAND


def _wheel_to_gait_prep_command(command: torch.Tensor) -> torch.Tensor:
    """构造 WHEEL->GAIT 准备段的原地抬高命令。"""
    out = torch.zeros_like(command)
    out[:, 4] = _GAIT_HEIGHT
    return out


def _mode_height(mode: TaskMode) -> float:
    """返回当前 mode 对应的固定机身高度 command。"""
    return _GAIT_HEIGHT if mode == TaskMode.GAIT else _WHEEL_HEIGHT


def _hold_script_command(term: object) -> None:
    """脚本 play 自己控制 mode 时间线，禁止 command term 中途重采样。"""
    cfg = getattr(term, "cfg", None)
    if cfg is not None and hasattr(cfg, "resampling_time_range"):
        cfg.resampling_time_range = (_SCRIPT_COMMAND_HOLD_S, _SCRIPT_COMMAND_HOLD_S)
    time_left = getattr(term, "time_left", None)
    if isinstance(time_left, torch.Tensor):
        time_left[:] = _SCRIPT_COMMAND_HOLD_S


def _overwrite_script_actor_obs(
    obs_dict: object,
    env: ManagerBasedRlEnv,
    current: TaskMode,
    prev: TaskMode,
    blend: float,
) -> object:
    """覆盖本步 policy 输入，避免 viewer 取 obs 慢一拍或沿用旧契约。"""
    if not isinstance(obs_dict, dict) or "actor" not in obs_dict:
        return obs_dict
    actor = obs_dict["actor"]
    if not isinstance(actor, torch.Tensor) or actor.ndim != 2 or actor.shape[1] != 42:
        return obs_dict
    out_actor = actor.clone()
    robot = env.scene["robot"]
    out_actor[:, 0:3] = _to_actor_device(
        robot.data.root_link_ang_vel_b * float(_OBS_CFG.ang_vel_scale),
        out_actor,
    )
    out_actor[:, 3:6] = _to_actor_device(robot.data.projected_gravity_b, out_actor)
    command = env.command_manager.get_command(_COMMAND_NAME)
    scale = torch.tensor(
        _OBS_CFG.command_scale,
        device=out_actor.device,
        dtype=out_actor.dtype,
    )
    out_actor[:, 6:11] = _to_actor_device(command[:, :5], out_actor) * scale
    out_actor[:, 11:15] = _to_actor_device(
        robot.data.joint_pos[:, JointGroup.LEGS] - robot.data.default_joint_pos[:, JointGroup.LEGS],
        out_actor,
    )
    out_actor[:, 15:19] = _to_actor_device(
        robot.data.joint_vel[:, JointGroup.LEGS] * float(_OBS_CFG.leg_vel_scale),
        out_actor,
    )
    out_actor[:, 19:21] = _to_actor_device(robot.data.joint_pos[:, JointGroup.WHEELS], out_actor)
    out_actor[:, 21:23] = _to_actor_device(
        robot.data.joint_vel[:, JointGroup.WHEELS] * float(_OBS_CFG.wheel_vel_scale),
        out_actor,
    )
    out_actor[:, 23:29] = _to_actor_device(env.action_manager.action, out_actor)
    out_actor = overwrite_task_mode_obs(out_actor, current, prev=prev, blend=blend)
    out = dict(obs_dict)
    out["actor"] = out_actor
    return out


def _zero_policy_action(obs_dict: object, env: ManagerBasedRlEnv) -> torch.Tensor:
    """构造准备段零动作：腿跟随当前 default_joint_pos，轮子速度目标为 0。"""
    if isinstance(obs_dict, dict):
        actor = obs_dict.get("actor")
        if isinstance(actor, torch.Tensor) and actor.ndim == 2:
            return torch.zeros(actor.shape[0], 6, device=actor.device, dtype=actor.dtype)
    return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)


def _to_actor_device(value: torch.Tensor, actor: torch.Tensor) -> torch.Tensor:
    """把运行期状态张量搬到 actor obs 的 device 和 dtype。"""
    return value.to(device=actor.device, dtype=actor.dtype)


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


def _set_termination_mode(env: ManagerBasedRlEnv, mode: TaskMode) -> None:
    """按 mode 切换 reset 触发阈值，保证脚本 GAIT 使用正式 GAIT 契约。"""
    bad_orientation = _termination_params(env, "bad_orientation")
    if bad_orientation is not None:
        bad_orientation["limit_angle"] = 0.5236
        bad_orientation["max_steps"] = 8 if mode == TaskMode.GAIT else 100
    low_base_height = _termination_params(env, "low_base_height")
    if low_base_height is not None:
        low_base_height["sensor_name"] = "base_height_sensor"
        low_base_height["min_height"] = 0.26 if mode == TaskMode.GAIT else 0.12
        low_base_height["max_steps"] = 25


def _set_wheel_to_gait_prep_termination_mode(env: ManagerBasedRlEnv) -> None:
    """准备段给 base_link 接触留足 grace，避免到位前被 reset 打断。"""
    _set_termination_mode(env, TaskMode.WHEEL)
    base_link_contact = _termination_params(env, "base_link_contact")
    if base_link_contact is not None:
        prep_steps = math.ceil(_WHEEL_TO_GAIT_PREP_S / float(env.step_dt))
        base_link_contact["delay_steps"] = max(prep_steps + 50, 50)


def _ensure_model_fields(env: ManagerBasedRlEnv, fields: tuple[str, ...]) -> None:
    """确保模型字段已按 env 维度展开，可在运行期写入。"""
    missing = tuple(field for field in fields if field not in env.sim.expanded_fields)
    if missing:
        env.sim.expand_model_fields(missing)


def _find_mj_joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    """按 entity 前缀后的关节名查找 MuJoCo joint id。"""
    for joint_id in range(model.njnt):
        full_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if full_name == joint_name or (
            full_name is not None and full_name.endswith(f"/{joint_name}")
        ):
            return int(joint_id)
    raise ValueError(f"找不到 MuJoCo joint：{joint_name}")


def _zero_wheel_joint_state(env: ManagerBasedRlEnv) -> None:
    """GAIT 锁轮时把轮轴位置和速度直接清零。"""
    robot = env.scene["robot"]
    zeros = torch.zeros(
        robot.data.joint_pos.shape[0],
        len(JointGroup.WHEELS),
        device=robot.data.joint_pos.device,
        dtype=robot.data.joint_pos.dtype,
    )
    robot.write_joint_state_to_sim(
        zeros,
        zeros,
        joint_ids=torch.tensor(JointGroup.WHEELS, device=zeros.device, dtype=torch.long),
    )


def _read_reset_base_height(env: ManagerBasedRlEnv) -> float | None:
    """读取当前 reset_root_state 的 base_height。"""
    params = _reset_event_params(env, "reset_root_state")
    value = params.get("base_height")
    return None if value is None else float(value)


def _set_reset_base_height(env: ManagerBasedRlEnv, base_height: float | None) -> None:
    """同步 reset_root_state 的 base_height 到 cfg 和运行期 EventManager。"""
    for params in _all_reset_event_params(env, "reset_root_state"):
        if base_height is None:
            params.pop("base_height", None)
        else:
            params["base_height"] = float(base_height)


def _read_reset_joint_params(env: ManagerBasedRlEnv) -> dict[str, object]:
    """读取当前 reset_joints 的姿态覆盖参数。"""
    params = _reset_event_params(env, "reset_joints")
    return {
        "joint_pos_override": params.get("joint_pos_override"),
        "update_default_joint_pos": bool(params.get("update_default_joint_pos", False)),
    }


def _set_reset_joint_params(
    env: ManagerBasedRlEnv,
    *,
    joint_pos_override: object,
    update_default_joint_pos: bool,
) -> None:
    """同步 reset_joints 的姿态覆盖参数到 cfg 和运行期 EventManager。"""
    for params in _all_reset_event_params(env, "reset_joints"):
        if joint_pos_override is None:
            params.pop("joint_pos_override", None)
        else:
            params["joint_pos_override"] = tuple(float(value) for value in joint_pos_override)
        params["update_default_joint_pos"] = bool(update_default_joint_pos)


def _reset_event_params(env: ManagerBasedRlEnv, name: str) -> dict[str, object]:
    """返回运行期 reset event params。"""
    return env.event_manager.get_term_cfg(name).params


def _all_reset_event_params(env: ManagerBasedRlEnv, name: str) -> tuple[dict[str, object], ...]:
    """返回源 cfg 和运行期 EventManager 的同名 reset event params。"""
    params: list[dict[str, object]] = []
    cfg_term = env.cfg.events.get(name)
    if cfg_term is not None:
        params.append(cfg_term.params)
    params.append(env.event_manager.get_term_cfg(name).params)
    return tuple(params)


def _termination_params(env: ManagerBasedRlEnv, name: str) -> dict[str, object] | None:
    """返回运行期 termination 参数，不存在时跳过。"""
    try:
        return env.termination_manager.get_term_cfg(name).params
    except ValueError:
        return None


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
