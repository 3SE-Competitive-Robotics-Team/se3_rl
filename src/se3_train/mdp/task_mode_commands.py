"""统一 Task Mode 指令生成器。

在基础速度/姿态/高度指令上追加跳跃和 Task Mode 信息：
  [vx, wz, pitch, roll, h, jump_flag, jump_target_height, jump_phase,
   task_mode_id, mode_blend, prev_task_mode_id]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from se3_shared import TASK_MODE_COUNT, TASK_MODE_NAMES, TaskMode
from se3_train.mdp.commands import BasicCommandCfg, BasicCommandTerm
from se3_train.mdp.jump_trajectories import (
    DEFAULT_JUMP_TRAJ_HEIGHTS,
    DEFAULT_JUMP_TRAJ_PATHS,
    JumpTrajLibrary,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_TASK_COMMAND_DIM = 11
_VZ_SUCCESS_THRESHOLD = 1.0

# terrain_types 列索引对应 sub_terrains 的 insertion order:
# 0=flat, 1=random_rough, 2=wave, 3=open_stairs, 4=random_spread_boxes, 5=discrete_obstacles
# 行对应 TaskMode: WHEEL=0, GAIT=1, WHEEL_LEG=2, GAIT_WHEEL=3, JUMP=4
MODE_TERRAIN_MASK = torch.tensor(
    [
        # flat  rough  wave  stairs  boxes  obstacles
        [1, 1, 1, 0, 0, 0],  # WHEEL
        [1, 1, 1, 1, 1, 1],  # GAIT
        [1, 1, 1, 1, 1, 1],  # WHEEL_LEG
        [1, 1, 1, 1, 1, 1],  # GAIT_WHEEL
        [1, 0, 0, 0, 0, 0],  # JUMP
    ],
    dtype=torch.float,
)


@dataclass
class TaskModeCommandCfg(BasicCommandCfg):
    """Task Mode 指令配置。"""

    jump_prob: float = 0.0
    """每个 policy step 启动一次跳跃窗口的概率（由课程或 mode 采样控制）。"""

    jump_cool_down_steps: int = 100
    """一次跳跃结束后的冷却步数，冷却期内不再启动新跳跃。"""

    jump_height_range: tuple[float, float] = (0.1, 0.6)
    """目标跳跃高度采样范围（m）。"""

    rsi_takeoff_prob: float = 1.0
    """jump_flag=1 的 episode 中使用参考轨迹起点初始化的概率。"""

    rsi_random_frame: bool = False
    """RSI 是否从参考轨迹随机帧初始化。"""

    rsi_frame_phase_range: tuple[float, float] = (0.0, 1.0)
    """随机 RSI 帧的相位范围，0=轨迹起点，1=轨迹末尾。"""

    traj_paths: tuple[str, ...] = DEFAULT_JUMP_TRAJ_PATHS
    """跳跃参考轨迹文件列表。"""

    traj_target_heights: tuple[float, ...] = DEFAULT_JUMP_TRAJ_HEIGHTS
    """参考轨迹对应的目标高度列表。"""

    mode_probabilities: tuple[float, ...] = (0.35, 0.15, 0.2, 0.15, 0.15)
    """5 个模式的采样概率:wheel/gait/wheel_leg/gait_wheel/jump。"""

    mode_switch_time_range: tuple[float, float] = (2.0, 5.0)
    """episode 内模式切换间隔（秒）。"""

    mode_blend_time_range: tuple[float, float] = (0.5, 1.0)
    """模式切换后的平滑时间（秒）。"""

    enable_mode_switch: bool = True
    """是否允许 episode 内切换模式。"""

    def build(self, env: ManagerBasedRlEnv) -> TaskModeCommandTerm:
        return TaskModeCommandTerm(self, env)


class TaskModeCommandTerm(BasicCommandTerm):
    """统一 Task Mode 指令项。"""

    cfg: TaskModeCommandCfg

    def __init__(self, cfg: TaskModeCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        old_command = self._command
        self._command = torch.zeros(self.num_envs, _TASK_COMMAND_DIM, device=self.device)
        self._command[:, : old_command.shape[1]] = old_command

        self._traj_library = JumpTrajLibrary.get(
            cfg.traj_paths, cfg.traj_target_heights, str(self.device)
        )
        self._jump_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._complete_jump_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._active_takeoff_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._rsi_takeoff_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._active_success_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._takeoff_recorded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._traj_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._jump_cool_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pre_resampled_for_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._resampling_for_reset = False

        self._mode = torch.full(
            (self.num_envs,), int(TaskMode.WHEEL), dtype=torch.long, device=self.device
        )
        self._prev_mode = self._mode.clone()
        self._mode_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_switch_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_blend_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._sync_mode_columns(torch.arange(self.num_envs, device=self.device))

    @property
    def jump_stage(self) -> torch.Tensor:
        """参考轨迹阶段张量 [num_envs]，值为 0/1/2。"""
        return self._jump_stage

    @property
    def traj_step(self) -> torch.Tensor:
        """轨迹计步器 [num_envs]，每步递增，用于索引参考轨迹。"""
        return self._traj_step

    def reference_root_velocity(self) -> torch.Tensor:
        """读取当前参考帧的 base 线速度。"""
        _, ref_vel, _, _, _, _ = self._traj_library.gather(self._command[:, 6], self._traj_step)
        return ref_vel

    def reference_joint_velocity(self) -> torch.Tensor:
        """读取当前参考帧的关节角速度。"""
        _, _, _, ref_q_vel, _, _ = self._traj_library.gather(self._command[:, 6], self._traj_step)
        return ref_q_vel

    def reference_takeoff_started(self, min_vz: float = 0.02) -> torch.Tensor:
        """参考起跳段是否已经开始。"""
        first_takeoff_step = self._traj_library.first_vz_above_step_for(self._command[:, 6], min_vz)
        return self._traj_step >= first_takeoff_step

    def reference_preload_active(self, max_vz: float = 0.05) -> torch.Tensor:
        """参考下降蓄力子相位。"""
        ref_vz = self.reference_root_velocity()[:, 2]
        return (self._jump_stage == 0) & (ref_vz <= max_vz)

    def reference_takeoff_active(
        self, min_vz: float = 0.05, min_window_steps: int = 16
    ) -> torch.Tensor:
        """参考起跳子相位：stance 末端的主动伸腿窗口。"""
        start_step = self._traj_library.takeoff_window_start_step_for(
            self._command[:, 6], min_vz, min_window_steps
        )
        return (self._jump_stage == 0) & (self._traj_step >= start_step)

    def set_reference_frame(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        """同步 RSI 使用的参考帧，保证观测相位和 reset 状态一致。"""
        if len(env_ids) == 0:
            return
        h_target = self._command[env_ids, 6]
        max_step = self._traj_library.n_steps_for(h_target) - 1
        frames = torch.minimum(frames, max_step)
        self._traj_step[env_ids] = frames
        self._jump_stage[env_ids] = self._traj_library.stage_for(h_target, frames)
        self._command[env_ids, 7] = frames.float() / torch.clamp(max_step.float(), min=1.0)

    def pre_resample_for_reset(self, env_ids: torch.Tensor) -> None:
        """在 reset 事件阶段预采样指令，供 TaskMode 内的 jump RSI 初始化读取。"""
        self._resampling_for_reset = True
        try:
            self._resample(env_ids)
            self._pre_resampled_for_reset[env_ids] = True
        finally:
            self._resampling_for_reset = False

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """重置指令项，并保留 reset 事件阶段已预采样的 TaskMode 指令。"""
        assert isinstance(env_ids, torch.Tensor)
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        pre_mask = self._pre_resampled_for_reset[env_ids]
        pre_ids = env_ids[pre_mask]
        fresh_ids = env_ids[~pre_mask]

        self.command_counter[env_ids] = 0
        self._resampling_for_reset = True
        try:
            if len(fresh_ids) > 0:
                self._resample(fresh_ids)
            if len(pre_ids) > 0:
                self.command_counter[pre_ids] = 1
                self._pre_resampled_for_reset[pre_ids] = False
        finally:
            self._resampling_for_reset = False

        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """重采样速度/姿态/高度，并按当前 mode 同步跳跃窗口。"""
        super()._resample_command(env_ids)
        if self._resampling_for_reset:
            self._sample_modes(env_ids)
            self._sample_mode_timers(env_ids)
            jump_ids = env_ids[self._mode[env_ids] == int(TaskMode.JUMP)]
            non_jump_ids = env_ids[self._mode[env_ids] != int(TaskMode.JUMP)]
            self._clear_non_jump_state(non_jump_ids)
            if len(jump_ids) > 0:
                self._start_jump(jump_ids)
            self._sync_mode_columns(env_ids)
            return

        non_jump_ids = env_ids[self._mode[env_ids] != int(TaskMode.JUMP)]
        self._clear_non_jump_state(non_jump_ids)
        jump_ids = env_ids[self._mode[env_ids] == int(TaskMode.JUMP)]
        inactive_jump_ids = jump_ids[
            (self._command[jump_ids, 5] <= 0.5) & (self._jump_cool_down[jump_ids] == 0)
        ]
        if len(inactive_jump_ids) > 0:
            self._start_jump(inactive_jump_ids)
        inactive = env_ids[self._command[env_ids, 5] <= 0.5]
        self._sample_jump_target_height(inactive)
        self._apply_mode_command_shape(env_ids)
        self._sync_mode_columns(env_ids)

    def _start_jump(self, env_ids: torch.Tensor) -> None:
        """启动跳跃窗口，并把 mode 同步成 jump。"""
        if len(env_ids) == 0:
            return
        self._sample_jump_target_height(env_ids)
        self._command[env_ids, 5] = 1.0
        self._command[env_ids, 7] = 0.0
        self._zero_locomotion_command(env_ids)
        self._jump_stage[env_ids] = 0
        self._traj_step[env_ids] = 0
        self._complete_jump_count[env_ids] = 0
        self._active_takeoff_count[env_ids] = 0
        self._rsi_takeoff_count[env_ids] = 0
        self._active_success_count[env_ids] = 0
        self._takeoff_recorded[env_ids] = False
        self._prev_mode[env_ids] = self._mode[env_ids]
        self._mode[env_ids] = int(TaskMode.JUMP)
        self._mode_elapsed_steps[env_ids] = 0
        self._sync_mode_columns(env_ids)

    def _clear_jump_command(self, env_ids: torch.Tensor) -> None:
        """跳跃结束后清理跳跃窗口，固定 JUMP 任务保留 JUMP 标签。"""
        if env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero().flatten()
        if len(env_ids) == 0:
            return
        self._command[env_ids, 5] = 0.0
        self._command[env_ids, 7] = 0.0
        self._jump_stage[env_ids] = 0
        self._traj_step[env_ids] = 0
        self._takeoff_recorded[env_ids] = False
        self._jump_cool_down[env_ids] = int(self.cfg.jump_cool_down_steps)
        self._prev_mode[env_ids] = int(TaskMode.JUMP)
        fixed_mode = self._configured_single_mode()
        self._mode[env_ids] = int(fixed_mode if fixed_mode is not None else TaskMode.WHEEL)
        self._mode_elapsed_steps[env_ids] = 0
        self._sample_mode_timers(env_ids)
        self._sync_mode_columns(env_ids)

    def _start_new_jumps(self) -> None:
        """TaskMode 由统一切换器决定 jump，避免额外 jump_prob 抢模式。"""
        self._jump_cool_down = torch.clamp(self._jump_cool_down - 1, min=0)

    def _update_command(self) -> None:
        """更新速度死区、模式切换和跳跃参考相位。"""
        super()._update_command()
        self._update_mode_switch()
        self._advance_traj_step()
        self._sync_mode_columns(torch.arange(self.num_envs, device=self.device))
        self._update_task_mode_metrics()

    def _sample_jump_target_height(self, env_ids: torch.Tensor) -> None:
        """为指定 env 采样目标跳跃高度。"""
        if len(env_ids) == 0:
            return
        lo, hi = self.cfg.jump_height_range
        self._command[env_ids, 6] = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo

    def _zero_locomotion_command(self, env_ids: torch.Tensor) -> None:
        """跳跃窗口固定为原地直立指令，避免和行走/yaw 指令冲突。"""
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0:4] = 0.0
        self._command[env_ids, 4] = (self.cfg.height_range[0] + self.cfg.height_range[1]) / 2.0

    def _advance_traj_step(self) -> None:
        """按参考轨迹时间推进 TaskMode 内部跳跃计步器。"""
        jump_envs = self._command[:, 5] > 0.5
        self._traj_step[~jump_envs] = 0
        self._jump_stage[~jump_envs] = 0

        if jump_envs.any():
            h_target = self._command[jump_envs, 6]
            max_step = self._traj_library.n_steps_for(h_target) - 1
            self._traj_step[jump_envs] = torch.minimum(self._traj_step[jump_envs] + 1, max_step)
            self._jump_stage[jump_envs] = self._traj_library.stage_for(
                h_target, self._traj_step[jump_envs]
            )
            self._update_reference_takeoff_metrics(jump_envs)
            motion_done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            motion_done[jump_envs] = self._traj_step[jump_envs] >= max_step
            self._complete_jump_count[motion_done] += 1
            self._clear_jump_command(jump_envs & motion_done)

        phase = torch.zeros(self.num_envs, device=self.device)
        if jump_envs.any():
            h_target = self._command[jump_envs, 6]
            max_step = self._traj_library.n_steps_for(h_target) - 1
            phase[jump_envs] = self._traj_step[jump_envs].float() / torch.clamp(
                max_step.float(), min=1.0
            )
        self._command[:, 7] = phase.clamp(0.0, 1.0)

    def _update_reference_takeoff_metrics(self, jump_envs: torch.Tensor) -> None:
        """按参考空中阶段记录 TaskMode 内的起跳诊断计数。"""
        robot = self._env.scene["robot"]
        vz_w = robot.data.root_link_lin_vel_w[:, 2]
        in_ref_air = jump_envs & (self._jump_stage == 1)
        newly_recorded = in_ref_air & ~self._takeoff_recorded

        rsi_takeoff = newly_recorded & (self._env.episode_length_buf <= 5)
        active_takeoff = newly_recorded & (self._env.episode_length_buf > 5) & (vz_w > 0.0)
        self._rsi_takeoff_count[rsi_takeoff] += 1
        self._active_takeoff_count[active_takeoff] += 1

        active_success = active_takeoff & (vz_w > _VZ_SUCCESS_THRESHOLD)
        self._active_success_count[active_success] += 1
        self._takeoff_recorded[newly_recorded] = True

    def _sample_modes(self, env_ids: torch.Tensor) -> None:
        """按配置概率采样新 mode，并根据 env 所在地形 mask 不兼容模式。

        jump_prob <= 0 时将 JUMP 模式概率强制清零，确保两条路径（step-level
        _start_new_jumps 和 episode-level _sample_modes）都受同一个配置项控制。
        """
        if len(env_ids) == 0:
            return
        probs = torch.tensor(self.cfg.mode_probabilities, device=self.device, dtype=torch.float)
        if probs.numel() != TASK_MODE_COUNT:
            raise ValueError(
                f"mode_probabilities 需要 {TASK_MODE_COUNT} 项，实际得到 {probs.numel()} 项。"
            )
        probs = torch.clamp(probs, min=0.0)
        probs[int(TaskMode.JUMP)] = max(float(self.cfg.jump_prob), 0.0)
        probs = probs / torch.clamp(probs.sum(), min=1.0e-6)

        terrain = self._env.scene.terrain
        if terrain is not None and hasattr(terrain, "terrain_types"):
            terrain_type_ids = terrain.terrain_types[env_ids]
            mask = MODE_TERRAIN_MASK.to(self.device)
            num_terrain_cols = mask.shape[1]
            safe_ids = torch.clamp(terrain_type_ids, max=num_terrain_cols - 1)
            per_env_mask = mask[:, safe_ids].T  # (N, num_modes)
            per_env_probs = probs.unsqueeze(0) * per_env_mask  # (N, num_modes)
            per_env_probs = per_env_probs / per_env_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            sampled = torch.multinomial(per_env_probs, 1).squeeze(1)
        else:
            sampled = torch.multinomial(probs, len(env_ids), replacement=True)

        self._prev_mode[env_ids] = self._mode[env_ids]
        self._mode[env_ids] = sampled.long()
        self._mode_elapsed_steps[env_ids] = 0

    def _clear_non_jump_state(self, env_ids: torch.Tensor) -> None:
        """清理非 jump mode 的跳跃窗口状态。"""
        if len(env_ids) == 0:
            return
        self._command[env_ids, 5] = 0.0
        self._command[env_ids, 7] = 0.0
        self._jump_stage[env_ids] = 0
        self._traj_step[env_ids] = 0
        self._jump_cool_down[env_ids] = 0
        self._complete_jump_count[env_ids] = 0
        self._active_takeoff_count[env_ids] = 0
        self._rsi_takeoff_count[env_ids] = 0
        self._active_success_count[env_ids] = 0
        self._takeoff_recorded[env_ids] = False
        self._sample_jump_target_height(env_ids)

    def _sample_mode_timers(self, env_ids: torch.Tensor) -> None:
        """采样 mode 切换和平滑步数。"""
        if len(env_ids) == 0:
            return
        dt = self._env.step_dt
        switch_lo, switch_hi = self.cfg.mode_switch_time_range
        blend_lo, blend_hi = self.cfg.mode_blend_time_range
        switch_s = (
            torch.rand(len(env_ids), device=self.device) * (switch_hi - switch_lo) + switch_lo
        )
        blend_s = torch.rand(len(env_ids), device=self.device) * (blend_hi - blend_lo) + blend_lo
        self._mode_switch_steps[env_ids] = torch.clamp((switch_s / dt).round().long(), min=1)
        self._mode_blend_steps[env_ids] = torch.clamp((blend_s / dt).round().long(), min=1)

    def _configured_single_mode(self) -> TaskMode | None:
        """读取固定单标签配置，用于保持纯任务的 mode 不漂移。"""
        if self.cfg.enable_mode_switch:
            return None
        probs = [max(float(value), 0.0) for value in self.cfg.mode_probabilities]
        if len(probs) != TASK_MODE_COUNT:
            return None
        probs[int(TaskMode.JUMP)] = max(float(self.cfg.jump_prob), 0.0)
        active_modes = [idx for idx, prob in enumerate(probs) if prob > 0.0]
        if len(active_modes) != 1:
            return None
        return TaskMode(active_modes[0])

    def _update_mode_switch(self) -> None:
        """episode 内按间隔切换 mode。"""
        self._mode_elapsed_steps += 1
        if not self.cfg.enable_mode_switch:
            return
        can_switch = self._mode_elapsed_steps >= self._mode_switch_steps
        can_switch &= self._command[:, 5] <= 0.5
        can_switch &= self._jump_cool_down == 0
        if not can_switch.any():
            return
        env_ids = can_switch.nonzero().flatten()
        self._sample_modes(env_ids)
        self._sample_mode_timers(env_ids)
        self._apply_mode_command_shape(env_ids)
        jump_ids = env_ids[self._mode[env_ids] == int(TaskMode.JUMP)]
        if len(jump_ids) > 0:
            self._start_jump(jump_ids)

    def _apply_mode_command_shape(self, env_ids: torch.Tensor) -> None:
        """按 mode 调整基础速度/姿态指令，避免目标互相冲突。"""
        if len(env_ids) == 0:
            return
        jump_ids = env_ids[self._mode[env_ids] == int(TaskMode.JUMP)]
        if len(jump_ids) > 0:
            self._zero_locomotion_command(jump_ids)

        gait_ids = env_ids[self._mode[env_ids] == int(TaskMode.GAIT)]
        if len(gait_ids) > 0:
            # 纯步态先限制速度，避免初期被高速轮式目标带偏。
            self._command[gait_ids, 0] = torch.clamp(self._command[gait_ids, 0], -1.0, 1.0)
            self._command[gait_ids, 1] = torch.clamp(self._command[gait_ids, 1], -1.5, 1.5)

        wheel_leg_ids = env_ids[self._mode[env_ids] == int(TaskMode.WHEEL_LEG)]
        if len(wheel_leg_ids) > 0:
            self._command[wheel_leg_ids, 0] = torch.clamp(
                self._command[wheel_leg_ids, 0], -1.5, 1.5
            )

        gait_wheel_ids = env_ids[self._mode[env_ids] == int(TaskMode.GAIT_WHEEL)]
        if len(gait_wheel_ids) > 0:
            self._command[gait_wheel_ids, 0] = torch.clamp(
                self._command[gait_wheel_ids, 0], -1.2, 1.2
            )

    def _sync_mode_columns(self, env_ids: torch.Tensor) -> None:
        """把内部 mode 状态写入 command 追加列。"""
        if len(env_ids) == 0:
            return
        self._command[env_ids, 8] = self._mode[env_ids].float()
        self._command[env_ids, 10] = self._prev_mode[env_ids].float()
        blend_steps = torch.clamp(self._mode_blend_steps[env_ids].float(), min=1.0)
        blend = torch.clamp(self._mode_elapsed_steps[env_ids].float() / blend_steps, 0.0, 1.0)
        self._command[env_ids, 9] = blend

    def _update_task_mode_metrics(self) -> None:
        """上报 Task Mode 分布和切换平滑指标。"""
        if not hasattr(self._env, "extras") or not isinstance(self._env.extras.get("log"), dict):
            return
        log = self._env.extras["log"]
        for mode in TaskMode:
            log[f"TaskMode/{TASK_MODE_NAMES[int(mode)]}_ratio"] = (
                (self._mode == int(mode)).float().mean().item()
            )
        log["TaskMode/mode_blend_mean"] = self._command[:, 9].mean().item()


__all__ = ["TaskModeCommandCfg", "TaskModeCommandTerm"]
