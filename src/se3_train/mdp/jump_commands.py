"""跳跃指令生成器。

在速度/姿态/高度指令基础上扩展 3 维跳跃信息：
  [vx, ωz, pitch, roll, h, jump_flag, jump_target_height, jump_phase]

jump_flag:          0/1，当前是否处于一次跳跃窗口
jump_target_height: 目标跳跃高度（m），0.1~0.6
jump_phase:        0→1 连续相位，按参考 motion 时间推进
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import recovery_state
from se3_train.mdp.commands import VelocityHeightCommandCfg, VelocityHeightCommandTerm
from se3_train.mdp.joint_indices import (
    active_leg_mirror_diffs,
    policy_leg_joint_ids,
    wheel_joint_ids,
)
from se3_train.mdp.jump_trajectories import (
    DEFAULT_JUMP_TRAJ_HEIGHTS,
    DEFAULT_JUMP_TRAJ_PATHS,
    JumpTrajLibrary,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 重力加速度（仿真用标准值）
_G = 9.81
_DEFAULT_STANDING_HEIGHT = SharedRobotConfig().default_base_height

# 成功起跳阈值：vz > 1.0 m/s，对应约 5cm 以上跳跃。
_VZ_SUCCESS_THRESHOLD = 1.0
_TAKEOFF_DIAG_EMA_ATTR = "_jump_takeoff_diag_ema"


def ideal_takeoff_vel(target_height: torch.Tensor) -> torch.Tensor:
    """由目标跳跃高度反算理想起跳 vz（纯抛体近似）。

    vz_ref = sqrt(2 * g * h)
    """
    return torch.sqrt(2.0 * _G * torch.clamp(target_height, min=0.01))


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """计算掩码内均值；无样本时返回 0，避免日志出现 NaN。"""
    if mask.any():
        return float(value[mask].mean().item())
    return 0.0


def _ema_value(previous: float | None, value: float, alpha: float = 0.05) -> float:
    """更新诊断 EMA；第一次命中窗口时直接采用当前值。"""
    if previous is None:
        return value
    return (1.0 - alpha) * previous + alpha * value


@dataclass
class JumpCommandCfg(VelocityHeightCommandCfg):
    """跳跃指令配置，继承速度/姿态/高度指令并扩展跳跃维度。"""

    jump_prob: float = 0.0
    """每个 policy step 启动一次跳跃窗口的概率 (由课程调度器动态修改)。"""

    jump_cool_down_steps: int = 100
    """一次跳跃结束后的冷却步数，冷却期内不再启动新跳跃。"""

    jump_height_range: tuple[float, float] = (0.1, 0.6)
    """目标跳跃高度采样范围 (m)。"""

    rsi_takeoff_prob: float = 1.0
    """jump_flag=1 的 episode 中使用参考轨迹起点初始化的概率。

    当前默认所有跳跃 episode 都从完整跳跃动作第 0 帧开始。
    """

    rsi_random_frame: bool = False
    """RSI 是否从参考轨迹随机帧初始化。

    False 时使用第 0 帧，适合只想给起跳初始状态的实验；True 时覆盖起跳、
    空中和落地全状态，适合 PreTrain 建立跳跃状态分布。
    """

    rsi_frame_phase_range: tuple[float, float] = (0.0, 1.0)
    """随机 RSI 帧的相位范围，0=轨迹起点，1=轨迹末尾。"""

    traj_paths: tuple[str, ...] = DEFAULT_JUMP_TRAJ_PATHS
    """跳跃参考轨迹文件列表。"""

    traj_target_heights: tuple[float, ...] = DEFAULT_JUMP_TRAJ_HEIGHTS
    """参考轨迹对应的目标高度列表。"""

    def build(self, env: ManagerBasedRlEnv) -> JumpCommandTerm:
        return JumpCommandTerm(self, env)


class JumpCommandTerm(VelocityHeightCommandTerm):
    """跳跃指令项，在速度/姿态/高度之外维护参考相位。

    指令维度：[vx, ωz, pitch, roll, h, jump_flag, jump_target_height, jump_phase]（8 维）

    jump_stage 来自参考轨迹帧，是跳跃奖励和窗口生命周期的唯一阶段来源。
    """

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)

        # 扩展指令张量到 8 维：[vx, ωz, pitch, roll, h, jump_flag, jump_target_height, jump_phase]
        # 第 7 维 jump_phase：0→1 连续相位，按参考 motion 时间推进
        self._command = torch.zeros(self.num_envs, 8, device=self.device)

        self._traj_library = JumpTrajLibrary.get(
            cfg.traj_paths, cfg.traj_target_heights, str(self.device)
        )

        # 参考轨迹阶段：0=stance/prep, 1=flight, 2=landing/end
        self._jump_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 完整跳跃计数：参考轨迹走到末帧即完成一次跳跃窗口。
        self._complete_jump_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 起跳诊断计数：用参考空中阶段 + 实际 vz 记录一次有效上升。
        self._active_takeoff_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # RSI 起跳计数：reset 后极短窗口内进入参考空中阶段，用于区分随机 RSI 注入样本。
        self._rsi_takeoff_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._takeoff_recorded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 主动成功起跳计数：主动起跳事件中 vz 超过成功阈值的次数。
        self._active_success_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # 轨迹计步器：跟踪每个 env 当前对应参考轨迹的第几帧
        # - reset 时清零
        # - 每步 +1，但仅在 jump_flag=1 时生效
        # - 上限由轨迹总步数决定（超出后夹住在末帧）
        self._traj_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 跳跃冷却计数：一次参考轨迹走完后进入冷却，避免连续触发导致落地恢复无窗口。
        self._jump_cool_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # MJLab reset 时事件早于 command_manager.reset 执行。跳跃 RSI 需要先知道
        # 新 episode 的 jump_flag，因此 reset 事件会提前采样一次指令，并用此标记
        # 避免 command_manager.reset 随后再次采样导致 RSI 与观测指令不一致。
        self._pre_resampled_for_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._resampling_for_reset = False

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
        """读取当前参考帧的关节速度。"""
        _, _, _, ref_q_vel, _, _ = self._traj_library.gather(self._command[:, 6], self._traj_step)
        return ref_q_vel

    def reference_takeoff_started(self, min_vz: float = 0.02) -> torch.Tensor:
        """参考起跳段是否已经开始。"""
        first_takeoff_step = self._traj_library.first_vz_above_step_for(self._command[:, 6], min_vz)
        return self._traj_step >= first_takeoff_step

    def reference_preload_active(self, max_vz: float = 0.05) -> torch.Tensor:
        """参考下降蓄力子相位：grounded 段内 base vz 尚未进入上升蹬地。"""
        ref_vz = self.reference_root_velocity()[:, 2]
        return (self._jump_stage == 0) & (ref_vz <= max_vz)

    def reference_takeoff_active(
        self, min_vz: float = 0.05, min_window_steps: int = 16
    ) -> torch.Tensor:
        """参考起跳子相位：stance 末端的主动伸腿窗口。

        参考 base vz 转正只出现在 stance 最后 3-6 帧，作为奖励门控会让主动蹬地
        梯度过于稀疏。这里保留 vz 阈值的相位含义，同时保证 stance 末端至少
        覆盖 min_window_steps 帧，让奖励有足够作用时间。
        """
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
        """在 reset 事件阶段预采样指令，供 RSI 初始化读取。"""
        self._resampling_for_reset = True
        try:
            self._resample(env_ids)
            self._pre_resampled_for_reset[env_ids] = True
        finally:
            self._resampling_for_reset = False

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """重置指令项，并保留 reset 事件阶段已预采样的 jump 指令。"""
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

        self._log_bad_orientation_diagnostics(env_ids)
        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """重新采样速度/姿态/高度指令。

        jump_flag 训练中由 step-level 启动逻辑维护；只有 reset 预采样阶段会
        同步抽一次 jump_flag，供 RSI 初始化读取。
        """
        # 先由父类采样 vx/ωz/pitch/roll/h（写入 self._command[:, 0:5]）
        super()._resample_command(env_ids)

        active_jump = self._command[env_ids, 5] > 0.5
        if active_jump.any():
            self._zero_locomotion_command(env_ids[active_jump])

        if self._resampling_for_reset:
            self._reset_jump_lifecycle(env_ids, sample_initial_jump=True)
            return

        # 普通 command 重采样不能中断正在进行的跳跃；只刷新非跳跃 env 的默认目标高度。
        inactive = self._command[env_ids, 5] <= 0.5
        if inactive.any():
            self._sample_jump_target_height(env_ids[inactive])

    def _sample_jump_target_height(self, env_ids: torch.Tensor) -> None:
        """为指定 env 采样目标跳跃高度。"""
        if len(env_ids) == 0:
            return
        lo, hi = self.cfg.jump_height_range
        self._command[env_ids, 6] = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo

    def _zero_locomotion_command(self, env_ids: torch.Tensor) -> None:
        """跳跃窗口固定为原地直立指令，避免原地跳轨迹与行走/yaw 指令冲突。"""
        if len(env_ids) == 0:
            return
        self._command[env_ids, 0:4] = 0.0
        self._command[env_ids, 4] = (self.cfg.height_range[0] + self.cfg.height_range[1]) / 2.0

    def _reset_jump_lifecycle(
        self, env_ids: torch.Tensor, sample_initial_jump: bool = False
    ) -> None:
        """重置跳跃窗口、冷却和诊断计数。"""
        if len(env_ids) == 0:
            return
        self._jump_stage[env_ids] = 0
        self._complete_jump_count[env_ids] = 0
        self._active_takeoff_count[env_ids] = 0
        self._rsi_takeoff_count[env_ids] = 0
        self._active_success_count[env_ids] = 0
        self._takeoff_recorded[env_ids] = False
        self._traj_step[env_ids] = 0
        self._jump_cool_down[env_ids] = 0
        self._command[env_ids, 5] = 0.0
        self._command[env_ids, 7] = 0.0
        self._sample_jump_target_height(env_ids)

        if sample_initial_jump:
            jump_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.jump_prob
            self._start_jump(env_ids[jump_mask])

    def _start_jump(self, env_ids: torch.Tensor) -> None:
        """启动一次跳跃窗口，并重置该窗口内的参考相位和诊断计数。"""
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

    def _start_new_jumps(self) -> None:
        """按 step-level 概率启动新跳跃，对齐 mondo 的 jump command 语义。"""
        self._jump_cool_down = torch.clamp(self._jump_cool_down - 1, min=0)
        can_start = (self._jump_cool_down == 0) & (self._command[:, 5] <= 0.5)
        if not can_start.any() or self.cfg.jump_prob <= 0.0:
            return

        start_mask = can_start & (
            torch.rand(self.num_envs, device=self.device) < self.cfg.jump_prob
        )
        self._start_jump(start_mask.nonzero().flatten())

    def _update_command(self) -> None:
        """更新速度死区，按 step-level 概率启动跳跃，并推进参考轨迹计步器。"""
        super()._update_command()
        self._apply_recovery_command()
        self._start_new_jumps()
        self._advance_traj_step()

    def _apply_recovery_command(self) -> None:
        """恢复 episode 固定为原地站立指令，避免重采样切回行走命令。"""
        recovery_mask = recovery_state.recovery_active_mask(self._env).to(device=self.device)
        if not recovery_mask.any():
            return
        command_height = float(
            getattr(self._env, "_recovery_command_height", _DEFAULT_STANDING_HEIGHT)
        )
        self._command[recovery_mask, 0:4] = 0.0
        self._command[recovery_mask, 4] = command_height
        self._command[recovery_mask, 5] = 0.0
        self._command[recovery_mask, 7] = 0.0

    def _advance_traj_step(self) -> None:
        """按参考轨迹时间推进计步器。

        traj_step/jump_phase/jump_stage 只由 reference motion 决定。
        真实接触不参与跳跃窗口生命周期。
        """
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

        # 更新 jump_phase（第 7 维）：完整 motion 相位，grounded 期也推进。
        phase = torch.zeros(self.num_envs, device=self.device)
        if jump_envs.any():
            h_target = self._command[jump_envs, 6]
            max_step = self._traj_library.n_steps_for(h_target) - 1
            phase[jump_envs] = self._traj_step[jump_envs].float() / torch.clamp(
                max_step.float(), min=1.0
            )
        self._command[:, 7] = phase.clamp(0.0, 1.0)

    def _update_reference_takeoff_metrics(self, jump_envs: torch.Tensor) -> None:
        """按参考空中阶段记录起跳诊断。"""
        robot = self._env.scene["robot"]
        vz_w = robot.data.root_link_lin_vel_w[:, 2]
        in_ref_air = jump_envs & (self._jump_stage == 1)
        newly_recorded = in_ref_air & ~self._takeoff_recorded

        # reset 后前 5 步视为 RSI 注入窗口；之后进入参考空中阶段且 vz>0 记为主动上升样本。
        rsi_takeoff = newly_recorded & (self._env.episode_length_buf <= 5)
        active_takeoff = newly_recorded & (self._env.episode_length_buf > 5) & (vz_w > 0.0)
        self._rsi_takeoff_count[rsi_takeoff] += 1
        self._active_takeoff_count[active_takeoff] += 1

        active_success = active_takeoff & (vz_w > _VZ_SUCCESS_THRESHOLD)
        self._active_success_count[active_success] += 1
        self._takeoff_recorded[newly_recorded] = True

    def _clear_jump_command(self, env_ids: torch.Tensor) -> None:
        """结束一次跳跃窗口，让同一个 env 回到普通行走指令。"""
        if env_ids.dtype == torch.bool:
            if not env_ids.any():
                return
        elif len(env_ids) == 0:
            return
        self._command[env_ids, 5] = 0.0
        self._command[env_ids, 7] = 0.0
        self._jump_stage[env_ids] = 0
        self._takeoff_recorded[env_ids] = False
        self._traj_step[env_ids] = 0
        self._jump_cool_down[env_ids] = int(self.cfg.jump_cool_down_steps)

    def _takeoff_reward_diagnostics(
        self,
        jump_flag: torch.Tensor,
        vz_w: torch.Tensor,
    ) -> dict[str, float]:
        """诊断蹬地窗口内的奖励稀疏度和惩罚量级。"""
        takeoff_window = jump_flag & self.reference_takeoff_active()

        h_target = self._command[:, 6]
        vz_ref = ideal_takeoff_vel(h_target).clamp_min(0.1)
        vz_progress = torch.clamp(vz_w / vz_ref, min=0.0, max=1.0)
        vz_tracking_threshold = 0.12 * vz_ref
        vz_tracking_error = torch.abs(vz_w - vz_ref)
        vz_tracking = 1.0 - torch.clamp(vz_tracking_error - 0.45, min=0.0) / vz_ref
        vz_tracking = torch.clamp(vz_tracking, min=0.0, max=1.0)
        vz_tracking_reward = torch.where(vz_w < vz_tracking_threshold, vz_progress, vz_tracking)

        robot = self._env.scene["robot"]
        leg_ids = policy_leg_joint_ids(robot)
        knee_indices = [leg_ids[1], leg_ids[3]]
        knee_vel = robot.data.joint_vel[:, knee_indices]
        extension_vel = torch.clamp(-torch.mean(knee_vel, dim=1), min=0.0)
        ref_q_vel = self.reference_joint_velocity()
        ref_knee_vel = ref_q_vel[:, [1, 4]]
        ref_extension_vel = torch.clamp(-torch.mean(ref_knee_vel, dim=1), min=0.05)
        impulse_reward = torch.clamp(extension_vel / ref_extension_vel, min=0.0, max=1.0)

        action_rate = torch.sum(
            (self._env.action_manager.action - self._env.action_manager.prev_action) ** 2,
            dim=1,
        )
        ang_vel = robot.data.root_link_ang_vel_b
        pg = robot.data.projected_gravity_b
        pitch = torch.atan2(pg[:, 0], -pg[:, 2])
        roll = torch.atan2(-pg[:, 1], -pg[:, 2])
        pitch_deg = torch.rad2deg(torch.abs(pitch))
        roll_deg = torch.rad2deg(torch.abs(roll))
        tilt = torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0))
        tilt_deg = torch.rad2deg(tilt)
        ang_vel_xy_sq = ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2
        yaw_error_sq = (ang_vel[:, 2] - self._command[:, 1]) ** 2

        vel = robot.data.root_link_lin_vel_w
        vxy_sq = vel[:, 0] ** 2 + vel[:, 1] ** 2
        horizontal_active = takeoff_window & (vz_w > 0.2 * vz_ref)
        horizontal_penalty = 1.0 - torch.exp(-vxy_sq / (0.25**2))

        soft_limits = robot.data.soft_joint_pos_limits
        if soft_limits is None:
            knee_limit_penalty = torch.zeros_like(vz_w)
        else:
            pos = robot.data.joint_pos[:, knee_indices]
            limits = soft_limits[:, knee_indices]
            knee_limit_penalty = torch.sum(-(pos - limits[:, :, 0]).clamp(max=0.0), dim=1)

        takeoff_total = takeoff_window.float().sum().clamp_min(1.0)
        current_metrics = {
            "Jump/diag_takeoff_positive_vz_ratio": float(
                ((takeoff_window & (vz_w > 0.0)).float().sum() / takeoff_total).item()
            ),
            "Jump/diag_takeoff_vz_progress": _mean_on_mask(vz_progress, takeoff_window),
            "Jump/diag_takeoff_vz_tracking_reward": _mean_on_mask(
                vz_tracking_reward,
                takeoff_window,
            ),
            "Jump/diag_takeoff_impulse_reward": _mean_on_mask(impulse_reward, takeoff_window),
            "Jump/diag_takeoff_action_rate_raw": _mean_on_mask(action_rate, takeoff_window),
            "Jump/diag_takeoff_pitch_deg": _mean_on_mask(pitch_deg, takeoff_window),
            "Jump/diag_takeoff_roll_deg": _mean_on_mask(roll_deg, takeoff_window),
            "Jump/diag_takeoff_tilt_deg": _mean_on_mask(tilt_deg, takeoff_window),
            "Jump/diag_takeoff_ang_vel_xy_sq": _mean_on_mask(ang_vel_xy_sq, takeoff_window),
            "Jump/diag_takeoff_yaw_error_sq": _mean_on_mask(yaw_error_sq, takeoff_window),
            "Jump/diag_takeoff_horizontal_penalty": _mean_on_mask(
                horizontal_penalty,
                horizontal_active,
            ),
            "Jump/diag_takeoff_knee_limit_penalty": _mean_on_mask(
                knee_limit_penalty,
                takeoff_window,
            ),
        }

        sample_masks = {
            "Jump/diag_takeoff_positive_vz_ratio": takeoff_window,
            "Jump/diag_takeoff_vz_progress": takeoff_window,
            "Jump/diag_takeoff_vz_tracking_reward": takeoff_window,
            "Jump/diag_takeoff_impulse_reward": takeoff_window,
            "Jump/diag_takeoff_action_rate_raw": takeoff_window,
            "Jump/diag_takeoff_pitch_deg": takeoff_window,
            "Jump/diag_takeoff_roll_deg": takeoff_window,
            "Jump/diag_takeoff_tilt_deg": takeoff_window,
            "Jump/diag_takeoff_ang_vel_xy_sq": takeoff_window,
            "Jump/diag_takeoff_yaw_error_sq": takeoff_window,
            "Jump/diag_takeoff_horizontal_penalty": horizontal_active,
            "Jump/diag_takeoff_knee_limit_penalty": takeoff_window,
        }
        ema_metrics = getattr(self, _TAKEOFF_DIAG_EMA_ATTR, None)
        if not isinstance(ema_metrics, dict):
            ema_metrics = {}
        for name, value in current_metrics.items():
            if sample_masks[name].any():
                ema_metrics[f"{name}_ema"] = _ema_value(ema_metrics.get(f"{name}_ema"), value)
        setattr(self, _TAKEOFF_DIAG_EMA_ATTR, ema_metrics)

        return current_metrics | {name: float(value) for name, value in ema_metrics.items()}

    def _symmetry_diagnostics(self, jump_flag: torch.Tensor) -> dict[str, float]:
        """上报左右腿镜像误差，直接观察静站对称性是否改善。"""
        robot = self._env.scene["robot"]
        cmd_speed = torch.linalg.norm(self._command[:, :2], dim=1)
        standing_mask = (~jump_flag) & (cmd_speed < 0.1)
        low_speed_mask = (~jump_flag) & (cmd_speed < 0.35)
        jump_mask = jump_flag

        q = robot.data.joint_pos
        pg = robot.data.projected_gravity_b
        pitch_deg = torch.rad2deg(torch.abs(torch.atan2(pg[:, 0], -pg[:, 2])))
        hip_diff, knee_diff = active_leg_mirror_diffs(robot, q)
        hip_abs = torch.abs(hip_diff)
        knee_abs = torch.abs(knee_diff)
        mirror_raw = 4.0 * hip_abs**2 + 1.5 * knee_abs**2
        wheel_vel = robot.data.joint_vel[:, list(wheel_joint_ids(robot))]
        # 左右轮 joint axis 相反，joint 广义速度同号更容易制造 yaw 扭转。
        wheel_yaw_drive_sq = (wheel_vel[:, 0] + wheel_vel[:, 1]) ** 2
        yaw_rate_abs = torch.abs(robot.data.root_link_ang_vel_b[:, 2])

        return {
            "Jump/diag_standing_symmetry_sample_ratio": standing_mask.float().mean().item(),
            "Jump/diag_flat_pitch_deg": _mean_on_mask(pitch_deg, ~jump_flag),
            "Jump/diag_standing_pitch_deg": _mean_on_mask(pitch_deg, standing_mask),
            "Jump/diag_standing_hip_abs_diff_rad": _mean_on_mask(hip_abs, standing_mask),
            "Jump/diag_standing_knee_abs_diff_rad": _mean_on_mask(knee_abs, standing_mask),
            "Jump/diag_standing_joint_mirror_raw": _mean_on_mask(mirror_raw, standing_mask),
            "Jump/diag_low_speed_joint_mirror_raw": _mean_on_mask(mirror_raw, low_speed_mask),
            "Jump/diag_jump_joint_mirror_raw": _mean_on_mask(mirror_raw, jump_mask),
            "Jump/diag_jump_wheel_yaw_drive_sq": _mean_on_mask(wheel_yaw_drive_sq, jump_mask),
            "Jump/diag_jump_yaw_rate_abs": _mean_on_mask(yaw_rate_abs, jump_mask),
        }

    def _update_metrics(self) -> None:
        """上报跳跃诊断指标到训练日志（通过 extras['log'] 传递给 wandb/tensorboard）。

        参考宇树 fzqver 的 diag_* 体系，分两类指标：

        【比率指标】反映本 iter 参考轨迹阶段分布：
        - Jump/airborne_ratio：参考空中阶段 env 占比
        - Jump/jump_flag_ratio：当前处于跳跃窗口的 env 占比

        【scale-independent 诊断指标】不受奖励权重影响，是跳跃能力的真实度量：
        - Jump/diag_mean_airborne_vz：参考空中阶段 env 的平均 vz（m/s）
        - Jump/diag_max_airborne_vz：参考空中阶段 env 的最大 vz
        - Jump/diag_jump_success_rate：参考空中阶段 vz > 1.0 m/s 的 env 比例
        - Jump/diag_tilt_airborne：参考空中阶段平均倾斜角（deg），越小姿态越稳
        - Jump/diag_pitch_airborne_deg：空中平均 pitch 绝对值（deg），用于定位前后点头
        - Jump/diag_roll_airborne_deg：空中平均 roll 绝对值（deg），用于区分左右倾斜

        【leg_contact 精细拆分指标】用于定位 leg_contact 来源，区分根因：
        - Jump/diag_leg_contact_jump：jump_flag=1 的 env 中腿部当前有接触的比例
        - Jump/diag_leg_contact_walk：jump_flag=0 的 env 中腿部当前有接触的比例
          两者之差反映跳跃任务特有的腿部接触问题
        - Jump/diag_leg_contact_rsi：jump_flag=1 且 episode_step <= rsi_window 的 env 中腿部接触比例
          专门检测 RSI 注入后的早期腿部触地（说明 RSI 姿态/速度导致起跳后立即失控）
        - Jump/diag_leg_contact_landing：jump_flag=1 且 stage==2（landing）的 env 中腿部接触比例
          专门检测着陆冲击时的腿部触地

        【对称性诊断指标】用于确认静站和低速时左右腿是否镜像：
        - Jump/diag_flat_pitch_deg：平地段（jump_flag=0）pitch 绝对值均值
        - Jump/diag_standing_pitch_deg：静站段（jump_flag=0 且低速）pitch 绝对值均值
        - Jump/diag_standing_joint_mirror_raw：静站期左右髋/膝镜像误差，越小越对称
        - Jump/diag_standing_hip_abs_diff_rad：静站期左右髋关节绝对差值
        - Jump/diag_standing_knee_abs_diff_rad：静站期左右膝关节绝对差值
        - Jump/diag_jump_wheel_yaw_drive_sq：跳跃期同号轮速扭腰驱动，越小越少 yaw 扭转
        - Jump/diag_jump_yaw_rate_abs：跳跃期 yaw 角速度绝对值
        """
        jump_flag = self._command[:, 5] > 0.5

        ref_grounded_ratio = (self._jump_stage == 0).float().mean().item()
        ref_airborne_ratio = (self._jump_stage == 1).float().mean().item()
        ref_landing_ratio = (self._jump_stage == 2).float().mean().item()
        ref_preload_ratio = (jump_flag & self.reference_preload_active()).float().mean().item()
        ref_takeoff_ratio = (jump_flag & self.reference_takeoff_active()).float().mean().item()
        grounded_ratio = ref_grounded_ratio
        airborne_ratio = ref_airborne_ratio
        landing_ratio = ref_landing_ratio
        jump_flag_ratio = jump_flag.float().mean().item()

        robot = self._env.scene["robot"]
        vz_w = robot.data.root_link_lin_vel_w[:, 2]
        airborne_mask = jump_flag & (self._jump_stage == 1)

        # scale-independent 诊断
        if airborne_mask.any():
            vz_air = vz_w[airborne_mask]
            mean_airborne_vz = vz_air.mean().item()
            max_airborne_vz = vz_air.max().item()
            jump_success_rate = (vz_air > _VZ_SUCCESS_THRESHOLD).float().mean().item()
            # 空中姿态：projected_gravity_z，-1=完全直立，+1=倒置
            pg = robot.data.projected_gravity_b[airborne_mask]
            pg_z = pg[:, 2]
            tilt_rad = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
            tilt_deg_airborne = torch.rad2deg(tilt_rad).mean().item()
            pitch_airborne = torch.atan2(pg[:, 0], -pg[:, 2])
            roll_airborne = torch.atan2(-pg[:, 1], -pg[:, 2])
            pitch_deg_airborne = torch.rad2deg(torch.abs(pitch_airborne)).mean().item()
            roll_deg_airborne = torch.rad2deg(torch.abs(roll_airborne)).mean().item()
        else:
            mean_airborne_vz = 0.0
            max_airborne_vz = 0.0
            jump_success_rate = 0.0
            tilt_deg_airborne = 0.0
            pitch_deg_airborne = 0.0
            roll_deg_airborne = 0.0

        active_takeoff_total = self._active_takeoff_count.float().sum()
        active_success_total = self._active_success_count.float().sum()
        if active_takeoff_total > 0:
            active_success_rate = (active_success_total / active_takeoff_total).item()
        else:
            active_success_rate = 0.0

        jump_flag_total = jump_flag.float().sum()
        if jump_flag_total > 0:
            active_takeoff_envs = ((self._active_takeoff_count > 0) & jump_flag).float().sum()
            active_takeoff_ratio_per_jump_flag = (active_takeoff_envs / jump_flag_total).item()
        else:
            active_takeoff_ratio_per_jump_flag = 0.0

        # leg_contact 精细拆分指标由 terminations.leg_contact() 在 termination 时序写入
        # （termination 在 reset 之前，传感器数据有效；此处不重复读取）

        if hasattr(self._env, "extras") and isinstance(self._env.extras.get("log"), dict):
            self._env.extras["log"].update(
                {
                    "Jump/grounded_ratio": grounded_ratio,
                    "Jump/airborne_ratio": airborne_ratio,
                    "Jump/landing_ratio": landing_ratio,
                    "Jump/ref_grounded_ratio": ref_grounded_ratio,
                    "Jump/ref_airborne_ratio": ref_airborne_ratio,
                    "Jump/ref_landing_ratio": ref_landing_ratio,
                    "Jump/ref_preload_ratio": ref_preload_ratio,
                    "Jump/ref_takeoff_ratio": ref_takeoff_ratio,
                    "Jump/jump_flag_ratio": jump_flag_ratio,
                    # scale-independent 诊断指标（对标宇树 diag_* 体系）
                    "Jump/diag_mean_airborne_vz": mean_airborne_vz,
                    "Jump/diag_max_airborne_vz": max_airborne_vz,
                    "Jump/diag_jump_success_rate": jump_success_rate,
                    "Jump/diag_tilt_airborne_deg": tilt_deg_airborne,
                    "Jump/diag_pitch_airborne_deg": pitch_deg_airborne,
                    "Jump/diag_roll_airborne_deg": roll_deg_airborne,
                    # 完整跳跃流程计数（每 iter 的均值）
                    # diag_complete_jumps：参考轨迹走到末帧的次数
                    # diag_active_takeoffs：参考空中阶段且实际 vz>0 的非 RSI 样本数
                    "Jump/diag_complete_jumps": self._complete_jump_count.float().mean().item(),
                    "Jump/diag_active_takeoffs": self._active_takeoff_count.float().mean().item(),
                    "Jump/diag_rsi_takeoffs": self._rsi_takeoff_count.float().mean().item(),
                    "Jump/diag_active_success_rate": active_success_rate,
                    "Jump/diag_active_takeoff_ratio_per_jump_flag": active_takeoff_ratio_per_jump_flag,
                }
            )
            self._env.extras["log"].update(self._takeoff_reward_diagnostics(jump_flag, vz_w))
            self._env.extras["log"].update(self._symmetry_diagnostics(jump_flag))
            # leg_contact 精细拆分由 terminations.leg_contact() 在 reset 之前写入
            # extras["_leg_contact_diag"]（跨 reset 存活），此处搬入 log 上报
            diag = self._env.extras.get("_leg_contact_diag")
            if isinstance(diag, dict):
                self._env.extras["log"].update(diag)


@dataclass
class RecoveryStandCommandCfg(JumpCommandCfg):
    """纯倒地自起任务的固定站立指令。"""

    target_height: float = _DEFAULT_STANDING_HEIGHT

    def build(self, env: ManagerBasedRlEnv) -> RecoveryStandCommandTerm:
        return RecoveryStandCommandTerm(self, env)


class RecoveryStandCommandTerm(JumpCommandTerm):
    """固定输出站立交接目标，不采样速度、姿态或跳跃相位。"""

    cfg: RecoveryStandCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """reset 时把所有指令写成固定站立目标。"""
        self._set_fixed_command(env_ids)
        self._reset_fixed_lifecycle(env_ids)

    def _update_command(self) -> None:
        """每步强制保持固定指令，防止父类课程或跳跃逻辑改写。"""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._set_fixed_command(env_ids)

    def _set_fixed_command(self, env_ids: torch.Tensor) -> None:
        """写入 [0,0,0,0,target_height,0,0,0]。"""
        if len(env_ids) == 0:
            return
        self._command[env_ids, :] = 0.0
        self._command[env_ids, 4] = float(self.cfg.target_height)
        self._standing_mask[env_ids] = True

    def _reset_fixed_lifecycle(self, env_ids: torch.Tensor) -> None:
        """清空跳跃生命周期缓存，保持 jump_commands 观测为 0。"""
        if len(env_ids) == 0:
            return
        self._jump_stage[env_ids] = 0
        self._complete_jump_count[env_ids] = 0
        self._active_takeoff_count[env_ids] = 0
        self._rsi_takeoff_count[env_ids] = 0
        self._active_success_count[env_ids] = 0
        self._takeoff_recorded[env_ids] = False
        self._traj_step[env_ids] = 0
        self._jump_cool_down[env_ids] = 0
        self._pre_resampled_for_reset[env_ids] = False

    def _update_metrics(self) -> None:
        """记录固定 command contract，便于前 5 iter 验收。"""
        if hasattr(self._env, "extras"):
            cmd = self._command
            self._env.extras.setdefault("log", {}).update(
                {
                    "RecoveryStand/command_vx": cmd[:, 0].mean().item(),
                    "RecoveryStand/command_yaw_rate": cmd[:, 1].mean().item(),
                    "RecoveryStand/command_pitch": cmd[:, 2].mean().item(),
                    "RecoveryStand/command_roll": cmd[:, 3].mean().item(),
                    "RecoveryStand/command_height": cmd[:, 4].mean().item(),
                    "RecoveryStand/command_jump_flag": cmd[:, 5].mean().item(),
                    "RecoveryStand/command_jump_target_height": cmd[:, 6].mean().item(),
                    "RecoveryStand/command_jump_phase": cmd[:, 7].mean().item(),
                }
            )
