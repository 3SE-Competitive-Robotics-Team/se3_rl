"""速度 + 姿态指令生成器。

指令:(lin_vel_x, ang_vel_yaw, pitch, roll, height)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from se3_train.mdp.height_default_cache import update_policy_default_from_height_cache

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class VelocityHeightCommandCfg(CommandTermCfg):
    """速度 + 姿态 + 高度指令生成器的配置。"""

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    ang_vel_yaw_range: tuple[float, float] = (-3.0, 3.0)
    pitch_range: tuple[float, float] = (-0.2, 0.2)
    roll_range: tuple[float, float] = (-0.1, 0.1)
    height_range: tuple[float, float] = (0.20, 0.32)
    standing_height_range: tuple[float, float] = (0.20, 0.32)
    lin_vel_deadband: float = 0.1
    yaw_deadband: float = 0.1
    standing_ratio: float = 0.1
    exclusive_linear_yaw_commands: bool = False
    linear_command_ratio: float = 0.5
    height_balance_schedule_enabled: bool = False
    height_balance_value: float = 0.22
    height_balance_stable_tilt_deg: float = 15.0
    height_balance_stable_ang_vel_xy: float = 0.8
    height_balance_stable_steps: int = 25
    height_balance_rise_rate: float = 0.1
    height_balance_completion_tolerance: float = 0.002
    height_balance_log_interval_steps: int = 64
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    height_resample_on_reset_only: bool = False
    """是否只在 reset 时采样高度指令；普通重采样只更新速度和姿态指令。"""

    def build(self, env: ManagerBasedRlEnv) -> VelocityHeightCommandTerm:
        return VelocityHeightCommandTerm(self, env)


class VelocityHeightCommandTerm(CommandTerm):
    """速度 + 姿态 + 高度的指令项。

    指令维度: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    """

    cfg: VelocityHeightCommandCfg
    _HEIGHT_PHASE_BALANCE = 0
    _HEIGHT_PHASE_RISE = 1
    _HEIGHT_PHASE_DONE = 2

    def __init__(self, cfg: VelocityHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 5 维指令: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._final_height_command = torch.zeros(self.num_envs, device=self.device)
        self._height_phase = torch.full(
            (self.num_envs,),
            self._HEIGHT_PHASE_DONE,
            device=self.device,
            dtype=torch.long,
        )
        self._height_stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._resampling_for_reset = False

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """重置指令项，并把跨 reset 缓存的终止诊断搬入日志。"""
        assert isinstance(env_ids, torch.Tensor)
        previous_resampling_for_reset = self._resampling_for_reset
        self._resampling_for_reset = True
        try:
            extras = super().reset(env_ids)
            self._log_bad_orientation_diagnostics(env_ids)
            return extras
        finally:
            self._resampling_for_reset = previous_resampling_for_reset

    def _log_bad_orientation_diagnostics(self, env_ids: torch.Tensor) -> None:
        """上报 bad_orientation 在恢复样本和平地样本中的拆分来源。"""
        diag = (
            self._env.extras.get("_bad_orientation_diag") if hasattr(self._env, "extras") else None
        )
        if not isinstance(diag, dict):
            return

        raw_bad = diag.get("raw_bad")
        counted_bad = diag.get("counted_bad")
        terminated = diag.get("terminated")
        recovery_mask = diag.get("recovery_mask")
        recovery_grace = diag.get("recovery_grace")
        tensors = (raw_bad, counted_bad, terminated, recovery_mask, recovery_grace)
        if not all(
            isinstance(item, torch.Tensor) and item.shape[0] == self.num_envs for item in tensors
        ):
            return

        reset_recovery = recovery_mask[env_ids]
        reset_flat = ~reset_recovery
        reset_terminated = terminated[env_ids]
        reset_raw_bad = raw_bad[env_ids]
        reset_counted_bad = counted_bad[env_ids]
        reset_grace = recovery_grace[env_ids]

        def _masked_rate(values: torch.Tensor, mask: torch.Tensor) -> float:
            if not mask.any():
                return 0.0
            return values[mask].float().mean().item()

        log = self._env.extras.setdefault("log", {})
        log.update(
            {
                "Episode_Termination/bad_orientation_recovery": (reset_terminated & reset_recovery)
                .float()
                .sum()
                .item(),
                "Episode_Termination/bad_orientation_flat": (reset_terminated & reset_flat)
                .float()
                .sum()
                .item(),
                "Recovery/bad_orientation_raw_rate": _masked_rate(reset_raw_bad, reset_recovery),
                "Recovery/bad_orientation_counted_rate": _masked_rate(
                    reset_counted_bad, reset_recovery
                ),
                "Recovery/bad_orientation_grace_rate": _masked_rate(reset_grace, reset_recovery),
                "Recovery/bad_orientation_termination_rate": _masked_rate(
                    reset_terminated, reset_recovery
                ),
            }
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样指令。"""
        n = len(env_ids)
        resample_height = not self.cfg.height_resample_on_reset_only or bool(
            getattr(self, "_resampling_for_reset", False)
        )

        # 确定哪些环境处于站立状态。
        standing_count = int(n * self.cfg.standing_ratio)
        standing_ids = env_ids[:standing_count]
        moving_ids = env_ids[standing_count:]

        self._standing_mask[standing_ids] = True
        self._standing_mask[moving_ids] = False

        # 站立环境:零速度,默认姿态,按站立高度范围采样。
        self._command[standing_ids, 0] = 0.0
        self._command[standing_ids, 1] = 0.0
        self._command[standing_ids, 2] = 0.0  # pitch = 0
        self._command[standing_ids, 3] = 0.0  # roll = 0
        if len(standing_ids) > 0 and resample_height:
            standing_height = (
                torch.rand(len(standing_ids), device=self.device)
                * (self.cfg.standing_height_range[1] - self.cfg.standing_height_range[0])
                + self.cfg.standing_height_range[0]
            )
            self._set_height_targets(standing_ids, standing_height)

        # 运动环境:随机速度 + 随机姿态 + 随机高度。
        if len(moving_ids) > 0:
            lin_vel = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.lin_vel_x_range[1] - self.cfg.lin_vel_x_range[0])
                + self.cfg.lin_vel_x_range[0]
            )
            yaw_vel = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.ang_vel_yaw_range[1] - self.cfg.ang_vel_yaw_range[0])
                + self.cfg.ang_vel_yaw_range[0]
            )
            if self.cfg.exclusive_linear_yaw_commands:
                linear_ratio = min(max(float(self.cfg.linear_command_ratio), 0.0), 1.0)
                linear_mode = torch.rand(len(moving_ids), device=self.device) < linear_ratio
                lin_vel = torch.where(linear_mode, lin_vel, torch.zeros_like(lin_vel))
                yaw_vel = torch.where(linear_mode, torch.zeros_like(yaw_vel), yaw_vel)
            pitch = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.pitch_range[1] - self.cfg.pitch_range[0])
                + self.cfg.pitch_range[0]
            )
            roll = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.roll_range[1] - self.cfg.roll_range[0])
                + self.cfg.roll_range[0]
            )
            self._command[moving_ids, 0] = lin_vel
            self._command[moving_ids, 1] = yaw_vel
            self._command[moving_ids, 2] = pitch
            self._command[moving_ids, 3] = roll
            if resample_height:
                height = (
                    torch.rand(len(moving_ids), device=self.device)
                    * (self.cfg.height_range[1] - self.cfg.height_range[0])
                    + self.cfg.height_range[0]
                )
                self._set_height_targets(moving_ids, height)

        if resample_height:
            update_policy_default_from_height_cache(
                self._env,
                "velocity_height",
                env_ids=env_ids,
                command=self._command,
            )

    def _update_command(self) -> None:
        """对速度指令施加死区。"""
        moving = ~self._standing_mask
        lin_vel = self._command[:, 0]
        yaw_vel = self._command[:, 1]

        # 将小速度置零(死区)。
        lin_vel = torch.where(
            moving & (torch.abs(lin_vel) < self.cfg.lin_vel_deadband),
            torch.zeros_like(lin_vel),
            lin_vel,
        )
        yaw_vel = torch.where(
            moving & (torch.abs(yaw_vel) < self.cfg.yaw_deadband),
            torch.zeros_like(yaw_vel),
            yaw_vel,
        )

        self._command[:, 0] = lin_vel
        self._command[:, 1] = yaw_vel
        self._update_height_balance_schedule()

    def _set_height_targets(self, env_ids: torch.Tensor, final_height: torch.Tensor) -> None:
        """设置最终高度，并生成当前有效高度指令。"""
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        final_height = final_height.to(device=self.device)
        self._final_height_command[env_ids] = final_height
        self._height_stable_steps[env_ids] = 0

        if not self.cfg.height_balance_schedule_enabled:
            self._command[env_ids, 4] = final_height
            self._height_phase[env_ids] = self._HEIGHT_PHASE_DONE
            return

        balance_height = torch.full_like(final_height, float(self.cfg.height_balance_value))
        effective_height = torch.minimum(final_height, balance_height)
        needs_rise = final_height > effective_height + float(
            self.cfg.height_balance_completion_tolerance
        )
        self._command[env_ids, 4] = effective_height
        self._height_phase[env_ids] = torch.where(
            needs_rise,
            torch.full_like(env_ids, self._HEIGHT_PHASE_BALANCE),
            torch.full_like(env_ids, self._HEIGHT_PHASE_DONE),
        )

    def _control_dt(self) -> float:
        """返回 policy 控制周期。"""
        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is not None:
            return max(float(step_dt), 1.0e-6)
        physics_dt = float(getattr(self._env, "physics_dt", 0.005))
        decimation = int(getattr(getattr(self._env, "cfg", object()), "decimation", 4))
        return max(physics_dt * decimation, 1.0e-6)

    def _update_height_balance_schedule(self) -> None:
        """先低高度稳定平衡，再平滑恢复到最终高度。"""
        if not self.cfg.height_balance_schedule_enabled:
            return

        active = self._height_phase != self._HEIGHT_PHASE_DONE
        if not active.any():
            self._log_height_balance_schedule()
            return

        robot = self._env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        tilt_limit = torch.deg2rad(
            torch.tensor(float(self.cfg.height_balance_stable_tilt_deg), device=self.device)
        )
        ang_vel_xy = torch.linalg.norm(robot.data.root_link_ang_vel_b[:, :2], dim=1)
        stable = (tilt < tilt_limit) & (
            ang_vel_xy < float(self.cfg.height_balance_stable_ang_vel_xy)
        )

        balance = self._height_phase == self._HEIGHT_PHASE_BALANCE
        stable_balance = balance & stable
        self._height_stable_steps[stable_balance] += 1
        self._height_stable_steps[balance & ~stable] = 0

        ready_to_rise = balance & (
            self._height_stable_steps >= int(self.cfg.height_balance_stable_steps)
        )
        self._height_phase[ready_to_rise] = self._HEIGHT_PHASE_RISE

        rising = self._height_phase == self._HEIGHT_PHASE_RISE
        if rising.any():
            previous_height = self._command[:, 4].clone()
            delta = float(self.cfg.height_balance_rise_rate) * self._control_dt()
            self._command[rising, 4] = torch.minimum(
                self._command[rising, 4] + delta,
                self._final_height_command[rising],
            )
            done = rising & (
                torch.abs(self._command[:, 4] - self._final_height_command)
                <= float(self.cfg.height_balance_completion_tolerance)
            )
            self._height_phase[done] = self._HEIGHT_PHASE_DONE
            changed = torch.nonzero(
                torch.abs(self._command[:, 4] - previous_height) > 0.0
            ).flatten()
            if len(changed) > 0:
                update_policy_default_from_height_cache(
                    self._env,
                    "velocity_height",
                    env_ids=changed,
                    command=self._command,
                )

        self._log_height_balance_schedule()

    def _log_height_balance_schedule(self) -> None:
        """上报两阶段高度指令调度诊断。"""
        interval = max(1, int(self.cfg.height_balance_log_interval_steps))
        if int(getattr(self._env, "common_step_counter", 0)) % interval != 0:
            return
        if not hasattr(self._env, "extras"):
            return

        final_height = self._final_height_command
        gap = torch.clamp(final_height - self._command[:, 4], min=0.0)
        log = self._env.extras.setdefault("log", {})
        log.update(
            {
                "Recovery/diag_height_phase_balance_rate": (
                    self._height_phase == self._HEIGHT_PHASE_BALANCE
                )
                .float()
                .mean()
                .item(),
                "Recovery/diag_height_phase_rise_rate": (
                    self._height_phase == self._HEIGHT_PHASE_RISE
                )
                .float()
                .mean()
                .item(),
                "Recovery/diag_height_phase_done_rate": (
                    self._height_phase == self._HEIGHT_PHASE_DONE
                )
                .float()
                .mean()
                .item(),
                "Recovery/diag_effective_command_height_mean_m": self._command[:, 4].mean().item(),
                "Recovery/diag_final_command_height_mean_m": final_height.mean().item(),
                "Recovery/diag_height_command_gap_m": gap.mean().item(),
                "Recovery/diag_height_balance_stable_steps": self._height_stable_steps.float()
                .mean()
                .item(),
            }
        )

    def _update_metrics(self) -> None:
        """更新指令指标。"""
        pass
