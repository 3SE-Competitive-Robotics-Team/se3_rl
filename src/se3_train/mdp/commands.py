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
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    height_resample_on_reset_only: bool = False
    """是否只在 reset 时采样高度指令；普通重采样只更新速度和姿态指令。"""
    constrain_diff_drive_commands: bool = False
    """是否按双轮差速轮速预算约束 vx/yaw 指令组合。"""
    diff_drive_wheel_radius: float = 0.06
    diff_drive_half_track: float = 0.20
    diff_drive_max_wheel_speed: float = 45.0
    diff_drive_wheel_speed_fraction: float = 1.0
    terrain_aware_height: bool = True
    """是否按当前 env 的地形台阶高度抬高 body height 指令下限。"""
    terrain_height_clearance: float = 0.0
    """机体碰撞盒下边缘相对单级台阶顶部的最小安全余量(m)。"""
    body_collision_bottom_offset: float = 0.0
    """机体碰撞盒下边缘相对 base_link frame 的 z 偏移(m)。"""
    terrain_step_height_type_names: tuple[str, ...] = (
        "forward_stairs",
        "inv_pyramid_stairs",
        "random_stairs",
    )
    """需要按单级台阶高度抬高 body height 的 terrain type 名称。"""

    def build(self, env: ManagerBasedRlEnv) -> VelocityHeightCommandTerm:
        return VelocityHeightCommandTerm(self, env)


class VelocityHeightCommandTerm(CommandTerm):
    """速度 + 姿态 + 高度的指令项。

    指令维度: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    """

    cfg: VelocityHeightCommandCfg

    def __init__(self, cfg: VelocityHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 5 维指令: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._resampling_for_reset = False
        self._pre_resampled_for_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """重置指令项，并保留 reset 事件阶段已预采样的指令。"""
        assert isinstance(env_ids, torch.Tensor)
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        pre_mask = self._pre_resampled_for_reset[env_ids]
        pre_ids = env_ids[pre_mask]
        fresh_ids = env_ids[~pre_mask]

        self.command_counter[env_ids] = 0
        previous_resampling_for_reset = self._resampling_for_reset
        self._resampling_for_reset = True
        try:
            if len(fresh_ids) > 0:
                self._resample(fresh_ids)
            if len(pre_ids) > 0:
                self.command_counter[pre_ids] = 1
                self._pre_resampled_for_reset[pre_ids] = False
            self._log_bad_orientation_diagnostics(env_ids)
            return extras
        finally:
            self._resampling_for_reset = previous_resampling_for_reset

    def pre_resample_for_reset(self, env_ids: torch.Tensor) -> None:
        """在 reset 事件写状态前预采样指令，保证关节默认姿态读取新 height。"""
        previous_resampling_for_reset = self._resampling_for_reset
        self._resampling_for_reset = True
        try:
            self._resample(env_ids)
            self._pre_resampled_for_reset[env_ids] = True
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

        # 用概率采样静站样本，避免单 env 重采样时 int(n * ratio) 被截断为 0。
        standing_prob = min(max(float(self.cfg.standing_ratio), 0.0), 1.0)
        standing_mask = torch.rand(n, device=self.device) < standing_prob
        standing_ids = env_ids[standing_mask]
        moving_ids = env_ids[~standing_mask]

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
            self._command[standing_ids, 4] = self._apply_terrain_aware_height(
                standing_ids,
                standing_height,
            )

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
            lin_vel, yaw_vel = self._constrain_diff_drive_command(lin_vel, yaw_vel)
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
                self._command[moving_ids, 4] = self._sample_terrain_aware_height(moving_ids)

        if resample_height:
            update_policy_default_from_height_cache(
                self._env,
                "velocity_height",
                env_ids=env_ids,
                command=self._command,
            )

    def _apply_terrain_aware_height(
        self,
        env_ids: torch.Tensor,
        sampled_height: torch.Tensor,
    ) -> torch.Tensor:
        """按当前地形单级高度抬高采样高度下限。"""
        if (
            not self.cfg.terrain_aware_height
            or len(env_ids) == 0
            or self.cfg.terrain_height_clearance <= 0.0
        ):
            return sampled_height

        min_height = self._terrain_aware_min_height(env_ids, sampled_height)
        max_height = torch.full_like(sampled_height, self.cfg.height_range[1])
        min_height = torch.minimum(min_height, max_height)
        return torch.maximum(sampled_height, min_height)

    def _sample_terrain_aware_height(self, env_ids: torch.Tensor) -> torch.Tensor:
        """在地形感知后的有效高度区间内均匀采样。"""
        count = len(env_ids)
        height_min = torch.full((count,), self.cfg.height_range[0], device=self.device)
        height_max = torch.full((count,), self.cfg.height_range[1], device=self.device)
        if self.cfg.terrain_aware_height and count > 0 and self.cfg.terrain_height_clearance > 0.0:
            height_min = self._terrain_aware_min_height(env_ids, height_min)
            height_min = torch.minimum(height_min, height_max)
        return torch.rand(count, device=self.device) * (height_max - height_min) + height_min

    def _terrain_aware_min_height(
        self,
        env_ids: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """由 terrain level/type 估算当前台阶需要的最低 body height。"""
        base_min = torch.full_like(reference, self.cfg.height_range[0])
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None:
            return base_min

        terrain_levels = getattr(terrain, "terrain_levels", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_origins = getattr(terrain, "terrain_origins", None)
        terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if (
            terrain_levels is None
            or terrain_types is None
            or terrain_origins is None
            or terrain_cfg is None
            or not getattr(terrain_cfg, "sub_terrains", None)
        ):
            return base_min

        num_rows = int(terrain_origins.shape[0])
        if num_rows <= 0:
            return base_min

        selected_names = set(self.cfg.terrain_step_height_type_names)
        sub_terrains = tuple(terrain_cfg.sub_terrains.items())
        levels = terrain_levels[env_ids].long()
        types = terrain_types[env_ids].long()
        lower, upper = terrain_cfg.difficulty_range
        difficulty_hi = (levels.float() + 1.0) / float(num_rows)
        difficulty_hi = float(lower) + (float(upper) - float(lower)) * difficulty_hi
        difficulty_hi = torch.clamp(difficulty_hi, float(lower), float(upper))

        step_height = torch.zeros_like(reference)
        for terrain_index, (terrain_name, sub_cfg) in enumerate(sub_terrains):
            if terrain_name not in selected_names:
                continue
            step_height_range = getattr(sub_cfg, "step_height_range", None)
            if step_height_range is None:
                continue
            terrain_mask = types == terrain_index
            if not torch.any(terrain_mask):
                continue

            step_low = float(step_height_range[0])
            step_high = float(step_height_range[1])
            difficulty = difficulty_hi[terrain_mask]
            if terrain_name == "random_stairs":
                step_height[terrain_mask] = step_high * (0.5 + 0.5 * difficulty)
            else:
                step_height[terrain_mask] = step_low + difficulty * (step_high - step_low)

        required = (
            step_height
            + float(self.cfg.terrain_height_clearance)
            - float(self.cfg.body_collision_bottom_offset)
        )
        return torch.maximum(base_min, required)

    def _constrain_diff_drive_command(
        self, lin_vel: torch.Tensor, yaw_vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按双轮差速轮速预算约束 vx/yaw，避免同时吃满直行和转向。"""
        if not self.cfg.constrain_diff_drive_commands:
            return lin_vel, yaw_vel

        wheel_radius = max(float(self.cfg.diff_drive_wheel_radius), 1.0e-6)
        half_track = max(float(self.cfg.diff_drive_half_track), 1.0e-6)
        wheel_speed_budget = (
            wheel_radius
            * max(float(self.cfg.diff_drive_max_wheel_speed), 1.0e-6)
            * max(float(self.cfg.diff_drive_wheel_speed_fraction), 1.0e-6)
        )

        lin_vel = torch.clamp(lin_vel, min=-wheel_speed_budget, max=wheel_speed_budget)
        yaw_low_cfg, yaw_high_cfg = self.cfg.ang_vel_yaw_range
        lower_from_left = (-wheel_speed_budget - lin_vel) / half_track
        upper_from_left = (wheel_speed_budget - lin_vel) / half_track
        lower_from_right = (lin_vel - wheel_speed_budget) / half_track
        upper_from_right = (lin_vel + wheel_speed_budget) / half_track
        yaw_low = torch.maximum(
            torch.full_like(lin_vel, float(yaw_low_cfg)),
            torch.maximum(lower_from_left, lower_from_right),
        )
        yaw_high = torch.minimum(
            torch.full_like(lin_vel, float(yaw_high_cfg)),
            torch.minimum(upper_from_left, upper_from_right),
        )
        yaw_span = torch.clamp(yaw_high - yaw_low, min=0.0)
        yaw_vel = yaw_low + torch.rand_like(yaw_vel) * yaw_span
        return lin_vel, yaw_vel

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

    def _update_metrics(self) -> None:
        """更新指令指标。"""
        pass
