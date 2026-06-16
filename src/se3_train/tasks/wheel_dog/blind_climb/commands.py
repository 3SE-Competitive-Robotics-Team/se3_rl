"""WheelDog 盲爬速度指令。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from . import terrain_progress

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """计算掩码内均值；无样本时返回 0。"""
    if mask.any():
        return float(value[mask].mean().item())
    return 0.0


@dataclass
class DogVelocityCommandCfg(CommandTermCfg):
    """盲爬速度指令配置。

    指令维度: [lin_vel_x, lin_vel_y, ang_vel_yaw]。
    """

    lin_vel_x_range: tuple[float, float] = (0.35, 0.70)
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0)
    flat_lin_vel_x_range: tuple[float, float] = (-2.0, 2.0)
    flat_lin_vel_y_range: tuple[float, float] = (-0.6, 0.6)
    flat_ang_vel_yaw_range: tuple[float, float] = (0.0, 0.0)
    lin_vel_deadband: float = 0.08
    yaw_deadband: float = 0.08
    standing_ratio: float = 0.0
    flat_standing_ratio: float = 0.08
    success_distance: float = terrain_progress.FINAL_SUCCESS_DISTANCE
    obstacle_window_before_high_edge: float = 0.25
    obstacle_window_after_high_edge: float = 0.35

    @dataclass
    class VizCfg:
        """速度跟踪调试可视化参数。"""

        z_offset: float = 0.25
        scale: float = 0.8
        command_color: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.65)
        actual_color: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.75)
        width: float = 0.015

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> DogVelocityCommandTerm:
        return DogVelocityCommandTerm(self, env)


class DogVelocityCommandTerm(CommandTerm):
    """WheelDog 盲爬速度指令项。"""

    cfg: DogVelocityCommandCfg

    def __init__(self, cfg: DogVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._robot = env.scene["robot"]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样速度指令。"""
        flat_mask = terrain_progress.is_flat_terrain(self._env, env_ids=env_ids)
        self._sample_group(
            env_ids[~flat_mask],
            standing_ratio=self.cfg.standing_ratio,
            lin_vel_x_range=self.cfg.lin_vel_x_range,
            lin_vel_y_range=self.cfg.lin_vel_y_range,
            ang_vel_yaw_range=self.cfg.ang_vel_yaw_range,
        )
        self._sample_group(
            env_ids[flat_mask],
            standing_ratio=self.cfg.flat_standing_ratio,
            lin_vel_x_range=self.cfg.flat_lin_vel_x_range,
            lin_vel_y_range=self.cfg.flat_lin_vel_y_range,
            ang_vel_yaw_range=self.cfg.flat_ang_vel_yaw_range,
        )

    def _sample_group(
        self,
        env_ids: torch.Tensor,
        *,
        standing_ratio: float,
        lin_vel_x_range: tuple[float, float],
        lin_vel_y_range: tuple[float, float],
        ang_vel_yaw_range: tuple[float, float],
    ) -> None:
        """按同一组指令范围为一批环境采样速度。"""
        if len(env_ids) == 0:
            return
        standing_count = int(len(env_ids) * standing_ratio)
        standing_ids = env_ids[:standing_count]
        moving_ids = env_ids[standing_count:]

        self._standing_mask[standing_ids] = True
        self._standing_mask[moving_ids] = False
        self._command[standing_ids] = 0.0

        if len(moving_ids) == 0:
            return
        self._command[moving_ids, 0] = self._sample_range(len(moving_ids), lin_vel_x_range)
        self._command[moving_ids, 1] = self._sample_range(len(moving_ids), lin_vel_y_range)
        self._command[moving_ids, 2] = self._sample_range(len(moving_ids), ang_vel_yaw_range)

    def _update_command(self) -> None:
        """对很小的速度指令施加死区。"""
        moving = ~self._standing_mask
        vx = self._command[:, 0]
        vy = self._command[:, 1]
        yaw = self._command[:, 2]

        speed = torch.linalg.norm(self._command[:, :2], dim=1)
        small_speed = moving & (speed < self.cfg.lin_vel_deadband)
        vx = torch.where(small_speed, torch.zeros_like(vx), vx)
        vy = torch.where(small_speed, torch.zeros_like(vy), vy)
        yaw = torch.where(
            moving & (torch.abs(yaw) < self.cfg.yaw_deadband),
            torch.zeros_like(yaw),
            yaw,
        )

        self._command[:, 0] = vx
        self._command[:, 1] = vy
        self._command[:, 2] = yaw

    def _update_metrics(self) -> None:
        """写入速度跟随诊断指标。"""
        if not hasattr(self._env, "extras") or not isinstance(self._env.extras.get("log"), dict):
            return

        lin_vel_b = self._robot.data.root_link_lin_vel_b
        ang_vel_b = self._robot.data.root_link_ang_vel_b
        cmd = self._command
        active = torch.linalg.norm(cmd[:, :2], dim=1) > self.cfg.lin_vel_deadband

        vx_err = lin_vel_b[:, 0] - cmd[:, 0]
        vy_err = lin_vel_b[:, 1] - cmd[:, 1]
        yaw_err = ang_vel_b[:, 2] - cmd[:, 2]
        vx_w = torch.nan_to_num(
            self._robot.data.root_link_lin_vel_w[:, 0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vz_w = torch.nan_to_num(
            self._robot.data.root_link_lin_vel_w[:, 2],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        height = self._robot.data.root_link_pos_w[:, 2]
        progress_x = self._robot.data.root_link_pos_w[:, 0] - self._env.scene.env_origins[:, 0]
        progress_x = torch.nan_to_num(progress_x, nan=0.0, posinf=0.0, neginf=0.0)
        target_progress = terrain_progress.current_success_distance(
            self._env,
            final_success_distance=self.cfg.success_distance,
        )
        facility_mask = terrain_progress.is_facility_terrain(self._env)
        obstacle_start, obstacle_end = terrain_progress.obstacle_window(
            self._env,
            before_high_edge=self.cfg.obstacle_window_before_high_edge,
            after_high_edge=self.cfg.obstacle_window_after_high_edge,
        )
        ramp_high_x = terrain_progress.ramp_high_progress(self._env)
        ramp_low_x = terrain_progress.ramp_low_progress(self._env)
        difficulty = terrain_progress.current_difficulty(self._env)
        lateral_offset = terrain_progress.lateral_offset(self._env)
        abs_lateral_offset = torch.abs(lateral_offset)
        in_corridor = terrain_progress.within_corridor(self._env)
        out_of_bounds = abs_lateral_offset > terrain_progress.CORRIDOR_HARD_HALF_WIDTH
        success = (progress_x > target_progress) & in_corridor
        obstacle_window = active & (progress_x > obstacle_start) & (progress_x < obstacle_end)
        pre_obstacle_window = active & (progress_x < obstacle_start)
        early_takeoff = pre_obstacle_window & (vz_w > 0.25)
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if isinstance(terrain_levels, torch.Tensor) and terrain_levels.numel() >= self.num_envs:
            terrain_level = float(terrain_levels[: self.num_envs].float().mean().item())
            high_level_ratio = float((terrain_levels[: self.num_envs] >= 35).float().mean().item())
            target_level_ratio = float(
                (terrain_levels[: self.num_envs] >= 39).float().mean().item()
            )
        else:
            terrain_level = 0.0
            high_level_ratio = 0.0
            target_level_ratio = 0.0
        flat_ratio = float((~facility_mask).float().mean().item())
        facility_ratio = float(facility_mask.float().mean().item())
        runup_distance = getattr(self._env, "_wheel_dog_last_runup_distance", None)
        if isinstance(runup_distance, torch.Tensor) and runup_distance.numel() >= self.num_envs:
            facility_runup = runup_distance[: self.num_envs][facility_mask]
            if facility_runup.numel() > 0:
                runup_mean = float(facility_runup.mean().item())
                runup_min = float(facility_runup.min().item())
                runup_max = float(facility_runup.max().item())
            else:
                runup_mean = 0.0
                runup_min = 0.0
                runup_max = 0.0
        else:
            runup_mean = 0.0
            runup_min = 0.0
            runup_max = 0.0

        self._env.extras["log"].update(
            {
                "WheelDog/diag_active_ratio": float(active.float().mean().item()),
                "WheelDog/diag_standing_ratio": float(self._standing_mask.float().mean().item()),
                "WheelDog/diag_flat_terrain_ratio": flat_ratio,
                "WheelDog/diag_facility_terrain_ratio": facility_ratio,
                "WheelDog/diag_cmd_vx_abs": float(torch.abs(cmd[:, 0]).mean().item()),
                "WheelDog/diag_cmd_vy_abs": float(torch.abs(cmd[:, 1]).mean().item()),
                "WheelDog/diag_actual_vx": float(lin_vel_b[:, 0].mean().item()),
                "WheelDog/diag_actual_vy": float(lin_vel_b[:, 1].mean().item()),
                "WheelDog/diag_world_vx": float(vx_w.mean().item()),
                "WheelDog/diag_forward_moving_ratio": float(
                    ((vx_w > 0.25) & active).float().mean().item()
                ),
                "WheelDog/diag_backward_moving_ratio": float(
                    ((vx_w < -0.25) & active).float().mean().item()
                ),
                "WheelDog/diag_vx_error_abs": float(torch.abs(vx_err).mean().item()),
                "WheelDog/diag_vy_error_abs": float(torch.abs(vy_err).mean().item()),
                "WheelDog/diag_vx_error_abs_active": _mean_on_mask(torch.abs(vx_err), active),
                "WheelDog/diag_vy_error_abs_active": _mean_on_mask(torch.abs(vy_err), active),
                "WheelDog/diag_yaw_error_abs": float(torch.abs(yaw_err).mean().item()),
                "WheelDog/diag_base_height": float(height.mean().item()),
                "WheelDog/diag_blind_climb_progress_x": _mean_on_mask(progress_x, facility_mask),
                "WheelDog/diag_blind_climb_success_ratio": _mean_on_mask(
                    success.float(),
                    facility_mask,
                ),
                "WheelDog/diag_current_success_distance": _mean_on_mask(
                    target_progress,
                    facility_mask,
                ),
                "WheelDog/diag_ramp_high_progress_x": float(ramp_high_x.mean().item()),
                "WheelDog/diag_ramp_low_progress_x": float(ramp_low_x.mean().item()),
                "WheelDog/diag_terrain_difficulty": float(difficulty.mean().item()),
                "WheelDog/diag_obstacle_window_ratio": _mean_on_mask(
                    obstacle_window.float(),
                    facility_mask,
                ),
                "WheelDog/diag_obstacle_window_vz": _mean_on_mask(vz_w, obstacle_window),
                "WheelDog/diag_obstacle_window_positive_vz": _mean_on_mask(
                    torch.clamp(vz_w, min=0.0),
                    obstacle_window,
                ),
                "WheelDog/diag_pre_obstacle_positive_vz": _mean_on_mask(
                    torch.clamp(vz_w, min=0.0),
                    pre_obstacle_window,
                ),
                "WheelDog/diag_early_takeoff_ratio": _mean_on_mask(
                    early_takeoff.float(),
                    facility_mask,
                ),
                "WheelDog/diag_lateral_offset_abs": float(abs_lateral_offset.mean().item()),
                "WheelDog/diag_in_corridor_ratio": _mean_on_mask(
                    in_corridor.float(),
                    facility_mask,
                ),
                "WheelDog/diag_out_of_corridor_ratio": _mean_on_mask(
                    out_of_bounds.float(),
                    facility_mask,
                ),
                "WheelDog/diag_terrain_level": terrain_level,
                "WheelDog/diag_high_level_ratio": high_level_ratio,
                "WheelDog/diag_target_level_ratio": target_level_ratio,
                "WheelDog/diag_runup_distance": runup_mean,
                "WheelDog/diag_runup_min": runup_min,
                "WheelDog/diag_runup_max": runup_max,
                "WheelDog/diag_cmd_x_limit": float(abs(self.cfg.lin_vel_x_range[1])),
                "WheelDog/diag_cmd_y_limit": float(abs(self.cfg.lin_vel_y_range[1])),
            }
        )

    def _sample_range(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        """在闭区间内均匀采样。"""
        lo, hi = value_range
        return torch.rand(count, device=self.device) * (hi - lo) + lo

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """绘制期望平移速度和实际平移速度箭头。"""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        commands = self.command.cpu().numpy()
        base_pos_ws = self._robot.data.root_link_pos_w.cpu().numpy()
        base_mat_ws = matrix_from_quat(self._robot.data.root_link_quat_w).cpu().numpy()
        lin_vel_bs = self._robot.data.root_link_lin_vel_b.cpu().numpy()

        local_offset = np.array([0.0, 0.0, self.cfg.viz.z_offset])
        for env_idx in env_indices:
            base_pos_w = base_pos_ws[env_idx]
            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            base_mat_w = base_mat_ws[env_idx]
            start = base_pos_w + base_mat_w @ local_offset
            command_vec_b = np.array([commands[env_idx, 0], commands[env_idx, 1], 0.0])
            actual_vec_b = np.array([lin_vel_bs[env_idx, 0], lin_vel_bs[env_idx, 1], 0.0])
            command_vec_b *= self.cfg.viz.scale
            actual_vec_b *= self.cfg.viz.scale

            if np.linalg.norm(command_vec_b) > 1e-4:
                visualizer.add_arrow(
                    start,
                    start + base_mat_w @ command_vec_b,
                    color=self.cfg.viz.command_color,
                    width=self.cfg.viz.width,
                    label="期望速度",
                )
            if np.linalg.norm(actual_vec_b) > 1e-4:
                visualizer.add_arrow(
                    start,
                    start + base_mat_w @ actual_vec_b,
                    color=self.cfg.viz.actual_color,
                    width=self.cfg.viz.width,
                    label="实际速度",
                )


__all__ = ["DogVelocityCommandCfg", "DogVelocityCommandTerm"]
