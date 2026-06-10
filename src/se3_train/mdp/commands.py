"""基础指令和速度 + 姿态指令生成器。

指令:(lin_vel_x, ang_vel_yaw, pitch, roll, height)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from se3_shared import TaskMode

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_GAIT_TERRAIN_TYPES = (
    (0, "flat"),
    (1, "random_grid"),
    (2, "random_spread_boxes"),
    (3, "open_stairs_up"),
    (4, "open_stairs_down"),
)


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """计算掩码内均值；无样本时返回 0。"""
    if mask.any():
        return float(value[mask].mean().item())
    return 0.0


def _ratio_on_mask(mask: torch.Tensor, denominator_mask: torch.Tensor) -> float:
    """计算 mask 在 denominator_mask 内的占比；无样本时返回 0。"""
    denominator = float(denominator_mask.float().sum().item())
    if denominator > 0:
        return float((mask & denominator_mask).float().sum().item() / denominator)
    return 0.0


def _terrain_types(env: ManagerBasedRlEnv) -> torch.Tensor | None:
    """读取每个 env 当前地形类型；没有地形课程时返回 None。"""
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return None
    if terrain_types.numel() < env.num_envs:
        return None
    return terrain_types[: env.num_envs].to(device=env.device, dtype=torch.long)


@dataclass
class BasicCommandCfg(CommandTermCfg):
    """基础指令配置，提供所有任务共享的速度、姿态和高度采样参数。"""

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

    @dataclass
    class VizCfg:
        """速度跟踪调试可视化参数。"""

        z_offset: float = 0.25
        scale: float = 0.8
        command_color: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.65)
        actual_color: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.75)
        width: float = 0.015

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> BasicCommandTerm:
        return BasicCommandTerm(self, env)


class BasicCommandTerm(CommandTerm):
    """基础指令项。

    指令维度: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    """

    cfg: BasicCommandCfg

    def __init__(self, cfg: BasicCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 5 维指令: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self._standing_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._robot = env.scene["robot"]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """为指定环境重新采样指令。"""
        n = len(env_ids)

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
        if len(standing_ids) > 0:
            standing_height = (
                torch.rand(len(standing_ids), device=self.device)
                * (self.cfg.standing_height_range[1] - self.cfg.standing_height_range[0])
                + self.cfg.standing_height_range[0]
            )
            self._command[standing_ids, 4] = standing_height

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
            height = (
                torch.rand(len(moving_ids), device=self.device)
                * (self.cfg.height_range[1] - self.cfg.height_range[0])
                + self.cfg.height_range[0]
            )

            self._command[moving_ids, 0] = lin_vel
            self._command[moving_ids, 1] = yaw_vel
            self._command[moving_ids, 2] = pitch
            self._command[moving_ids, 3] = roll
            self._command[moving_ids, 4] = height

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
        if not hasattr(self._env, "extras") or not isinstance(self._env.extras.get("log"), dict):
            return

        lin_vel_b = self._robot.data.root_link_lin_vel_b
        ang_vel_b = self._robot.data.root_link_ang_vel_b
        cmd = self._command

        moving = torch.abs(cmd[:, 0]) > self.cfg.lin_vel_deadband
        yawing = torch.abs(cmd[:, 1]) > self.cfg.yaw_deadband
        active = moving | yawing

        vx_error = lin_vel_b[:, 0] - cmd[:, 0]
        yaw_error = ang_vel_b[:, 2] - cmd[:, 1]
        lateral_vel_abs = torch.abs(lin_vel_b[:, 1])
        vz_abs = torch.abs(lin_vel_b[:, 2])

        pg = self._robot.data.projected_gravity_b
        pitch = torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0))
        roll = torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0))
        pitch_error = pitch - cmd[:, 2]
        roll_error = roll - cmd[:, 3]

        log = self._env.extras["log"]
        log.update(
            {
                "Command/diag_active_ratio": float(active.float().mean().item()),
                "Command/diag_standing_ratio": float(self._standing_mask.float().mean().item()),
                "Command/diag_cmd_vx_abs": float(torch.abs(cmd[:, 0]).mean().item()),
                "Command/diag_actual_vx": float(lin_vel_b[:, 0].mean().item()),
                "Command/diag_vx_error_abs": float(torch.abs(vx_error).mean().item()),
                "Command/diag_vx_error_abs_active": _mean_on_mask(torch.abs(vx_error), active),
                "Command/diag_cmd_yaw_abs": float(torch.abs(cmd[:, 1]).mean().item()),
                "Command/diag_actual_yaw": float(ang_vel_b[:, 2].mean().item()),
                "Command/diag_yaw_error_abs": float(torch.abs(yaw_error).mean().item()),
                "Command/diag_yaw_error_abs_active": _mean_on_mask(torch.abs(yaw_error), active),
                "Command/diag_lateral_vel_abs": float(lateral_vel_abs.mean().item()),
                "Command/diag_vz_abs": float(vz_abs.mean().item()),
                "Command/diag_pitch_error_abs_deg": float(
                    torch.rad2deg(torch.abs(pitch_error)).mean().item()
                ),
                "Command/diag_roll_error_abs_deg": float(
                    torch.rad2deg(torch.abs(roll_error)).mean().item()
                ),
            }
        )

        if cmd.shape[1] <= 8:
            return

        mode_ids = cmd[:, 8].round().long()
        mode_names = ("wheel", "gait", "wheel_leg", "gait_wheel", "jump")
        for mode_id, mode_name in enumerate(mode_names):
            mode_mask = mode_ids == mode_id
            mode_active = mode_mask & active
            log[f"TaskMode/diag_{mode_name}_vx_error_abs"] = _mean_on_mask(
                torch.abs(vx_error), mode_mask
            )
            log[f"TaskMode/diag_{mode_name}_vx_error_abs_active"] = _mean_on_mask(
                torch.abs(vx_error), mode_active
            )
            log[f"TaskMode/diag_{mode_name}_yaw_error_abs"] = _mean_on_mask(
                torch.abs(yaw_error), mode_mask
            )
            log[f"TaskMode/diag_{mode_name}_lateral_vel_abs"] = _mean_on_mask(
                lateral_vel_abs, mode_mask
            )

        self._update_gait_terrain_metrics(
            log=log,
            terrain_types=_terrain_types(self._env),
            gait_active=(mode_ids == int(TaskMode.GAIT)) & active,
            vx_error_abs=torch.abs(vx_error),
            yaw_error_abs=torch.abs(yaw_error),
            lateral_vel_abs=lateral_vel_abs,
            pitch_error_abs_deg=torch.rad2deg(torch.abs(pitch_error)),
            roll_error_abs_deg=torch.rad2deg(torch.abs(roll_error)),
            cmd_vx_abs=torch.abs(cmd[:, 0]),
            actual_vx=lin_vel_b[:, 0],
        )

    def _update_gait_terrain_metrics(
        self,
        log: dict,
        terrain_types: torch.Tensor | None,
        gait_active: torch.Tensor,
        vx_error_abs: torch.Tensor,
        yaw_error_abs: torch.Tensor,
        lateral_vel_abs: torch.Tensor,
        pitch_error_abs_deg: torch.Tensor,
        roll_error_abs_deg: torch.Tensor,
        cmd_vx_abs: torch.Tensor,
        actual_vx: torch.Tensor,
    ) -> None:
        """按 GAIT fine-tune 地形类型拆分速度和姿态误差。"""
        if terrain_types is None:
            return

        for terrain_id, terrain_name in _GAIT_TERRAIN_TYPES:
            terrain_mask = gait_active & (terrain_types == terrain_id)
            prefix = f"TaskMode/diag_gait_terrain_{terrain_name}"
            log[f"{prefix}_ratio"] = _ratio_on_mask(terrain_mask, gait_active)
            log[f"{prefix}_cmd_vx_abs"] = _mean_on_mask(cmd_vx_abs, terrain_mask)
            log[f"{prefix}_actual_vx"] = _mean_on_mask(actual_vx, terrain_mask)
            log[f"{prefix}_vx_error_abs_active"] = _mean_on_mask(vx_error_abs, terrain_mask)
            log[f"{prefix}_yaw_error_abs"] = _mean_on_mask(yaw_error_abs, terrain_mask)
            log[f"{prefix}_lateral_vel_abs"] = _mean_on_mask(lateral_vel_abs, terrain_mask)
            log[f"{prefix}_pitch_error_abs_deg"] = _mean_on_mask(pitch_error_abs_deg, terrain_mask)
            log[f"{prefix}_roll_error_abs_deg"] = _mean_on_mask(roll_error_abs_deg, terrain_mask)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """绘制期望线速度和实际线速度箭头。"""
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
            command_vec_b = np.array([commands[env_idx, 0], 0.0, 0.0]) * self.cfg.viz.scale
            actual_vec_b = (
                np.array([lin_vel_bs[env_idx, 0], lin_vel_bs[env_idx, 1], 0.0]) * self.cfg.viz.scale
            )

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


@dataclass
class VelocityHeightCommandCfg(BasicCommandCfg):
    """速度 + 姿态 + 高度指令生成器的配置。"""

    def build(self, env: ManagerBasedRlEnv) -> VelocityHeightCommandTerm:
        return VelocityHeightCommandTerm(self, env)


class VelocityHeightCommandTerm(BasicCommandTerm):
    """普通速度 + 姿态 + 高度指令项。"""

    cfg: VelocityHeightCommandCfg
