"""跳跃参考轨迹运行时缓存。

训练中的 reset、command 和 reward 必须读取同一份轨迹数据，避免相位、
初始状态和 tracking 目标各自漂移。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

DEFAULT_JUMP_TRAJ_PATHS = (
    "assets/trajectories/jump_0.1m.npz",
    "assets/trajectories/jump_0.2m.npz",
    "assets/trajectories/jump_0.3m.npz",
    "assets/trajectories/jump_0.4m.npz",
    "assets/trajectories/jump_0.5m.npz",
    "assets/trajectories/jump_0.6m.npz",
)
DEFAULT_JUMP_TRAJ_HEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


class JumpTrajectory:
    """单条参考轨迹在指定 device 上的 tensor 缓存。"""

    def __init__(self, traj_path: str, target_height: float, device: str) -> None:
        path = Path(traj_path)
        if not path.exists():
            raise FileNotFoundError(f"参考轨迹文件不存在: {traj_path}")

        data = np.load(str(path))
        required_fields = ("base_pos", "base_vel", "q_ref", "q_vel", "t_stance", "t_land", "dt")
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise RuntimeError(f"轨迹文件 {traj_path} 缺少字段: {', '.join(missing)}")

        base_vel_np = np.asarray(data["base_vel"], dtype=np.float32)
        self.base_pos = torch.tensor(data["base_pos"], dtype=torch.float32, device=device)
        self.base_vel = torch.tensor(base_vel_np, dtype=torch.float32, device=device)
        self.q_ref = torch.tensor(data["q_ref"], dtype=torch.float32, device=device)
        self.q_vel = torch.tensor(data["q_vel"], dtype=torch.float32, device=device)
        self.n_steps = int(self.base_pos.shape[0])
        self.dt = float(data["dt"])
        self.t_stance = float(data["t_stance"])
        self.t_land = float(data["t_land"])
        self.n_stance = max(1, min(round(self.t_stance / self.dt), self.n_steps))
        self.n_land = max(self.n_stance + 1, min(round(self.t_land / self.dt), self.n_steps - 1))
        self.target_height = float(target_height)

        stage = torch.ones(self.n_steps, dtype=torch.long, device=device)
        stage[: self.n_stance] = 0
        stage[self.n_land :] = 2
        self.stage_by_step = stage
        self._base_vz_np = base_vel_np[:, 2]
        self._first_vz_above_step_cache: dict[float, int] = {}

    def first_vz_above_step(self, min_vz: float) -> int:
        """返回参考 base vz 首次超过阈值的帧号。"""
        key = float(min_vz)
        cached = self._first_vz_above_step_cache.get(key)
        if cached is not None:
            return cached
        above = np.nonzero(self._base_vz_np > key)[0]
        step = int(above[0]) if above.size > 0 else self.n_stance
        self._first_vz_above_step_cache[key] = step
        return step

    def takeoff_window_start_step(self, min_vz: float, min_window_steps: int) -> int:
        """返回起跳奖励窗口起点，保证地面末段至少覆盖指定帧数。"""
        vz_start = self.first_vz_above_step(min_vz)
        window_start = max(0, self.n_stance - max(1, int(min_window_steps)))
        return min(vz_start, window_start)

    def get_step(
        self, step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量读取参考帧。"""
        idx = step.clamp(0, self.n_steps - 1)
        return (
            self.base_pos[idx],
            self.base_vel[idx],
            self.q_ref[idx],
            self.q_vel[idx],
            self.stage_by_step[idx],
        )


class JumpTrajLibrary:
    """多高度参考轨迹库，按 jump_target_height 选择最近轨迹。"""

    _instances: ClassVar[dict[str, JumpTrajLibrary]] = {}

    @classmethod
    def get(
        cls,
        traj_paths: tuple[str, ...] = DEFAULT_JUMP_TRAJ_PATHS,
        traj_target_heights: tuple[float, ...] = DEFAULT_JUMP_TRAJ_HEIGHTS,
        device: str = "cpu",
    ) -> JumpTrajLibrary:
        key = f"{'|'.join(traj_paths)}:{','.join(map(str, traj_target_heights))}:{device}"
        if key not in cls._instances:
            cls._instances[key] = cls(traj_paths, traj_target_heights, device)
        return cls._instances[key]

    def __init__(
        self,
        traj_paths: tuple[str, ...],
        traj_target_heights: tuple[float, ...],
        device: str,
    ) -> None:
        if len(traj_paths) != len(traj_target_heights):
            raise ValueError("traj_paths 与 traj_target_heights 长度必须一致")

        self.trajs = [
            JumpTrajectory(path, height, device)
            for path, height in zip(traj_paths, traj_target_heights, strict=True)
        ]
        self.heights = torch.tensor(traj_target_heights, dtype=torch.float32, device=device)
        self.n_steps = torch.tensor(
            [traj.n_steps for traj in self.trajs], dtype=torch.long, device=device
        )
        self.n_stance = torch.tensor(
            [traj.n_stance for traj in self.trajs], dtype=torch.long, device=device
        )
        self.n_land = torch.tensor(
            [traj.n_land for traj in self.trajs], dtype=torch.long, device=device
        )

    def nearest_index(self, target_height: torch.Tensor) -> torch.Tensor:
        """返回每个 env 最接近目标高度的轨迹索引。"""
        distance = torch.abs(target_height[:, None] - self.heights[None, :])
        return torch.argmin(distance, dim=1)

    def n_steps_for(self, target_height: torch.Tensor) -> torch.Tensor:
        """返回每个目标高度对应轨迹的总帧数。"""
        return self.n_steps[self.nearest_index(target_height)]

    def stage_for(self, target_height: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """返回每个目标高度和帧号对应的 reference stage。"""
        idx = self.nearest_index(target_height)
        stage = torch.zeros_like(step)
        for traj_idx, traj in enumerate(self.trajs):
            mask = idx == traj_idx
            if not mask.any():
                continue
            stage[mask] = traj.stage_by_step[step[mask].clamp(0, traj.n_steps - 1)]
        return stage

    def first_vz_above_step_for(self, target_height: torch.Tensor, min_vz: float) -> torch.Tensor:
        """返回参考 base vz 首次超过阈值的帧号。"""
        idx = self.nearest_index(target_height)
        first_step = torch.zeros(
            target_height.shape[0], dtype=torch.long, device=target_height.device
        )
        for traj_idx, traj in enumerate(self.trajs):
            mask = idx == traj_idx
            if not mask.any():
                continue
            first_step[mask] = traj.first_vz_above_step(min_vz)
        return first_step

    def takeoff_window_start_step_for(
        self, target_height: torch.Tensor, min_vz: float, min_window_steps: int
    ) -> torch.Tensor:
        """返回参考起跳奖励窗口起点，至少覆盖 stance 末端 min_window_steps 帧。"""
        idx = self.nearest_index(target_height)
        start_step = torch.zeros(
            target_height.shape[0], dtype=torch.long, device=target_height.device
        )
        for traj_idx, traj in enumerate(self.trajs):
            mask = idx == traj_idx
            if not mask.any():
                continue
            start_step[mask] = traj.takeoff_window_start_step(min_vz, min_window_steps)
        return start_step

    def gather(
        self,
        target_height: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """按目标高度批量取参考帧。"""
        idx = self.nearest_index(target_height)
        ref_pos = torch.zeros((target_height.shape[0], 3), device=target_height.device)
        ref_vel = torch.zeros((target_height.shape[0], 3), device=target_height.device)
        ref_q = torch.zeros((target_height.shape[0], 6), device=target_height.device)
        ref_q_vel = torch.zeros((target_height.shape[0], 6), device=target_height.device)
        ref_stage = torch.zeros(
            target_height.shape[0], dtype=torch.long, device=target_height.device
        )
        ref_height = self.heights[idx]

        for traj_idx, traj in enumerate(self.trajs):
            mask = idx == traj_idx
            if not mask.any():
                continue
            pos_i, vel_i, q_i, q_vel_i, stage_i = traj.get_step(step[mask])
            ref_pos[mask] = pos_i
            ref_vel[mask] = vel_i
            ref_q[mask] = q_i
            ref_q_vel[mask] = q_vel_i
            ref_stage[mask] = stage_i

        return ref_pos, ref_vel, ref_q, ref_q_vel, ref_height, ref_stage
