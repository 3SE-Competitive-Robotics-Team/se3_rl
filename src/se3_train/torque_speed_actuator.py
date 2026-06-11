"""支持实测非线性 T-N 包络的 MJLab PD actuator。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class TorqueSpeedCurveActuatorCfg(IdealPdActuatorCfg):
    """使用分段线性 ``|速度| -> 最大扭矩`` 包络的 actuator 配置。"""

    torque_speed_curve: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.torque_speed_curve) < 2:
            raise ValueError("torque_speed_curve 至少需要两个点")
        speeds = [float(point[0]) for point in self.torque_speed_curve]
        torques = [float(point[1]) for point in self.torque_speed_curve]
        if any(speed < 0.0 for speed in speeds):
            raise ValueError("torque_speed_curve 速度必须非负")
        if any(torque < 0.0 for torque in torques):
            raise ValueError("torque_speed_curve 扭矩必须非负")
        if any(next_speed <= speed for speed, next_speed in pairwise(speeds)):
            raise ValueError("torque_speed_curve 速度必须严格递增")
        if any(next_torque > torque for torque, next_torque in pairwise(torques)):
            raise ValueError("torque_speed_curve 扭矩必须单调不增")

    def build(
        self,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> TorqueSpeedCurveActuator:
        return TorqueSpeedCurveActuator(self, entity, target_ids, target_names)


class TorqueSpeedCurveActuator(IdealPdActuator[TorqueSpeedCurveActuatorCfg]):
    """在 MJLab 中按实测 T-N 曲线限制 PD 输出扭矩。"""

    def __init__(
        self,
        cfg: TorqueSpeedCurveActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._curve_speed: torch.Tensor | None = None
        self._curve_torque: torch.Tensor | None = None
        self._joint_vel: torch.Tensor | None = None

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        curve = torch.tensor(self.cfg.torque_speed_curve, dtype=torch.float, device=device)
        self._curve_speed = curve[:, 0].contiguous()
        self._curve_torque = curve[:, 1].contiguous()
        self._joint_vel = torch.zeros(
            data.nworld,
            len(self._target_names),
            dtype=torch.float,
            device=device,
        )

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self._joint_vel is not None
        self._joint_vel[:] = cmd.vel
        return super().compute(cmd)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        assert self._curve_speed is not None
        assert self._curve_torque is not None
        assert self._joint_vel is not None
        assert self.force_limit is not None

        speed = torch.abs(self._joint_vel).contiguous()
        upper = torch.searchsorted(self._curve_speed, speed).clamp(
            min=1,
            max=self._curve_speed.numel() - 1,
        )
        lower = upper - 1
        speed_lower = self._curve_speed[lower]
        speed_upper = self._curve_speed[upper]
        torque_lower = self._curve_torque[lower]
        torque_upper = self._curve_torque[upper]
        ratio = (speed - speed_lower) / (speed_upper - speed_lower)
        torque_limit = torque_lower + ratio * (torque_upper - torque_lower)
        torque_limit = torch.where(
            speed <= self._curve_speed[0],
            self._curve_torque[0],
            torque_limit,
        )
        torque_limit = torch.where(
            speed >= self._curve_speed[-1],
            self._curve_torque[-1],
            torque_limit,
        )
        torque_limit = torch.minimum(torque_limit, self.force_limit)
        return torch.clamp(effort, min=-torque_limit, max=torque_limit)
