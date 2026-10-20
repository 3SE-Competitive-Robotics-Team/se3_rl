"""训练端自定义执行器。

LaggedDcMotorActuator（2026-10-15）：在 mjlab DcMotorActuator 的输出上串一阶滞后，表示实车轮通道"力矩指令 → 地面实际推力"
之间仿真没有的环内滞后（电调电流环约 3–7 ms + 电流之后的传动链）。实车依据：只加 5–15 ms 轮滞后训练的策略
上车不再出现 8 Hz 静站极限环，3–10 ms 版仍振（2026-10-15/16 采集）；2026-10-20 起 Rough 系默认开启。控制律（速度环 + T-N 截顶）与 DcMotorActuator 逐位相同，
只在其后做 y ← y + α·(τ − y)，α = 1 − exp(−dt/τ_lag)，τ_lag 每 env 在 reset 时于 lag_tau_range_s 内均匀采样。

compute 可能在一个物理步内被调用多次（reset 前后的 write_data_to_sim），所以 compute 只算候选输出，
update(dt)（每个 sim.step 之后恰好一次）才提交滤波状态，保证每个物理步只推进一次。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab.entity import Entity


@dataclass(kw_only=True)
class LaggedDcMotorActuatorCfg(DcMotorActuatorCfg):
    """DcMotorActuatorCfg + 输出一阶滞后时间常数范围（s）。"""

    lag_tau_range_s: tuple[float, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        low, high = (float(v) for v in self.lag_tau_range_s)
        if not 0.0 < low <= high:
            raise ValueError(
                f"lag_tau_range_s 必须满足 0 < 下限 ≤ 上限，实际为 {self.lag_tau_range_s}"
            )

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> LaggedDcMotorActuator:
        return LaggedDcMotorActuator(self, entity, target_ids, target_names)


class LaggedDcMotorActuator(DcMotorActuator[LaggedDcMotorActuatorCfg]):
    """输出带一阶滞后的 DC 电机执行器；重写 compute，因此不参与 mjlab 的执行器融合。"""

    def __init__(
        self,
        cfg: LaggedDcMotorActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._physics_dt: float | None = None
        self.lag_tau: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._lagged: torch.Tensor | None = None
        self._pending: torch.Tensor | None = None

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_names))
        self._physics_dt = float(mj_model.opt.timestep)
        low, high = (float(v) for v in self.cfg.lag_tau_range_s)
        self.lag_tau = torch.full(shape, 0.5 * (low + high), device=device)
        self._alpha = torch.empty(shape, device=device)
        self._update_alpha(slice(None))
        self._lagged = torch.zeros(shape, device=device)
        self._pending = torch.zeros(shape, device=device)

    def _update_alpha(self, env_ids: torch.Tensor | slice) -> None:
        assert self.lag_tau is not None and self._alpha is not None
        self._alpha[env_ids] = 1.0 - torch.exp(-self._physics_dt / self.lag_tau[env_ids])

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self._alpha is not None and self._lagged is not None and self._pending is not None
        params = {name: getattr(self, name) for name in self.param_names}
        torque = type(self).control_law(params, cmd)
        torch.addcmul(self._lagged, self._alpha, torque - self._lagged, out=self._pending)
        return self._pending

    def update(self, dt: float) -> None:
        super().update(dt)
        assert self._lagged is not None and self._pending is not None
        if not math.isclose(dt, self._physics_dt, rel_tol=1e-6):
            raise RuntimeError(f"物理步长 {dt} 与初始化时的 {self._physics_dt} 不同，滞后系数失效")
        self._lagged.copy_(self._pending)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self.lag_tau is None:
            return
        assert self._alpha is not None and self._lagged is not None and self._pending is not None
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.lag_tau.shape[0], device=self.lag_tau.device)[
                env_ids if isinstance(env_ids, slice) else slice(None)
            ]
        else:
            ids = env_ids.to(device=self.lag_tau.device, dtype=torch.long)
        low, high = (float(v) for v in self.cfg.lag_tau_range_s)
        # 同一 env 的左右轮共用一个时间常数（同型号电调与传动链）；index_copy_/index_fill_ 不触发主机同步。
        tau = low + (high - low) * torch.rand(ids.numel(), 1, device=self.lag_tau.device)
        tau = tau.expand(-1, self.lag_tau.shape[1])
        self.lag_tau.index_copy_(0, ids, tau)
        self._alpha.index_copy_(0, ids, 1.0 - torch.exp(-self._physics_dt / tau))
        self._lagged.index_fill_(0, ids, 0.0)
        self._pending.index_fill_(0, ids, 0.0)
