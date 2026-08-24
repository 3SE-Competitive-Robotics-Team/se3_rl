"""为 MJLab 原生 DC 电机包络补充训练期越界统计。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator import DcMotorActuator, DcMotorActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


class _TnViolationAccumulatorMixin:
    """在 actuator 限幅前累计每个 physics substep 的 TN 包络峰值越界。"""

    def _initialize_tn_violation_accumulators(self) -> None:
        self._tn_violation_accumulators: dict[int, tuple[float, torch.Tensor]] = {}

    def register_tn_violation_accumulator(self, key: int, safe_tn_ratio: float) -> None:
        """注册一个按关节保存峰值越界的独立缓冲区。"""
        ratio = float(safe_tn_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"safe_tn_ratio 必须位于 (0, 1]，实际为 {ratio}")
        if key in self._tn_violation_accumulators:
            raise KeyError(f"TN 越界累积器 key={key} 已注册")

        force_limit = getattr(self, "force_limit", None)
        if not isinstance(force_limit, torch.Tensor):
            raise RuntimeError("actuator 尚未初始化 force_limit，无法注册 TN 越界累积器")
        self._tn_violation_accumulators[key] = (ratio, torch.zeros_like(force_limit))

    def consume_tn_violation_accumulator(self, key: int) -> torch.Tensor:
        """返回并清空一个 policy step 内累计的逐关节峰值越界。"""
        try:
            _, peak_violation = self._tn_violation_accumulators[key]
        except KeyError as exc:
            raise KeyError(f"TN 越界累积器 key={key} 未注册") from exc
        result = peak_violation.clone()
        peak_violation.zero_()
        return result

    def reset_tn_violation_accumulator(
        self,
        key: int,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """清空指定环境的一个 TN 越界累积器。"""
        try:
            _, peak_violation = self._tn_violation_accumulators[key]
        except KeyError as exc:
            raise KeyError(f"TN 越界累积器 key={key} 未注册") from exc
        peak_violation[slice(None) if env_ids is None else env_ids] = 0.0

    def _reset_all_tn_violation_accumulators(
        self,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        selected = slice(None) if env_ids is None else env_ids
        for _, peak_violation in self._tn_violation_accumulators.values():
            peak_violation[selected] = 0.0

    def _accumulate_tn_violation(
        self,
        effort: torch.Tensor,
        lower_effort: torch.Tensor,
        upper_effort: torch.Tensor,
    ) -> None:
        """按实际有符号上下界累计限幅前请求扭矩的越界量。"""
        for safe_tn_ratio, peak_violation in self._tn_violation_accumulators.values():
            # 包络跨过零点时，上下界按比例向零收缩；超速导致包络位于单侧时，
            # 收缩后的安全区仍严格落在物理可实现区间内。
            safe_lower = torch.maximum(lower_effort * safe_tn_ratio, lower_effort)
            safe_upper = torch.minimum(upper_effort * safe_tn_ratio, upper_effort)
            violation = torch.relu(effort - safe_upper) + torch.relu(safe_lower - effort)
            peak_violation.copy_(torch.maximum(peak_violation, violation))


@dataclass(kw_only=True)
class TnTrackedDcMotorActuatorCfg(DcMotorActuatorCfg):
    """保持 MJLab DC 电机物理包络并暴露 TN 越界统计的配置。"""

    def build(
        self,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> TnTrackedDcMotorActuator:
        return TnTrackedDcMotorActuator(self, entity, target_ids, target_names)


class TnTrackedDcMotorActuator(
    DcMotorActuator[TnTrackedDcMotorActuatorCfg],
    _TnViolationAccumulatorMixin,
):
    """带 physics-substep TN 越界统计的四象限线性 DC 电机。"""

    def __init__(
        self,
        cfg: TnTrackedDcMotorActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._initialize_tn_violation_accumulators()

    def effort_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回与 MJLab DC actuator 一致的有符号瞬时扭矩上下界。"""
        assert self.saturation_effort is not None
        assert self.velocity_limit_motor is not None
        assert self.force_limit is not None
        assert self._vel_at_effort_lim is not None
        assert self._joint_vel_clipped is not None

        velocity = torch.clamp(
            self._joint_vel_clipped,
            min=-self._vel_at_effort_lim,
            max=self._vel_at_effort_lim,
        )
        upper = self.saturation_effort * (1.0 - velocity / self.velocity_limit_motor)
        lower = self.saturation_effort * (-1.0 - velocity / self.velocity_limit_motor)
        return (
            torch.clamp(lower, min=-self.force_limit),
            torch.clamp(upper, max=self.force_limit),
        )

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        lower_effort, upper_effort = self.effort_bounds()
        self._accumulate_tn_violation(effort, lower_effort, upper_effort)
        return super()._clip_effort(effort)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._reset_all_tn_violation_accumulators(env_ids)
