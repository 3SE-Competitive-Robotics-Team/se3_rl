"""跳跃 mimic 指令项：8 维指令 + 参考时钟。

指令布局与部署契约一致：[lin_vel_x, ang_vel_yaw, pitch, roll, height, jump_flag, jump_target_height, jump_phase]。
跳跃中 jump_phase = 参考时刻 / phase_time_scale_s（J8 证明网络必须有时间信息，J10 证明一维相位可代替 20 维参考帧）。
参考时钟见 clock.JumpReferenceClock（rough 跳跃合入共用同一份）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from se3_train.mdp.commands import VelocityHeightCommandCfg, VelocityHeightCommandTerm

from .clock import JumpReferenceClock
from .reference import JumpReferenceLibrary, ReferenceFrame

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class JumpMimicCommandCfg(VelocityHeightCommandCfg):
    """跳跃 mimic 指令配置。"""

    reference_paths: tuple[str, ...] = ()
    """参考文件路径（env_cfg 按目标离地间隙给出）。"""
    min_idle_s: float = 1.0
    """回合开始或上一跳结束后至少站这么久才允许触发。"""
    trigger_rate_hz: float = 0.5
    """满足条件后每秒触发概率（泊松近似）。"""
    phase_time_scale_s: float = 1.5
    """jump_phase = 触发后参考时刻 / 该常数（跳跃中），给网络一维时间信息（见 env_cfg.JUMP_MIMIC_PHASE_TIME_SCALE_S）。"""

    def build(self, env: ManagerBasedRlEnv) -> JumpMimicCommandTerm:
        return JumpMimicCommandTerm(self, env)


class JumpMimicCommandTerm(VelocityHeightCommandTerm):
    """在速度 + 高度指令上叠加跳跃触发与参考时钟（时钟逻辑见 clock.JumpReferenceClock）。"""

    cfg: JumpMimicCommandCfg

    def __init__(self, cfg: JumpMimicCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 8, device=self.device)
        self.clock = JumpReferenceClock(
            JumpReferenceLibrary(cfg.reference_paths, self.device),
            self.num_envs,
            self.device,
            min_idle_s=cfg.min_idle_s,
            trigger_rate_hz=cfg.trigger_rate_hz,
            phase_time_scale_s=cfg.phase_time_scale_s,
        )

    # ---- 时钟状态（奖励 / 终止 / reset 事件读取） ----
    @property
    def library(self) -> JumpReferenceLibrary:
        return self.clock.library

    @property
    def active(self) -> torch.Tensor:
        return self.clock.active

    @property
    def ref_id(self) -> torch.Tensor:
        return self.clock.ref_id

    @property
    def ref_t(self) -> torch.Tensor:
        return self.clock.ref_t

    def reference(self, offset_s: float = 0.0) -> ReferenceFrame:
        """当前参考时刻 + offset 的参考帧；未在跳的 env 取站姿帧（第 0 帧）。"""
        return self.clock.reference(offset_s)

    # ---- 生命周期 ----
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """跳跃中的 env 不换速度 / 姿态指令（腾空时水平速度改不了，换了只会制造无法完成的指令）；reset 照常重采样。"""
        if bool(getattr(self, "_resampling_for_reset", False)):
            super()._resample_command(env_ids)
            return
        keep = env_ids[self.active[env_ids]]
        saved = self._command[keep, 0:4].clone()
        saved_standing = self._standing_mask[keep].clone()
        super()._resample_command(env_ids)
        self._command[keep, 0:4] = saved
        self._standing_mask[keep] = saved_standing

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids)
        assert isinstance(env_ids, torch.Tensor)
        self.clock.reset(env_ids)
        self._write_jump_dims()
        return extras

    def _update_command(self) -> None:
        super()._update_command()
        self.clock.step(float(self._env.step_dt))
        self._write_jump_dims()

    def deployment_jump_reference(self) -> dict:
        """部署契约顶层 jump_reference：参考数据 + 观测帧偏移与缩放（se3_runtime.jump_reference 解析）。"""
        from .mdp import DEFAULT_OFFSETS_STEPS, REFERENCE_SCALES

        return self.clock.deployment_payload(DEFAULT_OFFSETS_STEPS, REFERENCE_SCALES)

    def _write_jump_dims(self) -> None:
        self.clock.write_dims(self._command)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        log = self._env.extras.setdefault("log", {}) if hasattr(self._env, "extras") else None
        if isinstance(log, dict):
            log["Jump/active_rate"] = self.active.float().mean()


__all__ = ["JumpMimicCommandCfg", "JumpMimicCommandTerm"]
