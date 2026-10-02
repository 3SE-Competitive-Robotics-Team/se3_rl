"""跳跃 mimic 指令项：8 维指令 + 参考时钟。

指令布局与部署契约一致：[lin_vel_x, ang_vel_yaw, pitch, roll, height, jump_flag, jump_target_height, jump_phase]。
jump_phase 默认恒 0（J1 用户定：不把相位输入网络，参考进度靠参考帧观测）；J10 起可设 phase_time_scale_s，
跳跃中 jump_phase = 参考时刻 / 该常数（J8 证明网络必须有时间信息，J10 试一维相位代替 20 维参考帧）。参考时钟只在环境内部，按时间推进：
触发时从参考第 0 帧开始，每个 policy step 前进 step_dt，播完回到站立并进入冷却。
参考状态初始化（RSI）由 reset 事件调用 `start_reference` 预置，command reset 时保留（事件先于指令 reset）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from se3_train.mdp.commands import VelocityHeightCommandCfg, VelocityHeightCommandTerm

from .reference import DEFAULT_REFERENCE_PATHS, JumpReferenceLibrary, ReferenceFrame

PLAYBACK_END_TOLERANCE_S = 1.0e-6
"""参考播完判定容差（与 se3_runtime.jump_reference 同值）。"""

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class JumpMimicCommandCfg(VelocityHeightCommandCfg):
    """跳跃 mimic 指令配置。"""

    reference_paths: tuple[str, ...] = field(default_factory=lambda: DEFAULT_REFERENCE_PATHS)
    min_idle_s: float = 1.0
    """回合开始或上一跳结束后至少站这么久才允许触发。"""
    trigger_rate_hz: float = 0.5
    """满足条件后每秒触发概率（泊松近似）。"""
    phase_time_scale_s: float | None = None
    """J10：jump_phase = 触发后参考时刻 / 该常数（跳跃中），给网络一维时间信息；None 时 jump_phase 恒 0（J1–J9）。
    用固定常数而不是各条参考时长归一化，使同一相位值对应同一物理时刻（起蹬都在 0）。"""

    def build(self, env: ManagerBasedRlEnv) -> JumpMimicCommandTerm:
        return JumpMimicCommandTerm(self, env)


class JumpMimicCommandTerm(VelocityHeightCommandTerm):
    """在速度 + 高度指令上叠加跳跃触发与参考时钟。"""

    cfg: JumpMimicCommandCfg

    def __init__(self, cfg: JumpMimicCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 8, device=self.device)
        self.library = JumpReferenceLibrary(cfg.reference_paths, self.device)
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ref_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ref_t = torch.zeros(self.num_envs, device=self.device)
        self.idle_t = torch.zeros(self.num_envs, device=self.device)
        self._preset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ---- 参考读取 ----
    def reference(self, offset_s: float = 0.0) -> ReferenceFrame:
        """当前参考时刻 + offset 的参考帧；未在跳的 env 取站姿帧（第 0 帧）。"""
        t = torch.where(self.active, self.ref_t + float(offset_s), torch.zeros_like(self.ref_t))
        return self.library.frame(self.ref_id, t)

    def start_reference(self, env_ids: torch.Tensor, ref_id: torch.Tensor, t: torch.Tensor) -> None:
        """RSI：把指定 env 置为"正在播放 ref_id 的 t 时刻"，下一次 command reset 保留该状态。"""
        self.active[env_ids] = True
        self.ref_id[env_ids] = ref_id
        self.ref_t[env_ids] = t
        self.idle_t[env_ids] = 0.0
        self._preset[env_ids] = True
        self._write_jump_dims()

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
        fresh = env_ids[~self._preset[env_ids]]
        self.active[fresh] = False
        self.ref_t[fresh] = 0.0
        self.idle_t[fresh] = 0.0
        self._preset[env_ids] = False
        self._write_jump_dims()
        return extras

    def _update_command(self) -> None:
        super()._update_command()
        dt = float(self._env.step_dt)
        self.ref_t = torch.where(self.active, self.ref_t + dt, self.ref_t)
        # 留容差：float32 累加 73 × 0.02 = 1.4599999 < 1.46，否则比 runtime（float64）多播一步
        finished = self.active & (
            self.ref_t >= self.library.duration[self.ref_id] - PLAYBACK_END_TOLERANCE_S
        )
        self.active = self.active & ~finished
        self.ref_t = torch.where(finished, torch.zeros_like(self.ref_t), self.ref_t)
        self.idle_t = torch.where(self.active, torch.zeros_like(self.idle_t), self.idle_t + dt)
        self.idle_t = torch.where(finished, torch.zeros_like(self.idle_t), self.idle_t)
        eligible = (~self.active) & (self.idle_t >= float(self.cfg.min_idle_s))
        fire = eligible & (
            torch.rand(self.num_envs, device=self.device) < self.cfg.trigger_rate_hz * dt
        )
        if bool(fire.any()):
            ids = fire.nonzero().flatten()
            self.active[ids] = True
            self.ref_t[ids] = 0.0
            self.ref_id[ids] = torch.randint(
                0, self.library.num_refs, (len(ids),), device=self.device
            )
        self._write_jump_dims()

    def deployment_jump_reference(self) -> dict:
        """部署契约顶层 jump_reference：参考数据 + 观测帧偏移与缩放（se3_runtime.jump_reference 解析）。"""
        from .mdp import DEFAULT_OFFSETS_STEPS, REFERENCE_SCALES

        lib = self.library
        references = []
        for k in range(lib.num_refs):
            n = int(lib.length[k])
            references.append(
                {
                    "target_clearance": float(lib.target_clearance[k]),
                    "leg_len": lib.leg_len[k, :n, 0].tolist(),
                    "base_z": lib.base_z[k, :n].tolist(),
                    "base_vz": lib.base_vz[k, :n].tolist(),
                    "contact": lib.contact[k, :n].tolist(),
                }
            )
        payload = {
            "format": "se3.jump_ref.v1",
            "dt": lib.dt,
            "stand_height": lib.stand_height,
            "offsets_steps": list(DEFAULT_OFFSETS_STEPS),
            "scales": dict(REFERENCE_SCALES),
            "references": references,
        }
        if self.cfg.phase_time_scale_s is not None:
            payload["phase_time_scale_s"] = float(self.cfg.phase_time_scale_s)
        return payload

    def _write_jump_dims(self) -> None:
        self._command[:, 5] = self.active.float()
        self._command[:, 6] = self.library.target_clearance[self.ref_id] * self.active.float()
        scale = self.cfg.phase_time_scale_s
        if scale is None:
            self._command[:, 7] = 0.0
        else:
            self._command[:, 7] = (self.ref_t / float(scale)) * self.active.float()

    def _update_metrics(self) -> None:
        super()._update_metrics()
        log = self._env.extras.setdefault("log", {}) if hasattr(self._env, "extras") else None
        if isinstance(log, dict):
            log["Jump/active_rate"] = self.active.float().mean()


__all__ = ["JumpMimicCommandCfg", "JumpMimicCommandTerm"]
