"""跳跃参考时钟：触发、按时间推进参考、写 8 维指令的跳跃三维、导出部署参考（J 系列与 rough 共用）。

时钟只在环境内部，按时间推进：触发时从参考第 0 帧开始，每个 policy step 前进 step_dt，播完回到站立并重新累计站立
时长。未在跳的 env 取站姿帧（第 0 帧）。跳跃中 jump_phase = 参考时刻 / phase_time_scale_s。
"""

from __future__ import annotations

import torch

from .reference import JumpReferenceLibrary, ReferenceFrame

PLAYBACK_END_TOLERANCE_S = 1.0e-6
"""参考播完判定容差（与 se3_runtime.jump_reference 同值）：float32 累加 73 × 0.02 = 1.4599999 < 1.46，
不留容差会比 runtime（float64）多播一步。"""


class JumpReferenceClock:
    """一批 env 的参考播放状态。"""

    def __init__(
        self,
        library: JumpReferenceLibrary,
        num_envs: int,
        device: torch.device | str,
        *,
        min_idle_s: float,
        trigger_rate_hz: float,
        phase_time_scale_s: float,
    ) -> None:
        self.library = library
        self.device = device
        self.min_idle_s = float(min_idle_s)
        self.trigger_rate_hz = float(trigger_rate_hz)
        self.phase_time_scale_s = float(phase_time_scale_s)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ref_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ref_t = torch.zeros(num_envs, device=device)
        self.idle_t = torch.zeros(num_envs, device=device)

    def reference(self, offset_s: float = 0.0) -> ReferenceFrame:
        """当前参考时刻 + offset 的参考帧；未在跳的 env 取站姿帧（第 0 帧）。"""
        t = torch.where(self.active, self.ref_t + float(offset_s), torch.zeros_like(self.ref_t))
        return self.library.frame(self.ref_id, t)

    def reset(self, env_ids: torch.Tensor) -> None:
        """episode reset：回到不在跳、站立时长清零。"""
        # index_fill_ 把标量当 kernel 参数传入；`x[ids] = 0.0` 会先把标量从主机拷到 GPU 而同步。
        self.active.index_fill_(0, env_ids, False)
        self.ref_t.index_fill_(0, env_ids, 0.0)
        self.idle_t.index_fill_(0, env_ids, 0.0)

    def step(self, dt: float, allowed: torch.Tensor | None = None) -> torch.Tensor:
        """推进一个 policy step；allowed 给出可以触发的 env（None 为全部），返回本步新触发的 env id。"""
        self.ref_t = torch.where(self.active, self.ref_t + dt, self.ref_t)
        finished = self.active & (
            self.ref_t >= self.library.duration[self.ref_id] - PLAYBACK_END_TOLERANCE_S
        )
        self.active = self.active & ~finished
        self.ref_t = torch.where(finished, torch.zeros_like(self.ref_t), self.ref_t)
        self.idle_t = torch.where(self.active, torch.zeros_like(self.idle_t), self.idle_t + dt)
        self.idle_t = torch.where(finished, torch.zeros_like(self.idle_t), self.idle_t)
        eligible = (~self.active) & (self.idle_t >= self.min_idle_s)
        if allowed is not None:
            eligible = eligible & allowed
        fire = eligible & (
            torch.rand(len(self.active), device=self.device) < self.trigger_rate_hz * dt
        )
        ids = fire.nonzero().flatten()
        if len(ids) > 0:
            self.active.index_fill_(0, ids, True)
            self.ref_t.index_fill_(0, ids, 0.0)
            self.ref_id[ids] = torch.randint(
                0, self.library.num_refs, (len(ids),), device=self.device
            )
        return ids

    def write_dims(self, command: torch.Tensor) -> None:
        """写 8 维指令的 [5:8] = [jump_flag, jump_target_height, jump_phase]。"""
        active = self.active.float()
        command[:, 5] = active
        command[:, 6] = self.library.target_clearance[self.ref_id] * active
        command[:, 7] = (self.ref_t / self.phase_time_scale_s) * active

    def deployment_payload(self, offsets_steps: tuple[int, ...], scales: dict[str, float]) -> dict:
        """部署契约顶层 jump_reference（se3_runtime.jump_reference 解析）。"""
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
        return {
            "format": "se3.jump_ref.v1",
            "dt": lib.dt,
            "stand_height": lib.stand_height,
            "offsets_steps": list(offsets_steps),
            "scales": dict(scales),
            "references": references,
            "phase_time_scale_s": self.phase_time_scale_s,
        }


__all__ = ["PLAYBACK_END_TOLERANCE_S", "JumpReferenceClock"]
