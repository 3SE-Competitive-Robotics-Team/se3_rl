"""跳跃 flag 指令项：8 维指令 + 跳跃事件状态机（J5，2026-10-01 用户定）。

指令布局与部署契约一致：[lin_vel_x, ang_vel_yaw, pitch, roll, height, jump_flag, jump_target_height, jump_phase]。
不用参考轨迹、不用相位：触发后 jump_flag 置 1 保持 window_s 再置 0（与复旦 sim2sim 按键切入跳跃策略 0.55 s
同一思路，只是放进同一个策略），jump_target_height 在窗口内给出目标轮底离地间隙，jump_phase 恒 0。

一次跳跃事件（event）从触发开始，到"离地后再触地并稳住 settle_s"或"窗口结束仍未离地（漏跳）"结束；
事件期间冻结速度 / 姿态 / 高度指令，奖励侧据此屏蔽行走罚项。事件内只认第一次离地：
离地 → 记录腾空最高点 → 触地那一步给出一次性"最高点"信号（landing_now），之后再离地算二次起跳。
接触状态由轮子接触传感器给出；command 在 reward 之后更新，一次性信号由下一步的奖励读取后被清除。

起跳参考（J6，只进奖励不进观测）：触发那一刻按当前机身高度 z₀ 与目标间隙解析生成"蹬地 → 上升到最高点"的
机身高度 / 竖直速度参考：蹬地段恒加速度 a = v² / 2(z_to − z₀)，离地后抛体到最高点；起跳速度 v 与起跳机身高度
z_to 由 jump_ref_v1 参考的 meta 按目标间隙线性插值（与生成器同一来源）。这一段从 flag 上升沿开始、单调上升、没有
静止段，状态与时刻基本一一对应，所以网络不看参考也能跟上；最高点之后不再给参考。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from se3_train.mdp.commands import VelocityHeightCommandCfg, VelocityHeightCommandTerm
from se3_train.tasks.jump_mimic.reference import reference_paths

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

WHEEL_RADIUS_M = 0.06
GRAVITY = 9.81
TAKEOFF_TABLE_HEIGHTS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50)
"""起跳速度插值表取这几条 jump_ref_v1 参考的 meta（目标间隙 → v_takeoff / z_takeoff）。"""
MIN_PUSH_STROKE_M = 0.05
"""蹬地行程下限，防止站得过高时解析加速度发散。"""


@dataclass
class JumpFlagCommandCfg(VelocityHeightCommandCfg):
    """跳跃 flag 指令配置。"""

    min_idle_s: float = 1.0
    """回合开始或上一次事件结束后至少这么久才允许触发。"""
    trigger_rate_hz: float = 0.5
    """满足条件后每秒触发概率（泊松近似）。"""
    window_s: float = 0.5
    """jump_flag 保持为 1 的时长。"""
    settle_s: float = 0.5
    """触地后仍算在跳跃事件内（行走罚项继续屏蔽）的时长。"""
    max_event_s: float = 2.5
    """事件最长时长（防卡死：例如离地后迟迟不触地）。"""
    target_clearance_range: tuple[float, float] = (0.20, 0.50)
    """目标轮底离地间隙采样范围（m）。"""
    wheel_sensor_name: str = "wheel_sensor"
    contact_force_threshold: float = 1.0
    min_flight_clearance: float = 0.03
    """算作"离地"的最低轮底间隙（m）。只看接触力会把一步轮子卸载当成离地（首版 J5 第 64 轮实测：
    落地记录里最高点均值 0.009 m、漏跳 0 次，漏跳罚被假离地绕过），所以离地要求两轮无接触且较低一侧轮底高于此值。"""
    takeoff_table_paths: tuple[str, ...] = field(
        default_factory=lambda: reference_paths(TAKEOFF_TABLE_HEIGHTS)
    )

    def build(self, env: ManagerBasedRlEnv) -> JumpFlagCommandTerm:
        return JumpFlagCommandTerm(self, env)


class JumpFlagCommandTerm(VelocityHeightCommandTerm):
    """在速度 + 高度指令上叠加跳跃 flag 窗口与跳跃事件状态机。"""

    cfg: JumpFlagCommandCfg

    def __init__(self, cfg: JumpFlagCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 8, device=self.device)
        n, dev = self.num_envs, self.device
        self.event = torch.zeros(n, dtype=torch.bool, device=dev)
        self.event_t = torch.zeros(n, device=dev)
        self.target = torch.zeros(n, device=dev)
        self.airborne = torch.zeros(n, dtype=torch.bool, device=dev)
        self.liftoff_seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.settle_t = torch.zeros(n, device=dev)
        self.max_clearance = torch.zeros(n, device=dev)
        self.idle_t = torch.zeros(n, device=dev)
        # 一次性信号：由 command 更新写入，下一步奖励读取，再下一次 command 更新清除
        self.landing_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self.landing_clearance = torch.zeros(n, device=dev)
        self.missed_now = torch.zeros(n, dtype=torch.bool, device=dev)
        self._wheel_ids: list[int] | None = None
        # 起跳参考：触发时写入
        self.takeoff_z0 = torch.zeros(n, device=dev)
        self.takeoff_z_to = torch.zeros(n, device=dev)
        self.takeoff_v = torch.zeros(n, device=dev)
        self.takeoff_a = torch.ones(n, device=dev)
        self._table_targets, self._table_v, self._table_z = _load_takeoff_table(
            cfg.takeoff_table_paths, dev
        )

    # ---- 供奖励 / 观测读取的派生量 ----
    @property
    def flag(self) -> torch.Tensor:
        """jump_flag：事件中且处于窗口内。"""
        return self.event & (self.event_t < self.cfg.window_s)

    @property
    def before_landing(self) -> torch.Tensor:
        """事件中、尚未完成第一次落地（蹬地与腾空段）。"""
        return self.event & ~self.landed

    def base_height(self) -> torch.Tensor:
        """机身离地高度（平地：根节点 z − env 原点 z）。"""
        return (
            self._env.scene["robot"].data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        )

    def takeoff_reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """当前时刻的起跳参考 (z_ref, vz_ref, 有效)：从触发到参考最高点、且尚未第一次落地时有效。"""
        t = self.event_t
        t_push = self.takeoff_v / self.takeoff_a
        t_apex = t_push + self.takeoff_v / GRAVITY
        pushing = t < t_push
        tf = (t - t_push).clamp(min=0.0)
        z_ref = torch.where(
            pushing,
            self.takeoff_z0 + 0.5 * self.takeoff_a * t.square(),
            self.takeoff_z_to + self.takeoff_v * tf - 0.5 * GRAVITY * tf.square(),
        )
        vz_ref = torch.where(pushing, self.takeoff_a * t, self.takeoff_v - GRAVITY * tf)
        valid = self.before_landing & (t <= t_apex)
        return z_ref, vz_ref, valid

    def _start_takeoff_reference(self, env_ids: torch.Tensor) -> None:
        """按触发时机身高度与目标间隙写入解析起跳参考。"""
        target = self.target[env_ids]
        v = _interp(target, self._table_targets, self._table_v)
        z_to = _interp(target, self._table_targets, self._table_z)
        z0 = self.base_height()[env_ids]
        stroke = (z_to - z0).clamp(min=MIN_PUSH_STROKE_M)
        self.takeoff_z0[env_ids] = z_to - stroke
        self.takeoff_z_to[env_ids] = z_to
        self.takeoff_v[env_ids] = v
        self.takeoff_a[env_ids] = v.square() / (2.0 * stroke)

    def wheel_clearance(self) -> torch.Tensor:
        """较低一侧轮底离地间隙（平地：轮心 z − 半径 − env 原点 z）。"""
        robot = self._env.scene["robot"]
        if self._wheel_ids is None:
            self._wheel_ids = [robot.body_names.index(b) for b in ("l_wheel_Link", "r_wheel_Link")]
        z = robot.data.body_link_pos_w[:, self._wheel_ids, 2].amin(dim=-1)
        return z - WHEEL_RADIUS_M - self._env.scene.env_origins[:, 2]

    def wheel_contact(self) -> torch.Tensor:
        """两轮接触 [N, 2]。"""
        force = self._env.scene[self.cfg.wheel_sensor_name].data.force
        if force is None:
            return torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        return torch.linalg.norm(torch.nan_to_num(force), dim=-1) > self.cfg.contact_force_threshold

    # ---- 生命周期 ----
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """事件中的 env 不换速度 / 姿态 / 高度指令；reset 照常重采样。"""
        if bool(getattr(self, "_resampling_for_reset", False)):
            super()._resample_command(env_ids)
            return
        keep = env_ids[self.event[env_ids]]
        saved = self._command[keep, 0:5].clone()
        saved_standing = self._standing_mask[keep].clone()
        super()._resample_command(env_ids)
        self._command[keep, 0:5] = saved
        self._standing_mask[keep] = saved_standing

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids)
        assert isinstance(env_ids, torch.Tensor)
        self._clear_event(env_ids)
        self.idle_t[env_ids] = 0.0
        self.landing_now[env_ids] = False
        self.missed_now[env_ids] = False
        self._write_jump_dims()
        return extras

    def _clear_event(self, env_ids: torch.Tensor) -> None:
        self.event[env_ids] = False
        self.event_t[env_ids] = 0.0
        self.airborne[env_ids] = False
        self.liftoff_seen[env_ids] = False
        self.landed[env_ids] = False
        self.settle_t[env_ids] = 0.0
        self.max_clearance[env_ids] = 0.0

    def _update_command(self) -> None:
        super()._update_command()
        dt = float(self._env.step_dt)
        self.landing_now.zero_()
        self.missed_now.zero_()

        contact = self.wheel_contact().any(dim=-1)
        clearance = self.wheel_clearance()
        # 真离地：两轮无接触且轮底高于 min_flight_clearance（排除一步轮子卸载）
        airborne = ~contact & (clearance > self.cfg.min_flight_clearance)
        ev = self.event
        self.event_t = torch.where(ev, self.event_t + dt, self.event_t)

        # 第一次离地（只在落地前计）
        liftoff = ev & ~self.landed & airborne
        self.liftoff_seen |= liftoff
        flying = ev & self.liftoff_seen & ~self.landed
        self.max_clearance = torch.where(
            flying, torch.maximum(self.max_clearance, clearance), self.max_clearance
        )
        # 离地后第一次触地（任一轮有接触）→ 一次性落地信号，带出本次腾空最高点
        touchdown = flying & contact
        self.landing_now |= touchdown
        self.landing_clearance = torch.where(touchdown, self.max_clearance, self.landing_clearance)
        self.landed |= touchdown
        self.settle_t = torch.where(self.landed & ~touchdown, self.settle_t + dt, self.settle_t)
        self.airborne = airborne

        # 窗口结束仍未离地 → 漏跳，事件结束
        missed = ev & ~self.liftoff_seen & (self.event_t >= self.cfg.window_s)
        self.missed_now |= missed
        done = missed | (self.landed & (self.settle_t >= self.cfg.settle_s))
        done |= ev & (self.event_t >= self.cfg.max_event_s)
        if bool(done.any()):
            ids = done.nonzero().flatten()
            self._clear_event(ids)
            self.idle_t[ids] = 0.0

        # 触发
        self.idle_t = torch.where(self.event, torch.zeros_like(self.idle_t), self.idle_t + dt)
        eligible = (~self.event) & (self.idle_t >= float(self.cfg.min_idle_s))
        fire = eligible & (
            torch.rand(self.num_envs, device=self.device) < self.cfg.trigger_rate_hz * dt
        )
        if bool(fire.any()):
            ids = fire.nonzero().flatten()
            self._clear_event(ids)
            self.event[ids] = True
            lo, hi = self.cfg.target_clearance_range
            self.target[ids] = torch.rand(len(ids), device=self.device) * (hi - lo) + lo
            self._start_takeoff_reference(ids)
        self._write_jump_dims()

    def _write_jump_dims(self) -> None:
        flag = self.flag.float()
        self._command[:, 5] = flag
        self._command[:, 6] = self.target * flag
        self._command[:, 7] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        log = self._env.extras.setdefault("log", {}) if hasattr(self._env, "extras") else None
        if isinstance(log, dict):
            log["Jump/event_rate"] = self.event.float().mean()
            log["Jump/flag_rate"] = self.flag.float().mean()


def _load_takeoff_table(
    paths: tuple[str, ...], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从参考 meta 读 (目标间隙, 起跳速度, 起跳机身高度)，按目标间隙升序。"""
    rows = []
    for path in paths:
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
        rows.append((meta["target_clearance"], meta["v_takeoff"], meta["z_takeoff"]))
    rows.sort()
    table = torch.tensor(rows, dtype=torch.float32, device=device)
    return table[:, 0], table[:, 1], table[:, 2]


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """分段线性插值，区间外按端点两段外推。"""
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


__all__ = [
    "GRAVITY",
    "MIN_PUSH_STROKE_M",
    "TAKEOFF_TABLE_HEIGHTS",
    "WHEEL_RADIUS_M",
    "JumpFlagCommandCfg",
    "JumpFlagCommandTerm",
]
