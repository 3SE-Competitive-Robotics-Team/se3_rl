"""倒地起身脚本教师（2026-10-23 用户定）：6 月 recovery_teacher 路线在现闭链模型上的移植。

6 月版（gpu-8 Pod 上做、代码只剩 laptop 快照）是倒置时前髋固定方向扫动 + 动作层 α 混合 + 模仿奖励；直接照搬不行：
当时是开链模型、左右关节同号、腿部动作是直接关节角，而现在 Rough 是闭链、左右前杆镜像、腿部动作为 joint 语义
（四根主动杆绝对目标，夹角隐含）。动作段落照 R1 开环参考（se3_tools/recovery_reference.py，现模型实测可翻正）：

  ① 翻正：前杆按物理同向匀速转（policy 坐标左 +d·ω、右 −d·ω），后杆随前杆转、夹角以 fold_rate 限速收到折叠角
     （机械下限 + margin），轮速目标 0；倾角 < stop_tilt 结束；累计转过 max_sweep 仍没翻正，直接进保持段；
  ② 停顿 pause_s；
  ③ 反向收腿：前杆沿翻正的反方向转到 R1 蹲坐构型（与 sit_front 同余的最近位置，同向续转会再撬翻），夹角同时收到折叠角；
  ④ 保持 hold_s 后交还策略（站起来由策略负责）；每个 episode 只接一次，总时长超过 max_active_s 也交还。

接管条件：倒地出生、尚未起身（rewards.fall_constraint_active）、倾角 > enter_tilt，且不在对照子集里——倒地出生 env 里按 env 号
固定 off_env_fraction 不接教师，记录不靠教师的起身率（也起教师 dropout 的作用）。

混合在动作项里做（mdp.actions.SerialLegDelayedAction，夹紧之后、CTBC 与延迟之前）：执行动作 = π + α·active·(T − π)，
T 先按前杆周期整圈平移到与 π 同一圈的表示（joint 语义下同侧后杆一起平移，夹角不变），否则插值会插到半圈外。
PPO 存的仍是策略自己的采样动作，教师算环境的一部分；α 按 PPO 轮次 0.85 保持到 hold_iters、线性降到 mid_iter 的 alpha_mid、
再线性降到 end_iter 的 0（6 月实际训练用的调度）。配套 rewards.fall_teacher_imitation；critic 另有教师接管标记与 α 两维。

参数由开环探针定（.scratch/fall_teacher_probe_20261023/teacher_probe.py，α=1、10 s）：固定正向 2.0 rad/s、夹角 3 rad/s 限速折叠，
倒置 0.73 / 侧躺 0.75 翻正，四项峰值（腿速、机身角速度、轮速、腿正功率）都在旧拉格朗日限值内；夹角一步折叠腿速 p90 19–20 rad/s，
侧躺反向转大多翻正后又翻回去。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from . import events
from .rewards import fall_constraint_active

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from se3_train.mdp.actions import SerialLegDelayedAction

_IDLE, _FLIP, _PAUSE, _TUCK, _HOLD, _DONE = range(6)
_SEGMENT_NAMES = ("flip", "pause", "tuck", "hold")
_RECOVERY_LOG_CLASSES = ("side", "inverted")
_RECOVERY_LOG_HORIZONS_S = (3.0, 8.0)
_OFF_ENV_SEED = 20261023


@dataclass(kw_only=True)
class FallTeacherCfg:
    """倒地起身脚本教师的显式参数；默认值即 2026-10-23 用户定的探针推荐值。"""

    enter_tilt_deg: float = 60.0
    direction: float = 1.0
    sweep_speed_rad_s: float = 2.0
    max_sweep_rad: float = 2.0 * math.pi
    fold_rate_rad_s: float = 3.0
    fold_margin_rad: float = 0.02
    stop_tilt_deg: float = 20.0
    pause_s: float = 0.5
    tuck_speed_rad_s: float = 1.5
    sit_front_rad: tuple[float, float] = (0.292, -0.293)
    hold_s: float = 1.5
    max_active_s: float = 8.0
    off_env_fraction: float = 0.1
    alpha_start: float = 0.85
    alpha_mid: float = 0.45
    alpha_end: float = 0.0
    hold_iters: int = 600
    mid_iter: int = 1300
    end_iter: int = 1900
    steps_per_policy_iter: int = 24
    fixed_iteration: int | None = None

    def validate(self) -> None:
        """拒绝会让教师时序或动作语义静默漂移的配置。"""
        if self.direction not in (-1.0, 1.0):
            raise ValueError(f"direction 只能是 ±1，实际 {self.direction}")
        if not 0.0 < self.stop_tilt_deg < self.enter_tilt_deg <= 180.0:
            raise ValueError(
                f"须 0 < stop_tilt_deg < enter_tilt_deg ≤ 180，实际 {self.stop_tilt_deg} / {self.enter_tilt_deg}"
            )
        for name in ("sweep_speed_rad_s", "max_sweep_rad", "fold_rate_rad_s", "tuck_speed_rad_s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} 必须为正数，实际 {getattr(self, name)}")
        for name in ("fold_margin_rad", "pause_s", "hold_s"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} 不能为负，实际 {getattr(self, name)}")
        if self.max_active_s <= 0.0:
            raise ValueError(f"max_active_s 必须为正数，实际 {self.max_active_s}")
        if not 0.0 <= self.off_env_fraction < 1.0:
            raise ValueError(f"off_env_fraction 须在 [0, 1)，实际 {self.off_env_fraction}")
        if not 0.0 <= self.alpha_end <= self.alpha_mid <= self.alpha_start <= 1.0:
            raise ValueError("须 0 ≤ alpha_end ≤ alpha_mid ≤ alpha_start ≤ 1")
        if not 0 <= self.hold_iters <= self.mid_iter <= self.end_iter:
            raise ValueError("须 0 ≤ hold_iters ≤ mid_iter ≤ end_iter")
        if self.steps_per_policy_iter <= 0:
            raise ValueError(f"steps_per_policy_iter 必须为正数，实际 {self.steps_per_policy_iter}")

    def build(self, env: ManagerBasedRlEnv, term: SerialLegDelayedAction) -> FallTeacher:
        return FallTeacher(self, env, term)


def fall_teacher_alpha(current_iter: int, cfg: FallTeacherCfg) -> float:
    """α 调度：hold_iters 前保持 alpha_start，再分两段线性降到 mid_iter 的 alpha_mid、end_iter 的 alpha_end。"""
    if current_iter < cfg.hold_iters:
        return float(cfg.alpha_start)
    if current_iter < cfg.mid_iter:
        t = (current_iter - cfg.hold_iters) / max(cfg.mid_iter - cfg.hold_iters, 1)
        return float(cfg.alpha_start + t * (cfg.alpha_mid - cfg.alpha_start))
    if current_iter < cfg.end_iter:
        t = (current_iter - cfg.mid_iter) / max(cfg.end_iter - cfg.mid_iter, 1)
        return float(cfg.alpha_mid + t * (cfg.alpha_end - cfg.alpha_mid))
    return float(cfg.alpha_end)


class FallTeacher:
    """逐 env 的教师段落状态机、教师动作与混合；由 SerialLegDelayedAction 每个策略步调用一次 transform。"""

    def __init__(
        self, cfg: FallTeacherCfg, env: ManagerBasedRlEnv, term: SerialLegDelayedAction
    ) -> None:
        cfg.validate()
        if term.cfg.height_conditioned_action_default:
            raise ValueError("倒地教师按固定动作零点换算，不支持 height_conditioned_action_default")
        if term.cfg.leg_action_semantics not in ("joint", "active_rod"):
            raise ValueError(f"未知腿部动作语义 {term.cfg.leg_action_semantics}")
        self.cfg = cfg
        self._env = env
        self._term = term
        device = term.device
        n = env.num_envs
        self._joint_semantics = term.cfg.leg_action_semantics == "joint"
        self._leg_ids = term._leg_joint_ids
        self._scales = term._leg_action_scales.to(device=device, dtype=torch.float32)
        self._coeffs = term._active_rod_angle_coeffs.to(device=device, dtype=torch.float32)
        self._fold = float(term._active_rod_angle_limits[0]) + float(cfg.fold_margin_rad)
        self._mid = float(term._active_rod_angle_mid)
        clip = term.cfg.action_clip
        self._clip = None if clip is None else float(clip)
        self._side_sign = torch.tensor((1.0, -1.0), device=device)
        self._sit = torch.tensor(cfg.sit_front_rad, device=device, dtype=torch.float32)
        two_pi = 2.0 * math.pi
        # 整圈平移量（动作单位）：前杆 2π/scale；joint 语义下同侧后杆跟着平移 2π/scale，夹角不变。
        self._front_period = two_pi / self._scales[[0, 2]]
        self._back_period = two_pi / self._scales[[1, 3]]
        generator = torch.Generator(device="cpu").manual_seed(_OFF_ENV_SEED)
        self.off_env = (torch.rand(n, generator=generator) < float(cfg.off_env_fraction)).to(device)
        self.seg = torch.zeros(n, dtype=torch.long, device=device)
        self.seg_t = torch.zeros(n, device=device)
        self.active_t = torch.zeros(n, device=device)
        self.sweep = torch.zeros(n, device=device)
        self.theta = torch.zeros(n, 2, device=device)
        self.phi = torch.zeros(n, 2, device=device)
        self.tuck_target = torch.zeros(n, 2, device=device)
        self.used = torch.zeros(n, dtype=torch.bool, device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.teacher_action = torch.zeros(n, term.action_dim, device=device)
        self.alpha_value = 0.0

    def current_iteration(self) -> int:
        if self.cfg.fixed_iteration is not None:
            return max(0, int(self.cfg.fixed_iteration))
        return int(self._env.common_step_counter) // int(self.cfg.steps_per_policy_iter)

    def reset(self, env_ids: torch.Tensor) -> None:
        """episode reset：清段落与锁存（index_fill_ 不从主机拷标量，避免 GPU 同步）。"""
        for buffer in (self.seg, self.seg_t, self.active_t, self.sweep):
            buffer.index_fill_(0, env_ids, 0)
        for buffer in (self.used, self.active):
            buffer.index_fill_(0, env_ids, False)
        self.teacher_action.index_fill_(0, env_ids, 0.0)

    def transform(self, policy_actions: torch.Tensor) -> torch.Tensor:
        """返回执行动作 π + α·active·(T − π)；α = 0 后直接返回策略动作、不再推进状态机。"""
        alpha = fall_teacher_alpha(self.current_iteration(), self.cfg)
        self.alpha_value = alpha
        if alpha <= 0.0:
            self.active.fill_(False)
            return policy_actions
        self._step_state()
        teacher = self._align(self._teacher_action(), policy_actions)
        self.teacher_action.copy_(teacher)
        weight = (self.active.float() * alpha).unsqueeze(1)
        executed = policy_actions + weight * (teacher - policy_actions)
        if self._clip is not None:
            executed = torch.clamp(executed, -self._clip, self._clip)
        self._log(policy_actions, teacher, alpha)
        return executed

    def _step_state(self) -> None:
        cfg = self.cfg
        env = self._env
        dt = float(env.step_dt)
        robot = env.scene["robot"]
        tilt = torch.rad2deg(
            torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        )
        q = robot.data.joint_pos[:, self._leg_ids]
        coeffs = self._coeffs
        side = self._side_sign.unsqueeze(0)

        # 进入：每个 episode 只接一次
        enter = (
            fall_constraint_active(env)
            & ~self.off_env
            & ~self.used
            & (self.seg == _IDLE)
            & (tilt > float(cfg.enter_tilt_deg))
        )
        phi_now = torch.stack(
            (
                coeffs[0, 0] * q[:, 0] + coeffs[0, 1] * q[:, 1],
                coeffs[1, 0] * q[:, 2] + coeffs[1, 1] * q[:, 3],
            ),
            dim=1,
        )
        enter2 = enter.unsqueeze(1)
        self.theta = torch.where(enter2, q[:, [0, 2]], self.theta)
        self.phi = torch.where(enter2, phi_now, self.phi)
        for buffer_name in ("sweep", "seg_t", "active_t"):
            buffer = getattr(self, buffer_name)
            setattr(self, buffer_name, torch.where(enter, torch.zeros_like(buffer), buffer))
        self.seg = torch.where(enter, torch.full_like(self.seg, _FLIP), self.seg)
        self.used |= enter

        # ① 翻正：先判是否翻正，再推进；转满 max_sweep 仍没翻正直接进保持段
        flipping = self.seg == _FLIP
        flipped = flipping & (tilt < float(cfg.stop_tilt_deg))
        self.seg = torch.where(flipped, torch.full_like(self.seg, _PAUSE), self.seg)
        self.seg_t = torch.where(flipped, torch.zeros_like(self.seg_t), self.seg_t)
        advance = flipping & ~flipped
        inc = float(cfg.sweep_speed_rad_s) * dt * advance.float()
        self.theta = self.theta + side * (float(cfg.direction) * inc).unsqueeze(1)
        self.sweep = self.sweep + inc
        exhausted = advance & (self.sweep >= float(cfg.max_sweep_rad))
        self.seg = torch.where(exhausted, torch.full_like(self.seg, _HOLD), self.seg)
        self.seg_t = torch.where(exhausted, torch.zeros_like(self.seg_t), self.seg_t)

        # 翻正与停顿段夹角限速收向折叠角
        folding = ((self.seg == _FLIP) | (self.seg == _PAUSE)).unsqueeze(1)
        fold_step = float(cfg.fold_rate_rad_s) * dt
        self.phi = torch.where(
            folding, self.phi + torch.clamp(self._fold - self.phi, -fold_step, fold_step), self.phi
        )

        # ② 停顿 → ③ 收腿目标：沿翻正反方向转到与蹲坐前杆角同余的最近位置
        pausing = self.seg == _PAUSE
        self.seg_t = self.seg_t + dt * pausing.float()
        to_tuck = pausing & (self.seg_t >= float(cfg.pause_s))
        u = -side * float(cfg.direction)
        delta = torch.remainder(u * (self._sit.unsqueeze(0) - self.theta), 2.0 * math.pi)
        self.tuck_target = torch.where(
            to_tuck.unsqueeze(1), self.theta + u * delta, self.tuck_target
        )
        self.seg = torch.where(to_tuck, torch.full_like(self.seg, _TUCK), self.seg)

        tucking = (self.seg == _TUCK).unsqueeze(1)
        tuck_step = float(cfg.tuck_speed_rad_s) * dt
        self.theta = torch.where(
            tucking,
            self.theta + torch.clamp(self.tuck_target - self.theta, -tuck_step, tuck_step),
            self.theta,
        )
        self.phi = torch.where(
            tucking, self.phi + torch.clamp(self._fold - self.phi, -tuck_step, tuck_step), self.phi
        )
        tucked = (
            (self.seg == _TUCK)
            & ((self.tuck_target - self.theta).abs().amax(dim=1) < 1e-5)
            & ((self.phi - self._fold).abs().amax(dim=1) < 1e-5)
        )
        self.seg = torch.where(tucked, torch.full_like(self.seg, _HOLD), self.seg)
        self.seg_t = torch.where(tucked, torch.zeros_like(self.seg_t), self.seg_t)

        # ④ 保持 → 交还；总时长超限也交还
        holding = self.seg == _HOLD
        self.seg_t = self.seg_t + dt * holding.float()
        self.seg = torch.where(
            holding & (self.seg_t >= float(cfg.hold_s)), torch.full_like(self.seg, _DONE), self.seg
        )
        teaching = (self.seg >= _FLIP) & (self.seg <= _HOLD)
        self.active_t = self.active_t + dt * teaching.float()
        self.seg = torch.where(
            teaching & (self.active_t >= float(cfg.max_active_s)),
            torch.full_like(self.seg, _DONE),
            self.seg,
        )
        self.active = (self.seg >= _FLIP) & (self.seg <= _HOLD)

    def _teacher_action(self) -> torch.Tensor:
        """当前前杆 / 夹角指令换成策略动作空间（轮速目标 0）。"""
        default = self._term._entity.data.default_joint_pos[:, self._leg_ids]
        theta, phi, coeffs, scales = self.theta, self.phi, self._coeffs, self._scales
        if self._joint_semantics:
            back = torch.stack(
                (
                    (phi[:, 0] - coeffs[0, 0] * theta[:, 0]) / coeffs[0, 1],
                    (phi[:, 1] - coeffs[1, 0] * theta[:, 1]) / coeffs[1, 1],
                ),
                dim=1,
            )
            second = (back - default[:, [1, 3]]) / scales[[1, 3]]
        else:
            second = (phi - self._mid) / scales[[1, 3]]
        first = (theta - default[:, [0, 2]]) / scales[[0, 2]]
        teacher = torch.zeros_like(self.teacher_action)
        teacher[:, [0, 2]] = first
        teacher[:, [1, 3]] = second
        return teacher

    def _align(self, teacher: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        """把教师动作整圈平移到与策略动作同一圈（前杆最短角等价；joint 语义下同侧后杆同步平移）。"""
        turns = torch.round((teacher[:, [0, 2]] - policy[:, [0, 2]]) / self._front_period)
        aligned = teacher.clone()
        aligned[:, [0, 2]] = teacher[:, [0, 2]] - turns * self._front_period
        if self._joint_semantics:
            aligned[:, [1, 3]] = teacher[:, [1, 3]] - turns * self._back_period
        return aligned

    def _log(self, policy: torch.Tensor, teacher: torch.Tensor, alpha: float) -> None:
        """教师接管与各段占比、策略与教师的动作差，以及接 / 不接教师两组的起身率（全是张量，不同步 GPU）。"""
        env = self._env
        log = env.extras.setdefault("log", {})
        fall = getattr(env, events.FALL_SPAWN_MASK_ATTR, None)
        if fall is None:
            return
        n_fall = fall.float().sum().clamp(min=1.0)
        device = self.seg.device
        log["FallTeacher/alpha"] = torch.full((), alpha, device=device)
        log["FallTeacher/active_frac"] = (self.active & fall).float().sum() / n_fall
        log["FallTeacher/entered_frac"] = (self.used & fall).float().sum() / n_fall
        for index, name in enumerate(_SEGMENT_NAMES, start=_FLIP):
            log[f"FallTeacher/seg_{name}_frac"] = (
                (self.seg == index) & fall
            ).float().sum() / n_fall
        active = self.active.float()
        err = (teacher - policy).abs().mean(dim=1)
        log["FallTeacher/policy_teacher_abs_err"] = (err * active).sum() / active.sum().clamp(
            min=1.0
        )

        cache = getattr(env, events._FALL_CACHE_ATTR, None)
        if cache is None:
            return
        recovered = getattr(env, events._FALL_RECOVERED_ATTR)
        recover_time = getattr(env, events._FALL_RECOVER_TIME_ATTR)
        classes = getattr(env, events._FALL_CLASS_ATTR)
        age = env.episode_length_buf.float() * float(env.step_dt)
        for horizon in _RECOVERY_LOG_HORIZONS_S:
            old = fall & (age >= horizon)
            ok = old & recovered & (recover_time <= horizon)
            for class_name in _RECOVERY_LOG_CLASSES:
                if class_name not in cache.class_names:
                    continue
                in_class = classes == cache.class_names.index(class_name)
                for group, members in (("on", ~self.off_env), ("off", self.off_env)):
                    pool = old & in_class & members
                    log[f"FallTeacher/{group}_recovered_within_{horizon:.0f}s_{class_name}"] = (
                        ok & pool
                    ).float().sum() / pool.float().sum().clamp(min=1.0)


def fall_teacher_of(env: ManagerBasedRlEnv) -> FallTeacher | None:
    """取动作项上的倒地教师；没有动作项或没配教师时返回 None。"""
    manager = getattr(env, "action_manager", None)
    if manager is None or "delayed_action" not in manager.active_terms:
        return None
    teacher = getattr(manager.get_term("delayed_action"), "teacher", None)
    return teacher if isinstance(teacher, FallTeacher) else None


def fall_teacher_imitation(env: ManagerBasedRlEnv, sigma: float = 0.35) -> torch.Tensor:
    """模仿奖励（6 月 recovery_teacher_imitation 的算式）：exp(−mean_i((π_i − T_i)/max(|T_i|, 1))² / σ)·active·α。

    π 是夹紧后的策略动作（动作项 policy_action），T 是本步已对齐到 π 同一圈表示的教师动作；没有教师或 α = 0 时为 0。
    """
    teacher = fall_teacher_of(env)
    if teacher is None or teacher.alpha_value <= 0.0:
        return torch.zeros(env.num_envs, device=env.device)
    target = teacher.teacher_action
    policy = env.action_manager.get_term("delayed_action").policy_action
    scale = torch.clamp(target.abs(), min=1.0)
    mse = ((policy - target) / scale).square().mean(dim=1)
    reward = torch.exp(-mse / max(float(sigma), 1e-6))
    return reward * teacher.active.float() * float(teacher.alpha_value)


__all__ = [
    "FallTeacher",
    "FallTeacherCfg",
    "fall_teacher_alpha",
    "fall_teacher_imitation",
    "fall_teacher_of",
]
