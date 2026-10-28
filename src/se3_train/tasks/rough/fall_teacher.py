"""倒地起身脚本教师（2026-10-23 用户定）：6 月 recovery_teacher 路线在现闭链模型上的移植。

6 月版（gpu-8 Pod 上做、代码只剩 laptop 快照）是倒置时前髋扫动（每局 reset 随机 ±1 方向、整局固定）+ 动作层 α 混合
+ 模仿奖励；直接照搬不行：
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
再线性降到 end_iter 的 0（6 月实际训练用的调度）。配套 fall_teacher_imitation 模仿奖励；critic 另有教师接管标记与 α 两维。
状态机在观测时刻推进（prepare），观测组 fall_teacher_target 存下本步的接管标记、α 与教师动作，供 PPO 总损失里的模仿项
（se3_train.ppo 的 imitation_loss，2026-10-23 用户定）直接监督策略均值。

参数由开环探针定（.scratch/fall_teacher_probe_20261023/teacher_probe.py，α=1、10 s）：固定正向 2.0 rad/s、夹角 3 rad/s 限速折叠，
倒置 0.73 / 侧躺 0.75 翻正，四项峰值（腿速、机身角速度、轮速、腿正功率）都在旧拉格朗日限值内；夹角一步折叠腿速 p90 19–20 rad/s，
侧躺反向转大多翻正后又翻回去。

2026-10-27 用户定加回 6 月的随机方向版（FallTeacherCfg.june_random()）：上面这套固定方向 + 停顿 / 收腿 / 保持的整段轨迹决定了
起身时机身砸地（gyi9exiz 倒置峰值力中位约 1.2 kN，由它蒸馏的学生照样砸），加撞击罚（m6rxc7v9）后策略干脆不翻。随机方向版
只接倒置（倾角 > 135°，状态库倒置类边界）、每局 reset 随机 ±1 方向、π rad/s 扫到 2π；倾角 < 37°（6 月 pg_z < −0.8）后前杆停住，
连续 20 步仍在阈值内即交还策略（6 月这段继续扫，R1 实测闭链模型同向续转会再撬翻，故停住），不停顿、不收腿、不保持；
交还后再倒置可再接，累计接管超过 max_active_s 后本局不再接。夹角折叠沿用上面的限速折叠（现模型没有独立膝关节）。
同日用户定"夹角不限制"（fold='policy'，入口 Recover-RandDir-FreeFold-Impact）：教师只给前杆转动，后杆跟前杆同量转、
夹角保持策略自己给的值，模仿损失对后杆也只监督跟转量（se3_train.ppo 的 back_follow）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from se3_shared import RobotConfig

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
_DIRECTION_SEED = 20261027


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
    # 方向：fixed 用 direction；random_episode 每局 reset 随机 ±1（50/50，专用随机数发生器）、整局固定。
    direction_mode: Literal["fixed", "random_episode"] = "fixed"
    # 翻正之后：tuck 走停顿 → 反向收腿 → 保持（gyi9exiz）；upright 是 6 月做法——倾角 < stop_tilt_deg 后前杆停住，
    # 连续 exit_stable_steps 步仍在阈值内即交还，越出阈值继续扫；扫满 max_sweep 不进保持段，停在原角度等翻正或超时。
    release: Literal["tuck", "upright"] = "tuck"
    exit_stable_steps: int = 0
    # 交还后再满足进入条件可重新接管（6 月做法）；累计接管达到 max_active_s 后本局不再接。只配 release="upright"。
    reentry: bool = False
    # 夹角：limit 按 fold_rate_rad_s 限速收到折叠下限（教师规定夹角）；policy（2026-10-27 用户定"夹角不限制"）教师只管前杆转动，
    # 后杆目标 = 策略后杆 + 跟转量，使执行夹角恰为策略自己的夹角（见 back_follow_gain），只支持 joint 语义。
    fold: Literal["limit", "policy"] = "limit"
    # 模仿损失常驻（2026-10-28 用户定，同 10-24 PersistBC 思路，该实现 10-25 已删、复现用 5343bf96）：α 退到 0 后状态机继续推进、
    # 继续出模仿目标（只监督不执行）；监督目标组的权重列恒为 1（模仿损失不乘 α）。默认关，α = 0 后状态机停转。
    supervise_after_retire: bool = False
    # 限幅开环：前杆 / 夹角目标照原样积分后限制在实测 ±target_leash_rad 内（下一步从限幅值继续积分），只动接管中的 env。
    # 教师自己执行时跟踪误差 0.07–0.08 rad，L = 0.3 不触发；策略执行时目标最多领先 L，不会像纯开环那样越积越远
    # （PersistBC 标定：dqkkieq0 model_1000 上开环目标 − 实测 2 → 47 rad）。None = 纯开环。
    target_leash_rad: float | None = None
    # PersistBC 的 supervise_after_handoff（α = 0 后继续出模仿目标）与 target_leash_rad（目标限幅开环）2026-10-25 整理时删除，
    # 复现用 5343bf96。

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
        if self.direction_mode not in ("fixed", "random_episode"):
            raise ValueError(f"未知 direction_mode {self.direction_mode}")
        if self.release not in ("tuck", "upright"):
            raise ValueError(f"未知 release {self.release}")
        if self.release == "upright" and self.exit_stable_steps < 1:
            raise ValueError("release='upright' 须 exit_stable_steps ≥ 1")
        if self.release == "tuck" and (self.exit_stable_steps != 0 or self.reentry):
            raise ValueError("exit_stable_steps / reentry 只配 release='upright'")
        if self.fold not in ("limit", "policy"):
            raise ValueError(f"未知 fold {self.fold}")
        if self.target_leash_rad is not None and self.target_leash_rad <= 0.0:
            raise ValueError(f"target_leash_rad 必须为正数或 None，实际 {self.target_leash_rad}")
        if self.fold == "policy" and self.release != "upright":
            raise ValueError("fold='policy' 只配 release='upright'（收腿段要按夹角收到折叠角）")

    @classmethod
    def june_random(cls, **overrides) -> FallTeacherCfg:
        """6 月随机方向版（2026-10-27 用户定，见模块 docstring）：只接倒置、每局随机方向、π rad/s 扫到 2π、翻正即交还、可重接。"""
        params = dict(
            enter_tilt_deg=135.0,
            direction_mode="random_episode",
            sweep_speed_rad_s=math.pi,
            max_sweep_rad=2.0 * math.pi,
            stop_tilt_deg=math.degrees(math.acos(0.8)),
            release="upright",
            exit_stable_steps=20,
            reentry=True,
        )
        params.update(overrides)
        return cls(**params)

    def build(self, env: ManagerBasedRlEnv, term: SerialLegDelayedAction) -> FallTeacher:
        return FallTeacher(self, env, term)


def back_follow_gain(leg_scales: Sequence[float]) -> tuple[float, float]:
    """fold='policy' 的后杆跟转系数 k（左、右）：夹角 φ = c_f·q_前 + c_b·q_后，前杆动作改动 Δ 时后杆动作改动 −k·Δ 即保持 φ，
    k = c_f·s_前 / (c_b·s_后)。现模型左右系数 (1, −1) / (−1, 1)、四杆缩放相同，k = −1（后杆随前杆同向同量）。"""
    coeffs = RobotConfig().active_rod_angle_coeffs
    return tuple(
        float(coeffs[side][0])
        * float(leg_scales[2 * side])
        / (float(coeffs[side][1]) * float(leg_scales[2 * side + 1]))
        for side in (0, 1)
    )


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
        self.direction = torch.full((n,), float(cfg.direction), device=device)
        self._random_direction = cfg.direction_mode == "random_episode"
        # 多卡各 rank 的 env seed 不同（agent seed + rank），方向序列随之错开。
        self._direction_generator = torch.Generator(device=device).manual_seed(
            _DIRECTION_SEED + int(getattr(env.cfg, "seed", None) or 0)
        )
        if self._random_direction:
            self.direction.copy_(self._sample_direction(n))
        self.exit_count = torch.zeros(n, dtype=torch.long, device=device)
        self.timed_out = torch.zeros(n, dtype=torch.bool, device=device)
        self._free_fold = cfg.fold == "policy"
        if self._free_fold and not self._joint_semantics:
            raise ValueError("fold='policy' 只支持 joint 腿部动作语义")
        self._back_follow = torch.tensor(back_follow_gain(term.cfg.leg_scales), device=device)
        self.teacher_raw = torch.zeros(n, term.action_dim, device=device)
        self.teacher_action = torch.zeros(n, term.action_dim, device=device)
        self.alpha_value = 0.0
        self._fresh = False

    def current_iteration(self) -> int:
        if self.cfg.fixed_iteration is not None:
            return max(0, int(self.cfg.fixed_iteration))
        step = int(getattr(self._env, "common_step_counter", 0))
        return step // int(self.cfg.steps_per_policy_iter)

    def reset(self, env_ids: torch.Tensor) -> None:
        """episode reset：清段落与锁存（index_fill_ 不从主机拷标量，避免 GPU 同步），下次观测重新准备。"""
        for buffer in (self.seg, self.seg_t, self.active_t, self.sweep, self.exit_count):
            buffer.index_fill_(0, env_ids, 0)
        for buffer in (self.used, self.active, self.timed_out):
            buffer.index_fill_(0, env_ids, False)
        self.teacher_raw.index_fill_(0, env_ids, 0.0)
        self.teacher_action.index_fill_(0, env_ids, 0.0)
        if self._random_direction:
            self.direction.index_copy_(0, env_ids, self._sample_direction(env_ids.numel()))
        self._fresh = False

    def _sample_direction(self, count: int) -> torch.Tensor:
        """count 个 ±1（各 50%），用专用发生器，不动全局随机流。"""
        draw = torch.rand(
            count, generator=self._direction_generator, device=self._direction_generator.device
        )
        return torch.where(draw < 0.5, -1.0, 1.0)

    def prepare(self) -> None:
        """在观测时刻（上一步物理与 reset 之后、策略出动作之前）推进状态机并算出下一步的教师动作；每步只做一次。

        放在观测时刻而不是动作项里，是为了让存进 rollout 的模仿目标（observations.fall_teacher_target_obs）与
        策略这一步看到的观测严格同一时刻；两处读到的机器人状态相同，状态机轨迹与在动作项里推进一致。
        """
        if self._fresh:
            return
        self.alpha_value = fall_teacher_alpha(self.current_iteration(), self.cfg)
        if self.alpha_value <= 0.0 and not self.cfg.supervise_after_retire:
            self.active.fill_(False)
            self.teacher_raw.zero_()
        else:
            self._step_state()
            self.teacher_raw.copy_(self._teacher_action())
        self._fresh = True

    def transform(self, policy_actions: torch.Tensor) -> torch.Tensor:
        """返回执行动作 π + α·active·(T − π)；α = 0 后直接返回策略动作（supervise_after_retire 时仍算教师动作、记日志）。"""
        self.prepare()
        self._fresh = False
        alpha = self.alpha_value
        if alpha <= 0.0 and not self.cfg.supervise_after_retire:
            return policy_actions
        teacher = self._align(self.teacher_raw, policy_actions)
        if self._free_fold:
            # 后杆 = 策略后杆 + 跟转量：执行动作是 π 与 T 的线性插值、夹角对动作线性，故执行夹角恰为策略夹角
            teacher[:, [1, 3]] = policy_actions[:, [1, 3]] + self._back_follow * (
                policy_actions[:, [0, 2]] - teacher[:, [0, 2]]
            )
        self.teacher_action.copy_(teacher)
        if alpha <= 0.0:
            self._log(policy_actions, teacher, alpha)
            return policy_actions
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

        # 进入：默认每个 episode 只接一次；reentry 时交还后可重接，累计接管超时后不再接
        can_enter = ~self.timed_out if cfg.reentry else ~self.used
        enter = (
            fall_constraint_active(env)
            & ~self.off_env
            & can_enter
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
        # active_t 只在 reset 清零：单次接管时与进入时清零等价，重接时按整局累计
        for buffer_name in ("sweep", "seg_t"):
            buffer = getattr(self, buffer_name)
            setattr(self, buffer_name, torch.where(enter, torch.zeros_like(buffer), buffer))
        self.seg = torch.where(enter, torch.full_like(self.seg, _FLIP), self.seg)
        self.used |= enter

        flipping = self.seg == _FLIP
        upright = tilt < float(cfg.stop_tilt_deg)
        direction = self.direction.unsqueeze(1)
        if cfg.release == "tuck":
            # ① 翻正：先判是否翻正，再推进；转满 max_sweep 仍没翻正直接进保持段
            flipped = flipping & upright
            self.seg = torch.where(flipped, torch.full_like(self.seg, _PAUSE), self.seg)
            self.seg_t = torch.where(flipped, torch.zeros_like(self.seg_t), self.seg_t)
            advance = flipping & ~flipped
            inc = float(cfg.sweep_speed_rad_s) * dt * advance.float()
            self.theta = self.theta + side * (direction * inc.unsqueeze(1))
            self.sweep = self.sweep + inc
            exhausted = advance & (self.sweep >= float(cfg.max_sweep_rad))
            self.seg = torch.where(exhausted, torch.full_like(self.seg, _HOLD), self.seg)
            self.seg_t = torch.where(exhausted, torch.zeros_like(self.seg_t), self.seg_t)
        else:
            # ① 翻正（6 月做法）：阈值内前杆停住计数，连续 exit_stable_steps 步即交还；越出阈值清零续扫，扫满 max_sweep 停在原角度
            self.exit_count = torch.where(
                flipping & upright, self.exit_count + 1, torch.zeros_like(self.exit_count)
            )
            released = flipping & (self.exit_count >= int(cfg.exit_stable_steps))
            after = _IDLE if cfg.reentry else _DONE
            self.seg = torch.where(released, torch.full_like(self.seg, after), self.seg)
            self.exit_count = torch.where(
                released, torch.zeros_like(self.exit_count), self.exit_count
            )
            advance = flipping & ~upright
            inc = torch.minimum(
                float(cfg.sweep_speed_rad_s) * dt * advance.float(),
                (float(cfg.max_sweep_rad) - self.sweep).clamp(min=0.0),
            )
            self.theta = self.theta + side * (direction * inc.unsqueeze(1))
            self.sweep = self.sweep + inc

        # 翻正与停顿段夹角限速收向折叠角（fold='policy' 时夹角归策略，phi 只留进入时刻的值、不进执行动作）
        folding = ((self.seg == _FLIP) | (self.seg == _PAUSE)).unsqueeze(1) & (not self._free_fold)
        fold_step = float(cfg.fold_rate_rad_s) * dt
        self.phi = torch.where(
            folding, self.phi + torch.clamp(self._fold - self.phi, -fold_step, fold_step), self.phi
        )

        # ② 停顿 → ③ 收腿目标：沿翻正反方向转到与蹲坐前杆角同余的最近位置
        pausing = self.seg == _PAUSE
        self.seg_t = self.seg_t + dt * pausing.float()
        to_tuck = pausing & (self.seg_t >= float(cfg.pause_s))
        u = -side * direction
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
        timeout = teaching & (self.active_t >= float(cfg.max_active_s))
        self.seg = torch.where(timeout, torch.full_like(self.seg, _DONE), self.seg)
        self.timed_out |= timeout
        self.active = (self.seg >= _FLIP) & (self.seg <= _HOLD)

        # 限幅开环：目标限制在实测 ±L 内（只动接管中的 env）
        leash = cfg.target_leash_rad
        if leash is not None:
            act = self.active.unsqueeze(1)
            self.theta = torch.where(
                act, torch.clamp(self.theta, q[:, [0, 2]] - leash, q[:, [0, 2]] + leash), self.theta
            )
            self.phi = torch.where(
                act, torch.clamp(self.phi, phi_now - leash, phi_now + leash), self.phi
            )

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
        if self._random_direction:
            entered = self.used & fall
            log["FallTeacher/dir_pos_frac"] = (
                entered & (self.direction > 0)
            ).float().sum() / entered.float().sum().clamp(min=1.0)
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
                if self._random_direction:
                    # 随机方向：接过教师的 env 按本局方向拆开看起身率
                    for sign_name, sign in (
                        ("pos", self.direction > 0),
                        ("neg", self.direction < 0),
                    ):
                        pool = old & in_class & self.used & sign
                        log[
                            f"FallTeacher/dir_{sign_name}_recovered_within_{horizon:.0f}s_{class_name}"
                        ] = (ok & pool).float().sum() / pool.float().sum().clamp(min=1.0)


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
    "back_follow_gain",
    "fall_teacher_alpha",
    "fall_teacher_imitation",
    "fall_teacher_of",
]
