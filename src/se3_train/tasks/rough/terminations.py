"""Rough 专属终止项：倒地功率终止（2026-10-23 用户定，见 env_cfg.ROUGH_FALL_POWER_LIMIT_PENALTY）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.rewards import _policy_leg_torque_and_vel

from . import events
from .fall_teacher import fall_teacher_of
from .rewards import fall_constraint_active

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.termination_manager import TerminationTermCfg

# 日志里 M 段结局比例的指数平均时间常数（步）：每张卡每步结束的 M 段不到 1 个，按步算比例全是 0/0。
_LOG_EMA_STEPS = 100.0


class fall_power_limit:
    """倒地功率终止：M 内腿正功率连续 `steps` 步超过 `limit_w` 即终止（非 time_out，罚分见下）。

    M = 倒地出生、尚未起身（rewards.fall_constraint_active）。腿正功率 P⁺ = Σ₄ max(0, τ·q̇)：四根主动杆电机，
    与拉格朗日 leg_power 同口径，被地面反推的制动功率不算。连续计数在未超限（含 M 外）或 episode reset 时清零。
    罚分由奖励项 rewards.termination_term 单独计、fall_penalty 排除本项（env_cfg._apply_fall_power_limit）。

    日志：`FallPowerLimit/over_frac`（M 内当步超限的占比）；`FallPowerLimit/{all,on,off}_{terminated,recovered}_frac`：
    结束的 M 段里被本项终止 / 起身的比例（其余是别的终止或超时），每段在结束那一步记一次结局，约 100 步指数平均；
    on / off 按倒地教师的对照子集分，没有教师只记 all。Recovery/* 与 FallTeacher/* 的起身率只统计活到时限的 env，
    有本项终止时偏高，recovered_frac 把被终止的段算作没起身。
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        n = env.num_envs
        self._count = torch.zeros(n, dtype=torch.long, device=env.device)
        # 上一次调用时的 M 与本项终止，用来在这一次认出"已 reset 的 M 段"和"没 reset 就离开 M（起身）"。
        # 不在 reset() 里清：reset 发生在两次调用之间，清了就认不出上一段的结局。
        self._prev_m = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._prev_done = torch.zeros(n, dtype=torch.bool, device=env.device)
        # 指数平均累计的 M 段结局数：[本项终止, 起身, 其他] × [all, on, off]
        self._ema = torch.zeros(3, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._count[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, limit_w: float, steps: int) -> torch.Tensor:
        torque, leg_vel = _policy_leg_torque_and_vel(env, env.scene["robot"])
        power = torch.clamp(torque * leg_vel, min=0.0).sum(dim=1)
        active = fall_constraint_active(env)
        # nan 比较为 False：物理发散交给 catastrophic_state，不在这里算超限。
        over = active & (power > float(limit_w))
        self._count = torch.where(over, self._count + 1, torch.zeros_like(self._count))
        done = self._count >= int(steps)
        self._log(env, active, over, done)
        return done

    def _log(
        self, env: ManagerBasedRlEnv, active: torch.Tensor, over: torch.Tensor, done: torch.Tensor
    ) -> None:
        """全是张量，不同步 GPU。终止在 episode_length_buf 自增之后算，新 episode 第一步为 1。"""
        new_episode = env.episode_length_buf == 1
        ended_other = self._prev_m & new_episode & ~self._prev_done
        recovered = self._prev_m & ~new_episode & ~active
        self._prev_m.copy_(active)
        self._prev_done.copy_(done)
        fall = getattr(env, events.FALL_SPAWN_MASK_ATTR, None)
        if fall is None:
            return
        log = env.extras.setdefault("log", {})
        log["FallPowerLimit/over_frac"] = over.float().sum() / active.float().sum().clamp(min=1.0)
        teacher = fall_teacher_of(env)
        everyone = torch.ones_like(fall)
        if teacher is None:
            members = torch.stack((everyone, everyone, ~everyone))
            names: tuple[str, ...] = ("all",)
        else:
            members = torch.stack((everyone, ~teacher.off_env, teacher.off_env))
            names = ("all", "on", "off")
        outcomes = torch.stack((done, recovered, ended_other))
        counts = (outcomes.unsqueeze(1) & members.unsqueeze(0)).sum(dim=2)
        self._ema.mul_(1.0 - 1.0 / _LOG_EMA_STEPS).add_(counts.float())
        total = self._ema.sum(dim=0).clamp(min=1.0)
        for index, name in enumerate(names):
            log[f"FallPowerLimit/{name}_terminated_frac"] = self._ema[0, index] / total[index]
            log[f"FallPowerLimit/{name}_recovered_frac"] = self._ema[1, index] / total[index]


__all__ = ["fall_power_limit"]
