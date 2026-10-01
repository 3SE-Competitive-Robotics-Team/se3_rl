"""跳跃 flag 任务（J5）专属 MDP 项：按状态触发的跳跃奖励、事件屏蔽、critic 特权观测。

奖励不按时间逐帧模仿参考，而是按状态给分（思路取自复旦 jump 工程）：蹬地段奖励上升速度，腾空奖励离地与收腿，
触地那一步按本次腾空最高点与目标离地间隙的差给一次性奖励（高度可控靠这一项）；窗口结束仍未离地给一次性罚
（J1 教训：不跳不能比跳更划算）。这些项只在一次跳跃事件的第一次落地之前生效，二次起跳另罚。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp import rewards as flat_rewards
from se3_train.tasks.jump_mimic.mdp import leg_lengths

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from .commands import JumpFlagCommandTerm

TUCK_LEG_LENGTH_M = 0.16
"""腾空收腿目标虚拟腿长（参考 v1 空中最短 0.150 m；复旦 leg_tuck 目标 0.16）。"""


def _term(env: ManagerBasedRlEnv, command_name: str) -> JumpFlagCommandTerm:
    return env.command_manager.get_term(command_name)  # type: ignore[return-value]


# ---------------------------------------------------------------- 跳跃奖励
def jump_upward_velocity(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", max_vz: float = 3.0
) -> torch.Tensor:
    """第一次落地之前的机身上升速度 max(vz, 0)（蹬地与上升段的稠密塑形，复旦 line_z）。"""
    term = _term(env, command_name)
    vz = env.scene["robot"].data.root_link_lin_vel_w[:, 2]
    return vz.clamp(min=0.0, max=max_vz) * term.before_landing.float()


def jump_airborne(env: ManagerBasedRlEnv, command_name: str = "velocity_height") -> torch.Tensor:
    """第一次落地之前两轮离地（复旦 flight）。"""
    term = _term(env, command_name)
    return (term.airborne & term.before_landing).float()


def jump_tuck(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    target_leg_length: float = TUCK_LEG_LENGTH_M,
    sharpness: float = 4.0,
) -> torch.Tensor:
    """第一次落地之前腾空时收腿 exp(−k·Σ|L − L_tuck|)（复旦 leg_tuck）。"""
    term = _term(env, command_name)
    err = (leg_lengths(env) - target_leg_length).abs().sum(dim=-1)
    return torch.exp(-sharpness * err) * (term.airborne & term.before_landing).float()


def jump_apex_height(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", sigma: float = 0.08
) -> torch.Tensor:
    """一次性：触地那一步按本次腾空最高轮底离地间隙与目标的差给 exp(−Δ²/σ²)。"""
    term = _term(env, command_name)
    err = term.landing_clearance - term.target
    reward = torch.exp(-err.square() / sigma**2) * term.landing_now.float()
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        n = term.landing_now.float().sum().clamp(min=1.0)
        log["Jump/apex_clearance"] = (term.landing_clearance * term.landing_now.float()).sum() / n
        log["Jump/apex_error_abs"] = (err.abs() * term.landing_now.float()).sum() / n
        log["Jump/landings"] = term.landing_now.float().sum()
    return reward


def jump_missed(env: ManagerBasedRlEnv, command_name: str = "velocity_height") -> torch.Tensor:
    """一次性：窗口结束仍未离地（漏跳）。"""
    term = _term(env, command_name)
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        log["Jump/missed"] = term.missed_now.float().sum()
    return term.missed_now.float()


def jump_rehop(env: ManagerBasedRlEnv, command_name: str = "velocity_height") -> torch.Tensor:
    """事件中第一次落地之后再次两轮离地（二次起跳 / 落地弹跳）。"""
    term = _term(env, command_name)
    return (term.event & term.landed & term.airborne).float()


# ---------------------------------------------------------------- 屏蔽与行走项改写
def outside_jump_event(
    env: ManagerBasedRlEnv, inner, params: dict, command_name: str = "velocity_height"
) -> torch.Tensor:
    """任意奖励项在跳跃事件期间（触发 → 落地稳住）置零：轮子离地罚、高度罚、静站罚、接触力罚等。

    Flat 的这些项自身只按 jump_flag 屏蔽，而 flag 在窗口结束（往往还在空中）就回 0，所以要按事件再屏蔽一层。
    """
    return inner(env, **params) * (~_term(env, command_name).event).float()


def tracking_lin_vel_jump_event(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", **kwargs
) -> torch.Tensor:
    """Flat 速度跟踪，跳跃事件期间去掉核里的 vz 项（同 J3，屏蔽口径改成整个事件）。"""
    vz_weight = float(kwargs.pop("vz_weight", 2.0))
    event = _term(env, command_name).event
    weight = torch.where(event, torch.zeros_like(event, dtype=torch.float), vz_weight)
    return flat_rewards.tracking_lin_vel(env, command_name, vz_weight=weight, **kwargs)


# ---------------------------------------------------------------- 观测
def jump_event_state_obs(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height"
) -> torch.Tensor:
    """critic 特权：[事件中, 事件进度 t/max, 窗口内, 已离地, 已落地, 落地后进度, 目标间隙, 当前轮底间隙]。"""
    term = _term(env, command_name)
    ev = term.event.float()
    return torch.stack(
        [
            ev,
            (term.event_t / term.cfg.max_event_s).clamp(0.0, 1.0) * ev,
            term.flag.float(),
            term.liftoff_seen.float() * ev,
            term.landed.float() * ev,
            (term.settle_t / max(term.cfg.settle_s, 1.0e-6)).clamp(0.0, 1.0) * term.landed.float(),
            term.target * ev,
            term.wheel_clearance().clamp(-0.1, 1.0),
        ],
        dim=-1,
    )


__all__ = [
    "TUCK_LEG_LENGTH_M",
    "jump_airborne",
    "jump_apex_height",
    "jump_event_state_obs",
    "jump_missed",
    "jump_rehop",
    "jump_tuck",
    "jump_upward_velocity",
    "outside_jump_event",
    "tracking_lin_vel_jump_event",
]
