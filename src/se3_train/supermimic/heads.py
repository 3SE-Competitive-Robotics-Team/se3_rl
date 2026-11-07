"""世界模型的解析头：从（预测）状态算 30 维 actor 观测与能解析的奖励项；其余奖励项交给学习头。

每一项都按 env 配置里的**函数身份**映射到这里的纯张量实现，参数直接取 env 配置（缺省项从原函数签名读），
门控 helper 复用 `se3_train.mdp.rewards` 里同一批函数，不另写数字。某项能否解析以 GPU 逐项核对为准
（.scratch/supermimic/check_heads.py）；核对不过的项从 `_ANALYTIC_REWARDS` 去掉，自动落到学习头。

第 1 阶段只支持纯平地地形（`flat_only_terrains_cfg`）：按列生效的包装（off_column / column_scaled）在平地上
都等于原值，跳跃列不存在、`not_jumping` 恒为 1、`jump_env_only` 恒为 0。构造时检查地形，不是纯平地直接报错。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import cached_property

import torch
from mjlab.utils.noise import UniformNoiseCfg

from se3_shared import (
    COMMAND_FIELDS,
    NO_ATTITUDE_COMMAND_OBS_FIELDS,
    ObservationConfig,
    periodic_policy_action_second_difference_torch,
    policy_default_from_height_torch,
    policy_leg_phase_active_obs_torch,
    policy_leg_position_error_torch,
)
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import observations as obs_fns
from se3_train.mdp import rewards as flat_rewards
from se3_train.mdp.action_period import front_action_periods_from_env
from se3_train.tasks.jump_mimic import mdp as jump_mdp
from se3_train.tasks.rough import rewards as rough_rewards

from .kinematics import TreeKinematics
from .state import NQ, S_ANG, XI_DEFAULT_LEG, StateSpec

_OBS_CFG = ObservationConfig()
_SHARED_ROBOT = SharedRobotConfig()


# ====================================================================== 上下文


@dataclass
class HeadContext:
    """一步 transition 的输入：s 是到达的状态 s_{t+1}，其余量对应 env 在该步奖励计算时看到的值。"""

    spec: StateSpec
    s: torch.Tensor
    """(N, STATE_DIM) 到达状态。"""
    action: torch.Tensor
    """(N, 6) 本步动作（ActionManager.action 语义：未裁剪）。"""
    prev_action: torch.Tensor
    """(N, 6) 上一步动作（episode 首步为 0）。"""
    prev_prev_action: torch.Tensor
    """(N, 6) 上上步动作。"""
    command: torch.Tensor
    """(N, 8) 本步指令。"""
    xi: torch.Tensor
    """(N, XI_DIM) 原始（未归一化）上下文。"""
    episode_step: torch.Tensor
    """(N,) 奖励计算时的 episode_length_buf（reset 后首步为 1）。"""

    @cached_property
    def projected_gravity(self) -> torch.Tensor:
        return self.spec.projected_gravity(self.s)

    @cached_property
    def lin_vel_b(self) -> torch.Tensor:
        return self.spec.lin_vel_b(self.s)

    @property
    def ang_vel_b(self) -> torch.Tensor:
        return self.s[..., S_ANG]

    @cached_property
    def leg_pos(self) -> torch.Tensor:
        return self.spec.policy_leg_pos(self.s)

    @property
    def default_leg(self) -> torch.Tensor:
        return self.xi[..., XI_DEFAULT_LEG]


def _kwargs(func, params: dict | None) -> dict:
    """原函数签名里的缺省值 + 配置参数。"""
    out = {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    out.update(params or {})
    return out


# ====================================================================== 解析奖励（返回未乘权重的项值）


def _tracking_lin_vel(ctx: HeadContext, kw: dict) -> torch.Tensor:
    """tracking_lin_vel_terrain_vz 在平地列上 = tracking_lin_vel(vz_weight=vz_weight)。"""
    cmd = ctx.command
    error_x = ctx.lin_vel_b[:, 0] - cmd[:, 0]
    vz = ctx.lin_vel_b[:, 2]
    sigma = torch.where(
        cmd[:, 0].abs() < 0.2,
        torch.full_like(error_x, float(kw["sigma_stand"])),
        torch.full_like(error_x, float(kw["sigma_move"])),
    )
    reward = torch.exp(-(error_x**2 + float(kw["vz_weight"]) * vz**2) / sigma)
    if kw["use_upright_gate"]:
        reward = reward * flat_rewards._tracking_upright_gate(
            ctx.projected_gravity[:, 2], kw["tracking_upright_full_cos"]
        )
    return reward


def _tracking_ang_vel(ctx: HeadContext, kw: dict) -> torch.Tensor:
    cmd = ctx.command
    error = ctx.ang_vel_b[:, 2] - cmd[:, 1]
    gate = flat_rewards._tracking_upright_gate(
        ctx.projected_gravity[:, 2], kw["tracking_upright_full_cos"]
    )
    cmd_mag = cmd[:, 1].abs()
    sigma = float(kw["sigma"]) * (1.0 + float(kw["sigma_cmd_scale"]) * cmd_mag)
    reward_gate = gate if kw["use_upright_gate"] else torch.ones_like(gate)
    reward = torch.exp(-(error**2) / sigma) * reward_gate
    blend = min(max(float(kw["ratio_blend"]), 0.0), 1.0)
    if blend > 0.0:
        ratio = torch.clamp(1.0 - error.abs() / torch.clamp(cmd_mag, min=1.0), 0.0, 1.0)
        reward = (1.0 - blend) * reward + blend * ratio * reward_gate
    return reward


def _tracking_orientation_gravity(ctx: HeadContext, kw: dict) -> torch.Tensor:
    pitch, roll = ctx.command[:, 2], ctx.command[:, 3]
    target = torch.stack(
        (
            torch.sin(pitch),
            -torch.sin(roll) * torch.cos(pitch),
            -torch.cos(roll) * torch.cos(pitch),
        ),
        dim=1,
    )
    return torch.sum((ctx.projected_gravity - target) ** 2, dim=1)


def _base_height(ctx: HeadContext, kw: dict) -> torch.Tensor:
    """平地列上 = flat_base_height_penalty_no_jump：机身高度 = 根 z（相对 env 原点，平地地面在原点高度）。"""
    error = ctx.s[:, 2] - ctx.command[:, 4]
    max_error = kw.get("max_error")
    if max_error is not None:
        error = torch.clamp(error, -float(max_error), float(max_error))
    penalty = error**2 / float(kw["sigma"]) ** 2
    return penalty * (~(ctx.command[:, 5] > 0.5)).float()


def _ang_vel_xy(ctx: HeadContext, kw: dict) -> torch.Tensor:
    pg_z = ctx.projected_gravity[:, 2]
    gate = flat_rewards._upright_factor(pg_z) if kw["use_upright_gate"] else torch.ones_like(pg_z)
    return (ctx.ang_vel_b[:, 0] ** 2 + ctx.ang_vel_b[:, 1] ** 2) * gate


def _stand_still(ctx: HeadContext, kw: dict) -> torch.Tensor:
    gate = flat_rewards._upright_factor(ctx.projected_gravity[:, 2])
    default = policy_default_from_height_torch(ctx.command[:, 4], _SHARED_ROBOT).to(ctx.leg_pos)
    diff = policy_leg_position_error_torch(ctx.leg_pos, default)
    still = (torch.linalg.norm(ctx.command[:, :2], dim=1) <= float(kw["command_threshold"])).float()
    return torch.sum(diff**2, dim=1) * still * gate


def _action_rate(ctx: HeadContext, kw: dict) -> torch.Tensor:
    d2 = (ctx.action - ctx.prev_action) ** 2
    return float(kw["leg_scale"]) * d2[:, :4].sum(1) + float(kw["wheel_scale"]) * d2[:, 4:6].sum(1)


def _dof_pos_limits(ctx: HeadContext, kw: dict) -> torch.Tensor:
    lower, upper = _SHARED_ROBOT.active_rod_soft_angle_limits
    q = ctx.leg_pos
    total = torch.zeros_like(q[:, 0])
    for side, (front_coef, back_coef) in enumerate(_SHARED_ROBOT.active_rod_angle_coeffs):
        angle = float(front_coef) * q[:, 2 * side] + float(back_coef) * q[:, 2 * side + 1]
        total = (
            total + (-(angle - float(lower))).clamp(min=0.0) + (angle - float(upper)).clamp(min=0.0)
        )
    return total


def _is_alive(ctx: HeadContext, kw: dict) -> torch.Tensor:
    return torch.ones_like(ctx.s[:, 0])


def _tracking_lin_vel_narrow(ctx: HeadContext, kw: dict) -> torch.Tensor:
    error = ctx.lin_vel_b[:, 0] - ctx.command[:, 0]
    gate = flat_rewards._tracking_upright_gate(
        ctx.projected_gravity[:, 2], kw["tracking_upright_full_cos"]
    )
    return torch.exp(-(error**2) / float(kw["sigma"])) * gate


def _upward(ctx: HeadContext, kw: dict) -> torch.Tensor:
    return flat_rewards._upward_score(ctx.projected_gravity[:, 2])


# 函数身份 → 解析实现。包装函数（off_column / column_scaled / not_jumping）在 RewardHeads 里先拆开再查这张表。
_ANALYTIC_REWARDS = {
    rough_rewards.tracking_lin_vel_terrain_vz: _tracking_lin_vel,
    flat_rewards.tracking_ang_vel: _tracking_ang_vel,
    flat_rewards.tracking_orientation_gravity: _tracking_orientation_gravity,
    rough_rewards.base_height_penalty_window_on_terrain: _base_height,
    flat_rewards.ang_vel_xy: _ang_vel_xy,
    flat_rewards.stand_still: _stand_still,
    flat_rewards.action_rate: _action_rate,
    flat_rewards.dof_pos_limits: _dof_pos_limits,
    flat_rewards.is_alive: _is_alive,
    rough_rewards.tracking_lin_vel_narrow: _tracking_lin_vel_narrow,
    flat_rewards.upward: _upward,
}
# 需要额外上下文（周期参数 / FK）的两项单独处理
_ACTION_SMOOTHNESS = flat_rewards.action_smoothness
_WHEEL_FORE_AFT = rough_rewards.wheel_fore_aft_offset
_FALL_PENALTY = rough_rewards.is_terminated_except
_COLUMN_WRAPPERS = (rough_rewards.off_column, rough_rewards.column_scaled, jump_mdp.not_jumping)


def _require_flat_only_terrain(env) -> None:
    terrain = getattr(env.cfg.scene, "terrain", None)
    generator = getattr(terrain, "terrain_generator", None)
    names = set(getattr(generator, "sub_terrains", {}) or {})
    if names != {"flat"}:
        raise NotImplementedError(
            f"SuperMimic 第 1 阶段只支持纯平地地形，当前子地形 {sorted(names)}；"
            "全地形要把局部高度图接进世界模型（docs/plan/rough_supermimic_20261106.md 第 2 阶段）"
        )


@dataclass
class _Term:
    name: str
    weight: float
    kind: str  # "analytic" | "smoothness" | "fore_aft" | "fall" | "learned"
    func: object = None
    kwargs: dict | None = None


class RewardHeads:
    """把 env 的奖励项分成解析 / 学习 / 摔倒罚三类，并给出解析部分的加权和。"""

    def __init__(self, env, spec: StateSpec, kin: TreeKinematics, mj_model):
        _require_flat_only_terrain(env)
        self.spec = spec
        self.kin = kin
        self.step_dt = float(env.step_dt)
        self.front_periods = front_action_periods_from_env(env)
        self.left_wheel = kin.body_index(mj_model, "l_wheel_Link")
        self.right_wheel = kin.body_index(mj_model, "r_wheel_Link")
        manager = env.reward_manager
        self.term_names: list[str] = list(manager.active_terms)
        self.terms: list[_Term] = []
        for name in self.term_names:
            cfg = manager.get_term_cfg(name)
            func, params = cfg.func, dict(cfg.params or {})
            # 平地上按列包装等于原值（见模块 docstring）
            while func in _COLUMN_WRAPPERS:
                func, params = params["inner"], dict(params.get("params") or {})
            weight = float(cfg.weight)
            if func is _FALL_PENALTY:
                self.terms.append(_Term(name, weight, "fall"))
            elif func is _ACTION_SMOOTHNESS:
                self.terms.append(_Term(name, weight, "smoothness", func, _kwargs(func, params)))
            elif func is _WHEEL_FORE_AFT:
                self.terms.append(_Term(name, weight, "fore_aft", func, _kwargs(func, params)))
            elif func in _ANALYTIC_REWARDS:
                self.terms.append(_Term(name, weight, "analytic", func, _kwargs(func, params)))
            else:
                self.terms.append(_Term(name, weight, "learned"))
        self.learned_names = [t.name for t in self.terms if t.kind == "learned"]
        self.learned_index = [self.term_names.index(n) for n in self.learned_names]
        self.analytic_names = [t.name for t in self.terms if t.kind not in ("learned", "fall")]
        fall = [t for t in self.terms if t.kind == "fall"]
        self.fall_weight = sum(t.weight for t in fall)

    # ------------------------------------------------------------------ 逐项

    def _smoothness(self, ctx: HeadContext, kw: dict) -> torch.Tensor:
        acc = periodic_policy_action_second_difference_torch(
            ctx.action,
            ctx.prev_action,
            ctx.prev_prev_action,
            front_action_period=self.front_periods,
        )
        penalty = float(kw["leg_scale"]) * (acc[:, :4] ** 2).sum(1) + float(kw["wheel_scale"]) * (
            acc[:, 4:6] ** 2
        ).sum(1)
        penalty = torch.clamp(penalty, max=float(kw["max_penalty"]))
        gate = flat_rewards._near_upright_gate(
            ctx.projected_gravity[:, 2],
            gate_start_deg=float(kw["gate_start_deg"]),
            gate_full_deg=float(kw["gate_full_deg"]),
        )
        active = gate > 0.0
        if kw.get("command_name") is not None:
            active = active & ~(ctx.command[:, 5] > 0.5)
        startup = ctx.episode_step < 3
        return torch.where(startup | ~active, torch.zeros_like(penalty), penalty * gate)

    def _fore_aft(self, ctx: HeadContext, kw: dict) -> torch.Tensor:
        pos_b, _ = self.kin.forward(ctx.s[:, :NQ])
        dx = pos_b[:, self.left_wheel, 0] - pos_b[:, self.right_wheel, 0]
        excess = torch.clamp(dx.abs() - float(kw["dead_zone_m"]), min=0.0)
        gate = flat_rewards._tracking_upright_gate(
            ctx.projected_gravity[:, 2], kw["tracking_upright_full_cos"]
        )
        return excess**2 * gate

    def analytic_terms(self, ctx: HeadContext) -> dict[str, torch.Tensor]:
        """解析项的加权值（= weight × 项值，与 RewardManager._step_reward 同口径，未乘 dt）。"""
        out = {}
        for term in self.terms:
            if term.kind == "analytic":
                value = _ANALYTIC_REWARDS[term.func](ctx, term.kwargs)
            elif term.kind == "smoothness":
                value = self._smoothness(ctx, term.kwargs)
            elif term.kind == "fore_aft":
                value = self._fore_aft(ctx, term.kwargs)
            else:
                continue
            out[term.name] = term.weight * value
        return out

    def total_reward(
        self, ctx: HeadContext, learned: torch.Tensor, done_prob: torch.Tensor
    ) -> torch.Tensor:
        """一步总奖励（已乘 dt，与 env.reward_buf 同口径）。learned 为学习项的加权值 (N, K)。"""
        analytic = self.analytic_terms(ctx)
        total = learned.sum(-1) + self.fall_weight * done_prob
        for value in analytic.values():
            total = total + value
        return total * self.step_dt


# ====================================================================== 解析观测


def _obs_base_ang_vel(ctx: HeadContext) -> torch.Tensor:
    return ctx.ang_vel_b * _OBS_CFG.ang_vel_scale


def _obs_projected_gravity(ctx: HeadContext) -> torch.Tensor:
    return torch.clamp(ctx.projected_gravity, -1.0, 1.0)


_CMD_IDS = tuple(COMMAND_FIELDS.index(name) for name in NO_ATTITUDE_COMMAND_OBS_FIELDS)


def _obs_commands_vx_yaw_height(ctx: HeadContext) -> torch.Tensor:
    scale = torch.tensor(
        [_OBS_CFG.command_scale[i] for i in _CMD_IDS], device=ctx.s.device, dtype=ctx.s.dtype
    )
    return ctx.command[:, list(_CMD_IDS)] * scale


def _obs_leg_joint_pos(ctx: HeadContext) -> torch.Tensor:
    return policy_leg_phase_active_obs_torch(ctx.leg_pos, ctx.default_leg)


def _obs_leg_joint_vel(ctx: HeadContext) -> torch.Tensor:
    return ctx.spec.policy_leg_vel(ctx.s) * _OBS_CFG.leg_vel_scale


def _obs_wheel_vel(ctx: HeadContext) -> torch.Tensor:
    return ctx.spec.wheel_vel(ctx.s) * _OBS_CFG.wheel_vel_scale


def _obs_last_actions(ctx: HeadContext) -> torch.Tensor:
    return ctx.action


def _obs_jump_commands(ctx: HeadContext) -> torch.Tensor:
    return ctx.command[:, 5:8]


_ANALYTIC_OBS = {
    obs_fns.base_ang_vel_obs: _obs_base_ang_vel,
    obs_fns.projected_gravity_obs: _obs_projected_gravity,
    obs_fns.commands_vx_yaw_height_obs: _obs_commands_vx_yaw_height,
    obs_fns.leg_joint_pos_obs: _obs_leg_joint_pos,
    obs_fns.leg_joint_vel_obs: _obs_leg_joint_vel,
    obs_fns.wheel_vel_obs: _obs_wheel_vel,
    obs_fns.last_actions_obs: _obs_last_actions,
    obs_fns.jump_commands_obs: _obs_jump_commands,
}


class ObservationHead:
    """按 env actor 观测组的项顺序拼出解析观测，并按同一份噪声配置加噪。

    观测语义：s 是 obs 时刻的状态，action 是上一步动作（last_actions），command 是 obs 时刻的指令。
    """

    def __init__(self, env, group: str = "actor"):
        group_cfg = env.cfg.observations[group]
        if group_cfg.history_length:
            raise NotImplementedError("SuperMimic 第 1 阶段的 actor 观测不带历史")
        self.fns = []
        noise_lo, noise_hi = [], []
        self.term_names = []
        self.widths = []
        dims = env.observation_manager.group_obs_term_dim[group]
        for (name, term), dim in zip(group_cfg.terms.items(), dims, strict=True):
            if term.func not in _ANALYTIC_OBS:
                raise NotImplementedError(f"actor 观测项 {name}（{term.func}）没有解析实现")
            if (
                term.scale is not None
                or term.clip is not None
                or term.delay_max_lag
                or term.history_length
            ):
                raise NotImplementedError(
                    f"actor 观测项 {name} 带 scale / clip / delay / history，解析头未实现"
                )
            width = int(dim[-1])
            self.fns.append(_ANALYTIC_OBS[term.func])
            self.term_names.append(name)
            self.widths.append(width)
            lo = hi = 0.0
            if group_cfg.enable_corruption and term.noise is not None:
                if not isinstance(term.noise, UniformNoiseCfg) or term.noise.operation != "add":
                    raise NotImplementedError(f"actor 观测项 {name} 的噪声类型解析头未实现")
                lo, hi = float(term.noise.n_min), float(term.noise.n_max)
            noise_lo.append(torch.full((width,), lo))
            noise_hi.append(torch.full((width,), hi))
        self.noise_lo = torch.cat(noise_lo).to(env.device)
        self.noise_hi = torch.cat(noise_hi).to(env.device)
        self.dim = int(sum(self.widths))

    def clean(self, ctx: HeadContext) -> torch.Tensor:
        parts = [fn(ctx) for fn in self.fns]
        for name, part, width in zip(self.term_names, parts, self.widths, strict=True):
            if part.shape[-1] != width:
                raise RuntimeError(f"观测项 {name} 解析宽度 {part.shape[-1]} ≠ env {width}")
        return torch.cat(parts, dim=-1)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + torch.rand_like(obs) * (self.noise_hi - self.noise_lo) + self.noise_lo


__all__ = ["HeadContext", "ObservationHead", "RewardHeads"]
