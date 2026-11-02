"""recover 组（2026-10-24 用户定，env_cfg(recover_group=True)，见 env_cfg.ROUGH_RECOVER_* 注释）。

固定一部分 env 只练倒地起身、用自己的奖励，其余 env 只练走路、不再倒地出生。recover 组一局固定长度、每局都倒地出生：
  站稳 = 倾角 < 15° 且离出生点地面 ≥ 0.17 m 连续 stable_hold_s；截止 deadline_s 前没站稳、或站稳后倾角 > refall_tilt_deg，
  都算失败并终止（一次性罚分）；可选翻正截止 flip_deadline_s：此后还没站稳且倾角 > refall_tilt_deg 也算失败
  （躺着不动的代价从截止处的 −20·γ^400 ≈ −0.36 提到 −20·γ^150 ≈ −4.4，见 env_cfg.ROUGH_RECOVER_FLIP_DEADLINE_S）；
  到 episode_s 正常结束。两种结束都按真终止处理（不自举），recover 组的回报只含本局奖励，
  起身后能走路拿多少分不计入——早起身不再值钱，暴起就不再是最优解。
奖励（只对 recover 组）：站稳后每秒 +1、站稳那一步一次性 +20、失败一次性 −20，加腿 + 轮正功率 / 机身角速度 / 腿速 / 轮速
超限的线性罚与动作变化率，可选机身 / 轮子对地撞击超阈冲量罚（recover_impact_excess）；可选把站稳 +1/s 与 +20 换成
势函数整形（recover_potential_shaping，见 env_cfg.ROUGH_RECOVER_POTENTIAL_*）；走路组原有奖励在 recover 组上
整体抵消（recover_mask_walking_rewards，必须排在全部走路奖励之后、recover 奖励之前）。recover 组指令每步强制为零速（保留高度指令），站稳后与走路组的静站样本目标一致。
critic 另有 recover 分组、本局已过时间 / episode_s、是否已站稳三维（不进 actor / CTS 教师组）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from se3_shared import JointGroup
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.joint_indices import wheel_actuator_ids
from se3_train.mdp.rewards import _policy_leg_torque_and_vel

from . import events

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

RECOVER_GROUP_ATTR = "_se3_recover_group"
_STATE_ATTR = "_se3_recover_state"
_POTENTIAL_ATTR = "_se3_recover_potential"
_RECOVER_TERM_NAMES = ("recover_failed", "recover_finished")
_LOG_EMA_STEPS = 100.0


def recover_group_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """recover 组掩码；没有分组（未接 recover 组）时全 False。"""
    mask = getattr(env, RECOVER_GROUP_ATTR, None)
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask


def assign_recover_group(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...],
    fraction: float,
    seed: int = 20261024,
) -> None:
    """startup 事件：在倒地出生那几列（启动时的列归属）里按固定种子抽 fraction 的 env 作为 recover 组，整个训练不变。"""
    del env_ids
    from .columns import column_mask

    columns = column_mask(env, terrain_type_names)
    if columns is None:
        raise ValueError(f"找不到地形列 {terrain_type_names}，无法分 recover 组")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draw = (torch.rand(env.num_envs, generator=generator) < float(fraction)).to(env.device)
    setattr(env, RECOVER_GROUP_ATTR, columns & draw)


def force_recover_commands(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str
) -> None:
    """interval 事件（每步，指令更新之后、观测之前）：recover 组 vx / 横向 / yaw / 俯仰指令置零，高度指令保留。"""
    del env_ids
    mask = recover_group_mask(env)
    term = env.command_manager.get_term(command_name)
    term._command[:, 0:4].masked_fill_(mask.unsqueeze(1), 0.0)


class _RecoverState:
    """recover 组逐 env 的本局状态（站稳计时、站稳时刻、失败 / 结束），每个 env.step 在终止计算时更新一次。"""

    def __init__(self, env: ManagerBasedRlEnv, params: dict) -> None:
        self.params = params
        n, device = env.num_envs, env.device
        # 时长一律按步数比较（float 累加 dt 在 1.0 s 这种边界上会差一步）
        dt = float(env.step_dt)
        self.hold_steps = round(float(params["stable_hold_s"]) / dt)
        self.deadline_steps = round(float(params["deadline_s"]) / dt)
        self.episode_steps = round(float(params["episode_s"]) / dt)
        flip = params.get("flip_deadline_s")
        self.flip_deadline_steps = None if flip is None else round(float(flip) / dt)
        self.up_n = torch.zeros(n, dtype=torch.long, device=device)
        self.stable = torch.zeros(n, dtype=torch.bool, device=device)
        self.just_stable = torch.zeros(n, dtype=torch.bool, device=device)
        self.stable_t = torch.zeros(n, device=device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.timeout = torch.zeros(n, dtype=torch.bool, device=device)
        self.refall = torch.zeros(n, dtype=torch.bool, device=device)
        self.flip_late = torch.zeros(n, dtype=torch.bool, device=device)
        self.finished = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_step = -1
        # 指数平均的本局结局数：[成功结束, 超时, 又倒, 翻正超时, 站稳用时之和]
        self._ema = torch.zeros(5, device=device)

    def step(self, env: ManagerBasedRlEnv) -> None:
        step = int(env.common_step_counter)
        if step == self.last_step:
            return
        self.last_step = step
        p = self.params
        dt = float(env.step_dt)
        group = recover_group_mask(env)
        steps = env.episode_length_buf
        new_episode = steps == 1
        self.up_n.masked_fill_(new_episode, 0)
        self.stable_t.masked_fill_(new_episode, 0.0)
        self.stable.masked_fill_(new_episode, False)
        robot = env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
        ground = getattr(env, events._FALL_GROUND_Z_ATTR, None)
        height = robot.data.root_link_pos_w[:, 2] - (ground if ground is not None else 0.0)
        up = (tilt < float(p["stable_tilt_deg"])) & (height >= float(p["stable_height_m"]))
        counting = group & ~self.stable
        self.up_n = torch.where(
            counting, torch.where(up, self.up_n + 1, torch.zeros_like(self.up_n)), self.up_n
        )
        self.just_stable = counting & (self.up_n >= self.hold_steps)
        self.stable_t = torch.where(self.just_stable, steps.float() * dt, self.stable_t)
        self.stable |= self.just_stable
        self.timeout = group & ~self.stable & (steps >= self.deadline_steps)
        self.refall = group & self.stable & ~self.just_stable & (tilt > float(p["refall_tilt_deg"]))
        if self.flip_deadline_steps is not None:
            self.flip_late = (
                group
                & ~self.stable
                & ~self.timeout
                & (steps >= self.flip_deadline_steps)
                & (tilt > float(p["refall_tilt_deg"]))
            )
        self.failed = self.timeout | self.refall | self.flip_late
        self.finished = group & ~self.failed & (steps >= self.episode_steps)
        self._log(env)

    def _log(self, env: ManagerBasedRlEnv) -> None:
        """全是张量，不同步 GPU：本局结局按结束那一步计，约 100 步指数平均。"""
        success = self.finished & self.stable
        counts = torch.stack(
            (
                success.float().sum(),
                self.timeout.float().sum(),
                self.refall.float().sum(),
                self.flip_late.float().sum(),
                (self.stable_t * success.float()).sum(),
            )
        )
        self._ema.mul_(1.0 - 1.0 / _LOG_EMA_STEPS).add_(counts)
        ended = self._ema[:4].sum().clamp(min=1.0)
        log = env.extras.setdefault("log", {})
        log["Recover/success_frac"] = self._ema[0] / ended
        log["Recover/timeout_frac"] = self._ema[1] / ended
        log["Recover/refall_frac"] = self._ema[2] / ended
        if self.flip_deadline_steps is not None:
            log["Recover/flip_late_frac"] = self._ema[3] / ended
        log["Recover/stable_time_s"] = self._ema[4] / self._ema[0].clamp(min=1.0)


def recover_state(env: ManagerBasedRlEnv, params: dict | None = None) -> _RecoverState:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        if params is None:
            raise RuntimeError("recover 状态尚未由终止项 recover_failed 建立")
        state = _RecoverState(env, params)
        setattr(env, _STATE_ATTR, state)
    elif params is not None and state.params != params:
        raise ValueError("recover_failed / recover_finished 的参数须一致")
    return state


def recover_failed(
    env: ManagerBasedRlEnv,
    deadline_s: float,
    episode_s: float,
    stable_hold_s: float,
    stable_tilt_deg: float,
    stable_height_m: float,
    refall_tilt_deg: float,
    flip_deadline_s: float | None = None,
) -> torch.Tensor:
    """终止项（非 time_out）：recover 组截止前没站稳、站稳后又倒，或（flip_deadline_s 非 None）翻正截止后仍倾角
    > refall_tilt_deg。"""
    params = dict(
        deadline_s=deadline_s,
        episode_s=episode_s,
        stable_hold_s=stable_hold_s,
        stable_tilt_deg=stable_tilt_deg,
        stable_height_m=stable_height_m,
        refall_tilt_deg=refall_tilt_deg,
        flip_deadline_s=flip_deadline_s,
    )
    state = recover_state(env, params)
    state.step(env)
    return state.failed.clone()


def recover_finished(env: ManagerBasedRlEnv) -> torch.Tensor:
    """终止项（非 time_out，真终止不自举）：recover 组到 episode_s 正常结束。须排在 recover_failed 之后。"""
    state = recover_state(env)
    state.step(env)
    return state.finished.clone()


def recover_mask_walking_rewards(env: ManagerBasedRlEnv) -> torch.Tensor:
    """奖励项（权重 1）：把排在本项之前的全部走路奖励在 recover 组上抵消为 0（返回 −已累加奖励率）。"""
    manager = env.reward_manager
    rate = manager._reward_buf / (float(env.step_dt) if manager._scale_by_dt else 1.0)
    return torch.where(recover_group_mask(env), -rate, torch.zeros_like(rate))


def recover_stand(env: ManagerBasedRlEnv) -> torch.Tensor:
    """站稳后每秒 1（recover 组、本局未失败）。"""
    state = recover_state(env)
    return (state.stable & ~state.failed).float()


def recover_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    """站稳那一步为 1（权重取 +奖励 / step_dt 即一次性给奖励）。"""
    return recover_state(env).just_stable.float()


def recover_potential(env: ManagerBasedRlEnv, stable_height_m: float) -> torch.Tensor:
    """Φ ∈ [0, 1] = 0.5·(1 − 倾角 / 180°) + 0.5·min(离出生点地面高度 / stable_height_m, 1)，倾角按角度线性。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0)) / math.pi
    ground = getattr(env, events._FALL_GROUND_Z_ATTR, None)
    height = robot.data.root_link_pos_w[:, 2] - (ground if ground is not None else 0.0)
    return 0.5 * (1.0 - tilt) + 0.5 * torch.clamp(height / float(stable_height_m), 0.0, 1.0)


def snapshot_recover_potential(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, stable_height_m: float
) -> None:
    """step 事件（reset 与 forward 之后）：给刚 reset 的 env 记新局首个状态的 Φ(s₀)；其余 env 沿用奖励时刻记下的 Φ。"""
    del env_ids
    phi = recover_potential(env, stable_height_m)
    prev = getattr(env, _POTENTIAL_ATTR, None)
    if prev is None:
        setattr(env, _POTENTIAL_ATTR, phi)
        return
    prev.copy_(torch.where(env.episode_length_buf == 0, phi, prev))


def recover_potential_shaping(
    env: ManagerBasedRlEnv, gamma: float, stable_height_m: float
) -> torch.Tensor:
    """(γ·Φ(s′) − Φ(s)) / step_dt（权重即整形尺度 W），只在 recover 组；本步真终止的 env 取 Φ(s′) = 0。

    Φ(s′) 用奖励时刻的状态（与其余奖励同样滞后一个物理子步）并留作下一步的 Φ(s)，新局首步的 Φ(s₀) 由
    snapshot_recover_potential 补，逐局折扣求和严格等于 −W·Φ(s₀)。首次调用前没有 Φ(s)，整形记 0。
    """
    phi_next = recover_potential(env, stable_height_m)
    prev = getattr(env, _POTENTIAL_ATTR, None)
    setattr(env, _POTENTIAL_ATTR, phi_next)
    if prev is None:
        return torch.zeros_like(phi_next)
    terminal = env.termination_manager.terminated
    target = torch.where(terminal, torch.zeros_like(phi_next), float(gamma) * phi_next)
    shaping = (target - prev) / float(env.step_dt)
    return shaping * recover_group_mask(env).float()


def recover_only(env: ManagerBasedRlEnv, inner, params: dict) -> torch.Tensor:
    """任意函数型奖励项只在 recover 组上计。"""
    return inner(env, **params) * recover_group_mask(env).float()


def _wheel_ids(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    cached = getattr(env, "_se3_recover_wheel_ids", None)
    if cached is None:
        robot = env.scene["robot"]
        joints, _ = robot.find_joints(JointGroup.WHEEL_NAMES, preserve_order=True)
        cached = (
            torch.tensor(joints, device=env.device, dtype=torch.long),
            torch.tensor(wheel_actuator_ids(robot), device=env.device, dtype=torch.long),
        )
        env._se3_recover_wheel_ids = cached
    return cached


def recover_power_excess(env: ManagerBasedRlEnv, threshold_w: float) -> torch.Tensor:
    """max(0, 腿 + 轮正功率 − threshold_w)（W），只在 recover 组。"""
    robot = env.scene["robot"]
    torque, leg_vel = _policy_leg_torque_and_vel(env, robot)
    joints, actuators = _wheel_ids(env)
    wheel_power = robot.data.actuator_force[:, actuators] * robot.data.joint_vel[:, joints]
    power = torch.clamp(torque * leg_vel, min=0.0).sum(dim=1) + torch.clamp(
        wheel_power, min=0.0
    ).sum(dim=1)
    return torch.clamp(power - float(threshold_w), min=0.0) * recover_group_mask(env).float()


def recover_ang_vel_excess(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    """max(0, ‖ω_机身‖ − threshold)（rad/s），只在 recover 组。"""
    omega = torch.linalg.norm(env.scene["robot"].data.root_link_ang_vel_b, dim=1)
    return torch.clamp(omega - float(threshold), min=0.0) * recover_group_mask(env).float()


def recover_leg_speed_excess(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    """Σ腿 max(0, |q̇| − threshold)（rad/s），只在 recover 组。"""
    _torque, leg_vel = _policy_leg_torque_and_vel(env, env.scene["robot"])
    excess = torch.clamp(leg_vel.abs() - float(threshold), min=0.0).sum(dim=1)
    return excess * recover_group_mask(env).float()


def recover_wheel_speed_excess(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    """Σ轮 max(0, |ω_轮| − threshold)（rad/s），只在 recover 组。"""
    joints, _ = _wheel_ids(env)
    wheel = env.scene["robot"].data.joint_vel[:, joints].abs()
    return (
        torch.clamp(wheel - float(threshold), min=0.0).sum(dim=1) * recover_group_mask(env).float()
    )


def recover_impact_excess(
    env: ManagerBasedRlEnv, sensor_name: str, threshold_n: float
) -> torch.Tensor:
    """Σ_slot max(0, ‖F_对地‖ − threshold_n)（N），只在 recover 组；乘 step_dt 后即超阈冲量（N·s）。

    阈值按传感器的每个接触体分别比（wheel_sensor 两轮各一份，即单轮阈值）。机身（collision_sensor）与轮子
    （wheel_sensor）各挂一项；腿杆不计——要撬地翻身（hiphaauf 罚腿杆动作变怪）。
    """
    force = finite_contact_force_norm(env.scene.sensors[sensor_name].data.force)
    excess = torch.clamp(force - float(threshold_n), min=0.0).reshape(env.num_envs, -1).sum(dim=1)
    return excess * recover_group_mask(env).float()


def recover_group_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1] critic：是否 recover 组。"""
    return recover_group_mask(env).float().unsqueeze(1)


def recover_time_obs(env: ManagerBasedRlEnv, episode_s: float) -> torch.Tensor:
    """[N, 1] critic：recover 组本局已过时间 / episode_s（走路组为 0）。"""
    age = env.episode_length_buf.float() * float(env.step_dt) / float(episode_s)
    return (age * recover_group_mask(env).float()).unsqueeze(1)


def recover_stable_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[N, 1] critic：recover 组本局是否已站稳（观测时刻为上一次终止计算的结果；新局第一步为 0）。"""
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    stable = state.stable & ~(env.episode_length_buf == 0)
    return stable.float().unsqueeze(1)


__all__ = [
    "RECOVER_GROUP_ATTR",
    "assign_recover_group",
    "force_recover_commands",
    "recover_ang_vel_excess",
    "recover_failed",
    "recover_finished",
    "recover_group_mask",
    "recover_group_obs",
    "recover_impact_excess",
    "recover_leg_speed_excess",
    "recover_mask_walking_rewards",
    "recover_only",
    "recover_potential",
    "recover_potential_shaping",
    "recover_power_excess",
    "recover_stable_obs",
    "recover_stand",
    "recover_success",
    "recover_time_obs",
    "recover_wheel_speed_excess",
    "snapshot_recover_potential",
]
