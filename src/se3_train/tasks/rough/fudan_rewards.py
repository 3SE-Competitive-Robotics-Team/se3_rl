"""复旦 v3 奖励集（yly-true/fudan_rl_wheel_leg `8204e853` 上台阶 v3 快照），用于全地形统一奖励的对照实验。

复旦的奖励不分地形列：每一项乘权重、乘 dt 后逐项限幅到 [−dt, +dt]（`clip_single_reward=1`），即每项每秒贡献
最多 ±1，总和允许为负，没有存活奖励与终止罚。mjlab 的 RewardManager 只做"值 × 权重 × dt"、不支持逐项限幅，
所以这里每个函数自己完成"乘复旦权重 → 限幅到 ±clip"，注册时 RewardTermCfg 权重固定为 1：
每步贡献 dt·clip(权重·原值, ±1)，与复旦 `clip(原值·权重·dt, ±dt)` 逐项等价，`Episode_Reward/*` 就是限幅后的每秒贡献。

与复旦的已知口径差异（机器人与控制频率不同，按复旦原权重照搬、不做换算）：
- 控制频率 50 Hz（复旦 100 Hz）、腿动作缩放 0.25 rad、轮 45 rad/s（复旦 0.5 rad、10 rad/s），
  action_rate / action_smooth 的物理含义随之不同；
- 虚拟腿角 θ0 用髋关节（lf0/rf0_Link 原点）到轮心的连线在机身系里相对竖直的夹角，是闭链四连杆的等效
  虚拟腿，复旦是两连杆 FK；
- 高度参考复用 critic 的 77 点高度扫描（网格与复旦 measured_points 一致），高度指令沿用本仓库的 0.20–0.38 m；
- 复旦的 collision 项罚的接触体列表为空、恒为 0，这里不加。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared.action_history import periodic_policy_action_second_difference_torch
from se3_train.mdp.action_period import front_action_periods_from_env
from se3_train.mdp.joint_indices import wheel_actuator_ids, wheel_joint_ids
from se3_train.mdp.rewards import (
    _DEFAULT_ASSET_CFG,
    _policy_leg_acc,
    _policy_leg_torque_and_vel,
    dof_pos_limits,
)
from se3_train.mdp.terrain_height import ground_height_estimate

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg

# 复旦 v3 的 `tracking_sigma`；enhance 项用它的 10 倍作宽核。
FUDAN_TRACKING_SIGMA = 0.25


def _scaled_clip(raw: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    """复旦的逐项限幅：权重乘原值后夹到 ±clip（每秒量）。"""
    return torch.clamp(float(scale) * raw, -float(clip), float(clip))


def _vx_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 0] - env.scene["robot"].data.root_link_lin_vel_b[:, 0]


def tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
    sigma: float = FUDAN_TRACKING_SIGMA,
    shadow_func=None,
    shadow_params: dict | None = None,
) -> torch.Tensor:
    """1.3·exp(−e²/σ)，e 为 vx 跟踪误差。

    `shadow_func` 只为保留日志：原 tracking_lin_vel 会写速度课程读的
    `Locomotion/tracking_lin_vel_reward_curriculum` 与 `Rough/*` 诊断，删掉它速度课程就永远不推进；
    这里调用一次、丢弃返回值，不计入奖励。
    """
    if shadow_func is not None:
        shadow_func(env, **(shadow_params or {}))
    raw = 1.3 * torch.exp(-_vx_error(env, command_name).square() / float(sigma))
    return _scaled_clip(raw, scale, clip)


def tracking_lin_vel_enhance(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
    sigma: float = FUDAN_TRACKING_SIGMA,
) -> torch.Tensor:
    """1.3·(exp(−e²/(10σ)) − 1)：大误差段仍有梯度的有界负项。"""
    raw = 1.3 * (torch.exp(-_vx_error(env, command_name).square() / (10.0 * float(sigma))) - 1.0)
    return _scaled_clip(raw, scale, clip)


def tracking_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
    sigma: float = FUDAN_TRACKING_SIGMA,
) -> torch.Tensor:
    """exp(−e²/σ)，e 为 yaw 角速度跟踪误差。"""
    cmd = env.command_manager.get_command(command_name)
    error = cmd[:, 1] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return _scaled_clip(torch.exp(-error.square() / float(sigma)), scale, clip)


def _height_error(
    env: ManagerBasedRlEnv, command_name: str, window_sensor_name: str
) -> torch.Tensor:
    """机身 z − 机身周围窗口地面均值 − 高度指令。"""
    cmd = env.command_manager.get_command(command_name)
    base_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return base_z - ground_height_estimate(env, window_sensor_name) - cmd[:, 4]


def base_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    window_sensor_name: str,
    scale: float,
    clip: float = 1.0,
) -> torch.Tensor:
    """1.5·exp(−1000·e²)：高度窄核正奖励（约 ±3 cm）。"""
    error = _height_error(env, command_name, window_sensor_name)
    return _scaled_clip(1.5 * torch.exp(-1000.0 * error.square()), scale, clip)


def base_height_enhance(
    env: ManagerBasedRlEnv,
    command_name: str,
    window_sensor_name: str,
    scale: float,
    clip: float = 1.0,
) -> torch.Tensor:
    """exp(−e²/0.01) − 1：高度宽核有界负项（约 ±10 cm）。"""
    error = _height_error(env, command_name, window_sensor_name)
    return _scaled_clip(torch.exp(-error.square() / 0.01) - 1.0, scale, clip)


def orientation(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """投影重力的水平分量平方和。"""
    gravity = env.scene["robot"].data.projected_gravity_b[:, :2]
    return _scaled_clip(gravity.square().sum(dim=1), scale, clip)


def ang_vel_xy(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """roll/pitch 角速度平方和。"""
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b[:, :2]
    return _scaled_clip(ang_vel.square().sum(dim=1), scale, clip)


def lin_vel_z(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """机身竖直线速度平方。"""
    return _scaled_clip(env.scene["robot"].data.root_link_lin_vel_b[:, 2].square(), scale, clip)


def _body_ids(env: ManagerBasedRlEnv, names: tuple[str, str]) -> list[int]:
    attr = "_se3_fudan_body_ids_" + "_".join(names)
    cached = getattr(env, attr, None)
    if isinstance(cached, list):
        return cached
    ids, found = env.scene["robot"].find_bodies(names, preserve_order=True)
    if len(ids) != len(names):
        raise RuntimeError(f"必须找到 {names}，实际找到 {found}")
    setattr(env, attr, list(ids))
    return list(ids)


def virtual_leg_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
    """左右虚拟腿角 θ0，形状 [B, 2]：机身系里髋关节 → 轮心连线相对竖直向下的夹角，前摆为正。"""
    robot = env.scene["robot"]
    hip = _body_ids(env, ("lf0_Link", "rf0_Link"))
    wheel = _body_ids(env, ("l_wheel_Link", "r_wheel_Link"))
    delta_w = robot.data.body_link_pos_w[:, wheel, :] - robot.data.body_link_pos_w[:, hip, :]
    quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    delta_b = quat_apply_inverse(quat, delta_w.reshape(-1, 3)).reshape(env.num_envs, 2, 3)
    return torch.atan2(delta_b[..., 0], -delta_b[..., 2])


def nominal_state(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """左右虚拟腿角之差的平方（复旦 nominal_state 的负权重分支）。"""
    theta0 = virtual_leg_angles(env)
    return _scaled_clip((theta0[:, 0] - theta0[:, 1]).square(), scale, clip)


def dof_vel(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """四个腿部主动杆关节速度平方和（复旦只罚腿，不含轮）。"""
    _, vel = _policy_leg_torque_and_vel(env, env.scene["robot"])
    return _scaled_clip(vel.square().sum(dim=1), scale, clip)


def dof_acc(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """六个受控关节加速度平方和（含轮）。"""
    robot = env.scene["robot"]
    acc = torch.cat(
        (_policy_leg_acc(env, robot), robot.data.joint_acc[:, wheel_joint_ids(robot)]), dim=1
    )
    return _scaled_clip(acc.square().sum(dim=1), scale, clip)


def torques(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """六个执行器力矩平方和（含轮）。"""
    robot = env.scene["robot"]
    leg, _ = _policy_leg_torque_and_vel(env, robot)
    wheel = robot.data.actuator_force[:, wheel_actuator_ids(robot)]
    return _scaled_clip(torch.cat((leg, wheel), dim=1).square().sum(dim=1), scale, clip)


def action_rate(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """六维动作一阶差分平方和。"""
    delta = env.action_manager.action - env.action_manager.prev_action
    return _scaled_clip(delta.square().sum(dim=1), scale, clip)


class ActionSmooth(ManagerTermBase):
    """腿部四维动作二阶差分平方和；前两步动作按 env 缓存，reset 时清零（与复旦重置 last_actions 一致）。

    二阶差分沿用本仓库的周期处理（前关节动作按周期取最短差），与现有 action_smoothness 的动作语义一致。
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        del cfg
        shape = (env.num_envs, env.action_manager.total_action_dim)
        self._prev = torch.zeros(shape, device=env.device)
        self._prev_prev = torch.zeros(shape, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._prev[ids] = 0.0
        self._prev_prev[ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
        action = env.action_manager.action
        second = periodic_policy_action_second_difference_torch(
            action,
            self._prev,
            self._prev_prev,
            front_action_period=front_action_periods_from_env(env),
        )
        self._prev_prev.copy_(self._prev)
        self._prev.copy_(action)
        return _scaled_clip(second[:, :4].square().sum(dim=1), scale, clip)


def dof_pos_limits_clipped(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """腿部软限位越界量（沿用闭链的同侧两主动杆夹角口径）。"""
    return _scaled_clip(dof_pos_limits(env, _DEFAULT_ASSET_CFG), scale, clip)


# 复旦 v3 的非零权重（legged_robot_config.py 的 rewards.scales，collision 恒 0 不计）。
FUDAN_V3_SCALES: dict[str, float] = {
    "tracking_lin_vel": 1.0,
    "tracking_lin_vel_enhance": 1.0,
    "tracking_ang_vel": 1.0,
    "base_height": 1.0,
    "base_height_enhance": 1.0,
    "nominal_state": -1.0,
    "lin_vel_z": -0.1,
    "ang_vel_xy": -0.07,
    "orientation": -20.0,
    "dof_vel": -5e-5,
    "dof_acc": -2.5e-7,
    "torques": -1e-4,
    "action_rate": -0.05,
    "action_smooth": -0.05,
    "dof_pos_limits": -1.0,
}

__all__ = [
    "FUDAN_TRACKING_SIGMA",
    "FUDAN_V3_SCALES",
    "ActionSmooth",
    "action_rate",
    "ang_vel_xy",
    "base_height",
    "base_height_enhance",
    "dof_acc",
    "dof_pos_limits_clipped",
    "dof_vel",
    "lin_vel_z",
    "nominal_state",
    "orientation",
    "torques",
    "tracking_ang_vel",
    "tracking_lin_vel",
    "tracking_lin_vel_enhance",
    "virtual_leg_angles",
]
