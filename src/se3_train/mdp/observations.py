"""SE3 轮腿机器人的观测函数。

观测空间:
- actor: 31D (原 27D + pitch/roll/height cmd 扩展)
- critic: actor + 特权信息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import ObservationConfig, output_to_policy_pos_torch, output_to_policy_vel_torch
from se3_train.mdp.contact_utils import contact_force_nonfinite_env_mask, finite_contact_force_norm
from se3_train.mdp.joint_indices import (
    is_fourbar_surrogate_model,
    policy_leg_joint_ids,
    wheel_joint_ids,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 模块级观测配置，作为缩放系数的单一来源
_OBS_CFG = ObservationConfig()
_WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR = "_wheel_contact_force_nonfinite_total"
_WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR = "_wheel_contact_force_sample_total"


def _finite_clamp(value: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """把观测限制在有限范围内，避免单个发散 env 污染整批 PPO。"""
    bound = float(_OBS_CFG.clip_value if limit is None else limit)
    return torch.nan_to_num(value, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.root_link_ang_vel_b * _OBS_CFG.ang_vel_scale)


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.projected_gravity_b, limit=1.0)


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/姿态/高度指令,缩放 (2.0, 0.25, 5.0, 5.0, 5.0)。

    取前 5 维: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    兼容跳跃任务（8 维指令），跳跃扩展维度由 jump_commands_obs 单独输出。
    """
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor(list(_OBS_CFG.command_scale), device=cmd.device)
    return _finite_clamp(cmd[:, :5] * scale)


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部主动杆位置（相对默认位姿），4D。"""
    robot = env.scene["robot"]
    leg_ids = policy_leg_joint_ids(robot)
    if is_fourbar_surrogate_model(robot):
        pos = output_to_policy_pos_torch(robot.data.joint_pos[:, leg_ids])
        default_pos = output_to_policy_pos_torch(robot.data.default_joint_pos[:, leg_ids])
        return _finite_clamp(pos - default_pos)
    return _finite_clamp(
        robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
    )


def leg_joint_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部主动杆速度,缩放 0.25,4D。"""
    robot = env.scene["robot"]
    leg_ids = policy_leg_joint_ids(robot)
    if is_fourbar_surrogate_model(robot):
        vel = output_to_policy_vel_torch(
            robot.data.joint_pos[:, leg_ids],
            robot.data.joint_vel[:, leg_ids],
        )
        return _finite_clamp(vel * _OBS_CFG.leg_vel_scale)
    return _finite_clamp(
        robot.data.joint_vel[:, leg_ids] * _OBS_CFG.leg_vel_scale
    )


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节位置（MJCF 已修正轴方向,无需手动取反）。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.joint_pos[:, wheel_joint_ids(robot)])


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,缩放 0.05（MJCF 已修正轴方向）。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.joint_vel[:, wheel_joint_ids(robot)] * _OBS_CFG.wheel_vel_scale)


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一步的 6 个动作。"""
    return _finite_clamp(env.action_manager.action)


# --- Critic 特权观测 ---


def base_lin_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的线速度,3D（特权信息,actor 不可见）。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.root_link_lin_vel_b)


def wheel_contact_force_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """轮子地面接触力标量,2D（特权信息）。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    _record_wheel_contact_force_nonfinite(env, data.force)
    return _finite_clamp(finite_contact_force_norm(data.force))


def _record_wheel_contact_force_nonfinite(env: ManagerBasedRlEnv, force: torch.Tensor) -> None:
    """累计 wheel contact force 的非有限值触发次数，并写入训练日志。"""
    nonfinite_envs = int(contact_force_nonfinite_env_mask(force).sum().item())
    total = int(getattr(env, _WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR, 0)) + nonfinite_envs
    samples = int(getattr(env, _WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR, 0)) + env.num_envs
    setattr(env, _WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR, total)
    setattr(env, _WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR, samples)

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Debug/wheel_contact_force_nonfinite_last_envs": nonfinite_envs,
                "Debug/wheel_contact_force_nonfinite_total_envs": total,
                "Debug/wheel_contact_force_obs_total_envs": samples,
                "Debug/wheel_contact_force_nonfinite_rate": total / max(samples, 1),
            }
        )


def base_height_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """底盘离地高度（多射线取均值），标量（特权信息，critic 专用）。"""
    from mjlab.sensor import TerrainHeightSensor

    sensor: TerrainHeightSensor = env.scene[sensor_name]
    return torch.nan_to_num(sensor.data.heights, nan=0.0, posinf=0.0, neginf=0.0)


def jump_commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """跳跃指令观测，3D：[jump_flag, jump_target_height, jump_phase]。

    jump_flag:          0/1，本 episode 是否触发跳跃
    jump_target_height: 目标跳跃高度（m），范围 0.1~0.6
    jump_phase:         0→1 连续相位，从参考轨迹第 0 帧开始按 motion 时间推进

    要求指令必须为 8 维（JumpCommandTerm），否则直接报错。
    """
    cmd = env.command_manager.get_command("velocity_height")
    if cmd.shape[1] != 8:
        raise ValueError(
            f"jump_commands_obs 要求 8 维指令 (JumpCommandTerm),"
            f" 实际得到 {cmd.shape[1]} 维。请将 velocity_height 指令替换为 JumpCommandCfg。"
        )
    return _finite_clamp(cmd[:, 5:8])
