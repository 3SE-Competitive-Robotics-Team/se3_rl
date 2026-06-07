"""SE3 轮腿机器人的观测函数。

观测空间:
- actor: 31D (原 27D + pitch/roll/height cmd 扩展)
- critic: actor + 特权信息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import TASK_MODE_COUNT, TASK_MODE_SEMANTICS, JointGroup, ObservationConfig
from se3_train.mdp.contact_utils import contact_force_nonfinite_env_mask, finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 模块级观测配置，作为缩放系数的单一来源
_OBS_CFG = ObservationConfig()
_WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR = "_wheel_contact_force_nonfinite_total"
_WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR = "_wheel_contact_force_sample_total"


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return robot.data.root_link_ang_vel_b * _OBS_CFG.ang_vel_scale


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/姿态/高度指令,缩放 (2.0, 0.25, 5.0, 5.0, 5.0)。

    取前 5 维: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    兼容跳跃任务（8 维指令），跳跃扩展维度由 jump_commands_obs 单独输出。
    """
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor(list(_OBS_CFG.command_scale), device=cmd.device)
    return cmd[:, :5] * scale


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部关节位置（相对默认位姿），4D。"""
    robot = env.scene["robot"]
    return (
        robot.data.joint_pos[:, JointGroup.LEGS] - robot.data.default_joint_pos[:, JointGroup.LEGS]
    )


def leg_joint_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部关节速度,缩放 0.25,4D。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, JointGroup.LEGS] * _OBS_CFG.leg_vel_scale


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节位置（MJCF 已修正轴方向,无需手动取反）。"""
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, JointGroup.WHEELS]


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,缩放 0.05（MJCF 已修正轴方向）。"""
    robot = env.scene["robot"]
    return robot.data.joint_vel[:, JointGroup.WHEELS] * _OBS_CFG.wheel_vel_scale


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一步的 6 个动作。"""
    return env.action_manager.action


# --- Critic 特权观测 ---


def base_lin_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的线速度,3D（特权信息,actor 不可见）。"""
    robot = env.scene["robot"]
    return robot.data.root_link_lin_vel_b


def wheel_contact_force_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """轮子地面接触力标量,2D（特权信息）。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    _record_wheel_contact_force_nonfinite(env, data.force)
    return finite_contact_force_norm(data.force)


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
    return sensor.data.heights


def jump_commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """跳跃指令观测，3D：[jump_flag, jump_target_height, jump_phase]。

    jump_flag:          0/1，本 episode 是否触发跳跃
    jump_target_height: 目标跳跃高度（m），范围 0.1~0.6
    jump_phase:         0→1 连续相位，从参考轨迹第 0 帧开始按 motion 时间推进

    要求指令必须为 8 维（JumpCommandTerm），否则直接报错。
    """
    cmd = env.command_manager.get_command("velocity_height")
    if cmd.shape[1] < 8:
        raise ValueError(
            f"jump_commands_obs 要求至少 8 维指令 (JumpCommandTerm),"
            f" 实际得到 {cmd.shape[1]} 维。请将 velocity_height 指令替换为 JumpCommandCfg。"
        )
    return cmd[:, 5:8]


def task_mode_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Task Mode 观测，13D。

    布局：
    - current_semantic(4):[wheel_drive, leg_gait, obstacle_lift, jump]
    - prev_semantic(4):切换前模式语义
    - mode_blend(1)
    - jump_flag(1)
    - jump_target_height(1)
    - jump_phase(1)
    - jump_stage_norm(1)
    """
    cmd = env.command_manager.get_command("velocity_height")
    if cmd.shape[1] < 11:
        raise ValueError(
            f"task_mode_obs 要求 11 维指令 (TaskModeCommandTerm), 实际得到 {cmd.shape[1]} 维。"
        )

    mode_id = cmd[:, 8].round().long().clamp(0, TASK_MODE_COUNT - 1)
    prev_mode_id = cmd[:, 10].round().long().clamp(0, TASK_MODE_COUNT - 1)
    semantics = torch.tensor(TASK_MODE_SEMANTICS, device=cmd.device, dtype=cmd.dtype)
    current_semantic = semantics[mode_id]
    prev_semantic = semantics[prev_mode_id]
    mode_blend = cmd[:, 9:10]
    jump_obs = cmd[:, 5:8]

    term = env.command_manager.get_term("velocity_height")
    jump_stage = getattr(term, "jump_stage", None)
    if isinstance(jump_stage, torch.Tensor):
        jump_stage_norm = jump_stage.to(device=cmd.device, dtype=cmd.dtype).unsqueeze(1) / 2.0
    else:
        jump_stage_norm = torch.zeros(env.num_envs, 1, device=cmd.device)

    return torch.cat(
        [
            current_semantic,
            prev_semantic,
            mode_blend,
            jump_obs,
            jump_stage_norm,
        ],
        dim=1,
    )
