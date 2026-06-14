"""SE3 轮腿机器人的观测函数。

观测空间:
- actor: 31D (原 27D + pitch/roll/height cmd 扩展)
- critic: actor + 特权信息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import (
    TASK_MODE_COUNT,
    TASK_MODE_SEMANTICS,
    ObservationConfig,
    output_to_policy_pos_torch,
    output_to_policy_vel_torch,
)
from se3_train.mdp.contact_utils import contact_force_nonfinite_env_mask, finite_contact_force_norm
from se3_train.mdp.joint_indices import (
    is_closedchain_model,
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
_DEFAULT_CONTACT_DEBUG_LOG_INTERVAL_STEPS = 256


def _finite_clamp(value: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """把观测限制在有限范围内，避免单个发散 env 污染整批 PPO。"""
    bound = float(_OBS_CFG.clip_value if limit is None else limit)
    return torch.nan_to_num(value, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)


def _recovery_obs_model(env: ManagerBasedRlEnv) -> bool:
    """判断当前模型是否需要 recovery/fourbar 观测防发散保护。"""
    robot = env.scene["robot"]
    return is_closedchain_model(robot) or is_fourbar_surrogate_model(robot)


def _maybe_finite_clamp(
    env: ManagerBasedRlEnv,
    value: torch.Tensor,
    limit: float | None = None,
) -> torch.Tensor:
    """main openchain 保持原观测；recovery/fourbar 才做有限值保护。"""
    if _recovery_obs_model(env):
        return _finite_clamp(value, limit=limit)
    return value


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return _maybe_finite_clamp(env, robot.data.root_link_ang_vel_b * _OBS_CFG.ang_vel_scale)


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return _maybe_finite_clamp(env, robot.data.projected_gravity_b, limit=1.0)


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/姿态/高度指令,缩放 (2.0, 0.25, 5.0, 5.0, 5.0)。

    取前 5 维: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    兼容跳跃任务（8 维指令），跳跃扩展维度由 jump_commands_obs 单独输出。
    """
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor(list(_OBS_CFG.command_scale), device=cmd.device)
    return _maybe_finite_clamp(env, cmd[:, :5] * scale)


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部主动杆位置（相对默认位姿），4D。"""
    robot = env.scene["robot"]
    leg_ids = policy_leg_joint_ids(robot)
    if is_fourbar_surrogate_model(robot):
        pos = output_to_policy_pos_torch(robot.data.joint_pos[:, leg_ids])
        default_pos = output_to_policy_pos_torch(robot.data.default_joint_pos[:, leg_ids])
        return _finite_clamp(pos - default_pos)
    value = robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
    return _maybe_finite_clamp(env, value)


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
    return _maybe_finite_clamp(env, robot.data.joint_vel[:, leg_ids] * _OBS_CFG.leg_vel_scale)


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子位置观测；四连杆/闭链任务固定为 0，openchain 保持 main 语义。"""
    robot = env.scene["robot"]
    wheel_ids = wheel_joint_ids(robot)
    if not (is_closedchain_model(robot) or is_fourbar_surrogate_model(robot)):
        return robot.data.joint_pos[:, wheel_ids]
    return torch.zeros_like(robot.data.joint_pos[:, wheel_ids])


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,缩放 0.05（MJCF 已修正轴方向）。"""
    robot = env.scene["robot"]
    value = robot.data.joint_vel[:, wheel_joint_ids(robot)] * _OBS_CFG.wheel_vel_scale
    return _maybe_finite_clamp(env, value)


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一步的 6 个动作。"""
    return _maybe_finite_clamp(env, env.action_manager.action)


# --- Critic 特权观测 ---


def base_lin_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的线速度,3D（特权信息,actor 不可见）。"""
    robot = env.scene["robot"]
    return _maybe_finite_clamp(env, robot.data.root_link_lin_vel_b)


def wheel_contact_force_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """轮子地面接触力标量,2D（特权信息）。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    _record_wheel_contact_force_nonfinite(env, data.force)
    return _maybe_finite_clamp(env, finite_contact_force_norm(data.force))


def _record_wheel_contact_force_nonfinite(env: ManagerBasedRlEnv, force: torch.Tensor) -> None:
    """累计 wheel contact force 的非有限值触发次数，并写入训练日志。"""
    step = int(getattr(env, "common_step_counter", 0))
    interval = max(
        1,
        int(
            getattr(
                env,
                "_se3_contact_debug_log_interval_steps",
                _DEFAULT_CONTACT_DEBUG_LOG_INTERVAL_STEPS,
            )
        ),
    )
    if interval > 1 and (step - 1) % interval != 0:
        return

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
    if _recovery_obs_model(env):
        return torch.nan_to_num(sensor.data.heights, nan=0.0, posinf=0.0, neginf=0.0)
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
    return _maybe_finite_clamp(env, cmd[:, 5:8])


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

    obs = torch.cat(
        [
            current_semantic,
            prev_semantic,
            mode_blend,
            jump_obs,
            jump_stage_norm,
        ],
        dim=1,
    )
    return _maybe_finite_clamp(env, obs)
