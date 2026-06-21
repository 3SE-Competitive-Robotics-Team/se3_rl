"""SE3 轮腿机器人的观测函数。

观测空间:
- actor: 34D (腿部前杆使用 sin/cos 相位观测)
- critic: actor + 特权信息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import (
    ObservationConfig,
    output_to_policy_pos_torch,
    output_to_policy_vel_torch,
    policy_leg_phase_active_obs_torch,
)
from se3_train.mdp.contact_utils import (
    contact_force_nonfinite_env_mask,
    finite_contact_force_norm,
)
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
_DEFAULT_CONTACT_DEBUG_LOG_INTERVAL_STEPS = 256
_LEG_ENCODER_BIAS_ATTR = "_se3_leg_encoder_bias"


def _finite_clamp(value: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """把观测限制在有限范围内，避免单个发散 env 污染整批 PPO。"""
    bound = float(_OBS_CFG.clip_value if limit is None else limit)
    return torch.nan_to_num(value, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)


def _iteration_progress(
    env: ManagerBasedRlEnv,
    iteration_range: tuple[int, int],
    steps_per_policy_iter: int,
) -> float:
    """Return 0..1 progress through an iteration interval."""
    start, end = int(iteration_range[0]), int(iteration_range[1])
    if end <= start:
        return 1.0
    iteration = int(getattr(env, "common_step_counter", 0)) // max(1, int(steps_per_policy_iter))
    return min(max((iteration - start) / float(end - start), 0.0), 1.0)


def _interpolate_range(
    low_range: tuple[float, float],
    high_range: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    low_a, high_a = float(low_range[0]), float(low_range[1])
    low_b, high_b = float(high_range[0]), float(high_range[1])
    return (
        low_a + (low_b - low_a) * float(alpha),
        high_a + (high_b - high_a) * float(alpha),
    )


def _uniform_like_pos(pos: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
    low, high = float(value_range[0]), float(value_range[1])
    if high <= low:
        return torch.full_like(pos, low)
    return torch.empty_like(pos).uniform_(low, high)


def resample_leg_encoder_bias(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float] = (-0.01, 0.01),
    attr_name: str = _LEG_ENCODER_BIAS_ATTR,
) -> None:
    """Sample per-episode policy-joint encoder bias for [LF, LB, RF, RB]."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    if env_ids.numel() == 0:
        return
    bias = getattr(env, attr_name, None)
    if (
        not isinstance(bias, torch.Tensor)
        or bias.shape != (env.num_envs, 4)
        or bias.device != env.device
    ):
        bias = torch.zeros(env.num_envs, 4, device=env.device)
        setattr(env, attr_name, bias)
    low, high = float(bias_range[0]), float(bias_range[1])
    bias[env_ids] = torch.empty((env_ids.numel(), 4), device=env.device).uniform_(low, high)


def _apply_leg_encoder_error(
    env: ManagerBasedRlEnv,
    policy_pos: torch.Tensor,
    *,
    bias_range: tuple[float, float] | None,
    noise_range: tuple[float, float] | None,
    late_noise_range: tuple[float, float] | None,
    noise_ramp_iteration_range: tuple[int, int],
    steps_per_policy_iter: int,
    attr_name: str,
) -> torch.Tensor:
    """Apply encoder-level bias and white noise before sin/cos leg encoding."""
    noisy_pos = policy_pos
    if bias_range is not None:
        bias = getattr(env, attr_name, None)
        if (
            not isinstance(bias, torch.Tensor)
            or bias.shape != (env.num_envs, 4)
            or bias.device != policy_pos.device
        ):
            resample_leg_encoder_bias(env, None, bias_range=bias_range, attr_name=attr_name)
            bias = getattr(env, attr_name)
        noisy_pos = noisy_pos + bias.to(device=policy_pos.device, dtype=policy_pos.dtype)

    if noise_range is not None:
        active_noise_range = noise_range
        if late_noise_range is not None:
            alpha = _iteration_progress(
                env,
                noise_ramp_iteration_range,
                steps_per_policy_iter,
            )
            active_noise_range = _interpolate_range(noise_range, late_noise_range, alpha)
        noisy_pos = noisy_pos + _uniform_like_pos(noisy_pos, active_noise_range)
    return noisy_pos


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


def leg_joint_pos_obs(
    env: ManagerBasedRlEnv,
    encoder_bias_range: tuple[float, float] | None = None,
    encoder_noise_range: tuple[float, float] | None = None,
    late_encoder_noise_range: tuple[float, float] | None = None,
    noise_ramp_iteration_range: tuple[int, int] = (800, 1800),
    steps_per_policy_iter: int = 64,
    encoder_bias_attr: str = _LEG_ENCODER_BIAS_ATTR,
) -> torch.Tensor:
    """腿部主动杆相位和主动杆夹角观测，6D。"""
    robot = env.scene["robot"]
    leg_ids = policy_leg_joint_ids(robot)
    if is_fourbar_surrogate_model(robot):
        pos = output_to_policy_pos_torch(robot.data.joint_pos[:, leg_ids])
        default_pos = output_to_policy_pos_torch(robot.data.default_joint_pos[:, leg_ids])
    else:
        pos = robot.data.joint_pos[:, leg_ids]
        default_pos = robot.data.default_joint_pos[:, leg_ids]
    if encoder_bias_range is not None or encoder_noise_range is not None:
        pos = _apply_leg_encoder_error(
            env,
            pos,
            bias_range=encoder_bias_range,
            noise_range=encoder_noise_range,
            late_noise_range=late_encoder_noise_range,
            noise_ramp_iteration_range=noise_ramp_iteration_range,
            steps_per_policy_iter=steps_per_policy_iter,
            attr_name=encoder_bias_attr,
        )
    return _finite_clamp(
        policy_leg_phase_active_obs_torch(pos, default_pos)
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
    return _finite_clamp(robot.data.joint_vel[:, leg_ids] * _OBS_CFG.leg_vel_scale)


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """保留轮子位置观测槽位，但固定为 0，避免连续转角无限累计。"""
    robot = env.scene["robot"]
    wheel_ids = wheel_joint_ids(robot)
    return torch.zeros_like(robot.data.joint_pos[:, wheel_ids])


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
