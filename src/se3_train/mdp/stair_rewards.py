"""台阶训练奖励和诊断项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_stair_state(env: ManagerBasedRlEnv):
    """读取当前环境的 CTBC 状态机；没有台阶状态时返回 None。"""
    return getattr(env, "stair_climb_state", None)


def _upright_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """只在接近直立时施加行走类惩罚，避免倒地恢复样本被压制。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    return torch.clamp(-pg_z, 0.0, 0.7) / 0.7


def _ctbc_triggered_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 CTBC 前馈当前是否激活，兼容参考代码的 contact_triggered 名称。"""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    active_mask = getattr(state, "active_mask", None)
    if callable(active_mask):
        return active_mask().to(device=env.device, dtype=torch.bool)

    contact_triggered = getattr(state, "contact_triggered", None)
    if callable(contact_triggered):
        return contact_triggered().to(device=env.device, dtype=torch.bool)

    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def stair_climb_height(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    max_gain: float = 0.35,
) -> torch.Tensor:
    """奖励机器人相对指令站高的净爬升高度。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    base_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    climb = torch.clamp(base_z - cmd[:, 4], min=0.0, max=float(max_gain))
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/stair_climb_height": climb.mean().item(),
                "Stair/stair_base_z": base_z.mean().item(),
                "Stair/stair_command_height": cmd[:, 4].mean().item(),
            }
        )
    return climb


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    sigma: float = 0.25,
) -> torch.Tensor:
    """台阶前进速度跟踪奖励，保留向前梯度。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    error = torch.square(robot.data.root_link_lin_vel_b[:, 0] - cmd[:, 0])
    reward = torch.exp(-error / float(sigma))
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["Stair/forward_speed_error"] = torch.sqrt(error).mean().item()
    return reward


def stair_contact_diagnostics(
    env: ManagerBasedRlEnv,
    wheel_sensor_name: str = "wheel_sensor",
    leg_sensor_name: str = "leg_contact_sensor",
    collision_sensor_name: str = "collision_sensor",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """记录台阶训练关心的接触诊断，返回零奖励。"""
    wheel_contact = _contact_any(env, wheel_sensor_name, force_threshold)
    leg_contact = _contact_any(env, leg_sensor_name, force_threshold)
    collision_contact = _contact_any(env, collision_sensor_name, force_threshold)
    nonwheel = leg_contact | collision_contact
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/wheel_contact_rate": wheel_contact.float().mean().item(),
                "Stair/nonwheel_contact_rate": nonwheel.float().mean().item(),
                "Stair/leg_contact_rate": leg_contact.float().mean().item(),
                "Stair/base_collision_rate": collision_contact.float().mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


def ctbc_active_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 CTBC 当前是否激活，用于后续门控奖励。"""
    return _ctbc_triggered_mask(env)


def leg_torques_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内暂时豁免腿部力矩惩罚。"""
    from se3_train.mdp import rewards

    penalty = rewards.leg_torques(env, asset_cfg=asset_cfg)
    triggered = _ctbc_triggered_mask(env)
    return penalty * (~triggered).float() * _upright_gate(env)


def leg_power_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内暂时豁免腿部功率惩罚。"""
    from se3_train.mdp import rewards

    penalty = rewards.leg_power(env, asset_cfg=asset_cfg)
    triggered = _ctbc_triggered_mask(env)
    return penalty * (~triggered).float() * _upright_gate(env)


def stand_still_no_ctbc(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内不惩罚偏离默认站姿。"""
    from se3_train.mdp import rewards

    penalty = rewards.stand_still(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        default_height=default_height,
        height_tolerance=height_tolerance,
        asset_cfg=asset_cfg,
    )
    triggered = _ctbc_triggered_mask(env)
    return penalty * (~triggered).float()


def action_rate_no_ctbc(env: ManagerBasedRlEnv) -> torch.Tensor:
    """CTBC 抬腿周期内降低动作变化率惩罚，保留少量平滑约束。"""
    from se3_train.mdp import rewards

    penalty = rewards.action_rate(env)
    triggered = _ctbc_triggered_mask(env).float()
    scale = 1.0 - 0.8 * triggered
    return penalty * scale * _upright_gate(env)


def contact_forces_no_ctbc(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内放宽轮子接触力惩罚，仍约束极端硬撞。"""
    from se3_train.mdp import rewards

    penalty = rewards.contact_forces(
        env,
        threshold=threshold,
        sensor_name=sensor_name,
        asset_cfg=asset_cfg,
    )
    triggered = _ctbc_triggered_mask(env).float()
    return penalty * (1.0 - 0.5 * triggered)


def _contact_any(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force = torch.linalg.norm(data.force, dim=-1).reshape(env.num_envs, -1)
    return (force > float(force_threshold)).any(dim=1)
