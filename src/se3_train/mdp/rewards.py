"""SE3 轮腿机器人的奖励函数。

所有奖励函数接收 env 并返回 [num_envs] 的张量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from se3_shared import Joint, JointGroup
from se3_train.mdp.contact_utils import finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _recovery_reset_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 recovery reset 标记；普通任务没有该标记时全 False。"""
    mask = getattr(env, "_recovery_reset_mask", None)
    if not isinstance(mask, torch.Tensor) or mask.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return mask.to(device=env.device, dtype=torch.bool)


def _upright_factor(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """计算直立因子:clamp(-pg_z, 0, 0.7) / 0.7。"""
    return torch.clamp(-projected_gravity_z, 0.0, 0.7) / 0.7


def _recovery_penalty_gate(
    env: ManagerBasedRlEnv, projected_gravity_z: torch.Tensor
) -> torch.Tensor:
    """在宽限期内使用 1.0,否则使用直立因子。"""
    grace_steps = 74
    upright = _upright_factor(projected_gravity_z)
    in_grace = env.episode_length_buf < grace_steps
    return torch.where(in_grace, torch.ones_like(upright), upright)


def tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float = 2.0,
) -> torch.Tensor:
    """x 方向速度跟踪,将 v_z 折入同一 exp 核消除目标冲突。

    reward = exp(-(error_x² + vz_weight·v_z²) / sigma)
    低速时 sigma 收紧(adaptive),直立门控。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    lin_vel = robot.data.root_link_lin_vel_b
    error_x = lin_vel[:, 0] - cmd[:, 0]
    vz = lin_vel[:, 2]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    cmd_mag = torch.abs(cmd[:, 0])
    sigma = torch.where(cmd_mag < 0.2, sigma_stand, sigma_move)
    return torch.exp(-(error_x**2 + vz_weight * vz**2) / sigma) * gate


def tracking_ang_vel(env: ManagerBasedRlEnv, command_name: str, sigma: float) -> torch.Tensor:
    """偏航角速度跟踪的 exp(-error^2/sigma),直立门控。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_z = robot.data.root_link_ang_vel_b[:, 2]
    error = ang_vel_z - cmd[:, 1]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return torch.exp(-(error**2) / sigma) * gate


def tracking_orientation_l2(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """pitch/roll 姿态 L2 惩罚，提供 Tron 风格的持续回正梯度。

    L2 惩罚会随误差平方增长，小倾斜也有明确负反馈。行走任务保留 pitch/roll
    指令语义，惩罚相对目标姿态的误差；跳跃任务中 pitch/roll 指令固定为 0，
    因此等价于 flat orientation L2。
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pg = robot.data.projected_gravity_b

    current_pitch = torch.asin(torch.clamp(pg[:, 0], -1.0, 1.0))
    current_roll = torch.asin(torch.clamp(-pg[:, 1], -1.0, 1.0))

    pitch_error = current_pitch - cmd[:, 2]
    roll_error = current_roll - cmd[:, 3]
    return pitch_error**2 + roll_error**2


def tracking_height(
    env: ManagerBasedRlEnv, command_name: str, sigma: float, height_sensor_name: str
) -> torch.Tensor:
    """高度跟踪奖励,不门控（始终提供恢复梯度）。"""
    cmd = env.command_manager.get_command(command_name)

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    target_height = cmd[:, 4]
    error = torch.square(height - target_height)
    return torch.exp(-error / sigma)


def bad_tilt(
    env: ManagerBasedRlEnv,
    soft_limit_deg: float = 12.0,
    hard_limit_deg: float = 35.0,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """坏姿态 barrier 惩罚。

    小倾斜由 tracking_orientation_l2 处理；超过 soft_limit 后快速加重惩罚，
    避免策略把明显歪斜当成可接受状态。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    soft = torch.deg2rad(torch.tensor(float(soft_limit_deg), device=env.device))
    hard = torch.deg2rad(torch.tensor(float(hard_limit_deg), device=env.device))
    span = torch.clamp(hard - soft, min=1.0e-6)
    excess = torch.clamp((tilt - soft) / span, min=0.0)
    return torch.clamp(excess**2, max=float(max_penalty))


def lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座 z 方向速度的平方,直立门控。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    return robot.data.root_link_lin_vel_b[:, 2] ** 2 * gate


def ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    """横滚/俯仰角速度平方和,直立门控。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)
    ang_vel = robot.data.root_link_ang_vel_b
    return (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2) * gate


def angular_momentum(env: ManagerBasedRlEnv) -> torch.Tensor:
    """全身角动量范数平方,直立门控。

    使用 MuJoCo subtree_angmom[root_body] 获取整个机器人子树的角动量。
    对两轮倒立摆尤为重要——腿部激进动作产生的角动量脉冲会直接导致失稳。
    """
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    root_body_id = robot.data.indexing.root_body_id
    angmom = env.sim.data.subtree_angmom[:, root_body_id]
    return torch.sum(angmom**2, dim=-1) * gate


def leg_torques(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部执行器力矩平方和（position actuator 索引 0,1,2,3）。"""
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force[:, JointGroup.LEG_ACTUATORS]
    return torch.sum(torques**2, dim=1)


def wheel_torques(
    env: ManagerBasedRlEnv,
    max_torque: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """轮子执行器力矩超出额定值的平方和。

    max_torque: 轮子电机额定最大力矩 (N·m)。
    """
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force[:, JointGroup.WHEEL_ACTUATORS]
    excess = torch.clamp(torch.abs(torques) - max_torque, min=0.0)
    return torch.sum(excess**2, dim=1)


def leg_dof_acc(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部关节加速度平方和(排除轮子)。"""
    robot = env.scene[asset_cfg.name]
    acc = robot.data.joint_acc
    return torch.sum(acc[:, JointGroup.LEGS] ** 2, dim=1)


def leg_power(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """腿部关节 |力矩 * 速度| 之和。"""
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force[:, JointGroup.LEG_ACTUATORS]
    vel = robot.data.joint_vel[:, JointGroup.LEGS]
    return torch.sum(torch.abs(torques * vel), dim=1)


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """当前动作与上一动作差值的平方和。"""
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    ignore_recovery: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """站立时关节偏差平方和,高度自适应 sigma。

    当 cmd_height 偏离 default_height 时,惩罚自动放松,
    避免高度指令与姿态惩罚的结构性矛盾。
    衰减因子: exp(-height_tolerance * (cmd_h - default_h)²)
    """
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    diff = (
        robot.data.joint_pos[:, JointGroup.LEGS] - robot.data.default_joint_pos[:, JointGroup.LEGS]
    )
    reward = torch.sum(diff**2, dim=1)

    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    vel_scale = (cmd_norm <= command_threshold).float()

    height_deviation = cmd[:, 4] - default_height
    height_scale = torch.exp(-height_tolerance * height_deviation**2)

    result = reward * vel_scale * height_scale * gate
    if ignore_recovery:
        result = result * (~_recovery_reset_mask(env)).float()
    return result


def recovery_upright(env: ManagerBasedRlEnv) -> torch.Tensor:
    """倒地恢复期直立奖励。

    使用 projected_gravity 的 z 分量构造连续信号：侧躺约 0.5，直立为 1。
    这让策略在大倾角时也能得到非零恢复梯度。
    """
    robot = env.scene["robot"]
    active = _recovery_reset_mask(env)
    pg_z = robot.data.projected_gravity_b[:, 2]
    upright = torch.clamp((-pg_z + 1.0) * 0.5, 0.0, 1.0)

    if hasattr(env, "extras"):
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/reset_ratio": active.float().mean().item(),
                "Recovery/tilt_deg": tilt[active].mean().item() if active.any() else 0.0,
                "Recovery/upright_score": upright[active].mean().item() if active.any() else 0.0,
            }
        )

    return upright.square() * active.float()


def recovery_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    sigma: float = 0.04,
) -> torch.Tensor:
    """倒地恢复期 base 高度奖励，目标高度沿用当前站立高度指令。"""
    active = _recovery_reset_mask(env)
    cmd = env.command_manager.get_command(command_name)
    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    target_height = cmd[:, 4]
    reward = torch.exp(-torch.square(height - target_height) / float(sigma))

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/base_height_error_m": torch.abs(height - target_height)[active]
                .mean()
                .item()
                if active.any()
                else 0.0,
            }
        )

    return reward * active.float()


def recovery_stability(env: ManagerBasedRlEnv, ang_vel_weight: float = 0.25) -> torch.Tensor:
    """倒地恢复期稳定性惩罚：压制恢复后的大角速度和上下弹跳。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    lin_vel = robot.data.root_link_lin_vel_b
    ang_vel = robot.data.root_link_ang_vel_b
    penalty = lin_vel[:, 2].square() + float(ang_vel_weight) * torch.sum(ang_vel.square(), dim=1)
    return penalty * active.float()


def recovery_wheel_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """倒地恢复期轮子接地奖励，鼓励最终回到双轮支撑。"""
    active = _recovery_reset_mask(env)
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    contact = (force_mag > float(force_threshold)).float()
    return torch.mean(contact, dim=1) * active.float()


def recovery_success(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    upright_angle_deg: float = 15.0,
    height_tolerance: float = 0.05,
    ang_vel_threshold: float = 1.5,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """恢复成功奖励：直立、高度达标、角速度低且至少一个轮子接地。"""
    active = _recovery_reset_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    upright_limit = torch.deg2rad(torch.tensor(float(upright_angle_deg), device=env.device))
    upright = tilt < upright_limit

    cmd = env.command_manager.get_command(command_name)
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height_ok = torch.abs(height_sensor.data.heights[:, 0] - cmd[:, 4]) < float(height_tolerance)

    ang_vel_norm = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)
    stable = ang_vel_norm < float(ang_vel_threshold)

    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    if data.force is None:
        wheel_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        force_mag = finite_contact_force_norm(data.force)
        wheel_contact = (force_mag > float(force_threshold)).any(dim=1)

    success = active & upright & height_ok & stable & wheel_contact
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/success_rate": success.float().mean().item(),
                "Recovery/wheel_contact_rate": (active & wheel_contact).float().mean().item(),
            }
        )

    return success.float()


def joint_mirror(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """左右关节位置差的平均平方,直立门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    diff = (
        robot.data.joint_pos[:, [Joint.LF0, Joint.LF1]]
        - robot.data.joint_pos[:, [Joint.RF0, Joint.RF1]]
    )
    num_pairs = 2
    return torch.sum(diff**2, dim=1) / num_pairs * gate


def collision(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """受惩罚的身体接触计数,恢复惩罚门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    contact_count = (force_mag > 0.1).float().sum(dim=1)
    return contact_count * gate


def contact_forces(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """轮子接触力超过阈值的部分,除以 100 归一化,恢复门控。"""
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _recovery_penalty_gate(env, pg_z)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    excess = torch.clamp(force_mag - threshold, min=0.0) / 100.0
    return torch.sum(excess, dim=1) * gate


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    force_threshold: float,
    cmd_threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """静止时轮子接触,直立门控。"""
    cmd = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    pg_z = robot.data.projected_gravity_b[:, 2]
    gate = _upright_factor(pg_z)

    stationary = torch.abs(cmd[:, 0]) < cmd_threshold

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    has_contact = (force_mag > force_threshold).float()
    return torch.sum(has_contact, dim=1) * gate * stationary.float()


def dof_pos_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """关节限位违规惩罚(仅腿部关节:0,1,3,4)。"""
    robot = env.scene[asset_cfg.name]
    soft_limits = robot.data.soft_joint_pos_limits
    if soft_limits is None:
        return torch.zeros(env.num_envs, device=env.device)

    pos = robot.data.joint_pos[:, JointGroup.LEGS]
    limits = soft_limits[:, JointGroup.LEGS]

    out_of_limits = -(pos - limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (pos - limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
