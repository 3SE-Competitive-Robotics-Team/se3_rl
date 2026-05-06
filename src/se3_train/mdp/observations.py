"""SE3 轮腿机器人的自定义观测函数。

27D 观测向量,使用 VMC(虚拟模型控制)极坐标。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_L1 = 0.180
_L2 = 0.200


def _compute_vmc(joint_pos: torch.Tensor, joint_vel: torch.Tensor):
    """计算 VMC 状态(L0, theta0, L0_dot, theta0_dot),双腿。

    关节顺序:lf0, lf1, l_wheel, rf0, rf1, r_wheel
    左腿:索引 0, 1。右腿:索引 3, 4。
    """
    # 左腿。
    th1_l = joint_pos[:, 0]
    th2_l = joint_pos[:, 1]
    th1_dot_l = joint_vel[:, 0]
    th2_dot_l = joint_vel[:, 1]

    # 右腿。
    th1_r = joint_pos[:, 3]
    th2_r = joint_pos[:, 4]
    th1_dot_r = joint_vel[:, 3]
    th2_dot_r = joint_vel[:, 4]

    # 正运动学:末端执行器位置。
    end_x_l = _L1 * torch.cos(th1_l) - _L2 * torch.sin(th1_l + th2_l)
    end_y_l = _L1 * torch.sin(th1_l) + _L2 * torch.cos(th1_l + th2_l)
    end_x_r = _L1 * torch.cos(th1_r) - _L2 * torch.sin(th1_r + th2_r)
    end_y_r = _L1 * torch.sin(th1_r) + _L2 * torch.cos(th1_r + th2_r)

    # 极坐标。
    L0_l = torch.sqrt(end_x_l**2 + end_y_l**2)
    theta0_l = torch.atan2(end_x_l, end_y_l)
    L0_r = torch.sqrt(end_x_r**2 + end_y_r**2)
    theta0_r = torch.atan2(end_x_r, end_y_r)

    # 有限差分速度(fd_dt = 1ms,与参考实现对齐)。
    fd_dt = 0.001
    end_x_l_n = _L1 * torch.cos(th1_l + th1_dot_l * fd_dt) - _L2 * torch.sin(
        th1_l + th2_l + (th1_dot_l + th2_dot_l) * fd_dt
    )
    end_y_l_n = _L1 * torch.sin(th1_l + th1_dot_l * fd_dt) + _L2 * torch.cos(
        th1_l + th2_l + (th1_dot_l + th2_dot_l) * fd_dt
    )
    end_x_r_n = _L1 * torch.cos(th1_r + th1_dot_r * fd_dt) - _L2 * torch.sin(
        th1_r + th2_r + (th1_dot_r + th2_dot_r) * fd_dt
    )
    end_y_r_n = _L1 * torch.sin(th1_r + th1_dot_r * fd_dt) + _L2 * torch.cos(
        th1_r + th2_r + (th1_dot_r + th2_dot_r) * fd_dt
    )

    L0_l_n = torch.sqrt(end_x_l_n**2 + end_y_l_n**2)
    theta0_l_n = torch.atan2(end_x_l_n, end_y_l_n)
    L0_r_n = torch.sqrt(end_x_r_n**2 + end_y_r_n**2)
    theta0_r_n = torch.atan2(end_x_r_n, end_y_r_n)

    L0_dot_l = (L0_l_n - L0_l) / fd_dt
    L0_dot_r = (L0_r_n - L0_r) / fd_dt

    theta0_diff_l = theta0_l_n - theta0_l
    theta0_diff_l = torch.remainder(theta0_diff_l + torch.pi, 2 * torch.pi) - torch.pi
    theta0_dot_l = theta0_diff_l / fd_dt

    theta0_diff_r = theta0_r_n - theta0_r
    theta0_diff_r = torch.remainder(theta0_diff_r + torch.pi, 2 * torch.pi) - torch.pi
    theta0_dot_r = theta0_diff_r / fd_dt

    return (
        theta0_l,
        theta0_r,
        theta0_dot_l,
        theta0_dot_r,
        L0_l,
        L0_r,
        L0_dot_l,
        L0_dot_r,
    )


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return robot.data.root_link_ang_vel_b * 0.25


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/高度指令,缩放 (2.0, 0.25, 5.0)。"""
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor([2.0, 0.25, 5.0], device=cmd.device)
    return cmd * scale


def theta0_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """双腿的 VMC 腿部角度。"""
    robot = env.scene["robot"]
    theta0_l, theta0_r, _, _, _, _, _, _ = _compute_vmc(robot.data.joint_pos, robot.data.joint_vel)
    return torch.stack([theta0_l, theta0_r], dim=-1)


def theta0_dot_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """VMC 腿部角速度,缩放 0.05。"""
    robot = env.scene["robot"]
    _, _, theta0_dot_l, theta0_dot_r, _, _, _, _ = _compute_vmc(
        robot.data.joint_pos, robot.data.joint_vel
    )
    return torch.stack([theta0_dot_l, theta0_dot_r], dim=-1) * 0.05


def L0_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """双腿的 VMC 腿部长度,缩放 5.0。"""
    robot = env.scene["robot"]
    _, _, _, _, L0_l, L0_r, _, _ = _compute_vmc(robot.data.joint_pos, robot.data.joint_vel)
    return torch.stack([L0_l, L0_r], dim=-1) * 5.0


def L0_dot_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """VMC 腿部长度变化速度,缩放 0.25。"""
    robot = env.scene["robot"]
    _, _, _, _, _, _, L0_dot_l, L0_dot_r = _compute_vmc(robot.data.joint_pos, robot.data.joint_vel)
    return torch.stack([L0_dot_l, L0_dot_r], dim=-1) * 0.25


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节位置,右轮取反对齐方向。"""
    robot = env.scene["robot"]
    wheel_pos = robot.data.joint_pos[:, [2, 5]].clone()
    wheel_pos[:, 1] = -wheel_pos[:, 1]
    return wheel_pos


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,右轮取反对齐方向,缩放 0.05。"""
    robot = env.scene["robot"]
    wheel_vel = robot.data.joint_vel[:, [2, 5]].clone()
    wheel_vel[:, 1] = -wheel_vel[:, 1]
    return wheel_vel * 0.05


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一步的 6 个动作。"""
    return env.action_manager.action
