import numpy as np

from .forward import forward_kinematics


def compute_velocity(theta1, theta2, theta1_dot, theta2_dot, l1=0.180, l2=0.200, fd_dt=0.001):
    """通过前向差分计算 L0 和 theta0 的速度。

    Args:
        theta1: 大腿角度 [rad]
        theta2: 膝关节角度 [rad]
        theta1_dot: 大腿角速度 [rad/s]
        theta2_dot: 膝关节角速度 [rad/s]
        l1: 大腿长度 [m]
        l2: 小腿长度 [m]
        fd_dt: 有限差分时间步长 [s]

    Returns:
        L0_dot: 腿长变化率 [m/s]
        theta0_dot: 腿部角速度 [rad/s]
    """
    xp = _get_array_module(theta1)

    L0, theta0 = forward_kinematics(theta1, theta2, l1, l2)
    L0_next, theta0_next = forward_kinematics(
        theta1 + theta1_dot * fd_dt, theta2 + theta2_dot * fd_dt, l1, l2
    )

    L0_dot = (L0_next - L0) / fd_dt

    theta0_diff = theta0_next - theta0
    theta0_diff = xp.remainder(theta0_diff + xp.pi, 2 * xp.pi) - xp.pi
    theta0_dot = theta0_diff / fd_dt

    return L0_dot, theta0_dot


def _get_array_module(x):
    if hasattr(x, "is_cuda"):
        import torch
        return torch
    return np
