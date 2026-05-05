import numpy as np


def map_virtual_to_joint_torques(force, torque, theta1, theta2, l0, l1=0.180, l2=0.200):
    """将极坐标空间中的虚拟力/力矩映射为关节力矩。

    Args:
        force: 沿腿部轴线的虚拟力 [N]
        torque: 绕腿部支点的虚拟力矩 [Nm]
        theta1: 大腿角度 [rad]
        theta2: 膝关节角度 [rad]
        l0: 当前腿长 [m]
        l1: 大腿长度 [m]
        l2: 小腿长度 [m]

    Returns:
        T1: 大腿关节力矩 [Nm]
        T2: 膝关节力矩 [Nm]
    """
    xp = _get_array_module(theta1)

    if hasattr(l0, "is_cuda"):
        safe_l0 = xp.clamp(l0, min=1e-4)
    else:
        safe_l0 = xp.clip(l0, 1e-4, None)

    s1 = xp.sin(theta1)
    c1 = xp.cos(theta1)
    s12 = xp.sin(theta1 + theta2)
    c12 = xp.cos(theta1 + theta2)

    dx_dtheta1 = -l1 * s1 - l2 * c12
    dy_dtheta1 = l1 * c1 - l2 * s12
    dx_dtheta2 = -l2 * c12
    dy_dtheta2 = -l2 * s12

    inv_l0 = safe_l0**-1
    inv_l0_sq = safe_l0**-2

    dL0_dx = inv_l0 * (l1 * c1 - l2 * s12)
    dL0_dy = inv_l0 * (l1 * s1 + l2 * c12)
    dphi_dx = inv_l0_sq * (l1 * s1 + l2 * c12)
    dphi_dy = -inv_l0_sq * (l1 * c1 - l2 * s12)

    J11 = dL0_dx * dx_dtheta1 + dL0_dy * dy_dtheta1
    J12 = dL0_dx * dx_dtheta2 + dL0_dy * dy_dtheta2
    J21 = dphi_dx * dx_dtheta1 + dphi_dy * dy_dtheta1
    J22 = dphi_dx * dx_dtheta2 + dphi_dy * dy_dtheta2

    T1 = force * J11 + torque * J21
    T2 = force * J12 + torque * J22

    T1 = _nan_to_num(T1, xp)
    T2 = _nan_to_num(T2, xp)

    return T1, T2


def _nan_to_num(x, xp):
    if hasattr(x, "is_cuda"):
        import torch
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return xp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _get_array_module(x):
    if hasattr(x, "is_cuda"):
        import torch
        return torch
    return np
