import numpy as np


def forward_kinematics(theta1, theta2, l1=0.180, l2=0.200):
    """将关节角度转换为极坐标 (L0, theta0)。

    Args:
        theta1: 大腿角度 [rad]
        theta2: 膝关节角度 [rad]
        l1: 大腿长度 [m]
        l2: 小腿长度 [m]

    Returns:
        L0: 腿长 [m]
        theta0: 腿部角度 [rad]
    """
    xp = _get_array_module(theta1)

    end_x = l1 * xp.cos(theta1) - l2 * xp.sin(theta1 + theta2)
    end_y = l1 * xp.sin(theta1) + l2 * xp.cos(theta1 + theta2)

    L0 = xp.sqrt(end_x**2 + end_y**2)
    theta0 = xp.arctan2(end_x, end_y)

    return L0, theta0


def _get_array_module(x):
    """根据输入类型返回 numpy 或 torch 模块。"""
    if hasattr(x, "is_cuda"):
        import torch
        return torch
    return np
