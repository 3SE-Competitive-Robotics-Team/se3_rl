"""MuJoCo wxyz 四元数与角度工具。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def rotate(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).apply(vec_xyz)


def rotate_inverse(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).inv().apply(vec_xyz)


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return quat_xyzw_to_wxyz(Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat())


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """把角度包裹到单圈范围内，推荐按 ``[-π, π)`` 来理解返回值。"""

    arr = np.asarray(angle, dtype=np.float64)
    wrapped = (arr + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.shape == ():
        return float(wrapped)
    return wrapped


def extract_yaw(quat_wxyz: np.ndarray) -> float:
    """从 MuJoCo 的 ``wxyz`` 四元数中提取世界系 yaw。"""

    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"quat_wxyz shape mismatch: expected (4,), got {quat.shape}")
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))
