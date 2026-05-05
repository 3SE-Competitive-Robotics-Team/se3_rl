"""Quaternion helpers for MuJoCo wxyz data."""

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
