"""VMC kinematics matching the Isaac Gym training environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VMCState:
    l0: np.ndarray
    theta0: np.ndarray
    l0_dot: np.ndarray
    theta0_dot: np.ndarray


class VMCKinematics:
    def __init__(self, *, l1: float, l2: float) -> None:
        self.l1 = float(l1)
        self.l2 = float(l2)

    def forward(self, theta1, theta2) -> tuple[np.ndarray, np.ndarray]:
        theta1 = np.asarray(theta1, dtype=np.float64)
        theta2 = np.asarray(theta2, dtype=np.float64)
        end_x = self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2)
        end_y = self.l1 * np.sin(theta1) + self.l2 * np.cos(theta1 + theta2)
        l0 = np.sqrt(end_x**2 + end_y**2)
        theta0 = np.arctan2(end_x, end_y)
        return l0, theta0

    def velocities(self, theta1, theta2, theta1_dot, theta2_dot, *, dt: float) -> tuple[np.ndarray, np.ndarray]:
        theta1 = np.asarray(theta1, dtype=np.float64)
        theta2 = np.asarray(theta2, dtype=np.float64)
        theta1_dot = np.asarray(theta1_dot, dtype=np.float64)
        theta2_dot = np.asarray(theta2_dot, dtype=np.float64)
        l0, theta0 = self.forward(theta1, theta2)
        l0_next, theta0_next = self.forward(theta1 + theta1_dot * dt, theta2 + theta2_dot * dt)
        theta0_diff = (theta0_next - theta0 + np.pi) % (2.0 * np.pi) - np.pi
        return (l0_next - l0) / dt, theta0_diff / dt

    def state_from_dofs(self, dof_pos: np.ndarray, dof_vel: np.ndarray, *, fd_dt: float) -> tuple[np.ndarray, np.ndarray, VMCState]:
        theta1 = np.asarray([dof_pos[0], dof_pos[3]], dtype=np.float64)
        theta2 = np.asarray([dof_pos[1], dof_pos[4]], dtype=np.float64)
        theta1_dot = np.asarray([dof_vel[0], dof_vel[3]], dtype=np.float64)
        theta2_dot = np.asarray([dof_vel[1], dof_vel[4]], dtype=np.float64)
        l0, theta0 = self.forward(theta1, theta2)
        l0_dot, theta0_dot = self.velocities(theta1, theta2, theta1_dot, theta2_dot, dt=fd_dt)
        return theta1, theta2, VMCState(l0=l0, theta0=theta0, l0_dot=l0_dot, theta0_dot=theta0_dot)

    def map_virtual_to_joint_torques(
        self,
        *,
        force: np.ndarray,
        torque: np.ndarray,
        theta1: np.ndarray,
        theta2: np.ndarray,
        l0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        l0_safe = np.maximum(np.asarray(l0, dtype=np.float64), 1e-4)
        theta1 = np.asarray(theta1, dtype=np.float64)
        theta2 = np.asarray(theta2, dtype=np.float64)
        force = np.asarray(force, dtype=np.float64)
        torque = np.asarray(torque, dtype=np.float64)

        dx_dtheta1 = -self.l1 * np.sin(theta1) - self.l2 * np.cos(theta1 + theta2)
        dy_dtheta1 = self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2)
        dx_dtheta2 = -self.l2 * np.cos(theta1 + theta2)
        dy_dtheta2 = -self.l2 * np.sin(theta1 + theta2)

        d_l0_dx = l0_safe**-1 * (self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2))
        d_l0_dy = l0_safe**-1 * (self.l1 * np.sin(theta1) + self.l2 * np.cos(theta1 + theta2))
        d_phi_dx = l0_safe**-2 * (self.l1 * np.sin(theta1) + self.l2 * np.cos(theta1 + theta2))
        d_phi_dy = -l0_safe**-2 * (self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2))

        j11 = d_l0_dx * dx_dtheta1 + d_l0_dy * dy_dtheta1
        j12 = d_l0_dx * dx_dtheta2 + d_l0_dy * dy_dtheta2
        j21 = d_phi_dx * dx_dtheta1 + d_phi_dy * dy_dtheta1
        j22 = d_phi_dx * dx_dtheta2 + d_phi_dy * dy_dtheta2

        t1 = np.nan_to_num(force * j11 + torque * j21, nan=0.0, posinf=0.0, neginf=0.0)
        t2 = np.nan_to_num(force * j12 + torque * j22, nan=0.0, posinf=0.0, neginf=0.0)
        return t1, t2
