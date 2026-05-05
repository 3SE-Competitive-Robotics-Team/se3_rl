"""SE3 轮腿机器人的 VMC 动作项。

动作空间为 6D：[theta0_ref_L, l0_ref_L, wheel_vel_ref_L,
                theta0_ref_R, l0_ref_R, wheel_vel_ref_R]

通过 VMC 雅可比将虚拟极坐标空间指令映射为关节力矩。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class VMCActionTermCfg(ActionTermCfg):
    """VMC 动作项的配置。"""

    kp_theta: float = 22.0
    kd_theta: float = 1.5
    kp_l0: float = 2000.0
    kd_l0: float = 29.65
    wheel_kd: float = 0.1
    feedforward_mass: float = 12.61
    l0_offset: float = 0.22
    l1: float = 0.180
    l2: float = 0.200

    def build(self, env: ManagerBasedRlEnv) -> VMCActionTerm:
        return VMCActionTerm(self, env)


class VMCActionTerm(ActionTerm):
    """基于 VMC 的动作项，将极坐标空间指令映射为关节力矩。"""

    cfg: VMCActionTermCfg

    def __init__(self, cfg: VMCActionTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self._action_dim = 6
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # PD 增益。
        self._kp_theta = cfg.kp_theta
        self._kd_theta = cfg.kd_theta
        self._kp_l0 = cfg.kp_l0
        self._kd_l0 = cfg.kd_l0
        self._wheel_kd = cfg.wheel_kd
        self._feedforward_mass = cfg.feedforward_mass
        self._l0_offset = cfg.l0_offset
        self._l1 = cfg.l1
        self._l2 = cfg.l2

        # 重力常数。
        self._g = 9.81

        # 力矩缓冲区。
        self._torques = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """将原始动作缩放为 VMC 参考值。"""
        self._raw_actions[:] = actions

        # 缩放：theta0_ref = action * pi, l0_ref = action * 0.096 + 0.22,
        # wheel_vel_ref = action * 25.0。
        self._processed_actions[:, 0] = actions[:, 0] * 3.14  # theta0_ref_L
        self._processed_actions[:, 1] = actions[:, 1] * 0.096 + self._l0_offset  # l0_ref_L
        self._processed_actions[:, 2] = actions[:, 2] * 25.0  # wheel_vel_ref_L
        self._processed_actions[:, 3] = actions[:, 3] * 3.14  # theta0_ref_R
        self._processed_actions[:, 4] = actions[:, 4] * 0.096 + self._l0_offset  # l0_ref_R
        self._processed_actions[:, 5] = actions[:, 5] * 25.0  # wheel_vel_ref_R

    def apply_actions(self) -> None:
        """计算 VMC 力矩并写入仿真。"""
        robot = self._env.scene["robot"]
        joint_pos = robot.data.joint_pos  # [num_envs, 6]
        joint_vel = robot.data.joint_vel  # [num_envs, 6]
        pg = robot.data.projected_gravity_b  # [num_envs, 3]

        theta0_ref_l = self._processed_actions[:, 0]
        l0_ref_l = self._processed_actions[:, 1]
        wheel_vel_ref_l = self._processed_actions[:, 2]
        theta0_ref_r = self._processed_actions[:, 3]
        l0_ref_r = self._processed_actions[:, 4]
        wheel_vel_ref_r = self._processed_actions[:, 5]

        # 当前每条腿的 VMC 状态。
        th1_l = joint_pos[:, 0]
        th2_l = joint_pos[:, 1]
        th1_dot_l = joint_vel[:, 0]
        th2_dot_l = joint_vel[:, 1]

        th1_r = joint_pos[:, 3]
        th2_r = joint_pos[:, 4]
        th1_dot_r = joint_vel[:, 3]
        th2_dot_r = joint_vel[:, 4]

        # 正运动学计算当前 L0, theta0。
        end_x_l = self._l1 * torch.cos(th1_l) - self._l2 * torch.sin(th1_l + th2_l)
        end_y_l = self._l1 * torch.sin(th1_l) + self._l2 * torch.cos(th1_l + th2_l)
        L0_l = torch.sqrt(end_x_l**2 + end_y_l**2)
        theta0_l = torch.atan2(end_x_l, end_y_l)

        end_x_r = self._l1 * torch.cos(th1_r) - self._l2 * torch.sin(th1_r + th2_r)
        end_y_r = self._l1 * torch.sin(th1_r) + self._l2 * torch.cos(th1_r + th2_r)
        L0_r = torch.sqrt(end_x_r**2 + end_y_r**2)
        theta0_r = torch.atan2(end_x_r, end_y_r)

        # 有限差分速度。
        fd_dt = 0.005
        end_x_l_n = self._l1 * torch.cos(th1_l + th1_dot_l * fd_dt) - self._l2 * torch.sin(
            th1_l + th2_l + (th1_dot_l + th2_dot_l) * fd_dt
        )
        end_y_l_n = self._l1 * torch.sin(th1_l + th1_dot_l * fd_dt) + self._l2 * torch.cos(
            th1_l + th2_l + (th1_dot_l + th2_dot_l) * fd_dt
        )
        L0_l_n = torch.sqrt(end_x_l_n**2 + end_y_l_n**2)
        theta0_l_n = torch.atan2(end_x_l_n, end_y_l_n)
        L0_dot_l = (L0_l_n - L0_l) / fd_dt
        theta0_diff_l = theta0_l_n - theta0_l
        theta0_diff_l = torch.remainder(theta0_diff_l + torch.pi, 2 * torch.pi) - torch.pi
        theta0_dot_l = theta0_diff_l / fd_dt

        end_x_r_n = self._l1 * torch.cos(th1_r + th1_dot_r * fd_dt) - self._l2 * torch.sin(
            th1_r + th2_r + (th1_dot_r + th2_dot_r) * fd_dt
        )
        end_y_r_n = self._l1 * torch.sin(th1_r + th1_dot_r * fd_dt) + self._l2 * torch.cos(
            th1_r + th2_r + (th1_dot_r + th2_dot_r) * fd_dt
        )
        L0_r_n = torch.sqrt(end_x_r_n**2 + end_y_r_n**2)
        theta0_r_n = torch.atan2(end_x_r_n, end_y_r_n)
        L0_dot_r = (L0_r_n - L0_r) / fd_dt
        theta0_diff_r = theta0_r_n - theta0_r
        theta0_diff_r = torch.remainder(theta0_diff_r + torch.pi, 2 * torch.pi) - torch.pi
        theta0_dot_r = theta0_diff_r / fd_dt

        # 极坐标空间 PD 控制。
        torque_leg_l = self._kp_theta * (theta0_ref_l - theta0_l) - self._kd_theta * theta0_dot_l
        force_leg_l = self._kp_l0 * (l0_ref_l - L0_l) - self._kd_l0 * L0_dot_l

        torque_leg_r = self._kp_theta * (theta0_ref_r - theta0_r) - self._kd_theta * theta0_dot_r
        force_leg_r = self._kp_l0 * (l0_ref_r - L0_r) - self._kd_l0 * L0_dot_r

        # 动态前馈：F_ff = (m*g/2) * max(sin(theta0)*pg_x - cos(theta0)*pg_z, 0)。
        pg_x = pg[:, 0]
        pg_z = pg[:, 2]
        ff_l = (self._feedforward_mass * self._g / 2.0) * torch.clamp(
            torch.sin(theta0_l) * pg_x - torch.cos(theta0_l) * pg_z, min=0.0
        )
        ff_r = (self._feedforward_mass * self._g / 2.0) * torch.clamp(
            torch.sin(theta0_r) * pg_x - torch.cos(theta0_r) * pg_z, min=0.0
        )

        total_force_l = force_leg_l + ff_l
        total_force_r = force_leg_r + ff_r

        # 通过 VMC 雅可比将虚拟力/力矩映射为关节力矩。
        tau1_l, tau2_l = self._map_vmc_to_joint(
            total_force_l, torque_leg_l, th1_l, th2_l, L0_l
        )
        tau1_r, tau2_r = self._map_vmc_to_joint(
            total_force_r, torque_leg_r, th1_r, th2_r, L0_r
        )

        # 轮子力矩：kd * (wheel_vel_ref - wheel_vel)。
        wheel_vel_l = joint_vel[:, 2]
        wheel_vel_r = joint_vel[:, 5]
        wheel_tau_l = self._wheel_kd * (wheel_vel_ref_l - wheel_vel_l)
        wheel_tau_r = self._wheel_kd * (wheel_vel_ref_r - wheel_vel_r)

        # 组装力矩：[lf0, lf1, l_wheel, rf0, rf1, r_wheel]。
        self._torques[:, 0] = tau1_l
        self._torques[:, 1] = tau2_l
        self._torques[:, 2] = wheel_tau_l
        self._torques[:, 3] = tau1_r
        self._torques[:, 4] = tau2_r
        self._torques[:, 5] = wheel_tau_r

        robot.data.write_ctrl(self._torques)

    def _map_vmc_to_joint(self, force, torque, theta1, theta2, l0):
        """将极坐标空间的虚拟力/力矩映射为关节力矩。"""
        safe_l0 = torch.clamp(l0, min=1e-4)

        s1 = torch.sin(theta1)
        c1 = torch.cos(theta1)
        s12 = torch.sin(theta1 + theta2)
        c12 = torch.cos(theta1 + theta2)

        dx_dtheta1 = -self._l1 * s1 - self._l2 * c12
        dy_dtheta1 = self._l1 * c1 - self._l2 * s12
        dx_dtheta2 = -self._l2 * c12
        dy_dtheta2 = -self._l2 * s12

        inv_l0 = safe_l0**-1
        inv_l0_sq = safe_l0**-2

        dL0_dx = inv_l0 * (self._l1 * c1 - self._l2 * s12)
        dL0_dy = inv_l0 * (self._l1 * s1 + self._l2 * c12)
        dphi_dx = inv_l0_sq * (self._l1 * s1 + self._l2 * c12)
        dphi_dy = -inv_l0_sq * (self._l1 * c1 - self._l2 * s12)

        J11 = dL0_dx * dx_dtheta1 + dL0_dy * dy_dtheta1
        J12 = dL0_dx * dx_dtheta2 + dL0_dy * dy_dtheta2
        J21 = dphi_dx * dx_dtheta1 + dphi_dy * dy_dtheta1
        J22 = dphi_dx * dx_dtheta2 + dphi_dy * dy_dtheta2

        T1 = force * J11 + torque * J21
        T2 = force * J12 + torque * J22

        T1 = torch.nan_to_num(T1, nan=0.0, posinf=0.0, neginf=0.0)
        T2 = torch.nan_to_num(T2, nan=0.0, posinf=0.0, neginf=0.0)

        return T1, T2

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """将指定环境的原始动作重置为零。"""
        self._raw_actions[env_ids] = 0.0
        self._torques[env_ids] = 0.0
