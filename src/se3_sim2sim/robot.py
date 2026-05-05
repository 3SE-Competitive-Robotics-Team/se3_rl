"""MuJoCo robot runtime for the SE3 workflow."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .config import RobotConfig
from .diagnostics import model_diagnostics
from .kinematics import VMCKinematics
from .math_utils import euler_xyz_to_quat_wxyz, rotate, rotate_inverse
from .observation import ObservationBuilder
from .runtime_spec import RuntimeSpec, as_float64


class WheelLeggedRobot:
    def __init__(self, *, cfg: RobotConfig, runtime: RuntimeSpec) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.model_path = Path(cfg.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MJCF model not found: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if float(cfg.sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {cfg.sim_dt}")
        if int(cfg.control_decimation) < 1:
            raise ValueError(f"control_decimation must be >= 1, got {cfg.control_decimation}")
        self.model.opt.timestep = float(cfg.sim_dt)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(int(cfg.seed))
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = int(cfg.control_decimation)
        self.control_dt = self.sim_dt * self.decimation
        self.vmc = VMCKinematics(l1=cfg.l1, l2=cfg.l2)
        self.obs = ObservationBuilder(robot_cfg=cfg, runtime=runtime)

        self.joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in runtime.joint_names]
        self.actuator_ids = [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in runtime.actuator_names]
        self.joint_qpos = np.asarray([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int64)
        self.joint_qvel = np.asarray([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int64)

        self.default_dof_pos = as_float64(cfg.default_dof_pos)
        _, _, default_vmc = self.vmc.state_from_dofs(
            self.default_dof_pos,
            np.zeros_like(self.default_dof_pos),
            fd_dt=cfg.vmc_velocity_fd_dt,
        )
        self.default_l0 = default_vmc.l0
        self.default_theta0 = default_vmc.theta0
        self.action_scale = as_float64(cfg.action_scale)
        self.torque_limits = as_float64(cfg.torque_limits)
        self.command = np.asarray(cfg.command, dtype=np.float64)
        self.last_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_applied_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_policy_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_clipped_policy_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_ctrl = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_theta0_ref_raw = self.default_theta0.copy()
        self.last_theta0_ref_clipped = self.default_theta0.copy()
        self.last_theta0_ref = self.default_theta0.copy()
        self.last_l0_ref_raw = self.default_l0.copy()
        self.last_l0_ref = self.default_l0.copy()
        self.last_wheel_vel_ref_raw = np.zeros(2, dtype=np.float64)
        self.last_wheel_vel_ref_clipped = np.zeros(2, dtype=np.float64)
        self.last_wheel_vel_ref = np.zeros(2, dtype=np.float64)
        self.last_theta0_error = np.zeros(2, dtype=np.float64)
        self.last_l0_error = np.zeros(2, dtype=np.float64)
        self.last_wheel_vel_error = np.zeros(2, dtype=np.float64)
        self.action_delay_steps = max(0, int(cfg.action_delay_steps))
        self.action_fifo = np.zeros(
            (self.action_delay_steps + 1, runtime.policy.num_actions),
            dtype=np.float64,
        )
        self.prev_dof_pos = self.default_dof_pos.copy()
        self.step_count = 0
        self.reset()

    def reset(self, *, fixed: bool = True, randomize_root: bool = False) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.asarray([0.0, 0.0, self.cfg.base_height], dtype=np.float64)
        self.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[0:6] = 0.0
        self.data.qpos[self.joint_qpos] = self.default_dof_pos
        self.data.qvel[self.joint_qvel] = 0.0
        if (not fixed) or randomize_root:
            roll, pitch, yaw = self.rng.uniform(-0.25, 0.25, size=3)
            self.data.qpos[3:7] = euler_xyz_to_quat_wxyz(float(roll), float(pitch), float(yaw))
        mujoco.mj_forward(self.model, self.data)
        self.last_action.fill(0.0)
        self.last_applied_action.fill(0.0)
        self.last_policy_action.fill(0.0)
        self.last_clipped_policy_action.fill(0.0)
        self.action_fifo.fill(0.0)
        self.last_ctrl.fill(0.0)
        self.last_theta0_ref_raw[:] = self.default_theta0
        self.last_theta0_ref_clipped[:] = self.default_theta0
        self.last_theta0_ref[:] = self.default_theta0
        self.last_l0_ref_raw[:] = self.default_l0
        self.last_l0_ref[:] = self.default_l0
        self.last_wheel_vel_ref_raw.fill(0.0)
        self.last_wheel_vel_ref_clipped.fill(0.0)
        self.last_wheel_vel_ref.fill(0.0)
        self.last_theta0_error.fill(0.0)
        self.last_l0_error.fill(0.0)
        self.last_wheel_vel_error.fill(0.0)
        self.prev_dof_pos = self.dof_pos.copy()
        self.step_count = 0
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.runtime.policy.num_actions,):
            raise ValueError(f"action shape mismatch: expected {(self.runtime.policy.num_actions,)}, got {action.shape}")
        self.last_policy_action[:] = action
        action = np.clip(action, -100.0, 100.0)
        self.last_clipped_policy_action[:] = action
        self.last_action[:] = action
        for _ in range(self.decimation):
            self._refresh_state()
            self.action_fifo[1:] = self.action_fifo[:-1].copy()
            self.action_fifo[0] = action
            applied_action = self.action_fifo[self.action_delay_steps]
            self.last_applied_action[:] = applied_action
            ctrl = self._compute_vmc_torques(applied_action)
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
        self._refresh_state()
        self.step_count += 1
        obs = self.observation()
        reward = -abs(float(self.projected_gravity[2]) + 1.0)
        done, done_reason, fall_detected = self._termination_status()
        info = self.telemetry(reward=reward)
        info["done_reason"] = done_reason
        info["fall_detected"] = fall_detected
        return obs, reward, done, info

    def observation(self) -> np.ndarray:
        self._refresh_state()
        return self.obs.build(
            base_quat_wxyz=self.base_quat,
            base_ang_vel_world=self.base_ang_vel_world,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            command=self.command,
            theta0=self.theta0,
            theta0_dot=self.theta0_dot,
            l0=self.l0,
            l0_dot=self.l0_dot,
            action_obs=self.last_action,
        )

    def telemetry(self, *, reward: float | None = None) -> dict[str, object]:
        return {
            "step": int(self.step_count),
            "time": float(self.data.time),
            "height": float(self.data.qpos[2]),
            "tilt_deg": float(self.tilt_deg),
            "pitch_rad": float(np.arctan2(self.projected_gravity[0], -self.projected_gravity[2])),
            "reward": float(0.0 if reward is None else reward),
            "base_ang_vel_body": self.base_ang_vel_body.copy().tolist(),
            "base_ang_vel_world": self.base_ang_vel_world.copy().tolist(),
            "projected_gravity": self.projected_gravity.copy().tolist(),
            "dof_pos": self.dof_pos.copy().tolist(),
            "dof_vel": self.dof_vel.copy().tolist(),
            "theta0": self.theta0.copy().tolist(),
            "l0": self.l0.copy().tolist(),
            "l0_dot": self.l0_dot.copy().tolist(),
            "theta0_dot": self.theta0_dot.copy().tolist(),
            "wheel_vel": self.dof_vel[[2, 5]].copy().tolist(),
            "policy_action_raw": self.last_policy_action.copy().tolist(),
            "policy_action_clipped": self.last_clipped_policy_action.copy().tolist(),
            "theta0_ref_raw": self.last_theta0_ref_raw.copy().tolist(),
            "theta0_ref_clipped": self.last_theta0_ref_clipped.copy().tolist(),
            "theta0_ref": self.last_theta0_ref.copy().tolist(),
            "theta0_error": self.last_theta0_error.copy().tolist(),
            "l0_ref_raw": self.last_l0_ref_raw.copy().tolist(),
            "l0_ref": self.last_l0_ref.copy().tolist(),
            "l0_error": self.last_l0_error.copy().tolist(),
            "wheel_vel_ref_raw": self.last_wheel_vel_ref_raw.copy().tolist(),
            "wheel_vel_ref_clipped": self.last_wheel_vel_ref_clipped.copy().tolist(),
            "wheel_vel_ref": self.last_wheel_vel_ref.copy().tolist(),
            "wheel_vel_error": self.last_wheel_vel_error.copy().tolist(),
            "last_action": self.last_action.copy().tolist(),
            "applied_action": self.last_applied_action.copy().tolist(),
            "last_ctrl": self.last_ctrl.copy().tolist(),
            "fail_tilt_deg": float(self.cfg.fail_tilt_deg),
            "theta0_ref_limit": float(self.cfg.theta0_ref_limit),
            "wheel_vel_ref_limit": float(self.cfg.wheel_vel_ref_limit),
        }

    def diagnostics(self) -> dict[str, object]:
        return model_diagnostics(self.model)

    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos].copy()

    @property
    def dof_vel_raw(self) -> np.ndarray:
        return self.data.qvel[self.joint_qvel].copy()

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(-float(self.projected_gravity[2]), -1.0, 1.0))))

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"missing {obj_type.name} in MJCF: {name}")
        return int(idx)

    def _refresh_state(self) -> None:
        self.base_quat = self.data.qpos[3:7].copy()
        self.base_ang_vel_body = self.data.qvel[3:6].copy()
        self.base_ang_vel_world = rotate(self.base_quat, self.base_ang_vel_body)
        self.projected_gravity = rotate_inverse(self.base_quat, np.asarray([0.0, 0.0, -1.0]))
        raw_pos = self.dof_pos
        raw_vel = self.dof_vel_raw
        if self.cfg.dof_vel_use_pos_diff:
            diff = (raw_pos - self.prev_dof_pos + np.pi) % (2.0 * np.pi) - np.pi
            self.dof_vel = diff / self.sim_dt
        else:
            self.dof_vel = raw_vel
        self.prev_dof_pos = raw_pos.copy()
        self.theta1, self.theta2, vmc_state = self.vmc.state_from_dofs(
            raw_pos,
            self.dof_vel,
            fd_dt=self.cfg.vmc_velocity_fd_dt,
        )
        self.dof_pos_cached = raw_pos
        self.l0 = vmc_state.l0
        self.theta0 = vmc_state.theta0
        self.l0_dot = vmc_state.l0_dot
        self.theta0_dot = vmc_state.theta0_dot
        self._clip_vmc_velocity_state()

    def _clip_vmc_velocity_state(self) -> None:
        np.clip(
            self.theta0_dot,
            -float(self.cfg.theta0_dot_limit),
            float(self.cfg.theta0_dot_limit),
            out=self.theta0_dot,
        )
        np.clip(
            self.l0_dot,
            -float(self.cfg.l0_dot_limit),
            float(self.cfg.l0_dot_limit),
            out=self.l0_dot,
        )

    def _compute_vmc_torques(self, action: np.ndarray) -> np.ndarray:
        scaled = np.asarray(action, dtype=np.float64) * self.action_scale
        theta0_ref = np.asarray([scaled[0], scaled[3]], dtype=np.float64)
        theta0_ref_raw = theta0_ref.copy()
        np.clip(
            theta0_ref,
            -float(self.cfg.theta0_ref_limit),
            float(self.cfg.theta0_ref_limit),
            out=theta0_ref,
        )
        theta0_ref_clipped = theta0_ref.copy()
        l0_ref = np.asarray([scaled[1], scaled[4]], dtype=np.float64) + float(self.cfg.l0_offset)
        l0_ref_raw = l0_ref.copy()
        wheel_vel_ref = np.asarray([scaled[2], scaled[5]], dtype=np.float64)
        wheel_vel_ref_raw = wheel_vel_ref.copy()
        self._clip_wheel_vel_ref(wheel_vel_ref)
        wheel_vel_ref_clipped = wheel_vel_ref.copy()

        theta0_error = self._wrap_angle_diff(theta0_ref - self.theta0)
        l0_error = l0_ref - self.l0
        wheel_vel_error = wheel_vel_ref - self.dof_vel[[2, 5]]
        self.last_theta0_ref_raw[:] = theta0_ref_raw
        self.last_theta0_ref_clipped[:] = theta0_ref_clipped
        self.last_theta0_ref[:] = theta0_ref
        self.last_l0_ref_raw[:] = l0_ref_raw
        self.last_l0_ref[:] = l0_ref
        self.last_wheel_vel_ref_raw[:] = wheel_vel_ref_raw
        self.last_wheel_vel_ref_clipped[:] = wheel_vel_ref_clipped
        self.last_wheel_vel_ref[:] = wheel_vel_ref
        self.last_theta0_error[:] = theta0_error
        self.last_l0_error[:] = l0_error
        self.last_wheel_vel_error[:] = wheel_vel_error
        torque_leg = self.cfg.theta_kp * theta0_error - self.cfg.theta_kd * self.theta0_dot
        force_leg = self.cfg.l0_kp * l0_error - self.cfg.l0_kd * self.l0_dot

        cos_theta = np.cos(self.theta0)
        sin_theta = np.sin(self.theta0)
        gravity_along_leg = sin_theta * float(self.projected_gravity[0]) - cos_theta * float(self.projected_gravity[2])
        feedforward = (float(self.cfg.feedforward_mass) * 9.81 / 2.0) * np.maximum(gravity_along_leg, 0.0)
        total_force = np.nan_to_num(force_leg + feedforward, nan=0.0, posinf=0.0, neginf=0.0)

        t1, t2 = self.vmc.map_virtual_to_joint_torques(
            force=total_force,
            torque=torque_leg,
            theta1=self.theta1,
            theta2=self.theta2,
            l0=self.l0,
        )
        wheel_torque = self.cfg.wheel_kd * wheel_vel_error
        torques = np.asarray([t1[0], t2[0], wheel_torque[0], t1[1], t2[1], wheel_torque[1]], dtype=np.float64)
        torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        self.last_ctrl[:] = torques
        return torques

    def _clip_wheel_vel_ref(self, wheel_vel_ref: np.ndarray) -> None:
        np.clip(
            wheel_vel_ref,
            -float(self.cfg.wheel_vel_ref_limit),
            float(self.cfg.wheel_vel_ref_limit),
            out=wheel_vel_ref,
        )

    @staticmethod
    def _wrap_angle_diff(diff: np.ndarray) -> np.ndarray:
        return (np.asarray(diff, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    def _termination_status(self) -> tuple[bool, str, bool]:
        invalid_state = not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.ctrl).all()
        )
        if invalid_state:
            return True, "invalid_state", False

        fall_detected = bool(
            self.tilt_deg > float(self.cfg.fail_tilt_deg)
            or float(self.data.qpos[2]) < float(self.cfg.fail_height_m)
        )
        if bool(self.cfg.terminate_on_fall) and fall_detected:
            return True, "fall", True
        return False, "running", fall_detected
