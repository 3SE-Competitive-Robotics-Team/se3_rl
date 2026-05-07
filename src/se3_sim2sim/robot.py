"""MuJoCo robot runtime for the SE3 workflow (joint-position control)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .config import RobotConfig
from .diagnostics import model_diagnostics
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
        self.model = self._build_model(str(self.model_path))
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
        self.obs = ObservationBuilder(robot_cfg=cfg, runtime=runtime)

        self.joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in runtime.joint_names]
        self.joint_qpos = np.asarray(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int64
        )
        self.joint_qvel = np.asarray(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int64
        )

        self.default_dof_pos = as_float64(cfg.default_dof_pos)
        self.action_scale = as_float64(cfg.action_scale)
        self.torque_limits = as_float64(cfg.torque_limits)
        self.command = np.asarray(cfg.command, dtype=np.float64)
        self.last_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_applied_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_policy_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_clipped_policy_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_ctrl = np.zeros(6, dtype=np.float64)
        self.action_delay_steps = max(0, int(cfg.action_delay_steps))
        self.action_fifo = np.zeros(
            (self.action_delay_steps + 1, runtime.policy.num_actions),
            dtype=np.float64,
        )
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
        self.step_count = 0
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.runtime.policy.num_actions,):
            expected = (self.runtime.policy.num_actions,)
            raise ValueError(f"action shape mismatch: expected {expected}, got {action.shape}")
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
            ctrl = self._compute_pd_torques(applied_action)
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
            "policy_action_raw": self.last_policy_action.copy().tolist(),
            "policy_action_clipped": self.last_clipped_policy_action.copy().tolist(),
            "last_action": self.last_action.copy().tolist(),
            "applied_action": self.last_applied_action.copy().tolist(),
            "last_ctrl": self.last_ctrl.copy().tolist(),
            "fail_tilt_deg": float(self.cfg.fail_tilt_deg),
        }

    def diagnostics(self) -> dict[str, object]:
        return model_diagnostics(self.model)

    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos].copy()

    @property
    def dof_vel(self) -> np.ndarray:
        return self.data.qvel[self.joint_qvel].copy()

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(-float(self.projected_gravity[2]), -1.0, 1.0))))

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"missing {obj_type.name} in MJCF: {name}")
        return int(idx)

    @staticmethod
    def _build_model(xml_path: str) -> mujoco.MjModel:
        """加载 MJCF 并程序化添加 motor actuator（MJCF 中已删除 <actuator> 段）。"""
        spec = mujoco.MjSpec.from_file(xml_path)
        joint_names = [
            "lf0_Joint",
            "lf1_Joint",
            "l_wheel_Joint",
            "rf0_Joint",
            "rf1_Joint",
            "r_wheel_Joint",
        ]
        torque_limits = [30.0, 30.0, 3.3, 30.0, 30.0, 3.3]
        for jname, tlim in zip(joint_names, torque_limits, strict=True):
            act = spec.add_actuator()
            act.name = f"{jname}_motor"
            act.target = jname
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gear = np.array([1.0, 0, 0, 0, 0, 0])
            act.ctrlrange = np.array([-tlim, tlim])
            act.ctrllimited = True
        return spec.compile()

    def _refresh_state(self) -> None:
        self.base_quat = self.data.qpos[3:7].copy()
        self.base_ang_vel_body = self.data.qvel[3:6].copy()
        self.base_ang_vel_world = rotate(self.base_quat, self.base_ang_vel_body)
        self.projected_gravity = rotate_inverse(self.base_quat, np.asarray([0.0, 0.0, -1.0]))

    def _compute_pd_torques(self, action: np.ndarray) -> np.ndarray:
        """关节位置 PD + 轮子速度控制。

        action[0:4] = 腿部关节位置增量（相对默认位姿）
        action[4:6] = 轮子速度目标
        """
        scaled = np.asarray(action, dtype=np.float64) * self.action_scale

        # 腿部: q_target = action * scale + q_default
        leg_default = self.default_dof_pos[[0, 1, 3, 4]]
        q_target = scaled[:4] + leg_default
        q_current = self.dof_pos[[0, 1, 3, 4]]
        dq_current = self.dof_vel[[0, 1, 3, 4]]
        tau_legs = (
            float(self.cfg.leg_kp) * (q_target - q_current) - float(self.cfg.leg_kd) * dq_current
        )

        # 轮子: vel_target = action * scale, tau = kd * (vel_target - vel)
        vel_target = scaled[4:6]
        vel_current = self.dof_vel[[2, 5]]
        tau_wheels = float(self.cfg.wheel_kd) * (vel_target - vel_current)

        torques = np.asarray(
            [tau_legs[0], tau_legs[1], tau_wheels[0], tau_legs[2], tau_legs[3], tau_wheels[1]],
            dtype=np.float64,
        )
        torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        self.last_ctrl[:] = torques
        return torques

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
