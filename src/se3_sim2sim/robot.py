"""SE3 sim2sim 的 MuJoCo 机器人运行时。"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from se3_shared import (
    FOURBAR_SURROGATE_MARKER,
    ActionDelayConfig,
    JointGroup,
    PolicyActionDecoder,
    Termination,
    output_to_policy_pos_np,
    output_to_policy_vel_np,
    policy_leg_position_error_np,
    policy_to_output_torque_np,
)
from se3_shared import RobotConfig as SharedRobotConfig
from se3_shared.motor import DM8009P, M3508_C620_14

from .closed_chain import ClosedChainClosureSolver
from .config import RobotConfig
from .diagnostics import model_diagnostics
from .math_utils import euler_xyz_to_quat_wxyz, extract_yaw, rotate, rotate_inverse, wrap_angle
from .observation import ObservationBuilder
from .runtime_spec import RuntimeSpec, as_float64
from .stair_ctbc import StairCtbcRuntime
from .yaw_pid import YawPidController

_SHARED_ROBOT = SharedRobotConfig()
_RESET_FLOOR_CLEARANCE_M = 0.01
_BASE_CONTACT_PENETRATION_M = 0.001
_GROUND_GEOM_GROUP = 2


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """读取模型中的非 freejoint 关节名。"""
    names: list[str] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            names.append(name)
    return tuple(names)


def _model_has_joints(model: mujoco.MjModel, names: tuple[str, ...]) -> bool:
    """判断模型是否包含一组关节。"""
    available = set(_model_joint_names(model))
    return all(name in available for name in names)


def _model_site_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """读取模型中的 site 名称。"""
    names: list[str] = []
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name:
            names.append(name)
    return tuple(names)


def _policy_joint_names_for_model(model: mujoco.MjModel) -> tuple[str, ...]:
    """闭链使用主动杆；开链显式回退到旧 lf1/rf1 语义。"""
    if _model_has_joints(model, JointGroup.POLICY_LEG_NAMES):
        return JointGroup.POLICY_JOINT_NAMES
    return (*JointGroup.OPENCHAIN_LEG_NAMES, *JointGroup.WHEEL_NAMES)


def _tn_clip(
    effort: np.ndarray,
    velocity: np.ndarray,
    saturation_effort: float,
    velocity_limit: float,
    effort_limit: float,
) -> np.ndarray:
    """线性 T-N 包络 clamp — 与 MJLab DcMotorActuatorCfg._clip_effort 一致。"""
    vel_at_effort_lim = velocity_limit * (1.0 + effort_limit / saturation_effort)
    vel_clipped = np.clip(velocity, -vel_at_effort_lim, vel_at_effort_lim)
    top = saturation_effort * (1.0 - vel_clipped / velocity_limit)
    bottom = saturation_effort * (-1.0 - vel_clipped / velocity_limit)
    max_eff = np.minimum(top, effort_limit)
    min_eff = np.maximum(bottom, -effort_limit)
    return np.clip(effort, min_eff, max_eff)


class WheelLeggedRobot:
    def __init__(
        self,
        *,
        cfg: RobotConfig,
        runtime: RuntimeSpec,
        termination: Termination | None = None,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.termination = termination if termination is not None else Termination()
        self.model_path = Path(cfg.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MJCF model not found: {self.model_path}")
        self.model = self._build_model(str(self.model_path), cfg)
        # sim_dt > 0 和 control_decimation >= 1 已由 RobotConfig(BaseModel) 在构造时校验
        self.model.opt.timestep = float(cfg.sim_dt)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 100
        self.data = mujoco.MjData(self.model)
        self.fourbar_surrogate = FOURBAR_SURROGATE_MARKER in set(_model_site_names(self.model))
        (
            self._ground_geom_ids,
            self._base_geom_ids,
            self._leg_geom_ids,
            self._left_leg_geom_ids,
            self._right_leg_geom_ids,
            self._wheel_geom_ids,
            self._left_wheel_geom_ids,
            self._right_wheel_geom_ids,
        ) = self._build_contact_geom_groups()
        self._stair_geom_ids = self._build_stair_geom_ids()
        self.rng = np.random.default_rng(int(cfg.seed))
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = int(cfg.control_decimation)
        self.control_dt = self.sim_dt * self.decimation
        self.yaw_pid = YawPidController(cfg.yaw_pid)
        self.stair_ctbc = (
            StairCtbcRuntime(cfg.stair_ctbc, control_dt=self.control_dt)
            if bool(cfg.stair_ctbc.enabled)
            else None
        )

        self.policy_joint_names = _policy_joint_names_for_model(self.model)
        self.joint_ids = [
            self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.policy_joint_names
        ]
        self.joint_qpos = np.asarray(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int64
        )
        self.joint_qvel = np.asarray(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int64
        )
        if self.policy_joint_names == JointGroup.POLICY_JOINT_NAMES:
            self.default_dof_pos = as_float64(cfg.default_dof_pos)
        else:
            default_map = _SHARED_ROBOT.default_model_joint_pos
            self.default_dof_pos = np.asarray(
                [default_map[name] for name in self.policy_joint_names], dtype=np.float64
            )
        self.obs = ObservationBuilder(
            robot_cfg=cfg,
            runtime=runtime,
            default_dof_pos=self.default_dof_pos,
            fourbar_surrogate=self.fourbar_surrogate,
        )
        self.action_scale = as_float64(cfg.action_scale)
        self.active_rod_action_semantics = (
            self.policy_joint_names == JointGroup.POLICY_JOINT_NAMES or self.fourbar_surrogate
        )
        self.action_decoder = PolicyActionDecoder(
            robot_cfg=_SHARED_ROBOT,
            action_scale=self.action_scale,
            height_conditioned_action_default=(
                bool(cfg.height_conditioned_action_default) and self.active_rod_action_semantics
            ),
            active_rod_semantics=self.active_rod_action_semantics,
            active_rod_target_lower_preload_margin=cfg.active_rod_target_lower_preload_margin,
            active_rod_target_upper_preload_margin=cfg.active_rod_target_upper_preload_margin,
            dtype=np.float64,
        )
        self.leg_kp = float(cfg.leg_kp)
        self.leg_kd = float(cfg.leg_kd)
        self.wheel_kd = float(cfg.wheel_kd)
        self.torque_limits = as_float64(cfg.torque_limits)
        self.motor_actuator_names = tuple(f"{name}_motor" for name in self.policy_joint_names)
        self.motor_ctrl_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.motor_actuator_names],
            dtype=np.int64,
        )
        self.output_joint_names = tuple(
            name
            for name in JointGroup.OUTPUT_LEG_NAMES
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        )
        self.output_joint_qpos = np.asarray(
            [
                self.model.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in self.output_joint_names
            ],
            dtype=np.int64,
        )
        self.command = np.asarray(cfg.command, dtype=np.float64)
        self.last_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_applied_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_policy_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_clipped_policy_action = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_ctbc_action_delta = np.zeros(runtime.policy.num_actions, dtype=np.float64)
        self.last_ctrl = np.zeros(6, dtype=np.float64)
        self.reset_floor_lift_m = 0.0
        self.closed_chain_reset_position_residual_m = 0.0
        self.closed_chain_reset_velocity_residual = 0.0
        # 线速度缓存，_refresh_state 更新
        self.base_lin_vel_world = np.zeros(3, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)
        self.action_delay_cfg: ActionDelayConfig = cfg.action_delay
        self.min_action_delay_steps, self.max_action_delay_steps = (
            self.action_delay_cfg.step_bounds(self.sim_dt)
        )
        self.action_delay_steps = 0
        self.action_fifo = np.zeros(
            (self.max_action_delay_steps + 1, runtime.policy.num_actions),
            dtype=np.float64,
        )
        self.step_count = 0
        self.pre_policy_settle_info: dict[str, object] = {"enabled": False}
        self.reset()

    def reset(self, *, fixed: bool = True, randomize_root: bool = False) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        base_height = (
            self.cfg.base_height
            if self.cfg.initial_base_height is None
            else self.cfg.initial_base_height
        )
        self.data.qpos[0:3] = np.asarray([0.0, 0.0, float(base_height)], dtype=np.float64)
        roll = float(self.cfg.initial_roll_rad)
        pitch = float(self.cfg.initial_pitch_rad)
        yaw = float(self.cfg.initial_yaw_rad)
        self.data.qvel[0:6] = 0.0
        self.data.qvel[3:6] = np.asarray(self.cfg.initial_ang_vel_rad_s, dtype=np.float64)
        self._apply_default_joint_positions()
        self._apply_initial_wheel_joint_positions()
        closure_solver = self._close_initial_chain_positions()
        self.data.qvel[:] = 0.0
        self.data.qvel[3:6] = np.asarray(self.cfg.initial_ang_vel_rad_s, dtype=np.float64)
        self._apply_initial_dof_vel()
        self._close_initial_chain_velocities(closure_solver)
        if (not fixed) or randomize_root:
            roll_offset, pitch_offset, yaw_offset = self.rng.uniform(-0.25, 0.25, size=3)
            roll += float(roll_offset)
            pitch += float(pitch_offset)
            yaw += float(yaw_offset)
        self.data.qpos[3:7] = euler_xyz_to_quat_wxyz(roll, pitch, yaw)
        mujoco.mj_forward(self.model, self.data)
        self._lift_root_to_clear_floor()
        self._refresh_state()
        self.last_action.fill(0.0)
        self.last_applied_action.fill(0.0)
        self.last_policy_action.fill(0.0)
        self.last_clipped_policy_action.fill(0.0)
        self.last_ctbc_action_delta.fill(0.0)
        self.action_fifo.fill(0.0)
        self.action_delay_steps = self._sample_action_delay_steps()
        self.last_ctrl.fill(0.0)
        if self.stair_ctbc is not None:
            self.stair_ctbc.reset()
        self._apply_initial_policy_io_state()
        yaw_rate_cmd = self.yaw_pid.reset(self.base_yaw)
        if self.cfg.yaw_pid.enabled:
            self.command[1] = yaw_rate_cmd
        self.step_count = 0
        self.pre_policy_settle_info = {"enabled": False}
        return self.observation()

    def reset_policy_io_state(self) -> None:
        """清空 policy 输入输出历史，用于模拟真机 output disabled 时的 GRU reset。"""
        self.last_action.fill(0.0)
        self.last_applied_action.fill(0.0)
        self.last_policy_action.fill(0.0)
        self.last_clipped_policy_action.fill(0.0)
        self.last_ctbc_action_delta.fill(0.0)
        self.action_fifo.fill(0.0)
        self.last_ctrl.fill(0.0)

    def update_stair_ctbc(self, *, iteration: int | None) -> None:
        """用当前轮-台阶立面接触更新 stair CTBC 状态机。"""

        if self.stair_ctbc is None:
            return
        self.stair_ctbc.update(
            self._stair_riser_wheel_contact_score(),
            iteration=iteration,
        )

    def settle_base_before_policy(
        self,
        *,
        max_s: float,
        contact_steps: int,
    ) -> dict[str, object]:
        """policy 推理前用零控制推进物理，直到 base_link 连续接地。"""
        base_clearance_before = float(self._min_collision_geom_z_for(self._base_geom_ids))
        base_snap_down_m = 0.0
        if self._base_geom_ids and base_clearance_before > -_BASE_CONTACT_PENETRATION_M:
            base_snap_down_m = base_clearance_before + _BASE_CONTACT_PENETRATION_M
            self.data.qpos[2] -= base_snap_down_m
            mujoco.mj_forward(self.model, self.data)
            self._refresh_state()
        base_clearance_after_snap = float(self._min_collision_geom_z_for(self._base_geom_ids))
        max_steps = max(0, round(float(max_s) / self.sim_dt))
        required_contact_steps = max(1, int(contact_steps))
        consecutive_contact_steps = 0
        steps = 0
        base_touching = bool(self._base_geom_ids) and base_clearance_after_snap <= 0.0
        base_contact = bool(self._ground_contact_state()["base_contact"])
        if max_steps > 0:
            for _ in range(max_steps + 1):
                contact = self._ground_contact_state()
                base_contact = bool(contact["base_contact"])
                if base_contact:
                    consecutive_contact_steps += 1
                    if consecutive_contact_steps >= required_contact_steps:
                        break
                else:
                    consecutive_contact_steps = 0
                if steps >= max_steps:
                    break
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)
                steps += 1
                self._refresh_state()

        settle_time_s = float(steps * self.sim_dt)
        info = {
            "enabled": True,
            "success": bool(base_touching or consecutive_contact_steps >= required_contact_steps),
            "steps": int(steps),
            "time_s": settle_time_s,
            "base_touching": bool(base_touching),
            "base_contact": bool(base_contact),
            "base_clearance_before_snap": base_clearance_before,
            "base_snap_down_m": float(base_snap_down_m),
            "base_clearance_after_snap": base_clearance_after_snap,
            "base_clearance": float(self._min_collision_geom_z_for(self._base_geom_ids)),
            "required_contact_steps": int(required_contact_steps),
            "consecutive_contact_steps": int(consecutive_contact_steps),
        }
        self.pre_policy_settle_info = info
        self.data.time = 0.0
        self.step_count = 0
        self.last_action.fill(0.0)
        self.last_applied_action.fill(0.0)
        self.last_policy_action.fill(0.0)
        self.last_clipped_policy_action.fill(0.0)
        self.last_ctbc_action_delta.fill(0.0)
        self.action_fifo.fill(0.0)
        self.last_ctrl.fill(0.0)
        self._refresh_state()
        return info

    def update_yaw_command(self) -> float:
        """按当前 yaw 更新 policy command 中的 yaw 维度。"""

        if not self.cfg.yaw_pid.enabled:
            return float(self.command[1])
        self._refresh_state()
        self.command[1] = self.yaw_pid.update(self.base_yaw, self.control_dt)
        return float(self.command[1])

    def _apply_stair_ctbc(self, action: np.ndarray) -> np.ndarray:
        """对最终执行 action 注入 stair CTBC，保持 last_action 为策略原始输出。"""

        self.last_ctbc_action_delta.fill(0.0)
        if self.stair_ctbc is None:
            return np.asarray(action, dtype=np.float64).reshape(6)
        execution_action = self.stair_ctbc.apply(self, action)
        self.last_ctbc_action_delta[:] = self.stair_ctbc.action_delta
        return execution_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.runtime.policy.num_actions,):
            expected = (self.runtime.policy.num_actions,)
            raise ValueError(f"action shape mismatch: expected {expected}, got {action.shape}")
        self.last_policy_action[:] = action
        if self.cfg.action_clip is not None:
            clip = float(self.cfg.action_clip)
            action = np.clip(action, -clip, clip)
        self.last_clipped_policy_action[:] = action
        self.last_action[:] = action
        execution_action = self._apply_stair_ctbc(action)
        for _ in range(self.decimation):
            self._refresh_state()
            self.action_fifo[1:] = self.action_fifo[:-1].copy()
            self.action_fifo[0] = execution_action
            applied_action = self.action_fifo[self.action_delay_steps]
            self.last_applied_action[:] = applied_action
            ctrl = self._compute_pd_torques(applied_action)
            self.data.ctrl[self.motor_ctrl_ids] = ctrl
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

    def step_hold_current(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """按 deploy hold_current 语义推进一步：保持当前腿部目标，轮速目标为 0。"""
        self.reset_policy_io_state()
        self._refresh_state()
        leg_hold_target = self.dof_pos[JointGroup.CTRL_LEGS].copy()
        for _ in range(self.decimation):
            self._refresh_state()
            ctrl = self._compute_hold_current_torques(leg_hold_target)
            self.data.ctrl[self.motor_ctrl_ids] = ctrl
            mujoco.mj_step(self.model, self.data)
        self._refresh_state()
        self.step_count += 1
        obs = self.observation()
        reward = -abs(float(self.projected_gravity[2]) + 1.0)
        done, done_reason, fall_detected = self._termination_status()
        info = self.telemetry(reward=reward)
        info["done_reason"] = done_reason
        info["fall_detected"] = fall_detected
        info["target_mode"] = "hold_current"
        return obs, reward, done, info

    def step_no_torque(self) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """按真机 output disabled 语义推进一步：电机失能，不输出力矩。"""
        self.reset_policy_io_state()
        self._refresh_state()
        for _ in range(self.decimation):
            self.data.ctrl[self.motor_ctrl_ids] = 0.0
            self.last_ctrl.fill(0.0)
            mujoco.mj_step(self.model, self.data)
        self._refresh_state()
        self.step_count += 1
        obs = self.observation()
        reward = -abs(float(self.projected_gravity[2]) + 1.0)
        done, done_reason, fall_detected = self._termination_status()
        info = self.telemetry(reward=reward)
        info["done_reason"] = done_reason
        info["fall_detected"] = fall_detected
        info["target_mode"] = "no_torque"
        return obs, reward, done, info

    def observation(self) -> np.ndarray:
        self._refresh_state()
        obs = self.obs.build(
            base_quat_wxyz=self.base_quat,
            base_ang_vel_world=self.base_ang_vel_world,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            command=self.command,
            action_obs=self.last_action,
        )
        if self.stair_ctbc is not None:
            obs = obs.copy()
            obs[self.runtime.observation_slices["jump_commands"]] = self.stair_ctbc.obs()
        return obs

    def apply_root_velocity_delta(self, delta_velocity: np.ndarray) -> None:
        """给 floating base 广义速度一次性叠加扰动。"""
        delta = np.asarray(delta_velocity, dtype=np.float64)
        if delta.shape != (6,) or not np.isfinite(delta).all():
            raise ValueError(f"root velocity delta must be finite shape (6,), got {delta}")
        self.data.qvel[0:6] += delta
        self._refresh_state()

    def telemetry(self, *, reward: float | None = None) -> dict[str, object]:
        wheel_radius = 0.059  # m，与 MJCF wheelRadius 一致
        wheel_vel = self.dof_vel[JointGroup.CTRL_WHEELS]  # rad/s，[l, r]
        # 左右轮 joint axis 相反，前进速度对应广义轮速 l-r，而不是 l+r。
        wheel_lin_vel = float(0.5 * (wheel_vel[0] - wheel_vel[1]) * wheel_radius)

        # 轮子最低点离地高度：轮子中心 z - 轮子半径
        # 反映实际离地间隙，跳跃时比 baselink 高度更直观
        l_wheel_pos = np.asarray(self.data.body("l_wheel_Link").xpos, dtype=np.float64).copy()
        r_wheel_pos = np.asarray(self.data.body("r_wheel_Link").xpos, dtype=np.float64).copy()
        l_wheel_z = float(l_wheel_pos[2])
        r_wheel_z = float(r_wheel_pos[2])
        wheel_clearance_l = l_wheel_z - wheel_radius  # 左轮最低点离地高度
        wheel_clearance_r = r_wheel_z - wheel_radius  # 右轮最低点离地高度
        wheel_clearance = min(wheel_clearance_l, wheel_clearance_r)  # 取两轮最低
        wheel_delta_b = rotate_inverse(self.base_quat, l_wheel_pos - r_wheel_pos)
        wheel_lateral_distance = abs(float(wheel_delta_b[1]))
        wheel_fore_aft_offset = abs(float(wheel_delta_b[0]))
        dof_pos = self.dof_pos
        output_leg_pos = self.output_leg_pos
        leg_mirror_error = max(
            abs(float(dof_pos[0] - dof_pos[2])),
            abs(float(dof_pos[1] - dof_pos[3])),
        )
        output_leg_mirror_error = (
            max(
                abs(float(output_leg_pos[0] - output_leg_pos[2])),
                abs(float(output_leg_pos[1] - output_leg_pos[3])),
            )
            if output_leg_pos.shape == (4,)
            else 0.0
        )
        contact = self._ground_contact_state()
        leg_clearance = self._min_collision_geom_z_for(self._leg_geom_ids)
        base_clearance = self._min_collision_geom_z_for(self._base_geom_ids)

        telemetry = {
            "step": int(self.step_count),
            "time": float(self.data.time),
            "height": float(self.data.qpos[2]),  # baselink z 高度
            "reset_floor_lift_m": float(self.reset_floor_lift_m),
            "wheel_clearance": float(wheel_clearance),  # 轮子最低点离地高度
            "wheel_clearance_left": float(wheel_clearance_l),  # 左轮最低点离地高度
            "wheel_clearance_right": float(wheel_clearance_r),  # 右轮最低点离地高度
            "leg_clearance": float(leg_clearance),  # 腿部碰撞几何最低点离地高度
            "base_clearance": float(base_clearance),  # base 碰撞几何最低点离地高度
            "wheel_lateral_distance": float(wheel_lateral_distance),
            "wheel_fore_aft_offset": float(wheel_fore_aft_offset),
            "leg_mirror_error": float(leg_mirror_error),
            "output_leg_mirror_error": float(output_leg_mirror_error),
            "wheel_contact": float(contact["wheel_contact"]),
            "wheel_full_contact": float(contact["wheel_full_contact"]),
            "wheel_contact_left": float(contact["wheel_contact_left"]),
            "wheel_contact_right": float(contact["wheel_contact_right"]),
            "leg_contact": float(contact["leg_contact"]),
            "leg_contact_left": float(contact["leg_contact_left"]),
            "leg_contact_right": float(contact["leg_contact_right"]),
            "base_contact": float(contact["base_contact"]),
            "nonwheel_contact": float(contact["nonwheel_contact"]),
            "tilt_deg": float(self.tilt_deg),
            "roll_rad": float(np.arctan2(-self.projected_gravity[1], -self.projected_gravity[2])),
            "pitch_rad": float(np.arctan2(self.projected_gravity[0], -self.projected_gravity[2])),
            "base_yaw": float(self.base_yaw),
            "roll_deg": float(
                np.degrees(np.arctan2(-self.projected_gravity[1], -self.projected_gravity[2]))
            ),
            "pitch_deg": float(
                np.degrees(np.arctan2(self.projected_gravity[0], -self.projected_gravity[2]))
            ),
            "yaw_deg": float(np.degrees(self.base_yaw)),
            "reward": float(0.0 if reward is None else reward),
            "command": self.command.copy().tolist(),
            "command_lin_vel_x": float(self.command[0]),
            "command_yaw_rate": float(self.command[1]),
            "command_height": float(self.command[4]),
            "leg_kp": float(self.leg_kp),
            "leg_kd": float(self.leg_kd),
            "wheel_kd": float(self.wheel_kd),
            "base_lin_vel_x": float(self.base_lin_vel_body[0]),
            "wheel_lin_vel": wheel_lin_vel,
            "base_ang_vel_body": self.base_ang_vel_body.copy().tolist(),
            "base_ang_vel_world": self.base_ang_vel_world.copy().tolist(),
            "projected_gravity": self.projected_gravity.copy().tolist(),
            "dof_pos": dof_pos.copy().tolist(),
            "dof_vel": self.dof_vel.copy().tolist(),
            "policy_joint_names": list(self.policy_joint_names),
            "output_leg_pos": output_leg_pos.copy().tolist(),
            "output_joint_names": list(self.output_joint_names),
            "policy_action_raw": self.last_policy_action.copy().tolist(),
            "policy_action_clipped": self.last_clipped_policy_action.copy().tolist(),
            "last_action": self.last_action.copy().tolist(),
            "ctbc_action_delta": self.last_ctbc_action_delta.copy().tolist(),
            "applied_action": self.last_applied_action.copy().tolist(),
            "last_ctrl": self.last_ctrl.copy().tolist(),
            "action_delay_steps": int(self.action_delay_steps),
            "action_delay_s": float(
                self.action_delay_cfg.actual_delay_s(self.action_delay_steps, self.sim_dt)
            ),
            "action_delay_config": self.action_delay_cfg.model_dump(),
            "fail_tilt_deg": float(self.termination.fail_tilt_deg),
            "closed_chain_reset_position_residual_m": float(
                self.closed_chain_reset_position_residual_m
            ),
            "closed_chain_reset_velocity_residual": float(
                self.closed_chain_reset_velocity_residual
            ),
        }
        if self.cfg.yaw_pid.enabled:
            yaw_pid = self.yaw_pid.telemetry()
            yaw_pid["current_yaw"] = float(self.base_yaw)
            yaw_pid["error"] = float(wrap_angle(yaw_pid["target_yaw"] - float(self.base_yaw)))
            telemetry["yaw_pid"] = yaw_pid
        if self.stair_ctbc is not None:
            telemetry.update(self.stair_ctbc.telemetry())
        return telemetry

    def diagnostics(self) -> dict[str, object]:
        return model_diagnostics(self.model)

    def _lift_root_to_clear_floor(self) -> None:
        """如果 reset 姿态穿地，则整体上抬到可碰撞几何体离地。"""
        min_z = self._min_collision_geom_z()
        self.reset_floor_lift_m = 0.0
        if min_z < _RESET_FLOOR_CLEARANCE_M:
            lift = float(_RESET_FLOOR_CLEARANCE_M - min_z)
            self.data.qpos[2] += lift
            self.reset_floor_lift_m = lift
            mujoco.mj_forward(self.model, self.data)

    def _min_collision_geom_z(self) -> float:
        return self._min_collision_geom_z_for(set(range(self.model.ngeom)))

    def _min_collision_geom_z_for(self, geom_ids: set[int]) -> float:
        """计算一组可碰撞几何体最低点离地高度。"""
        min_z = float("inf")
        for geom_id in geom_ids:
            if (
                geom_id in self._ground_geom_ids
                or self.model.geom_type[geom_id]
                in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
                or self.model.geom_contype[geom_id] == 0
            ):
                continue
            min_z = min(min_z, self._geom_min_z(geom_id))
        return 0.0 if min_z == float("inf") else float(min_z)

    def _build_contact_geom_groups(
        self,
    ) -> tuple[set[int], set[int], set[int], set[int], set[int], set[int], set[int], set[int]]:
        """按 body/geom 名称缓存接地诊断需要的碰撞几何体分组。"""
        ground: set[int] = set()
        base: set[int] = set()
        legs: set[int] = set()
        left_legs: set[int] = set()
        right_legs: set[int] = set()
        wheels: set[int] = set()
        left_wheels: set[int] = set()
        right_wheels: set[int] = set()

        for geom_id in range(self.model.ngeom):
            if (
                int(self.model.geom_contype[geom_id]) == 0
                and int(self.model.geom_conaffinity[geom_id]) == 0
            ):
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            name = f"{body_name}/{geom_name}".lower()
            geom_type = int(self.model.geom_type[geom_id])

            if (
                body_id == 0
                or geom_type
                in (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
                or "terrain" in name
                or "floor" in name
                or "ground" in name
            ):
                ground.add(int(geom_id))
            if "base_link" in name:
                base.add(int(geom_id))
            if "l_wheel" in name:
                wheels.add(int(geom_id))
                left_wheels.add(int(geom_id))
                continue
            if "r_wheel" in name:
                wheels.add(int(geom_id))
                right_wheels.add(int(geom_id))
                continue
            if "lf0" in name or "lf1" in name:
                legs.add(int(geom_id))
                left_legs.add(int(geom_id))
            elif "rf0" in name or "rf1" in name:
                legs.add(int(geom_id))
                right_legs.add(int(geom_id))

        return ground, base, legs, left_legs, right_legs, wheels, left_wheels, right_wheels

    def _build_stair_geom_ids(self) -> set[int]:
        """缓存程序化台阶几何 id，用于 CTBC 立面触发。"""

        stair_geoms: set[int] = set()
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name.startswith("stair_terrain_step_"):
                stair_geoms.add(int(geom_id))
        return stair_geoms

    def _stair_riser_wheel_contact_score(self) -> np.ndarray:
        """估计左右轮撞到台阶立面的接触强度。"""

        score = np.zeros(2, dtype=np.float64)
        if self.stair_ctbc is None or not self._stair_geom_ids:
            return score

        fallback_score = float(self.cfg.stair_ctbc.force_threshold_n) + 1.0
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self._stair_geom_ids:
                wheel_geom = geom2
            elif geom2 in self._stair_geom_ids:
                wheel_geom = geom1
            else:
                continue

            if wheel_geom in self._left_wheel_geom_ids:
                side = 0
            elif wheel_geom in self._right_wheel_geom_ids:
                side = 1
            else:
                continue

            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            if normal.shape != (3,) or not np.isfinite(normal).all():
                continue
            if abs(float(normal[2])) > 0.5:
                continue

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_idx, contact_force)
            contact_score = abs(float(contact_force[0]))
            if not math.isfinite(contact_score) or contact_score <= 0.0:
                contact_score = fallback_score
            score[side] += contact_score
        return score

    def _ground_contact_state(self) -> dict[str, bool]:
        """返回轮子、腿和 base 是否正在与地面接触。"""
        left_wheel = False
        right_wheel = False
        left_leg = False
        right_leg = False
        base = False

        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self._ground_geom_ids:
                other = geom2
            elif geom2 in self._ground_geom_ids:
                other = geom1
            else:
                continue

            left_wheel = left_wheel or other in self._left_wheel_geom_ids
            right_wheel = right_wheel or other in self._right_wheel_geom_ids
            left_leg = left_leg or other in self._left_leg_geom_ids
            right_leg = right_leg or other in self._right_leg_geom_ids
            base = base or other in self._base_geom_ids

        leg = left_leg or right_leg
        wheel = left_wheel or right_wheel
        return {
            "wheel_contact": wheel,
            "wheel_full_contact": left_wheel and right_wheel,
            "wheel_contact_left": left_wheel,
            "wheel_contact_right": right_wheel,
            "leg_contact": leg,
            "leg_contact_left": left_leg,
            "leg_contact_right": right_leg,
            "base_contact": base,
            "nonwheel_contact": leg or base,
        }

    def _geom_min_z(self, geom_id: int) -> float:
        geom_type = self.model.geom_type[geom_id]
        geom_pos = self.data.geom_xpos[geom_id]
        geom_mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.model.geom_dataid[geom_id])
            vert_adr = int(self.model.mesh_vertadr[mesh_id])
            vert_num = int(self.model.mesh_vertnum[mesh_id])
            vertices = self.model.mesh_vert[vert_adr : vert_adr + vert_num]
            return float(np.min(vertices @ geom_mat[2, :]) + geom_pos[2])

        if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            axis_z = abs(float(geom_mat[2, 2]))
            radial_z = float(np.sqrt(max(0.0, 1.0 - axis_z * axis_z)))
            half_length = float(geom_size[1])
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_length += float(geom_size[0])
            vertical_extent = half_length * axis_z + float(geom_size[0]) * radial_z
            return float(geom_pos[2] - vertical_extent)

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            vertical_extent = float(np.dot(np.abs(geom_mat[2, :]), geom_size[:3]))
            return float(geom_pos[2] - vertical_extent)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_pos[2] - geom_size[0])

        return float(geom_pos[2])

    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos].copy()

    @property
    def dof_vel(self) -> np.ndarray:
        return self.data.qvel[self.joint_qvel].copy()

    @property
    def output_leg_pos(self) -> np.ndarray:
        return self.data.qpos[self.output_joint_qpos].copy()

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(-float(self.projected_gravity[2]), -1.0, 1.0))))

    @property
    def base_yaw(self) -> float:
        return float(self._base_yaw)

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"missing {obj_type.name} in MJCF: {name}")
        return int(idx)

    @staticmethod
    def _build_model(xml_path: str, cfg: RobotConfig) -> mujoco.MjModel:
        """加载 MJCF 并程序化添加 motor actuator（与训练端 DcMotorActuatorCfg 一致）。

        PD 控制在 Python 层计算，T-N 包络 clamp 也在 Python 层执行。
        MuJoCo actuator 仅作为力矩执行器，不做内部 PD。
        训练端 actuator 顺序: 先 leg（4个），再 wheel（2个）。
        """
        spec = mujoco.MjSpec.from_file(xml_path)
        if cfg.stair_terrain:
            WheelLeggedRobot._add_stair_terrain_geoms(spec, cfg)
        if cfg.rough_terrain:
            WheelLeggedRobot._add_rough_terrain(spec, cfg)

        joint_names = tuple(joint.name for joint in spec.joints if joint.name)
        site_names = tuple(site.name for site in spec.sites if site.name)
        fourbar_surrogate = FOURBAR_SURROGATE_MARKER in site_names
        if all(name in joint_names for name in JointGroup.POLICY_LEG_NAMES):
            leg_joint_names = JointGroup.POLICY_LEG_NAMES
        else:
            leg_joint_names = JointGroup.OPENCHAIN_LEG_NAMES
        wheel_joint_names = JointGroup.WHEEL_NAMES

        for jname in leg_joint_names:
            act = spec.add_actuator()
            act.name = f"{jname}_motor"
            act.target = jname
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.dyntype = mujoco.mjtDyn.mjDYN_NONE
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.gainprm[0] = 1.0
            act.forcelimited = not fourbar_surrogate
            if act.forcelimited:
                act.forcerange[:] = np.array([-DM8009P.rated_torque, DM8009P.rated_torque])
            act.ctrllimited = False
            act.inheritrange = 0.0

        for jname in wheel_joint_names:
            act = spec.add_actuator()
            act.name = f"{jname}_motor"
            act.target = jname
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.dyntype = mujoco.mjtDyn.mjDYN_NONE
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.gainprm[0] = 1.0
            act.forcelimited = True
            act.forcerange[:] = np.array([-M3508_C620_14.rated_torque, M3508_C620_14.rated_torque])
            act.ctrllimited = False
            act.inheritrange = 0.0

        model = spec.compile()
        WheelLeggedRobot._assign_ground_geom_group(model)
        return model

    @staticmethod
    def _add_rough_terrain(spec: mujoco.MjSpec, cfg: RobotConfig) -> None:
        """用 MJLab 的 ROUGH_TERRAINS_CFG 生成 sim2sim rough terrain。"""

        from mjlab.terrains.config import ROUGH_TERRAINS_CFG
        from mjlab.terrains.terrain_generator import TerrainGenerator

        WheelLeggedRobot._disable_default_floor_contacts(spec)
        terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
        terrain_cfg.seed = int(cfg.seed)
        terrain_cfg.size = tuple(float(v) for v in cfg.rough_terrain_size_m)
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            sub_cfg = terrain_cfg.sub_terrains.get(name)
            if sub_cfg is not None and hasattr(sub_cfg, "step_height_range"):
                sub_cfg.step_height_range = tuple(
                    float(v) for v in cfg.rough_stair_step_height_range
                )
        if str(cfg.rough_terrain_type) == "mixed":
            generator = TerrainGenerator(terrain_cfg, device="cpu")
            generator.compile(spec)
            terrain_body = spec.body("terrain")
            level = max(0, min(int(terrain_cfg.num_rows) - 1, int(cfg.rough_terrain_level)))
            col = int(getattr(generator, "_num_cols", terrain_cfg.num_cols)) // 2
            selected_origin = np.asarray(generator.terrain_origins[level, col], dtype=np.float64)
            terrain_body.pos = (-selected_origin).tolist()
            for geom in terrain_body.geoms:
                geom.group = _GROUND_GEOM_GROUP
                if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
                    continue
                geom.contype = 2
                geom.conaffinity = 1
            return

        terrain_body = spec.worldbody.add_body(name="terrain")
        sub_cfg = replace(terrain_cfg.sub_terrains[str(cfg.rough_terrain_type)])
        sub_cfg.size = tuple(float(v) for v in cfg.rough_terrain_size_m)

        level = max(0, min(9, int(cfg.rough_terrain_level)))
        lower, upper = (float(v) for v in terrain_cfg.difficulty_range)
        difficulty = lower + (upper - lower) * ((float(level) + 0.5) / 10.0)
        rng = np.random.default_rng(int(cfg.seed) + 7919 + 104729 * level)
        output = sub_cfg.function(difficulty, spec, rng)

        offset = -np.asarray(output.origin, dtype=np.float64)
        for index, terrain_geom in enumerate(output.geometries):
            geom = terrain_geom.geom
            if geom is None:
                continue
            geom.name = f"rough_terrain_{index}"
            geom.pos = np.asarray(geom.pos, dtype=np.float64) + offset
            geom.mass = 0.0
            geom.group = _GROUND_GEOM_GROUP
            geom.contype = 2
            geom.conaffinity = 1
            if terrain_geom.color is not None:
                geom.rgba[:] = terrain_geom.color

        for index, geom in enumerate(terrain_body.geoms):
            if not geom.name:
                geom.name = f"rough_terrain_{index}"
            geom.mass = 0.0

    @staticmethod
    def _disable_default_floor_contacts(spec: mujoco.MjSpec) -> None:
        """rough terrain 开启时关闭 MJCF 默认平面碰撞，避免覆盖 MJLab 地形。"""

        for geom in spec.worldbody.geoms:
            name = str(geom.name or "").lower()
            if (
                int(geom.type) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                or "floor" in name
                or "ground" in name
            ):
                geom.contype = 0
                geom.conaffinity = 0
                geom.rgba[3] = 0.0

    @staticmethod
    def _add_stair_terrain_geoms(spec: mujoco.MjSpec, cfg: RobotConfig) -> None:
        """在原生 MuJoCo sim2sim 中添加与训练课程同尺度的台阶碰撞体。"""

        low, high = (float(v) for v in cfg.stair_step_height_range)
        level = max(0, min(9, int(cfg.stair_terrain_level)))
        step_height = low + (float(level) / 9.0) * (high - low)
        step_depth = float(cfg.stair_step_depth_m)
        half_width = float(cfg.stair_half_width_m)
        start_x = float(cfg.stair_start_x_m)
        for idx in range(int(cfg.stair_step_count)):
            height = step_height * float(idx + 1)
            spec.worldbody.add_geom(
                name=f"stair_terrain_step_{idx}",
                type=int(mujoco.mjtGeom.mjGEOM_BOX),
                pos=[
                    start_x + idx * step_depth + 0.5 * step_depth,
                    0.0,
                    0.5 * height,
                ],
                size=[0.5 * step_depth, half_width, 0.5 * height],
                contype=2,
                conaffinity=1,
                condim=3,
                group=_GROUND_GEOM_GROUP,
                friction=[0.8, 0.005, 0.0001],
                rgba=[0.35, 0.37, 0.33, 1.0],
            )

    @staticmethod
    def _assign_ground_geom_group(model: mujoco.MjModel) -> None:
        for geom_id in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
            geom_type = int(model.geom_type[geom_id])
            if (
                geom_type in (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
                or "floor" in name
                or "ground" in name
            ):
                model.geom_group[geom_id] = _GROUND_GEOM_GROUP

    def _apply_default_joint_positions(self) -> None:
        """把闭链被动输出和 policy 关节一起写到默认站姿。"""
        default_map = _SHARED_ROBOT.default_model_joint_pos
        for joint_name, value in default_map.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(value)
        if self.cfg.initial_leg_joint_pos is None:
            return
        initial_leg_pos = np.asarray(self.cfg.initial_leg_joint_pos, dtype=np.float64).reshape(-1)
        if initial_leg_pos.shape == (2,):
            left_front, left_back = initial_leg_pos
            initial_leg_pos = np.asarray(
                [left_front, left_back, left_front, left_back],
                dtype=np.float64,
            )
        elif initial_leg_pos.shape != (4,):
            raise ValueError(
                "initial_leg_joint_pos must contain 2 mirrored or 4 policy-order values"
            )
        if not np.isfinite(initial_leg_pos).all():
            raise ValueError("initial_leg_joint_pos must be finite")
        for joint_name, value in zip(self.policy_joint_names[:4], initial_leg_pos, strict=True):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise ValueError(f"missing leg joint in MJCF: {joint_name}")
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(value)

    def _apply_initial_wheel_joint_positions(self) -> None:
        """覆写 deploy telemetry 中记录的轮子连续关节位置。"""
        if self.cfg.initial_wheel_joint_pos is None:
            return
        wheel_pos = np.asarray(self.cfg.initial_wheel_joint_pos, dtype=np.float64).reshape(-1)
        if wheel_pos.shape != (2,) or not np.isfinite(wheel_pos).all():
            raise ValueError("initial_wheel_joint_pos must contain 2 finite values")
        for joint_name, value in zip(JointGroup.WHEEL_NAMES, wheel_pos, strict=True):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise ValueError(f"missing wheel joint in MJCF: {joint_name}")
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(value)

    def _apply_initial_dof_vel(self) -> None:
        """覆写 deploy telemetry 中记录的 6 维 policy-order 关节速度。"""
        self.data.qvel[self.joint_qvel] = 0.0
        if self.cfg.initial_dof_vel is None:
            return
        dof_vel = np.asarray(self.cfg.initial_dof_vel, dtype=np.float64).reshape(-1)
        if dof_vel.shape != (6,) or not np.isfinite(dof_vel).all():
            raise ValueError("initial_dof_vel must contain 6 finite policy-order values")
        self.data.qvel[self.joint_qvel] = dof_vel

    def _apply_initial_policy_io_state(self) -> None:
        """覆写 reset 后的 last_action/action FIFO，使初始 obs 与 deploy obs 对齐。"""
        if self.cfg.initial_last_action is None:
            return
        last_action = np.asarray(self.cfg.initial_last_action, dtype=np.float64).reshape(-1)
        expected = self.runtime.policy.num_actions
        if last_action.shape != (expected,) or not np.isfinite(last_action).all():
            raise ValueError(f"initial_last_action must contain {expected} finite values")
        self.last_action[:] = last_action
        self.last_applied_action[:] = last_action
        self.last_policy_action[:] = last_action
        self.last_clipped_policy_action[:] = last_action
        self.action_fifo[:] = last_action

    def _close_initial_chain_positions(self) -> ClosedChainClosureSolver | None:
        """让 deploy 初始主动关节对应的闭链被动关节落在同一机构分支上。"""
        self.closed_chain_reset_position_residual_m = 0.0
        if self.cfg.initial_leg_joint_pos is None:
            return None
        solver = ClosedChainClosureSolver.try_create(model=self.model, data=self.data)
        if solver is None:
            return None
        self.closed_chain_reset_position_residual_m = solver.solve_positions()
        return solver

    def _close_initial_chain_velocities(
        self,
        solver: ClosedChainClosureSolver | None,
    ) -> None:
        """根据主动关节速度反解闭链被动关节速度。"""
        self.closed_chain_reset_velocity_residual = 0.0
        if solver is None or self.cfg.initial_dof_vel is None:
            return
        self.closed_chain_reset_velocity_residual = solver.solve_velocities()

    def _refresh_state(self) -> None:
        self.base_quat = self.data.qpos[3:7].copy()
        self.base_lin_vel_world = self.data.qvel[0:3].copy()
        self.base_lin_vel_body = rotate_inverse(self.base_quat, self.base_lin_vel_world)
        self.base_ang_vel_body = self.data.qvel[3:6].copy()
        self.base_ang_vel_world = rotate(self.base_quat, self.base_ang_vel_body)
        self.projected_gravity = rotate_inverse(self.base_quat, np.asarray([0.0, 0.0, -1.0]))
        self._base_yaw = extract_yaw(self.base_quat)

    def _compute_pd_torques(self, action: np.ndarray) -> np.ndarray:
        """计算 PD 转矩并应用 T-N 包络 clamp，输出直接写入 data.ctrl。

        与训练端 DcMotorActuatorCfg 行为一致：
        - 腿部: torque = kp*(pos_target - pos) + kd*(0 - vel), clamp by DM8009P T-N
        - 轮子: torque = kd*(vel_target - vel), clamp by M3508_C620_14 T-N
        """
        action = np.asarray(action, dtype=np.float64)

        decoded_action = self.action_decoder.decode(
            action,
            command_height=float(self.command[4]),
            fallback_default=self.default_dof_pos[JointGroup.CTRL_LEGS],
        )
        return self._compute_decoded_target_torques(
            leg_target=decoded_action.leg_target,
            wheel_vel_target=decoded_action.wheel_vel_target,
        )

    def _compute_decoded_target_torques(
        self,
        *,
        leg_target: np.ndarray,
        wheel_vel_target: np.ndarray,
    ) -> np.ndarray:
        """从部署物理目标计算电机力矩。"""
        leg_target = np.asarray(leg_target, dtype=np.float64).reshape(4)
        wheel_vel_target = np.asarray(wheel_vel_target, dtype=np.float64).reshape(2)

        dof_pos = self.dof_pos
        dof_vel = self.dof_vel

        output_leg_pos = dof_pos[JointGroup.CTRL_LEGS]
        output_leg_vel = dof_vel[JointGroup.CTRL_LEGS]
        if self.fourbar_surrogate:
            policy_pos = output_to_policy_pos_np(output_leg_pos)
            policy_vel = output_to_policy_vel_np(output_leg_pos, output_leg_vel)
            policy_target = leg_target
            policy_error = policy_leg_position_error_np(policy_target, policy_pos)
            policy_torque = self.leg_kp * policy_error
            policy_torque -= self.leg_kd * policy_vel
            policy_torque = _tn_clip(
                policy_torque,
                policy_vel,
                DM8009P.stall_torque,
                DM8009P.no_load_speed,
                DM8009P.rated_torque,
            )
            leg_torque = policy_to_output_torque_np(policy_pos, policy_torque)
            leg_vel = policy_vel
        else:
            if self.active_rod_action_semantics:
                leg_pos_err = policy_leg_position_error_np(leg_target, output_leg_pos)
            else:
                leg_pos_err = leg_target - output_leg_pos
            leg_vel = output_leg_vel
            leg_torque = self.leg_kp * leg_pos_err - self.leg_kd * leg_vel
        if not self.fourbar_surrogate:
            leg_torque = _tn_clip(
                leg_torque,
                leg_vel,
                DM8009P.stall_torque,
                DM8009P.no_load_speed,
                DM8009P.rated_torque,
            )

        wheel_vel = dof_vel[JointGroup.CTRL_WHEELS]
        wheel_torque = self.wheel_kd * (wheel_vel_target - wheel_vel)
        wheel_torque = M3508_C620_14.clip_effort_np(wheel_torque, wheel_vel)

        # MuJoCo actuator 顺序由 _build_model 固定为 legs(4) + wheels(2)。
        ctrl = np.concatenate([leg_torque, wheel_torque])
        self.last_ctrl[:] = ctrl
        return ctrl

    def _compute_hold_current_torques(self, leg_hold_target: np.ndarray) -> np.ndarray:
        """计算 output disabled 时的保持目标力矩。"""
        dof_pos = self.dof_pos
        dof_vel = self.dof_vel
        leg_target = np.asarray(leg_hold_target, dtype=np.float64).reshape(4)
        leg_pos = dof_pos[JointGroup.CTRL_LEGS]
        leg_vel = dof_vel[JointGroup.CTRL_LEGS]
        leg_torque = self.leg_kp * (leg_target - leg_pos)
        leg_torque -= self.leg_kd * leg_vel
        leg_torque = _tn_clip(
            leg_torque,
            leg_vel,
            DM8009P.stall_torque,
            DM8009P.no_load_speed,
            DM8009P.rated_torque,
        )

        wheel_vel = dof_vel[JointGroup.CTRL_WHEELS]
        wheel_torque = self.wheel_kd * (0.0 - wheel_vel)
        wheel_torque = M3508_C620_14.clip_effort_np(wheel_torque, wheel_vel)

        ctrl = np.concatenate([leg_torque, wheel_torque])
        self.last_ctrl[:] = ctrl
        return ctrl

    def _sample_action_delay_steps(self) -> int:
        if self.min_action_delay_steps == self.max_action_delay_steps:
            return int(self.min_action_delay_steps)
        return int(
            self.rng.integers(
                int(self.min_action_delay_steps),
                int(self.max_action_delay_steps) + 1,
            )
        )

    def _termination_status(self) -> tuple[bool, str, bool]:
        invalid_state = not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.ctrl).all()
        )
        if invalid_state:
            return True, "invalid_state", False

        fall_detected = bool(
            self.tilt_deg > float(self.termination.fail_tilt_deg)
            or float(self.data.qpos[2]) < float(self.termination.fail_height_m)
        )
        if bool(self.termination.terminate_on_fall) and fall_detected:
            return True, "fall", True
        return False, "running", fall_detected
