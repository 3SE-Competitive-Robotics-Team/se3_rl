"""不依赖旧脚本路径的完整 sim2sim workflow。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from se3_train.mdp.jump_trajectories import DEFAULT_JUMP_TRAJ_HEIGHTS, DEFAULT_JUMP_TRAJ_PATHS

from .config import RunConfig
from .course import CourseType, create_course
from .diagnostics import rollout_diagnostics
from .mujoco_viewer import CompositeViewer, MujocoViewer
from .policy import PolicyRuntime
from .rerun_viewer import RerunViewer
from .robot import WheelLeggedRobot
from .runtime_spec import RuntimeSpec
from .teleop_input import CommandInputSource


class Sim2SimWorkflow:
    def __init__(self, cfg: RunConfig, *, command_source: CommandInputSource | None = None) -> None:
        self.cfg = cfg.resolved()
        self.runtime = RuntimeSpec(task=self.cfg.robot.task)
        self.robot = WheelLeggedRobot(cfg=self.cfg.robot, runtime=self.runtime)
        if self.cfg.policy.checkpoint is None:
            raise RuntimeError("启动 workflow 前必须先解析 policy checkpoint")
        self.policy = PolicyRuntime(
            checkpoint=self.cfg.policy.checkpoint,
            device=self.cfg.policy.device,
            runtime=self.runtime,
        )
        control_dt = self.cfg.robot.sim_dt * self.cfg.robot.control_decimation
        self._course = create_course(self.cfg.course, control_dt)
        self.command_source = command_source
        self.viewer = self._make_viewer()

    def run(self) -> dict[str, object]:
        obs = self.robot.reset(fixed=self.cfg.fixed_reset, randomize_root=self.cfg.randomize_root)
        deploy_init_obs_check = self._deploy_telemetry_initial_obs_check(obs)
        if deploy_init_obs_check.get("enabled"):
            print(
                "[deploy telemetry init] "
                f"mode={deploy_init_obs_check.get('mode', '')} "
                f"sample={deploy_init_obs_check.get('selected_sample_index', '')} "
                f"max_obs_err={float(deploy_init_obs_check['max_abs_error']):.3e}"
            )
        pre_policy_settle = {"enabled": False}
        if self.cfg.robot.settle_base_before_policy:
            pre_policy_settle = self.robot.settle_base_before_policy(
                max_s=self.cfg.robot.pre_policy_settle_max_s,
                contact_steps=self.cfg.robot.pre_policy_settle_contact_steps,
            )
            print(
                "[pre-policy settle] "
                f"success={pre_policy_settle['success']} "
                f"steps={pre_policy_settle['steps']} "
                f"time={float(pre_policy_settle['time_s']):.3f}s "
                f"base_touching={pre_policy_settle.get('base_touching', False)} "
                f"base_contact={pre_policy_settle['base_contact']} "
                f"snap_down={float(pre_policy_settle.get('base_snap_down_m', 0.0)):.4f}m "
                f"base_clearance={float(pre_policy_settle['base_clearance']):+.4f}m"
            )
            obs = self.robot.observation()
        initial_policy_io = self._policy_io_diagnostics(obs)
        samples: list[dict[str, float]] = []
        model_diag = self.robot.diagnostics()

        rc_sched = self.cfg.robot.rc_switch
        rc_events = tuple(rc_sched.events)
        _next_rc_event = 0
        _rc_output_enabled = bool(rc_sched.initial_output_enabled)
        _policy_memory_clean = True
        if not _rc_output_enabled:
            self.policy.reset()
            self.robot.reset_policy_io_state()
            obs = self.robot.observation()

        if self.viewer is not None:
            self.viewer.log_model(self.robot.model)
            initial_telemetry = self.robot.telemetry()
            initial_telemetry["rc_switch_r"] = 1.0 if _rc_output_enabled else 0.0
            initial_telemetry["output_enabled"] = 1.0 if _rc_output_enabled else 0.0
            initial_telemetry["rc_switch_event"] = 0.0
            initial_telemetry["rc_policy_reset"] = 0.0
            initial_telemetry["rc_off_mode"] = str(rc_sched.off_mode)
            self.viewer.log_state(
                self.robot.model, self.robot.data, step=0, telemetry=initial_telemetry
            )

        max_steps = int(self.cfg.max_steps)
        step_iter = range(1, max_steps + 1) if max_steps > 0 else itertools.count(1)
        done_reason = "max_steps" if max_steps > 0 else "interrupted"

        # -----------------------------------------------------------------------
        # 跳跃参考相位（对齐训练端 JumpCommandTerm）
        #
        # sim2sim 不再根据接触力维护 airborne/landing 状态机。训练端已经把
        # jump_stage 作为唯一阶段来源，这里只按参考轨迹步数推进 jump_phase，
        # 轨迹结束后清除 jump_flag 并回到行走。
        # -----------------------------------------------------------------------
        sched = self.cfg.robot.jump_schedule
        script_events = tuple(sched.events)
        control_dt = self.cfg.robot.sim_dt * self.cfg.robot.control_decimation

        _interval_steps_remaining = 0  # 距离下次触发跳跃的剩余步数（定时模式）
        _next_script_event = 0  # 下一个按绝对时间触发的跳跃事件索引
        _traj_step = 0
        _traj_steps = self._trajectory_steps_for_height(float(self.robot.command[6]))
        _prev_action: np.ndarray | None = None
        _prev_applied_action: np.ndarray | None = None

        # 初始化：静态模式下读取用户命令中的 jump_flag；调度模式下初始不跳
        if sched.enabled or script_events:
            self.robot.command[5] = 0.0
            self.robot.command[6] = (
                float(script_events[0].target_height)
                if script_events
                else float(sched.target_height)
            )
            _traj_steps = self._trajectory_steps_for_height(float(self.robot.command[6]))
        if sched.enabled:
            _interval_steps_remaining = max(1, round(sched.interval_s / control_dt))
        _jump_active = self.robot.command[5] > 0.5  # 当前是否处于跳跃意图激活状态

        # --- 历程初始化 ---
        if self._course is not None and self.cfg.course.mode == CourseType.JUMP_SWEEP:
            # jump-sweep：前进速度为 0，接管跳跃调度
            self.robot.command[0] = 0.0
            sched.enabled = True
            self.robot.command[6] = self._course.current_height
            _traj_steps = self._trajectory_steps_for_height(float(self.robot.command[6]))
            _interval_steps_remaining = max(
                1, round(self.cfg.course.jump_sweep_interval_s / control_dt)
            )

        if self._course is not None:
            if self.cfg.course.mode == CourseType.WALK_SWEEP:
                print(
                    f"[历程] walk-sweep: vx 0.1→0.6 m/s 每档 {self.cfg.course.walk_sweep_segment_duration_s}s"
                )
            elif self.cfg.course.mode == CourseType.JUMP_SWEEP:
                heights_str = ", ".join(f"{h:.1f}m" for h in self.cfg.course.jump_sweep_heights)
                print(
                    f"[历程] jump-sweep: {heights_str}, 间隔 {self.cfg.course.jump_sweep_interval_s}s"
                )
            elif self.cfg.course.mode == CourseType.UPRIGHT_VELOCITY_SWEEP:
                commands_str = ", ".join(
                    f"vx={vx:+.1f},yaw={yaw:+.1f}"
                    for vx, yaw in self.cfg.course.upright_velocity_sweep_commands
                )
                print(
                    "[历程] upright-velocity-sweep: "
                    f"{commands_str}, 每档 {self.cfg.course.upright_velocity_sweep_segment_duration_s}s"
                )

        try:
            for step in step_iter:
                if self._viewer_is_closed(self.viewer):
                    done_reason = "viewer_closed"
                    break
                sim_time_s = (step - 1) * control_dt
                if self.command_source is not None:
                    self.command_source.pace(sim_time_s)
                # --- jump_phase 更新（对齐训练端参考轨迹）---
                if _jump_active:
                    self.robot.command[5] = 1.0
                    self.robot.command[7] = float(
                        min(_traj_step, _traj_steps - 1) / max(_traj_steps - 1, 1)
                    )
                    _traj_step += 1
                    if _traj_step >= _traj_steps:
                        _traj_step = 0
                        _jump_active = False
                        self.robot.command[5] = 0.0
                        self.robot.command[7] = 0.0
                        if sched.enabled:
                            _interval_steps_remaining = max(1, round(sched.interval_s / control_dt))
                        # 跳跃历程：每次跳跃完成后切换到下一高度
                        if (
                            self._course is not None
                            and self.cfg.course.mode == CourseType.JUMP_SWEEP
                        ):
                            self._course.advance()
                            self.robot.command[6] = self._course.current_height
                            _traj_steps = self._trajectory_steps_for_height(
                                float(self.robot.command[6])
                            )
                            if self._course.done:
                                sched.enabled = False
                else:
                    self.robot.command[5] = 0.0
                    self.robot.command[7] = 0.0

                # --- 定时跳跃调度 ---
                course_done = (
                    self._course is not None
                    and self.cfg.course.mode == CourseType.JUMP_SWEEP
                    and self._course.done
                )
                if sched.enabled and not _jump_active and not course_done:
                    _interval_steps_remaining -= 1
                    if _interval_steps_remaining <= 0:
                        _traj_steps = self._trajectory_steps_for_height(
                            float(self.robot.command[6])
                        )
                        _jump_active = True  # 触发下一次跳跃
                elif script_events and not _jump_active:
                    if _next_script_event < len(script_events):
                        event = script_events[_next_script_event]
                        if sim_time_s + 1.0e-9 >= float(event.trigger_time_s):
                            self.robot.command[6] = float(event.target_height)
                            _traj_steps = self._trajectory_steps_for_height(
                                float(self.robot.command[6])
                            )
                            _jump_active = True  # 按脚本触发下一次跳跃
                            _next_script_event += 1

                # --- 行走历程：每步推进 vx ---
                if self._course is not None and self.cfg.course.mode == CourseType.WALK_SWEEP:
                    self._course.step()
                    self.robot.command[0] = self._course.current_vx

                if self.cfg.robot.yaw_pid.enabled:
                    self.robot.update_yaw_command()
                    obs = self.robot.observation()
                if (
                    self._course is not None
                    and self.cfg.course.mode == CourseType.UPRIGHT_VELOCITY_SWEEP
                ):
                    self._course.step()
                    vx, yaw = self._course.current_command
                    self.robot.command[0] = vx
                    self.robot.command[1] = yaw
                    obs = self.robot.observation()

                rc_switch_event = 0.0
                while _next_rc_event < len(rc_events):
                    event = rc_events[_next_rc_event]
                    if sim_time_s + 1.0e-9 < float(event.trigger_time_s):
                        break
                    _rc_output_enabled = bool(event.output_enabled)
                    _next_rc_event += 1
                    rc_switch_event = 1.0

                if self.command_source is not None:
                    command_update = self.command_source.poll(sim_time_s)
                    if command_update.quit_requested:
                        done_reason = "user_quit"
                        break
                    self.robot.command[0] = float(command_update.lin_vel_x)
                    self.robot.command[1] = float(command_update.yaw_rate)
                    self.robot.command[4] = float(command_update.command_height)
                    if command_update.toggle_output:
                        _rc_output_enabled = not _rc_output_enabled
                        rc_switch_event = 1.0
                    obs = self.robot.observation()

                rc_policy_reset = 0.0
                if _rc_output_enabled:
                    action = self.policy.act(obs)
                    _policy_memory_clean = False
                    obs, reward, done, info = self.robot.step(action)
                    info["target_mode"] = "policy"
                else:
                    if not _policy_memory_clean:
                        self.policy.reset()
                        self.robot.reset_policy_io_state()
                        obs = self.robot.observation()
                        _policy_memory_clean = True
                        rc_policy_reset = 1.0
                    if rc_sched.off_mode == "hold-current":
                        obs, reward, done, info = self.robot.step_hold_current()
                    else:
                        obs, reward, done, info = self.robot.step_no_torque()
                info["rc_switch_r"] = 1.0 if _rc_output_enabled else 0.0
                info["output_enabled"] = 1.0 if _rc_output_enabled else 0.0
                info["rc_switch_event"] = rc_switch_event
                info["rc_policy_reset"] = rc_policy_reset
                info["rc_off_mode"] = str(rc_sched.off_mode)
                action_now = np.asarray(info["last_action"], dtype=np.float64)
                applied_action_now = np.asarray(info["applied_action"], dtype=np.float64)
                action_delta = (
                    np.zeros_like(action_now) if _prev_action is None else action_now - _prev_action
                )
                applied_action_delta = (
                    np.zeros_like(applied_action_now)
                    if _prev_applied_action is None
                    else applied_action_now - _prev_applied_action
                )
                _prev_action = action_now
                _prev_applied_action = applied_action_now
                sample = {
                    "step": float(step),
                    "time": float(info["time"]),
                    "height": float(info["height"]),
                    "reset_floor_lift_m": float(info.get("reset_floor_lift_m", 0.0)),
                    "wheel_clearance": float(info.get("wheel_clearance", 0.0)),
                    "wheel_clearance_left": float(info.get("wheel_clearance_left", 0.0)),
                    "wheel_clearance_right": float(info.get("wheel_clearance_right", 0.0)),
                    "wheel_lateral_distance": float(info.get("wheel_lateral_distance", 0.0)),
                    "wheel_fore_aft_offset": float(info.get("wheel_fore_aft_offset", 0.0)),
                    "leg_mirror_error": float(info.get("leg_mirror_error", 0.0)),
                    "leg_clearance": float(info.get("leg_clearance", 0.0)),
                    "base_clearance": float(info.get("base_clearance", 0.0)),
                    "wheel_contact": float(info.get("wheel_contact", 0.0)),
                    "wheel_full_contact": float(info.get("wheel_full_contact", 0.0)),
                    "wheel_contact_left": float(info.get("wheel_contact_left", 0.0)),
                    "wheel_contact_right": float(info.get("wheel_contact_right", 0.0)),
                    "leg_contact": float(info.get("leg_contact", 0.0)),
                    "leg_contact_left": float(info.get("leg_contact_left", 0.0)),
                    "leg_contact_right": float(info.get("leg_contact_right", 0.0)),
                    "base_contact": float(info.get("base_contact", 0.0)),
                    "nonwheel_contact": float(info.get("nonwheel_contact", 0.0)),
                    "tilt_deg": float(info["tilt_deg"]),
                    "roll_deg": float(info.get("roll_deg", 0.0)),
                    "pitch_deg": float(info.get("pitch_deg", 0.0)),
                    "yaw_deg": float(info.get("yaw_deg", 0.0)),
                    "roll_rate_rad_s": float(info["base_ang_vel_body"][0]),
                    "pitch_rate_rad_s": float(info["base_ang_vel_body"][1]),
                    "yaw_rate_rad_s": float(info["base_ang_vel_body"][2]),
                    "reward": float(reward),
                    "command_lin_vel_x": float(info.get("command_lin_vel_x", 0.0)),
                    "command_yaw_rate": float(info.get("command_yaw_rate", 0.0)),
                    "command_height": float(info.get("command_height", 0.0)),
                    "base_lin_vel_x": float(info["base_lin_vel_x"]),
                    "wheel_lin_vel": float(info["wheel_lin_vel"]),
                    "action_delta_l2": float(np.linalg.norm(action_delta)),
                    "action_delta_max_abs": float(np.max(np.abs(action_delta))),
                    "action_delta_sq_sum": float(np.sum(np.square(action_delta))),
                    "applied_action_delta_l2": float(np.linalg.norm(applied_action_delta)),
                    "applied_action_delta_max_abs": float(np.max(np.abs(applied_action_delta))),
                    "applied_action_delta_sq_sum": float(np.sum(np.square(applied_action_delta))),
                    "action_delay_steps": float(info["action_delay_steps"]),
                    "action_delay_s": float(info["action_delay_s"]),
                    "rc_switch_r": float(info.get("rc_switch_r", 1.0)),
                    "output_enabled": float(info.get("output_enabled", 1.0)),
                    "rc_switch_event": float(info.get("rc_switch_event", 0.0)),
                    "rc_policy_reset": float(info.get("rc_policy_reset", 0.0)),
                    "rc_off_mode_no_torque": 1.0
                    if str(info.get("rc_off_mode", "hold-current")) == "no-torque"
                    else 0.0,
                    "closed_chain_reset_position_residual_m": float(
                        info.get("closed_chain_reset_position_residual_m", 0.0)
                    ),
                    "closed_chain_reset_velocity_residual": float(
                        info.get("closed_chain_reset_velocity_residual", 0.0)
                    ),
                }
                samples.append(sample)
                if self.viewer is not None and step % max(1, int(self.cfg.viewer.log_every)) == 0:
                    self.viewer.log_state(
                        self.robot.model, self.robot.data, step=step, telemetry=info
                    )
                    if self._viewer_is_closed(self.viewer):
                        done_reason = "viewer_closed"
                        break
                if int(self.cfg.print_every) > 0 and step % int(self.cfg.print_every) == 0:
                    course_info = ""
                    if self._course is not None:
                        report = self._course.report()
                        if self.cfg.course.mode == CourseType.WALK_SWEEP:
                            course_info = f" course_vx={report.get('vx', 0.0):.2f}"
                        elif self.cfg.course.mode == CourseType.JUMP_SWEEP:
                            course_info = (
                                f" jump_h={report.get('height', 0.0):.1f}"
                                f" ({report.get('index', 0)}/{report.get('total', 0)})"
                            )
                        elif self.cfg.course.mode == CourseType.UPRIGHT_VELOCITY_SWEEP:
                            course_info = (
                                f" course_vx={float(report.get('vx', 0.0)):+.2f}"
                                f" course_yaw={float(report.get('yaw', 0.0)):+.2f}"
                            )
                    rc_info = ""
                    if (
                        self.command_source is not None
                        or rc_events
                        or not bool(rc_sched.initial_output_enabled)
                    ):
                        rc_info = (
                            f" rc={'on' if _rc_output_enabled else 'off'}"
                            f" mode={info.get('target_mode', 'policy')}"
                            f" reset={int(float(info.get('rc_policy_reset', 0.0)))}"
                        )
                    line = (
                        f"step={step:05d} time={float(info['time']):.3f}"
                        f"{course_info}"
                        f"{rc_info}"
                        f" cmd_h={float(info.get('command_height', 0.0)):.3f}"
                        f" base_h={float(info['height']):.3f} "
                        f"wheel_clr={float(info.get('wheel_clearance', 0.0)):.3f} "
                        f"tilt={float(info['tilt_deg']):.2f} "
                        f"reward={float(reward):+.4f}"
                    )
                    if self.cfg.print_debug:
                        yaw = info.get("yaw_pid")
                        yaw_debug = ""
                        if isinstance(yaw, dict):
                            yaw_debug = (
                                f" yaw_current={float(yaw['current_yaw']):+.3f}"
                                f" yaw_target={float(yaw['target_yaw']):+.3f}"
                                f" yaw_error={float(yaw['error']):+.3f}"
                                f" yaw_cmd={float(yaw['command']):+.3f}"
                            )
                        line += (
                            f" dof_pos={self._fmt(info['dof_pos'])}"
                            f" raw_action={self._fmt(info['policy_action_raw'])}"
                            f" action={self._fmt(info['last_action'])}"
                            f" applied={self._fmt(info['applied_action'])}"
                            f" ctrl={self._fmt(info['last_ctrl'])}"
                            f"{yaw_debug}"
                        )
                    print(line)
                if done:
                    done_reason = str(info.get("done_reason", "done"))
                    break
        except KeyboardInterrupt:
            done_reason = "interrupted"

        summary = {
            "config": self.cfg.to_dict(),
            "runtime": self.runtime.to_dict(),
            "policy": {
                "checkpoint": str(self.policy.checkpoint_path),
                "iteration": self.policy.iteration,
                "policy_type": self.policy.policy_type,
                "spec": self.policy.spec.to_dict(),
            },
            "model_diagnostics": model_diag,
            "pre_policy_settle": pre_policy_settle,
            "initial_policy_io": initial_policy_io,
            "deploy_telemetry_init_obs_check": deploy_init_obs_check,
            "rollout": rollout_diagnostics(samples),
            "done_reason": done_reason,
        }
        if script_events:
            summary["jump_events"] = self._jump_event_diagnostics(samples, script_events)
        if (
            self.command_source is not None
            or rc_events
            or not bool(rc_sched.initial_output_enabled)
        ):
            summary["rc_switch"] = self._rc_switch_diagnostics(
                samples,
                rc_events,
                bool(rc_sched.initial_output_enabled),
            )
        if self.cfg.course.mode == CourseType.UPRIGHT_VELOCITY_SWEEP:
            summary["upright_velocity_sweep"] = self._upright_velocity_sweep_diagnostics(
                samples,
                self.cfg.course.upright_velocity_sweep_commands,
            )
        if self.cfg.course.mode == CourseType.WALK_SWEEP:
            summary["walk_sweep"] = self._walk_sweep_diagnostics(
                samples,
                self.cfg.course.walk_sweep_velocities,
            )
        if self.cfg.json_output is not None:
            self._write_json(self.cfg.json_output, summary)
        if self.viewer is not None:
            self.viewer.close()
        return summary

    def _make_viewer(self) -> RerunViewer | MujocoViewer | CompositeViewer | None:
        if self.cfg.viewer.mode == "none":
            return None
        if self.cfg.viewer.mode == "mujoco":
            key_callback = getattr(self.command_source, "key_callback", None)
            if not callable(key_callback):
                key_callback = None
            viewers: list[object] = []
            if self.cfg.viewer.record_to_rrd is not None:
                viewers.append(
                    RerunViewer(
                        app_id=self.cfg.viewer.app_id,
                        spawn=False,
                        address=self.cfg.viewer.address,
                        record_to_rrd=self.cfg.viewer.record_to_rrd,
                        memory_limit=self.cfg.viewer.memory_limit,
                        follow_body=self.cfg.viewer.follow_body,
                        geom_view=self.cfg.viewer.geom_view,
                    )
                )
            viewers.append(
                MujocoViewer(
                    model=self.robot.model,
                    data=self.robot.data,
                    key_callback=key_callback,
                    pose_joint_names=self.robot.policy_joint_names[:4],
                    follow_body=self.cfg.viewer.follow_body,
                    geom_view=self.cfg.viewer.geom_view,
                )
            )
            return CompositeViewer(viewers) if len(viewers) > 1 else viewers[0]
        return RerunViewer(
            app_id=self.cfg.viewer.app_id,
            spawn=bool(self.cfg.viewer.spawn),
            address=self.cfg.viewer.address,
            record_to_rrd=self.cfg.viewer.record_to_rrd,
            memory_limit=self.cfg.viewer.memory_limit,
            follow_body=self.cfg.viewer.follow_body,
            geom_view=self.cfg.viewer.geom_view,
        )

    @staticmethod
    def _viewer_is_closed(viewer: object | None) -> bool:
        """用 duck typing 识别 MuJoCo 窗口关闭状态，兼容 CompositeViewer。"""
        if viewer is None:
            return False
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            return any(Sim2SimWorkflow._viewer_is_closed(child) for child in children)
        is_closed = getattr(viewer, "is_closed", None)
        if callable(is_closed):
            return bool(is_closed())
        closed = getattr(viewer, "closed", None)
        if closed is not None:
            return bool(closed)
        return bool(getattr(viewer, "_closed", False))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _deploy_telemetry_initial_obs_check(self, obs: np.ndarray) -> dict[str, object]:
        """对比 reset obs 和选中的 deploy telemetry 行。"""
        reference = self.cfg.deploy_telemetry_reference_obs
        init_summary = self.cfg.deploy_telemetry_init
        if reference is None:
            return {"enabled": False}
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        got = np.asarray(obs, dtype=np.float64).reshape(-1)
        if ref.shape != got.shape:
            return {
                "enabled": True,
                "shape_mismatch": True,
                "reference_shape": list(ref.shape),
                "obs_shape": list(got.shape),
            }
        diff = got - ref
        slices = self.runtime.observation_slices
        by_term = {
            name: float(np.max(np.abs(diff[sl]))) if diff[sl].size else 0.0
            for name, sl in slices.items()
        }
        return {
            "enabled": True,
            "shape_mismatch": False,
            "max_abs_error": float(np.max(np.abs(diff))),
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "by_term_max_abs_error": by_term,
            "mode": "" if init_summary is None else str(init_summary.get("mode", "")),
            "selected_sample_index": (
                None if init_summary is None else init_summary.get("selected_sample_index")
            ),
            "selected_line_no": None
            if init_summary is None
            else init_summary.get("selected_line_no"),
        }

    def _policy_io_diagnostics(self, obs: np.ndarray) -> dict[str, object]:
        """记录初始 policy 输入的切片统计，便于定位 obs contract 错位。"""
        arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        terms: dict[str, object] = {}
        for name, sl in self.runtime.observation_slices.items():
            values = arr[sl]
            term: dict[str, object] = {
                "start": int(sl.start),
                "stop": int(sl.stop),
                "size": int(values.size),
                "min": float(np.min(values)) if values.size else 0.0,
                "max": float(np.max(values)) if values.size else 0.0,
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
            }
            if values.size <= 6:
                term["values"] = values.tolist()
            terms[name] = term
        return {
            "num_obs": int(arr.size),
            "expected_num_obs": int(self.runtime.policy.num_obs),
            "finite": bool(np.isfinite(arr).all()),
            "terms": terms,
        }

    @staticmethod
    def _fmt(values: object) -> str:
        if not isinstance(values, list):
            return str(values)
        return "[" + ",".join(f"{float(v):+.3f}" for v in values) + "]"

    @staticmethod
    def _rc_switch_diagnostics(
        samples: list[dict[str, float]],
        events: tuple[object, ...],
        initial_output_enabled: bool,
    ) -> dict[str, object]:
        """统计遥控器开关脚本对 policy 输出和 GRU reset 的影响。"""
        if not samples:
            return {
                "initial_output_enabled": bool(initial_output_enabled),
                "events": [],
                "samples": 0,
            }
        output_enabled = np.asarray([s["output_enabled"] for s in samples], dtype=np.float64)
        resets = np.asarray([s["rc_policy_reset"] for s in samples], dtype=np.float64)
        action_delta = np.asarray([s["action_delta_l2"] for s in samples], dtype=np.float64)
        switch_events = np.asarray([s["rc_switch_event"] for s in samples], dtype=np.float64)
        event_reports: list[dict[str, object]] = []
        for event in events:
            trigger = float(event.trigger_time_s)
            after = [s for s in samples if trigger <= float(s["time"]) <= trigger + 0.2]
            event_reports.append(
                {
                    "trigger_time_s": trigger,
                    "output_enabled": bool(event.output_enabled),
                    "samples_200ms": len(after),
                    "max_action_delta_l2_200ms": float(
                        max((s["action_delta_l2"] for s in after), default=0.0)
                    ),
                    "max_applied_action_delta_l2_200ms": float(
                        max((s["applied_action_delta_l2"] for s in after), default=0.0)
                    ),
                    "policy_reset_seen_200ms": bool(
                        any(float(s["rc_policy_reset"]) > 0.5 for s in after)
                    ),
                }
            )
        return {
            "initial_output_enabled": bool(initial_output_enabled),
            "events": event_reports,
            "samples": len(samples),
            "enabled_rate": float(np.mean(output_enabled > 0.5)),
            "switch_count": int(np.sum(switch_events > 0.5)),
            "policy_reset_count": int(np.sum(resets > 0.5)),
            "max_action_delta_l2": float(np.max(action_delta)),
        }

    @staticmethod
    def _jump_event_diagnostics(
        samples: list[dict[str, float]], events: tuple[object, ...]
    ) -> list[dict[str, object]]:
        """统计每次脚本跳跃触发后的局部表现。"""
        result: list[dict[str, object]] = []
        for event in events:
            start_s = float(event.trigger_time_s)
            end_s = start_s + 2.0
            window = [s for s in samples if start_s <= float(s["time"]) <= end_s]
            if not window:
                result.append(
                    {
                        "trigger_time_s": start_s,
                        "target_height": float(event.target_height),
                        "samples": 0,
                    }
                )
                continue
            height = np.asarray([s["height"] for s in window], dtype=np.float64)
            left = np.asarray([s["wheel_clearance_left"] for s in window], dtype=np.float64)
            right = np.asarray([s["wheel_clearance_right"] for s in window], dtype=np.float64)
            tilt = np.asarray([s["tilt_deg"] for s in window], dtype=np.float64)
            roll = np.asarray([s["roll_deg"] for s in window], dtype=np.float64)
            pitch = np.asarray([s["pitch_deg"] for s in window], dtype=np.float64)
            yaw = np.asarray([s["yaw_deg"] for s in window], dtype=np.float64)
            action_rate = np.asarray([s["action_delta_sq_sum"] for s in window], dtype=np.float64)
            applied_action_rate = np.asarray(
                [s["applied_action_delta_sq_sum"] for s in window], dtype=np.float64
            )
            phase_diagnostics = Sim2SimWorkflow._jump_event_phase_diagnostics(window, start_s)
            result.append(
                {
                    "trigger_time_s": start_s,
                    "target_height": float(event.target_height),
                    "samples": len(window),
                    "max_base_height": float(np.max(height)),
                    "max_wheel_clearance_left": float(np.max(left)),
                    "max_wheel_clearance_right": float(np.max(right)),
                    "max_wheel_clearance_abs_diff": float(np.max(np.abs(left - right))),
                    "max_tilt_deg": float(np.max(tilt)),
                    "max_abs_roll_deg": float(np.max(np.abs(roll))),
                    "max_abs_pitch_deg": float(np.max(np.abs(pitch))),
                    "max_abs_yaw_deg": float(np.max(np.abs(yaw))),
                    "mean_action_delta_sq_sum": float(np.mean(action_rate)),
                    "max_action_delta_sq_sum": float(np.max(action_rate)),
                    "mean_applied_action_delta_sq_sum": float(np.mean(applied_action_rate)),
                    "max_applied_action_delta_sq_sum": float(np.max(applied_action_rate)),
                    "phases": phase_diagnostics,
                }
            )
        return result

    @staticmethod
    def _walk_sweep_diagnostics(
        samples: list[dict[str, float]],
        velocities: tuple[float, ...],
    ) -> list[dict[str, object]]:
        """按 walk-sweep 的速度档位统计稳态跟踪、姿态和接触情况。"""
        result: list[dict[str, object]] = []
        for vx_cmd in velocities:
            window = [
                s
                for s in samples
                if abs(float(s.get("command_lin_vel_x", 0.0)) - float(vx_cmd)) < 1.0e-6
            ]
            if not window:
                result.append(
                    {
                        "command_lin_vel_x": float(vx_cmd),
                        "samples": 0,
                    }
                )
                continue
            steady = window[max(0, len(window) // 4) :]

            def values(key: str, steady_window: list[dict[str, float]] = steady) -> np.ndarray:
                return np.asarray([s.get(key, 0.0) for s in steady_window], dtype=np.float64)

            base_vx = values("base_lin_vel_x")
            wheel_lin = values("wheel_lin_vel")
            yaw_rate = values("yaw_rate_rad_s")
            tilt = values("tilt_deg")
            height = values("height")
            leg_contact = values("leg_contact")
            wheel_contact = values("wheel_contact")
            wheel_full_contact = values("wheel_full_contact")
            nonwheel_contact = values("nonwheel_contact")
            leg_clearance = values("leg_clearance")
            base_clearance = values("base_clearance")
            velocity_error = np.abs(base_vx - float(vx_cmd))
            wheel_velocity_error = np.abs(wheel_lin - float(vx_cmd))
            result.append(
                {
                    "command_lin_vel_x": float(vx_cmd),
                    "samples": len(window),
                    "steady_samples": len(steady),
                    "mean_base_lin_vel_x": float(np.mean(base_vx)),
                    "mean_abs_velocity_error": float(np.mean(velocity_error)),
                    "mean_wheel_lin_vel": float(np.mean(wheel_lin)),
                    "mean_abs_wheel_velocity_error": float(np.mean(wheel_velocity_error)),
                    "mean_abs_yaw_rate_rad_s": float(np.mean(np.abs(yaw_rate))),
                    "mean_tilt_deg": float(np.mean(tilt)),
                    "max_tilt_deg": float(np.max(tilt)),
                    "mean_height": float(np.mean(height)),
                    "wheel_contact_rate": float(np.mean(wheel_contact > 0.5)),
                    "wheel_full_contact_rate": float(np.mean(wheel_full_contact > 0.5)),
                    "leg_contact_rate": float(np.mean(leg_contact > 0.5)),
                    "nonwheel_contact_rate": float(np.mean(nonwheel_contact > 0.5)),
                    "min_leg_clearance": float(np.min(leg_clearance)),
                    "min_base_clearance": float(np.min(base_clearance)),
                }
            )
        return result

    @staticmethod
    def _upright_velocity_sweep_diagnostics(
        samples: list[dict[str, float]],
        commands: tuple[tuple[float, float], ...],
    ) -> list[dict[str, object]]:
        """按验收速度分组统计轮式平地运动是否被腿部接触套利。"""
        result: list[dict[str, object]] = []
        for vx_cmd, yaw_cmd in commands:
            window = [
                s
                for s in samples
                if abs(float(s.get("command_lin_vel_x", 0.0)) - float(vx_cmd)) < 1.0e-6
                and abs(float(s.get("command_yaw_rate", 0.0)) - float(yaw_cmd)) < 1.0e-6
            ]
            if not window:
                result.append(
                    {
                        "command_lin_vel_x": float(vx_cmd),
                        "command_yaw_rate": float(yaw_cmd),
                        "samples": 0,
                    }
                )
                continue
            steady = window[max(0, len(window) // 4) :]

            def values(key: str, steady_window: list[dict[str, float]] = steady) -> np.ndarray:
                return np.asarray([s.get(key, 0.0) for s in steady_window], dtype=np.float64)

            base_vx = values("base_lin_vel_x")
            yaw_rate = values("yaw_rate_rad_s")
            wheel_lin = values("wheel_lin_vel")
            leg_contact = values("leg_contact")
            wheel_contact = values("wheel_contact")
            wheel_full_contact = values("wheel_full_contact")
            nonwheel_contact = values("nonwheel_contact")
            leg_clearance = values("leg_clearance")
            velocity_error = np.abs(base_vx - float(vx_cmd))
            yaw_error = np.abs(yaw_rate - float(yaw_cmd))
            result.append(
                {
                    "command_lin_vel_x": float(vx_cmd),
                    "command_yaw_rate": float(yaw_cmd),
                    "samples": len(window),
                    "steady_samples": len(steady),
                    "mean_base_lin_vel_x": float(np.mean(base_vx)),
                    "mean_abs_velocity_error": float(np.mean(velocity_error)),
                    "mean_yaw_rate": float(np.mean(yaw_rate)),
                    "mean_abs_yaw_error": float(np.mean(yaw_error)),
                    "mean_wheel_lin_vel": float(np.mean(wheel_lin)),
                    "mean_abs_wheel_lin_vel": float(np.mean(np.abs(wheel_lin))),
                    "wheel_contact_rate": float(np.mean(wheel_contact > 0.5)),
                    "wheel_full_contact_rate": float(np.mean(wheel_full_contact > 0.5)),
                    "leg_contact_rate": float(np.mean(leg_contact > 0.5)),
                    "nonwheel_contact_rate": float(np.mean(nonwheel_contact > 0.5)),
                    "min_leg_clearance": float(np.min(leg_clearance)),
                }
            )
        return result

    @staticmethod
    def _jump_event_phase_diagnostics(
        window: list[dict[str, float]],
        trigger_time_s: float,
    ) -> dict[str, dict[str, float | int]]:
        """把一次跳跃按观测到的高度曲线切成四段，定位 pitch 来源。

        训练端的 jump_stage 来自参考轨迹，sim2sim 这里故意只用真实 rollout：
        - takeoff：触发后 0.35s，观察离地前后 pitch rate 是否被打出来
        - early_air：轮子首次明显离地后的 0.25s，观察离地瞬间姿态
        - apex：最高点附近 ±0.12s，观察空中顶点是否仍然前后点头
        - landing：最高点后首次回到接近起跳高度附近，观察落地姿态
        """
        if not window:
            return {}

        time = np.asarray([s["time"] for s in window], dtype=np.float64)
        height = np.asarray([s["height"] for s in window], dtype=np.float64)
        clearance = np.asarray([s["wheel_clearance"] for s in window], dtype=np.float64)

        base_height = float(height[0])
        apex_idx = int(np.argmax(height))
        apex_time = float(time[apex_idx])

        airborne_candidates = np.nonzero(clearance > 0.015)[0]
        airborne_idx = int(airborne_candidates[0]) if airborne_candidates.size > 0 else apex_idx
        airborne_time = float(time[airborne_idx])

        post_apex = np.arange(apex_idx, len(window))
        landing_candidates = post_apex[height[post_apex] <= base_height + 0.035]
        landing_idx = int(landing_candidates[0]) if landing_candidates.size > 0 else len(window) - 1
        landing_time = float(time[landing_idx])

        masks = {
            "takeoff": (time >= trigger_time_s) & (time <= trigger_time_s + 0.35),
            "early_air": (time >= airborne_time) & (time <= airborne_time + 0.25),
            "apex": (time >= apex_time - 0.12) & (time <= apex_time + 0.12),
            "landing": (time >= landing_time - 0.18) & (time <= landing_time + 0.18),
        }
        return {name: Sim2SimWorkflow._phase_stats(window, mask) for name, mask in masks.items()}

    @staticmethod
    def _phase_stats(
        window: list[dict[str, float]],
        mask: np.ndarray,
    ) -> dict[str, float | int]:
        """计算单个跳跃阶段的姿态、角速度和动作变化统计。"""
        if not bool(np.any(mask)):
            return {"samples": 0}

        def values(key: str) -> np.ndarray:
            return np.asarray([s[key] for s in window], dtype=np.float64)[mask]

        pitch = values("pitch_deg")
        roll = values("roll_deg")
        tilt = values("tilt_deg")
        height = values("height")
        wheel_clearance = values("wheel_clearance")
        pitch_rate = values("pitch_rate_rad_s")
        yaw_rate = values("yaw_rate_rad_s")
        action_rate = values("action_delta_sq_sum")
        return {
            "samples": int(np.count_nonzero(mask)),
            "max_base_height": float(np.max(height)),
            "max_wheel_clearance": float(np.max(wheel_clearance)),
            "mean_abs_pitch_deg": float(np.mean(np.abs(pitch))),
            "max_abs_pitch_deg": float(np.max(np.abs(pitch))),
            "mean_abs_roll_deg": float(np.mean(np.abs(roll))),
            "max_abs_roll_deg": float(np.max(np.abs(roll))),
            "mean_tilt_deg": float(np.mean(tilt)),
            "max_tilt_deg": float(np.max(tilt)),
            "mean_abs_pitch_rate_rad_s": float(np.mean(np.abs(pitch_rate))),
            "max_abs_pitch_rate_rad_s": float(np.max(np.abs(pitch_rate))),
            "mean_abs_yaw_rate_rad_s": float(np.mean(np.abs(yaw_rate))),
            "mean_action_delta_sq_sum": float(np.mean(action_rate)),
            "max_action_delta_sq_sum": float(np.max(action_rate)),
        }

    def _trajectory_steps_for_height(self, target_height: float) -> int:
        """按目标高度读取最近参考轨迹的真实帧数，避免 sim2sim 写死旧 125 帧。"""
        heights = np.asarray(DEFAULT_JUMP_TRAJ_HEIGHTS, dtype=np.float64)
        idx = int(np.argmin(np.abs(heights - float(target_height))))
        path = Path(DEFAULT_JUMP_TRAJ_PATHS[idx])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return int(self.cfg.robot.jump_state_machine.trajectory_steps)
        data = np.load(path)
        return int(data["base_pos"].shape[0])


def run_sim2sim(
    cfg: RunConfig,
    *,
    command_source: CommandInputSource | None = None,
) -> dict[str, object]:
    return Sim2SimWorkflow(cfg, command_source=command_source).run()
