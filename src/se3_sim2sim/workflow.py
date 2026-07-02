"""不依赖旧脚本路径的完整 sim2sim workflow。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from se3_shared import front_action_periods_from_scales, periodic_policy_action_delta_np
from se3_train.mdp.jump_trajectories import DEFAULT_JUMP_TRAJ_HEIGHTS, DEFAULT_JUMP_TRAJ_PATHS

from .config import RECOVERY_COMMAND_HEIGHT_M, RECOVERY_POSE_RP_RAD, RunConfig
from .course import CourseType, create_course
from .diagnostics import rollout_diagnostics
from .mujoco_viewer import CompositeViewer, MujocoViewer
from .policy import PolicyRuntime
from .rerun_viewer import RerunViewer
from .robot import WheelLeggedRobot
from .runtime_spec import RuntimeSpec
from .teleop_input import CommandInputSource
from .viser_viewer import ViserViewer


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
        if self.command_source is None and hasattr(self.viewer, "poll"):
            self.command_source = self.viewer  # Viser GUI sliders drive sim2sim commands.
        self._viewer_default_initial_roll_rad = float(self.cfg.robot.initial_roll_rad)
        self._viewer_default_initial_pitch_rad = float(self.cfg.robot.initial_pitch_rad)
        self._viewer_default_initial_yaw_rad = float(self.cfg.robot.initial_yaw_rad)
        self._viewer_default_initial_base_height = self.cfg.robot.initial_base_height
        self._viewer_default_command = tuple(float(value) for value in self.cfg.robot.command)
        self._viewer_default_yaw_pid_enabled = bool(self.cfg.robot.yaw_pid.enabled)
        self.startup_contract = self._startup_contract()

    def _startup_contract(self) -> dict[str, object]:
        contract = self.robot.startup_contract()
        contract.update(
            {
                "checkpoint": str(self.policy.checkpoint_path),
                "policy_iteration": self.policy.iteration,
                "policy_type": self.policy.policy_type,
                "num_obs": int(self.policy.spec.num_obs),
                "num_actions": int(self.policy.spec.num_actions),
                "viewer_mode": str(self.cfg.viewer.mode),
                "fixed_reset": bool(self.cfg.fixed_reset),
                "randomize_root": bool(self.cfg.randomize_root),
                "initial_roll_rad": float(self.cfg.robot.initial_roll_rad),
                "initial_pitch_rad": float(self.cfg.robot.initial_pitch_rad),
                "initial_yaw_rad": float(self.cfg.robot.initial_yaw_rad),
                "initial_base_height": self.cfg.robot.initial_base_height,
            }
        )
        return contract

    def _print_startup_contract(self) -> None:
        contract = self.startup_contract
        rough = contract.get("rough_terrain")
        rough_text = ""
        if isinstance(rough, dict) and rough.get("enabled"):
            rough_text = (
                f" rough={rough.get('selected_type')}@L{rough.get('level')}"
                f"/col{rough.get('column')}"
            )
        print(
            "[sim2sim contract] "
            f"model={contract['model_path']} "
            f"policy={contract['policy_type']}@{contract['policy_iteration']} "
            f"scale={contract['action_scale']} clip={contract['action_clip']} "
            f"height_default={contract['height_conditioned_action_default']} "
            f"active_rod={contract['active_rod_action_semantics']}"
            f"{rough_text}"
        )

    def run(self) -> dict[str, object]:
        self._print_startup_contract()
        obs = self.robot.reset(fixed=self.cfg.fixed_reset, randomize_root=self.cfg.randomize_root)
        deploy_init_obs_check = self._deploy_telemetry_initial_obs_check(obs)
        if deploy_init_obs_check.get("enabled"):
            line = (
                "[deploy telemetry init] "
                f"mode={deploy_init_obs_check.get('mode', '')} "
                f"sample={deploy_init_obs_check.get('selected_sample_index', '')}"
            )
            if deploy_init_obs_check.get("shape_mismatch"):
                line += (
                    f" shape_mismatch reference={deploy_init_obs_check.get('reference_shape')} "
                    f"obs={deploy_init_obs_check.get('obs_shape')}"
                )
            else:
                line += f" max_obs_err={float(deploy_init_obs_check.get('max_abs_error', 0.0)):.3e}"
            print(line)
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
        max_steps = int(self.cfg.max_steps)
        samples: list[dict[str, float]] = []
        collect_samples = not (max_steps == 0 and self.cfg.viewer.mode == "viser")
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
        stair_hold_active = False

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
            episode_step_offset = 0
            viewer_step_offset = 0
            for step in step_iter:
                if self._viewer_is_closed(self.viewer):
                    done_reason = "viewer_closed"
                    break
                checkpoint_request = self._consume_viewer_checkpoint_request(self.viewer)
                if checkpoint_request is not None:
                    try:
                        next_policy = PolicyRuntime(
                            checkpoint=checkpoint_request,
                            device=self.cfg.policy.device,
                            runtime=self.runtime,
                        )
                    except Exception as exc:
                        self._notify_viewer_checkpoint_failed(
                            self.viewer,
                            checkpoint_request,
                            str(exc),
                        )
                    else:
                        self.policy = next_policy
                        self.cfg.policy.checkpoint = next_policy.checkpoint_path
                        self._notify_viewer_checkpoint_loaded(
                            self.viewer,
                            next_policy.checkpoint_path,
                            next_policy.iteration,
                        )
                        print(
                            "[viser] loaded checkpoint "
                            f"{next_policy.checkpoint_path.name} iter={next_policy.iteration}"
                        )
                        episode_step_offset = step
                        viewer_step_offset = step
                        self._apply_viewer_reset_mode("normal")
                        obs = self.robot.reset(
                            fixed=self.cfg.fixed_reset,
                            randomize_root=self.cfg.randomize_root,
                        )
                        self.robot.command[:] = np.asarray(
                            self.cfg.robot.command,
                            dtype=np.float64,
                        )
                        self.policy.reset()
                        self.robot.reset_policy_io_state()
                        obs = self.robot.observation()
                        _next_rc_event = 0
                        _rc_output_enabled = bool(rc_sched.initial_output_enabled)
                        _policy_memory_clean = True
                        _interval_steps_remaining = (
                            max(1, round(sched.interval_s / control_dt)) if sched.enabled else 0
                        )
                        _next_script_event = 0
                        _traj_step = 0
                        _traj_steps = self._trajectory_steps_for_height(
                            float(self.robot.command[6])
                        )
                        _jump_active = self.robot.command[5] > 0.5
                        _prev_action = None
                        _prev_applied_action = None
                        self._notify_viewer_reset(self.viewer, "checkpoint_switch")
                        if self.viewer is not None:
                            reset_telemetry = self.robot.telemetry(reward=0.0)
                            reset_telemetry["rc_switch_r"] = 1.0 if _rc_output_enabled else 0.0
                            reset_telemetry["output_enabled"] = 1.0 if _rc_output_enabled else 0.0
                            reset_telemetry["rc_switch_event"] = 0.0
                            reset_telemetry["rc_policy_reset"] = 1.0
                            reset_telemetry["rc_off_mode"] = str(rc_sched.off_mode)
                            reset_telemetry["target_mode"] = "checkpoint_switch"
                            self.viewer.log_state(
                                self.robot.model,
                                self.robot.data,
                                step=0,
                                telemetry=reset_telemetry,
                            )
                        continue
                reset_mode = self._consume_viewer_reset_request(self.viewer)
                if reset_mode is not None:
                    episode_step_offset = step
                    viewer_step_offset = step
                    applied_reset_mode = self._apply_viewer_reset_mode(reset_mode)
                    obs = self.robot.reset(
                        fixed=self.cfg.fixed_reset,
                        randomize_root=self.cfg.randomize_root,
                    )
                    self.robot.command[:] = np.asarray(self.cfg.robot.command, dtype=np.float64)
                    self.policy.reset()
                    self.robot.reset_policy_io_state()
                    obs = self.robot.observation()
                    _next_rc_event = 0
                    _rc_output_enabled = bool(rc_sched.initial_output_enabled)
                    _policy_memory_clean = True
                    _interval_steps_remaining = (
                        max(1, round(sched.interval_s / control_dt)) if sched.enabled else 0
                    )
                    _next_script_event = 0
                    _traj_step = 0
                    _traj_steps = self._trajectory_steps_for_height(float(self.robot.command[6]))
                    _jump_active = self.robot.command[5] > 0.5
                    _prev_action = None
                    _prev_applied_action = None
                    self._notify_viewer_reset(self.viewer, applied_reset_mode)
                    if self.viewer is not None:
                        reset_telemetry = self.robot.telemetry(reward=0.0)
                        reset_telemetry["rc_switch_r"] = 1.0 if _rc_output_enabled else 0.0
                        reset_telemetry["output_enabled"] = 1.0 if _rc_output_enabled else 0.0
                        reset_telemetry["rc_switch_event"] = 0.0
                        reset_telemetry["rc_policy_reset"] = 1.0
                        reset_telemetry["rc_off_mode"] = str(rc_sched.off_mode)
                        reset_telemetry["target_mode"] = applied_reset_mode
                        self.viewer.log_state(
                            self.robot.model,
                            self.robot.data,
                            step=0,
                            telemetry=reset_telemetry,
                        )
                    continue
                terrain_level_request = self._consume_viewer_terrain_level_request(self.viewer)
                if terrain_level_request is not None:
                    try:
                        rough_info = self.robot.set_rough_terrain_level(terrain_level_request)
                    except Exception as exc:
                        self._notify_viewer_terrain_level_failed(
                            self.viewer,
                            terrain_level_request,
                            str(exc),
                        )
                        print(
                            "[viser] terrain level switch failed "
                            f"L{int(terrain_level_request)}: {exc}"
                        )
                    else:
                        episode_step_offset = step
                        viewer_step_offset = step
                        obs = self.robot.reset(
                            fixed=self.cfg.fixed_reset,
                            randomize_root=self.cfg.randomize_root,
                        )
                        self.robot.command[:] = np.asarray(
                            self.cfg.robot.command,
                            dtype=np.float64,
                        )
                        self.policy.reset()
                        self.robot.reset_policy_io_state()
                        obs = self.robot.observation()
                        _next_rc_event = 0
                        _rc_output_enabled = bool(rc_sched.initial_output_enabled)
                        _policy_memory_clean = True
                        _interval_steps_remaining = (
                            max(1, round(sched.interval_s / control_dt)) if sched.enabled else 0
                        )
                        _next_script_event = 0
                        _traj_step = 0
                        _traj_steps = self._trajectory_steps_for_height(
                            float(self.robot.command[6])
                        )
                        _jump_active = self.robot.command[5] > 0.5
                        _prev_action = None
                        _prev_applied_action = None
                        self._notify_viewer_terrain_level_changed(self.viewer, rough_info)
                        mode = f"rough_terrain_level:{int(rough_info.get('level', 0))}"
                        self._notify_viewer_reset(self.viewer, mode)
                        print(
                            "[viser] switched rough terrain "
                            f"level={rough_info.get('level')} "
                            f"type={rough_info.get('selected_type')}"
                        )
                        if self.viewer is not None:
                            reset_telemetry = self.robot.telemetry(reward=0.0)
                            reset_telemetry["rc_switch_r"] = 1.0 if _rc_output_enabled else 0.0
                            reset_telemetry["output_enabled"] = 1.0 if _rc_output_enabled else 0.0
                            reset_telemetry["rc_switch_event"] = 0.0
                            reset_telemetry["rc_policy_reset"] = 1.0
                            reset_telemetry["rc_off_mode"] = str(rc_sched.off_mode)
                            reset_telemetry["target_mode"] = mode
                            self.viewer.log_state(
                                self.robot.model,
                                self.robot.data,
                                step=0,
                                telemetry=reset_telemetry,
                            )
                    continue
                episode_step = max(0, step - 1 - episode_step_offset)
                sim_time_s = episode_step * control_dt
                push_delta = self._consume_viewer_push_request(self.viewer)
                if push_delta is not None:
                    self.robot.apply_root_velocity_delta(np.asarray(push_delta, dtype=np.float64))
                    obs = self.robot.observation()
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
                    if self.cfg.robot.yaw_pid.enabled:
                        if command_update.yaw_target_rad is not None:
                            self.robot.yaw_pid.set_target_yaw(float(command_update.yaw_target_rad))
                        self.robot.update_yaw_command()
                    else:
                        self.robot.command[1] = float(command_update.yaw_rate)
                    self.robot.command[4] = float(command_update.command_height)
                    if command_update.toggle_output:
                        _rc_output_enabled = not _rc_output_enabled
                        rc_switch_event = 1.0
                    obs = self.robot.observation()
                if stair_hold_active:
                    self.robot.command[0] = float(self.cfg.stair_hold_vx)
                    obs = self.robot.observation()

                gru_hidden_reset = 0.0
                gru_hidden_norm_before = None
                gru_hidden_norm_after = None
                if self._consume_viewer_gru_hidden_reset_request(self.viewer):
                    hidden_norm = getattr(self.policy, "hidden_state_norm", None)
                    if callable(hidden_norm):
                        gru_hidden_norm_before = hidden_norm()
                    self.policy.reset()
                    if callable(hidden_norm):
                        gru_hidden_norm_after = hidden_norm()
                    _policy_memory_clean = True
                    gru_hidden_reset = 1.0
                    self._notify_viewer_gru_hidden_reset(
                        self.viewer,
                        step=max(0, step - viewer_step_offset),
                        before_norm=gru_hidden_norm_before,
                        after_norm=gru_hidden_norm_after,
                    )
                    before_text = (
                        "-"
                        if gru_hidden_norm_before is None
                        else f"{float(gru_hidden_norm_before):.6f}"
                    )
                    after_text = (
                        "-"
                        if gru_hidden_norm_after is None
                        else f"{float(gru_hidden_norm_after):.6f}"
                    )
                    print(
                        "[viser] reset GRU hidden "
                        f"step={max(0, step - viewer_step_offset)} "
                        f"norm={before_text}->{after_text}"
                    )
                    obs = self.robot.observation()

                policy_iteration = self._policy_iteration_int()
                self.robot.update_stair_ctbc(iteration=policy_iteration)
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
                info["gru_hidden_reset"] = gru_hidden_reset
                info["gru_hidden_norm_before_reset"] = (
                    0.0 if gru_hidden_norm_before is None else float(gru_hidden_norm_before)
                )
                info["gru_hidden_norm_after_reset"] = (
                    0.0 if gru_hidden_norm_after is None else float(gru_hidden_norm_after)
                )
                info["rc_off_mode"] = str(rc_sched.off_mode)
                action_now = np.asarray(info["last_action"], dtype=np.float64)
                applied_action_now = np.asarray(info["applied_action"], dtype=np.float64)
                front_action_period = front_action_periods_from_scales(self.robot.action_scale)
                action_delta = (
                    np.zeros_like(action_now)
                    if _prev_action is None
                    else periodic_policy_action_delta_np(
                        action_now,
                        _prev_action,
                        front_action_period=front_action_period,
                    )
                )
                applied_action_delta = (
                    np.zeros_like(applied_action_now)
                    if _prev_applied_action is None
                    else periodic_policy_action_delta_np(
                        applied_action_now,
                        _prev_applied_action,
                        front_action_period=front_action_period,
                    )
                )
                _prev_action = action_now
                _prev_applied_action = applied_action_now
                sample = {
                    "step": float(step),
                    "time": float(info["time"]),
                    "height": float(info["height"]),
                    "base_x": float(info.get("base_x", 0.0)),
                    "base_y": float(info.get("base_y", 0.0)),
                    "reset_floor_lift_m": float(info.get("reset_floor_lift_m", 0.0)),
                    "wheel_x_left": float(info.get("wheel_x_left", 0.0)),
                    "wheel_x_right": float(info.get("wheel_x_right", 0.0)),
                    "wheel_z_left": float(info.get("wheel_z_left", 0.0)),
                    "wheel_z_right": float(info.get("wheel_z_right", 0.0)),
                    "wheel_bottom_z_left": float(info.get("wheel_bottom_z_left", 0.0)),
                    "wheel_bottom_z_right": float(info.get("wheel_bottom_z_right", 0.0)),
                    "stair_step_height_m": float(info.get("stair_step_height_m", 0.0)),
                    "stair_step_depth_m": float(info.get("stair_step_depth_m", 0.0)),
                    "stair_start_x_m": float(info.get("stair_start_x_m", 0.0)),
                    "stair_step_count": float(info.get("stair_step_count", 0.0)),
                    "stair_half_width_m": float(info.get("stair_half_width_m", 0.0)),
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
                    "wheel_stair_contact_left": float(info.get("wheel_stair_contact_left", 0.0)),
                    "wheel_stair_contact_right": float(info.get("wheel_stair_contact_right", 0.0)),
                    "wheel_floor_contact_left": float(info.get("wheel_floor_contact_left", 0.0)),
                    "wheel_floor_contact_right": float(info.get("wheel_floor_contact_right", 0.0)),
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
                    "ctbc_trigger": float(info.get("ctbc_trigger", 0.0)),
                    "ctbc_left_active": float(info.get("ctbc_left_active", 0.0)),
                    "ctbc_right_active": float(info.get("ctbc_right_active", 0.0)),
                    "ctbc_phase_left": float(info.get("ctbc_phase_left", 0.0)),
                    "ctbc_phase_right": float(info.get("ctbc_phase_right", 0.0)),
                    "ctbc_contact_left": float(info.get("ctbc_contact_left", 0.0)),
                    "ctbc_contact_right": float(info.get("ctbc_contact_right", 0.0)),
                    "ctbc_complete_ff_cycles": float(info.get("ctbc_complete_ff_cycles", 0.0)),
                    "action_delay_steps": float(info["action_delay_steps"]),
                    "action_delay_s": float(info["action_delay_s"]),
                    "rc_switch_r": float(info.get("rc_switch_r", 1.0)),
                    "output_enabled": float(info.get("output_enabled", 1.0)),
                    "rc_switch_event": float(info.get("rc_switch_event", 0.0)),
                    "rc_policy_reset": float(info.get("rc_policy_reset", 0.0)),
                    "gru_hidden_reset": float(info.get("gru_hidden_reset", 0.0)),
                    "gru_hidden_norm_before_reset": float(
                        info.get("gru_hidden_norm_before_reset", 0.0)
                    ),
                    "gru_hidden_norm_after_reset": float(
                        info.get("gru_hidden_norm_after_reset", 0.0)
                    ),
                    "rc_off_mode_no_torque": 1.0
                    if str(info.get("rc_off_mode", "hold-current")) == "no-torque"
                    else 0.0,
                    "stair_hold_active": 1.0 if stair_hold_active else 0.0,
                    "closed_chain_reset_position_residual_m": float(
                        info.get("closed_chain_reset_position_residual_m", 0.0)
                    ),
                    "closed_chain_reset_velocity_residual": float(
                        info.get("closed_chain_reset_velocity_residual", 0.0)
                    ),
                }
                if collect_samples:
                    samples.append(sample)
                if self.cfg.stair_hold_on_support and not stair_hold_active:
                    support_step = self._stair_support_step(info)
                    if support_step >= int(self.cfg.stair_hold_min_step):
                        stair_hold_active = True
                if self.viewer is not None and step % max(1, int(self.cfg.viewer.log_every)) == 0:
                    viewer_step = max(0, step - viewer_step_offset)
                    self.viewer.log_state(
                        self.robot.model, self.robot.data, step=viewer_step, telemetry=info
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
            "startup_contract": self.startup_contract,
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

    def _apply_viewer_reset_mode(self, mode: str | None) -> str:
        reset_mode = "normal" if mode is None else str(mode)
        if reset_mode in RECOVERY_POSE_RP_RAD and reset_mode != "standing":
            roll, pitch = RECOVERY_POSE_RP_RAD[reset_mode]
            self.cfg.robot.initial_roll_rad = float(roll)
            self.cfg.robot.initial_pitch_rad = float(pitch)
            self.cfg.robot.initial_yaw_rad = self._viewer_default_initial_yaw_rad
            self.cfg.robot.initial_base_height = 0.16
            self.cfg.robot.command = (
                0.0,
                0.0,
                0.0,
                0.0,
                float(RECOVERY_COMMAND_HEIGHT_M),
                0.0,
                0.0,
                0.0,
            )
            self.cfg.robot.yaw_pid.enabled = False
            return f"recovery_reset:{reset_mode}"

        self.cfg.robot.initial_roll_rad = self._viewer_default_initial_roll_rad
        self.cfg.robot.initial_pitch_rad = self._viewer_default_initial_pitch_rad
        self.cfg.robot.initial_yaw_rad = self._viewer_default_initial_yaw_rad
        self.cfg.robot.initial_base_height = self._viewer_default_initial_base_height
        self.cfg.robot.command = self._viewer_default_command
        self.cfg.robot.yaw_pid.enabled = self._viewer_default_yaw_pid_enabled
        return "normal"

    def _policy_iteration_int(self) -> int | None:
        """返回当前 checkpoint 的迭代数，无法解析时交给 CTBC 使用默认值。"""

        try:
            return int(self.policy.iteration)
        except (TypeError, ValueError):
            return None

    def _make_viewer(self) -> RerunViewer | MujocoViewer | ViserViewer | CompositeViewer | None:
        if self.cfg.viewer.mode == "none":
            return None
        if self.cfg.viewer.mode == "viser":
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
                ViserViewer(
                    model=self.robot.model,
                    control_dt=self.cfg.robot.sim_dt * self.cfg.robot.control_decimation,
                    port=self.cfg.viewer.port,
                    geom_view=self.cfg.viewer.geom_view,
                    checkpoint_path=self.policy.checkpoint_path,
                    policy_iteration=self.policy.iteration,
                    initial_command=self.cfg.robot.command,
                    yaw_pid_enabled=self.cfg.robot.yaw_pid.enabled,
                    initial_yaw_target_rad=self.cfg.robot.yaw_pid.target_yaw_rad,
                    rough_terrain_info=self.robot.rough_terrain_info,
                )
            )
            return CompositeViewer(viewers) if len(viewers) > 1 else viewers[0]
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
    def _consume_viewer_reset_request(viewer: object | None) -> str | None:
        """从 viewer 或组合 viewer 中消费 reset 请求。"""
        if viewer is None:
            return None
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                requested = Sim2SimWorkflow._consume_viewer_reset_request(child)
                if requested is not None:
                    return requested
            return None
        consume_typed = getattr(viewer, "consume_reset_request", None)
        if callable(consume_typed):
            requested = consume_typed()
            return None if requested is None else str(requested)
        consume = getattr(viewer, "consume_reset_requested", None)
        if callable(consume):
            return "normal" if bool(consume()) else None
        return None

    @staticmethod
    def _consume_viewer_reset_requested(viewer: object | None) -> bool:
        """从 viewer 或组合 viewer 中消费 reset 请求。"""
        if viewer is None:
            return False
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            return any(Sim2SimWorkflow._consume_viewer_reset_requested(child) for child in children)
        consume = getattr(viewer, "consume_reset_requested", None)
        if callable(consume):
            return bool(consume())
        return False

    @staticmethod
    def _consume_viewer_checkpoint_request(viewer: object | None) -> Path | None:
        """从 viewer 或组合 viewer 中消费 checkpoint 切换请求。"""
        if viewer is None:
            return None
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                requested = Sim2SimWorkflow._consume_viewer_checkpoint_request(child)
                if requested is not None:
                    return requested
            return None
        consume = getattr(viewer, "consume_checkpoint_request", None)
        if callable(consume):
            requested = consume()
            return None if requested is None else Path(requested)
        return None

    @staticmethod
    def _consume_viewer_push_request(
        viewer: object | None,
    ) -> tuple[float, float, float, float, float, float] | None:
        """从 viewer 或组合 viewer 中消费一次性 root velocity 扰动。"""
        if viewer is None:
            return None
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                requested = Sim2SimWorkflow._consume_viewer_push_request(child)
                if requested is not None:
                    return requested
            return None
        consume = getattr(viewer, "consume_push_request", None)
        if callable(consume):
            requested = consume()
            if requested is None:
                return None
            delta = tuple(float(value) for value in requested)
            if len(delta) != 6:
                return None
            return delta
        return None

    @staticmethod
    def _consume_viewer_gru_hidden_reset_request(viewer: object | None) -> bool:
        """从 viewer 或组合 viewer 中消费一次 GRU hidden reset 请求。"""
        if viewer is None:
            return False
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            return any(
                Sim2SimWorkflow._consume_viewer_gru_hidden_reset_request(child)
                for child in children
            )
        consume = getattr(viewer, "consume_gru_hidden_reset_request", None)
        if callable(consume):
            return bool(consume())
        return False

    @staticmethod
    def _consume_viewer_terrain_level_request(viewer: object | None) -> int | None:
        """从 viewer 或组合 viewer 中消费 rough terrain level 切换请求。"""
        if viewer is None:
            return None
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                requested = Sim2SimWorkflow._consume_viewer_terrain_level_request(child)
                if requested is not None:
                    return requested
            return None
        consume = getattr(viewer, "consume_terrain_level_request", None)
        if callable(consume):
            requested = consume()
            return None if requested is None else int(requested)
        return None

    @staticmethod
    def _notify_viewer_reset(viewer: object | None, mode: str = "normal") -> None:
        """通知 viewer 主线程已经完成 reset。"""
        if viewer is None:
            return
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                Sim2SimWorkflow._notify_viewer_reset(child, mode)
            return
        notify = getattr(viewer, "notify_reset", None)
        if callable(notify):
            notify(mode)

    @staticmethod
    def _notify_viewer_gru_hidden_reset(
        viewer: object | None,
        *,
        step: int,
        before_norm: float | None,
        after_norm: float | None,
    ) -> None:
        """通知 viewer 主线程已经完成 GRU hidden reset。"""
        if viewer is None:
            return
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                Sim2SimWorkflow._notify_viewer_gru_hidden_reset(
                    child,
                    step=step,
                    before_norm=before_norm,
                    after_norm=after_norm,
                )
            return
        notify = getattr(viewer, "notify_gru_hidden_reset", None)
        if callable(notify):
            notify(step=step, before_norm=before_norm, after_norm=after_norm)

    @staticmethod
    def _notify_viewer_terrain_level_changed(
        viewer: object | None,
        info: dict[str, object],
    ) -> None:
        """通知 viewer rough terrain level 已切换。"""
        if viewer is None:
            return
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                Sim2SimWorkflow._notify_viewer_terrain_level_changed(child, info)
            return
        notify = getattr(viewer, "notify_terrain_level_changed", None)
        if callable(notify):
            notify(info)

    @staticmethod
    def _notify_viewer_terrain_level_failed(
        viewer: object | None,
        level: int,
        message: str,
    ) -> None:
        """通知 viewer rough terrain level 切换失败。"""
        if viewer is None:
            return
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                Sim2SimWorkflow._notify_viewer_terrain_level_failed(child, level, message)
            return
        notify = getattr(viewer, "notify_terrain_level_failed", None)
        if callable(notify):
            notify(level, message)

    @staticmethod
    def _notify_viewer_checkpoint_loaded(
        viewer: object | None,
        checkpoint_path: Path,
        policy_iteration: object,
    ) -> None:
        """通知 viewer 主线程已经完成 checkpoint 切换。"""
        if viewer is None:
            return
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                Sim2SimWorkflow._notify_viewer_checkpoint_loaded(
                    child,
                    checkpoint_path,
                    policy_iteration,
                )
            return
        notify = getattr(viewer, "notify_checkpoint_loaded", None)
        if callable(notify):
            notify(checkpoint_path, policy_iteration)

    @staticmethod
    def _notify_viewer_checkpoint_failed(
        viewer: object | None,
        checkpoint_path: Path,
        message: str,
    ) -> None:
        """通知 viewer 主线程 checkpoint 切换失败。"""
        if viewer is None:
            return
        children = getattr(viewer, "_viewers", None)
        if children is not None:
            for child in children:
                Sim2SimWorkflow._notify_viewer_checkpoint_failed(
                    child,
                    checkpoint_path,
                    message,
                )
            return
        notify = getattr(viewer, "notify_checkpoint_failed", None)
        if callable(notify):
            notify(checkpoint_path, message)

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
    def _stair_support_step(info: dict[str, object]) -> int:
        """返回当前左右轮共同支撑的台阶级数，未共同支撑时返回 0。"""
        step_height = float(info.get("stair_step_height_m", 0.0))
        step_depth = float(info.get("stair_step_depth_m", 0.0))
        start_x = float(info.get("stair_start_x_m", 0.0))
        step_count = round(float(info.get("stair_step_count", 0.0)))
        if step_height <= 0.0 or step_depth <= 0.0 or step_count <= 0:
            return 0
        if float(info.get("wheel_stair_contact_left", 0.0)) <= 0.5:
            return 0
        if float(info.get("wheel_stair_contact_right", 0.0)) <= 0.5:
            return 0

        left_step = int(
            np.clip(
                np.floor((float(info.get("wheel_x_left", 0.0)) - start_x) / step_depth) + 1,
                0,
                step_count,
            )
        )
        right_step = int(
            np.clip(
                np.floor((float(info.get("wheel_x_right", 0.0)) - start_x) / step_depth) + 1,
                0,
                step_count,
            )
        )
        if left_step <= 0 or left_step != right_step:
            return 0
        z_tol = max(0.035, 0.35 * step_height)
        left_err = abs(float(info.get("wheel_bottom_z_left", 0.0)) - left_step * step_height)
        right_err = abs(float(info.get("wheel_bottom_z_right", 0.0)) - right_step * step_height)
        if left_err > z_tol or right_err > z_tol:
            return 0
        return left_step

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
