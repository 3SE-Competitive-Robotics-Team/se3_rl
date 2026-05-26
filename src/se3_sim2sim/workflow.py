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
from .policy import PolicyRuntime
from .rerun_viewer import RerunViewer
from .robot import WheelLeggedRobot
from .runtime_spec import RuntimeSpec


class Sim2SimWorkflow:
    def __init__(self, cfg: RunConfig) -> None:
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
        self.viewer = self._make_viewer()
        control_dt = self.cfg.robot.sim_dt * self.cfg.robot.control_decimation
        self._course = create_course(self.cfg.course, control_dt)

    def run(self) -> dict[str, object]:
        obs = self.robot.reset(fixed=self.cfg.fixed_reset, randomize_root=self.cfg.randomize_root)
        samples: list[dict[str, float]] = []
        model_diag = self.robot.diagnostics()
        if self.viewer is not None:
            self.viewer.log_model(self.robot.model)
            self.viewer.log_state(
                self.robot.model, self.robot.data, step=0, telemetry=self.robot.telemetry()
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

        try:
            for step in step_iter:
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
                    sim_time_s = (step - 1) * control_dt
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
                action = self.policy.act(obs)
                obs, reward, done, info = self.robot.step(action)
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
                    "wheel_clearance": float(info.get("wheel_clearance", 0.0)),
                    "wheel_clearance_left": float(info.get("wheel_clearance_left", 0.0)),
                    "wheel_clearance_right": float(info.get("wheel_clearance_right", 0.0)),
                    "tilt_deg": float(info["tilt_deg"]),
                    "roll_deg": float(info.get("roll_deg", 0.0)),
                    "pitch_deg": float(info.get("pitch_deg", 0.0)),
                    "yaw_deg": float(info.get("yaw_deg", 0.0)),
                    "roll_rate_rad_s": float(info["base_ang_vel_body"][0]),
                    "pitch_rate_rad_s": float(info["base_ang_vel_body"][1]),
                    "yaw_rate_rad_s": float(info["base_ang_vel_body"][2]),
                    "reward": float(reward),
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
                }
                samples.append(sample)
                if self.viewer is not None and step % max(1, int(self.cfg.viewer.log_every)) == 0:
                    self.viewer.log_state(
                        self.robot.model, self.robot.data, step=step, telemetry=info
                    )
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
                    line = (
                        f"step={step:05d} time={float(info['time']):.3f}"
                        f"{course_info}"
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
            "rollout": rollout_diagnostics(samples),
            "done_reason": done_reason,
        }
        if script_events:
            summary["jump_events"] = self._jump_event_diagnostics(samples, script_events)
        if self.cfg.json_output is not None:
            self._write_json(self.cfg.json_output, summary)
        if self.viewer is not None:
            self.viewer.close()
        return summary

    def _make_viewer(self) -> RerunViewer | None:
        if self.cfg.viewer.mode == "none":
            return None
        return RerunViewer(
            app_id=self.cfg.viewer.app_id,
            spawn=bool(self.cfg.viewer.spawn),
            address=self.cfg.viewer.address,
            record_to_rrd=self.cfg.viewer.record_to_rrd,
            memory_limit=self.cfg.viewer.memory_limit,
            follow_body=self.cfg.viewer.follow_body,
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _fmt(values: object) -> str:
        if not isinstance(values, list):
            return str(values)
        return "[" + ",".join(f"{float(v):+.3f}" for v in values) + "]"

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


def run_sim2sim(cfg: RunConfig) -> dict[str, object]:
    return Sim2SimWorkflow(cfg).run()
