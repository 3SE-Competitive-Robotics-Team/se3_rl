"""不依赖旧脚本路径的完整 sim2sim workflow。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from .config import RunConfig
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
        try:
            for step in step_iter:
                if self.cfg.robot.yaw_pid.enabled:
                    self.robot.update_yaw_command()
                    obs = self.robot.observation()
                action = self.policy.act(obs)
                obs, reward, done, info = self.robot.step(action)
                sample = {
                    "step": float(step),
                    "time": float(info["time"]),
                    "height": float(info["height"]),
                    "tilt_deg": float(info["tilt_deg"]),
                    "reward": float(reward),
                    "action_delay_steps": float(info["action_delay_steps"]),
                    "action_delay_s": float(info["action_delay_s"]),
                }
                samples.append(sample)
                if self.viewer is not None and step % max(1, int(self.cfg.viewer.log_every)) == 0:
                    self.viewer.log_state(
                        self.robot.model, self.robot.data, step=step, telemetry=info
                    )
                if int(self.cfg.print_every) > 0 and step % int(self.cfg.print_every) == 0:
                    line = (
                        f"step={step:05d} time={float(info['time']):.3f} "
                        f"height={float(info['height']):.3f} tilt={float(info['tilt_deg']):.2f} "
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
                "spec": self.policy.spec.to_dict(),
            },
            "model_diagnostics": model_diag,
            "rollout": rollout_diagnostics(samples),
            "done_reason": done_reason,
        }
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


def run_sim2sim(cfg: RunConfig) -> dict[str, object]:
    return Sim2SimWorkflow(cfg).run()
