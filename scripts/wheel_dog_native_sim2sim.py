"""WheelDog flat checkpoint 的最小 native MuJoCo sim2sim 验证。

用途：
    uv run python scripts/wheel_dog_native_sim2sim.py \
      --checkpoint logs/rsl_rl/se3_wheel_dog_flat/2026-06-12_11-33-55/model_4999.pt

说明：
    现有 `se3-sim2sim` 是 SerialLeg 6DOF 运行契约，不能加载 WheelDog 53D/16D
    checkpoint。本脚本只实现 WheelDog flat 的最小闭环：native MuJoCo 动力学、
    训练同款 53D actor 观测、16D 动作、PD 控制和 T-N 力矩裁剪。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.runtime_spec import PolicyArchitectureSpec, RuntimeSpec
from se3_train.tasks.wheel_dog.robot_cfg import (
    DOG_BASE_HEIGHT,
    DOG_DEFAULT_JOINT_POS,
    DOG_JOINT_NAMES,
    DOG_LEG_JOINT_NAMES,
    DOG_WHEEL_JOINT_NAMES,
)

_MODEL_PATH = Path("assets/robots/minidog/mjcf/minidog_16dof_20kg.xml")
_GRAVITY_W = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
_ACTION_CLIP = 100.0
_DEFAULT_COMMANDS = "0,0,0;1,0,0;2,0,0;4,0,0;-4,0,0;0,1.5,0;0,-1.5,0"


def _wheel_dog_runtime() -> RuntimeSpec:
    """构造 WheelDog 53D/16D checkpoint 的最小运行契约。"""
    policy = replace(
        PolicyArchitectureSpec(),
        num_obs=53,
        num_actions=16,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
    )
    return RuntimeSpec(
        task="wheel_dog_flat",
        spec_name="se3/wheel_dog_flat",
        policy=policy,
        joint_names=DOG_JOINT_NAMES,
        actuator_names=DOG_JOINT_NAMES,
    )


def _parse_commands(raw: str) -> list[tuple[float, float, float]]:
    """解析 `vx,vy,yaw;...` 形式的速度指令列表。"""
    commands: list[tuple[float, float, float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(v.strip()) for v in item.split(",")]
        if len(parts) == 2:
            parts.append(0.0)
        if len(parts) != 3:
            raise ValueError(f"速度指令必须是 vx,vy 或 vx,vy,yaw: {item!r}")
        commands.append((parts[0], parts[1], parts[2]))
    if not commands:
        raise ValueError("至少需要一个速度指令")
    return commands


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    """wxyz 四元数转 body 到 world 旋转矩阵。"""
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _tn_clip(
    effort: float,
    velocity: float,
    *,
    saturation_effort: float,
    velocity_limit: float,
    effort_limit: float,
) -> float:
    """与 MJLab DcMotorActuatorCfg 一致的线性 T-N 力矩裁剪。"""
    vel_at_effort_lim = velocity_limit * (1.0 + effort_limit / saturation_effort)
    vel = float(np.clip(velocity, -vel_at_effort_lim, vel_at_effort_lim))
    top = saturation_effort * (1.0 - vel / velocity_limit)
    bottom = saturation_effort * (-1.0 - vel / velocity_limit)
    max_effort = min(top, effort_limit)
    min_effort = max(bottom, -effort_limit)
    return float(np.clip(effort, min_effort, max_effort))


class WheelDogNativeSim:
    """WheelDog flat checkpoint 的 native MuJoCo 单机 rollout。"""

    def __init__(self, *, checkpoint: Path, model_path: Path, device: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.policy = PolicyRuntime(
            checkpoint=checkpoint,
            device=device,
            runtime=_wheel_dog_runtime(),
        )
        self.base_id = self._body_id("base_link")
        self.joint_qpos = np.asarray(
            [self.model.jnt_qposadr[self._joint_id(name)] for name in DOG_JOINT_NAMES],
            dtype=np.int64,
        )
        self.joint_qvel = np.asarray(
            [self.model.jnt_dofadr[self._joint_id(name)] for name in DOG_JOINT_NAMES],
            dtype=np.int64,
        )
        self.actuator_by_joint = self._actuator_by_joint()
        self.default_joint_pos = np.asarray(DOG_DEFAULT_JOINT_POS, dtype=np.float64)
        self.last_action = np.zeros(16, dtype=np.float32)

    def reset(self) -> None:
        """重置到 WheelDog flat 默认站姿。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (0.0, 0.0, DOG_BASE_HEIGHT)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.joint_qpos] = self.default_joint_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.last_action[:] = 0.0
        self.policy.reset()
        mujoco.mj_forward(self.model, self.data)

    def run_command(
        self,
        command: tuple[float, float, float],
        *,
        max_steps: int,
        print_every: int,
    ) -> dict[str, float | int | list[float]]:
        """对单个固定速度指令做 rollout 并返回摘要。"""
        self.reset()
        samples: list[dict[str, float]] = []
        for step in range(max_steps):
            obs = self._observation(command)
            action = self.policy.act(obs)
            action = np.clip(action, -_ACTION_CLIP, _ACTION_CLIP).astype(np.float32)
            self._apply_action(action)
            for _ in range(4):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.last_action[:] = action
            telemetry = self._telemetry(command)
            samples.append(telemetry)
            if print_every > 0 and (step + 1) % print_every == 0:
                print(
                    f"[wheel-dog-sim2sim] step={step + 1:04d} "
                    f"cmd=({command[0]:+.2f},{command[1]:+.2f}) "
                    f"vx={telemetry['vx_body']:+.2f} vy={telemetry['vy_body']:+.2f} "
                    f"h={telemetry['height']:.3f} tilt={telemetry['tilt_deg']:.1f}"
                )
        return self._summary(command, samples)

    def _observation(self, command: tuple[float, float, float]) -> np.ndarray:
        """构造 WheelDog flat 53D actor 观测。"""
        vel6 = self._base_velocity(local=True)
        base_ang_vel = vel6[:3] * 0.25
        projected_gravity = _quat_to_matrix_wxyz(self.data.qpos[3:7]).T @ _GRAVITY_W
        command_obs = np.asarray(command, dtype=np.float64) * np.asarray((2.0, 2.0, 0.25))
        joint_pos = self.data.qpos[self.joint_qpos]
        joint_vel = self.data.qvel[self.joint_qvel]
        leg_indices = [DOG_JOINT_NAMES.index(name) for name in DOG_LEG_JOINT_NAMES]
        leg_joint_pos = joint_pos[leg_indices] - self.default_joint_pos[leg_indices]
        obs = np.concatenate(
            (
                base_ang_vel,
                projected_gravity,
                command_obs,
                leg_joint_pos,
                joint_vel * 0.05,
                self.last_action,
            )
        )
        return obs.astype(np.float32, copy=False)

    def _apply_action(self, action: np.ndarray) -> None:
        """把 16D action 转成 native MuJoCo actuator ctrl。"""
        self.data.ctrl[:] = 0.0
        joint_pos = self.data.qpos[self.joint_qpos]
        joint_vel = self.data.qvel[self.joint_qvel]
        joint_index = {name: idx for idx, name in enumerate(DOG_JOINT_NAMES)}

        for action_idx, joint_name in enumerate(DOG_LEG_JOINT_NAMES):
            idx = joint_index[joint_name]
            scale = 0.125 if "abad" in joint_name else 0.25
            target = self.default_joint_pos[idx] + scale * float(action[action_idx])
            torque = 80.0 * (target - joint_pos[idx]) - 2.0 * joint_vel[idx]
            torque = _tn_clip(
                torque,
                float(joint_vel[idx]),
                saturation_effort=50.0,
                velocity_limit=22.4,
                effort_limit=50.0,
            )
            self.data.ctrl[self.actuator_by_joint[joint_name]] = torque

        wheel_offset = len(DOG_LEG_JOINT_NAMES)
        for wheel_idx, joint_name in enumerate(DOG_WHEEL_JOINT_NAMES):
            idx = joint_index[joint_name]
            target_vel = 5.0 * float(action[wheel_offset + wheel_idx])
            torque = 0.6 * (target_vel - joint_vel[idx])
            torque = _tn_clip(
                torque,
                float(joint_vel[idx]),
                saturation_effort=12.0,
                velocity_limit=79.3,
                effort_limit=12.0,
            )
            self.data.ctrl[self.actuator_by_joint[joint_name]] = torque

    def _telemetry(self, command: tuple[float, float, float]) -> dict[str, float]:
        """读取当前 rollout 状态。"""
        vel6 = self._base_velocity(local=True)
        projected_gravity = _quat_to_matrix_wxyz(self.data.qpos[3:7]).T @ _GRAVITY_W
        tilt = math.degrees(math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0))))
        return {
            "height": float(self.data.qpos[2]),
            "tilt_deg": float(tilt),
            "vx_body": float(vel6[3]),
            "vy_body": float(vel6[4]),
            "vx_error": float(vel6[3] - command[0]),
            "vy_error": float(vel6[4] - command[1]),
        }

    def _summary(
        self,
        command: tuple[float, float, float],
        samples: list[dict[str, float]],
    ) -> dict[str, float | int | list[float]]:
        """汇总单个速度指令 rollout。"""
        tail = samples[max(0, len(samples) // 2) :]
        final = samples[-1]

        def mean(name: str) -> float:
            return float(np.mean([sample[name] for sample in tail]))

        def max_abs(name: str) -> float:
            return float(np.max(np.abs([sample[name] for sample in samples])))

        return {
            "command": [float(v) for v in command],
            "steps": len(samples),
            "final_height": float(final["height"]),
            "final_tilt_deg": float(final["tilt_deg"]),
            "mean_vx_body": mean("vx_body"),
            "mean_vy_body": mean("vy_body"),
            "mean_abs_vx_error": float(np.mean(np.abs([s["vx_error"] for s in tail]))),
            "mean_abs_vy_error": float(np.mean(np.abs([s["vy_error"] for s in tail]))),
            "max_tilt_deg": max_abs("tilt_deg"),
            "min_height": float(np.min([sample["height"] for sample in samples])),
        }

    def _base_velocity(self, *, local: bool) -> np.ndarray:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_id,
            vel,
            int(local),
        )
        return vel

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"找不到 body: {name}")
        return int(body_id)

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"找不到 joint: {name}")
        return int(joint_id)

    def _actuator_by_joint(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for act_id in range(self.model.nu):
            joint_id = int(self.model.actuator(act_id).trnid[0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is not None:
                result[joint_name] = int(act_id)
        missing = sorted(set(DOG_JOINT_NAMES) - set(result))
        if missing:
            raise ValueError(f"以下关节没有 actuator: {missing}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="WheelDog native MuJoCo sim2sim")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--commands",
        default=_DEFAULT_COMMANDS,
        help="速度指令列表，格式为 'vx,vy,yaw;vx,vy,yaw'。",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    sim = WheelDogNativeSim(
        checkpoint=args.checkpoint.resolve(),
        model_path=args.model.resolve(),
        device=str(args.device),
    )
    summaries = [
        sim.run_command(command, max_steps=int(args.max_steps), print_every=int(args.print_every))
        for command in _parse_commands(args.commands)
    ]
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        "summaries": summaries,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")


if __name__ == "__main__":
    main()
