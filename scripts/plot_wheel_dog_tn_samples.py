"""采样 WheelDog flat 策略的关节 T-N 工作点。

脚本会在 flat play 指令范围内构造速度网格，跑本地 checkpoint，并把
abad、hip、knee、wheel 四类关节的速度-力矩点叠到对应电机 T-N 包络上。

用法:
    uv run python scripts/plot_wheel_dog_tn_samples.py
    uv run python scripts/plot_wheel_dog_tn_samples.py --checkpoint logs/.../model_2300.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import se3_train  # noqa: F401  # 触发任务注册
from se3_train.tasks.wheel_dog.flat import TASK_ID

_COMMAND_NAME = "base_velocity"
_EXPERIMENT_NAME = "se3_wheel_dog_flat"

_JOINT_TYPES: dict[str, tuple[str, ...]] = {
    "abad": (
        "fl_abad_joint",
        "fr_abad_joint",
        "hl_abad_joint",
        "hr_abad_joint",
    ),
    "hip": (
        "fl_hip_joint",
        "fr_hip_joint",
        "hl_hip_joint",
        "hr_hip_joint",
    ),
    "knee": (
        "fl_knee_joint",
        "fr_knee_joint",
        "hl_knee_joint",
        "hr_knee_joint",
    ),
    "wheel": (
        "fl_wheel_joint",
        "fr_wheel_joint",
        "hl_wheel_joint",
        "hr_wheel_joint",
    ),
}

_LIMB_COLORS = {
    "fl": "#1f77b4",
    "fr": "#ff7f0e",
    "hl": "#2ca02c",
    "hr": "#d62728",
}


@dataclass(frozen=True)
class TnSpec:
    """当前任务 actuator 的 T-N 包络参数。"""

    name: str
    saturation_effort: float
    velocity_limit: float
    effort_limit: float


def _tn_envelope(spec: TnSpec, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算线性 T-N 包络上下界，与 MJLab DcMotorActuatorCfg 一致。"""
    saturation_effort = float(spec.saturation_effort)
    velocity_limit = float(spec.velocity_limit)
    effort_limit = float(spec.effort_limit)

    vel_at_effort_lim = velocity_limit * (1.0 + effort_limit / saturation_effort)
    vel_clipped = np.clip(velocity, -vel_at_effort_lim, vel_at_effort_lim)
    top = saturation_effort * (1.0 - vel_clipped / velocity_limit)
    bottom = saturation_effort * (-1.0 - vel_clipped / velocity_limit)
    return np.minimum(top, effort_limit), np.maximum(bottom, -effort_limit)


def _latest_checkpoint(repo_root: Path, experiment_name: str) -> Path:
    """按 run 修改时间和 checkpoint 数字后缀选择本地最新模型。"""
    root = repo_root / "logs" / "rsl_rl" / experiment_name
    runs = (
        [run for run in root.iterdir() if run.is_dir() and any(run.glob("model_*.pt"))]
        if root.exists()
        else []
    )
    if not runs:
        raise FileNotFoundError(f"未找到本地 checkpoint: {root}")
    latest_run = max(runs, key=lambda path: (path.stat().st_mtime, path.name))
    return max(latest_run.glob("model_*.pt"), key=_checkpoint_iteration).resolve()


def _checkpoint_iteration(path: Path) -> int:
    """解析 model_<iter>.pt 的迭代号。"""
    try:
        return int(path.stem.removeprefix("model_"))
    except ValueError:
        return -1


def _command_grid(
    vx_points: int,
    vy_points: int,
    vx_range: tuple[float, float],
    vy_range: tuple[float, float],
) -> np.ndarray:
    """构造覆盖 WheelDog flat play 范围的速度指令网格。"""
    vx = np.linspace(vx_range[0], vx_range[1], vx_points, dtype=np.float32)
    vy = np.linspace(vy_range[0], vy_range[1], vy_points, dtype=np.float32)
    return np.asarray([(x, y, 0.0) for x in vx for y in vy], dtype=np.float32)


def _tn_specs_from_env_cfg(env_cfg) -> dict[str, TnSpec]:
    """从 WheelDog 任务配置读取各类关节的实际 actuator T-N 参数。"""
    robot_cfg = env_cfg.scene.entities["robot"]
    joint_to_spec: dict[str, TnSpec] = {}
    for actuator in robot_cfg.articulation.actuators:
        spec = TnSpec(
            name=actuator.__class__.__name__,
            saturation_effort=float(actuator.saturation_effort),
            velocity_limit=float(actuator.velocity_limit),
            effort_limit=float(actuator.effort_limit),
        )
        for joint_name in actuator.target_names_expr:
            joint_to_spec[joint_name] = spec

    result: dict[str, TnSpec] = {}
    for joint_type, joint_names in _JOINT_TYPES.items():
        specs = {joint_to_spec[name] for name in joint_names}
        if len(specs) != 1:
            raise RuntimeError(f"{joint_type} 关节对应了多套 actuator 参数: {specs}")
        result[joint_type] = next(iter(specs))
    return result


def _set_fixed_commands(env: ManagerBasedRlEnv, commands: np.ndarray) -> None:
    """把每个 env 固定到一个速度指令，禁止指令项自动重采样。"""
    term = env.command_manager.get_term(_COMMAND_NAME)
    cmd = torch.as_tensor(commands, device=env.device, dtype=term.command.dtype)
    term._command[:] = cmd
    term._standing_mask[:] = torch.linalg.norm(cmd[:, :2], dim=1) < term.cfg.lin_vel_deadband
    term.time_left[:] = 1.0e9
    term._update_command()


def _resolve_indices(robot) -> tuple[dict[str, int], dict[str, int]]:
    """建立关节速度列和执行器力矩列的名称映射。"""
    joint_index = {name: idx for idx, name in enumerate(robot.joint_names)}
    actuator_index = {name: idx for idx, name in enumerate(robot.actuator_names)}
    missing = sorted(set(joint_index) - set(actuator_index))
    if missing:
        raise RuntimeError(f"以下关节没有对应 actuator_force 列: {missing}")
    return joint_index, actuator_index


def _collect_samples(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    """加载策略并采集所有速度网格下的关节速度-力矩点。"""
    repo_root = Path.cwd()
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else _latest_checkpoint(repo_root, _EXPERIMENT_NAME)
    )

    env_cfg = load_env_cfg(TASK_ID, play=True)
    cmd_cfg = env_cfg.commands[_COMMAND_NAME]
    vx_range = tuple(float(v) for v in cmd_cfg.lin_vel_x_range)
    vy_range = tuple(float(v) for v in cmd_cfg.lin_vel_y_range)
    commands = _command_grid(args.vx_points, args.vy_points, vx_range, vy_range)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg.scene.num_envs = int(commands.shape[0])
    env_cfg.terminations = {}

    agent_cfg = load_rl_cfg(TASK_ID)
    tn_specs = _tn_specs_from_env_cfg(env_cfg)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    _set_fixed_commands(env, commands)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    joint_index, actuator_index = _resolve_indices(robot)

    rows: list[dict[str, object]] = []
    try:
        with torch.inference_mode():
            total_steps = args.warmup_steps + args.record_steps
            for step in range(total_steps):
                actions = policy(obs)
                obs, _reward, dones, _extras = wrapped.step(actions)
                if torch.any(dones):
                    _set_fixed_commands(env, commands)
                    obs = wrapped.get_observations()
                if step < args.warmup_steps:
                    continue

                record_step = step - args.warmup_steps
                joint_vel = robot.data.joint_vel.detach().cpu().numpy()
                actuator_force = robot.data.actuator_force.detach().cpu().numpy()
                for env_idx, command in enumerate(commands):
                    for joint_type, joint_names in _JOINT_TYPES.items():
                        for joint_name in joint_names:
                            limb = joint_name.split("_", maxsplit=1)[0]
                            rows.append(
                                {
                                    "step": record_step,
                                    "env_idx": env_idx,
                                    "vx_cmd_mps": float(command[0]),
                                    "vy_cmd_mps": float(command[1]),
                                    "joint_type": joint_type,
                                    "limb": limb,
                                    "joint_name": joint_name,
                                    "velocity_rad_s": float(
                                        joint_vel[env_idx, joint_index[joint_name]]
                                    ),
                                    "torque_nm": float(
                                        actuator_force[env_idx, actuator_index[joint_name]]
                                    ),
                                }
                            )
                if args.progress and (record_step + 1) % args.progress == 0:
                    print(
                        f"[tn] recorded {record_step + 1}/{args.record_steps} control steps",
                        flush=True,
                    )
    finally:
        wrapped.close()

    meta = {
        "task_id": TASK_ID,
        "checkpoint": str(checkpoint),
        "device": device,
        "num_commands": int(commands.shape[0]),
        "vx_points": int(args.vx_points),
        "vy_points": int(args.vy_points),
        "warmup_steps": int(args.warmup_steps),
        "record_steps": int(args.record_steps),
        "control_dt_s": float(env_cfg.decimation * env_cfg.sim.mujoco.timestep),
        "vx_range_mps": list(vx_range),
        "vy_range_mps": list(vy_range),
        "joint_index_by_name": joint_index,
        "actuator_index_by_name": actuator_index,
        "tn_specs_by_joint_type": {
            joint_type: asdict(spec) for joint_type, spec in tn_specs.items()
        },
    }
    return rows, meta


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    """保存原始采样点。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, object]], meta: dict[str, object], output: Path) -> None:
    """保存每类关节的速度、力矩范围摘要。"""
    summary: dict[str, object] = {"meta": meta, "joint_types": {}}
    for joint_type in _JOINT_TYPES:
        vel = np.asarray(
            [float(row["velocity_rad_s"]) for row in rows if row["joint_type"] == joint_type],
            dtype=np.float64,
        )
        torque = np.asarray(
            [float(row["torque_nm"]) for row in rows if row["joint_type"] == joint_type],
            dtype=np.float64,
        )
        spec = TnSpec(**meta["tn_specs_by_joint_type"][joint_type])
        top, bottom = _tn_envelope(spec, vel)
        outside = np.logical_or(torque > top + 1.0e-6, torque < bottom - 1.0e-6)
        summary["joint_types"][joint_type] = {
            "count": len(vel),
            "velocity_min_rad_s": float(vel.min()),
            "velocity_max_rad_s": float(vel.max()),
            "torque_min_nm": float(torque.min()),
            "torque_max_nm": float(torque.max()),
            "tn_outside_ratio": float(outside.mean()) if len(outside) else 0.0,
            "saturation_effort_nm": float(spec.saturation_effort),
            "velocity_limit_rad_s": float(spec.velocity_limit),
            "effort_limit_nm": float(spec.effort_limit),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def _plot(rows: list[dict[str, object]], meta: dict[str, object], output: Path) -> None:
    """绘制四类关节的 T-N 工作点。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_by_type = dict(zip(_JOINT_TYPES, axes.flatten(), strict=True))

    for joint_type, ax in axes_by_type.items():
        spec = TnSpec(**meta["tn_specs_by_joint_type"][joint_type])
        vel_axis = np.linspace(-spec.velocity_limit * 1.35, spec.velocity_limit * 1.35, 500)
        top, bottom = _tn_envelope(spec, vel_axis)
        ax.fill_between(vel_axis, bottom, top, color="#dbeafe", alpha=0.7, label="T-N envelope")
        ax.plot(vel_axis, top, color="#2563eb", linewidth=1.4)
        ax.plot(vel_axis, bottom, color="#2563eb", linewidth=1.4)
        ax.axhline(spec.effort_limit, color="#16a34a", linestyle="--", linewidth=1.0)
        ax.axhline(-spec.effort_limit, color="#16a34a", linestyle="--", linewidth=1.0)
        ax.axvline(spec.velocity_limit, color="#f97316", linestyle=":", linewidth=1.0)
        ax.axvline(-spec.velocity_limit, color="#f97316", linestyle=":", linewidth=1.0)

        for limb, color in _LIMB_COLORS.items():
            selected = [
                row for row in rows if row["joint_type"] == joint_type and row["limb"] == limb
            ]
            if not selected:
                continue
            vel = np.asarray([float(row["velocity_rad_s"]) for row in selected])
            torque = np.asarray([float(row["torque_nm"]) for row in selected])
            ax.scatter(vel, torque, s=4, alpha=0.18, color=color, linewidths=0, label=limb.upper())

        ax.set_title(
            f"{joint_type.upper()} | sat={spec.saturation_effort:g}Nm, "
            f"eff={spec.effort_limit:g}Nm, vel={spec.velocity_limit:g}rad/s"
        )
        ax.set_xlabel("joint velocity (rad/s)")
        ax.set_ylabel("torque (N·m)")
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="#6b7280", linewidth=0.6)
        ax.axvline(0, color="#6b7280", linewidth=0.6)

    handles, labels = axes_by_type["abad"].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "WheelDog flat policy joint T-N samples\n"
        f"{Path(str(meta['checkpoint'])).name} | "
        f"{meta['num_commands']} command cases | "
        f"{meta['record_steps']} recorded control steps",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="采样 WheelDog 关节 T-N 工作点")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="checkpoint 路径，默认取最新本地模型"
    )
    parser.add_argument("--device", default=None, help="运行设备，默认自动选择 cuda:0 或 cpu")
    parser.add_argument("--vx-points", type=int, default=9, help="vx 网格点数量")
    parser.add_argument("--vy-points", type=int, default=5, help="vy 网格点数量")
    parser.add_argument("--warmup-steps", type=int, default=40, help="记录前预热控制步数")
    parser.add_argument("--record-steps", type=int, default=180, help="实际记录控制步数")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("scripts/wheel_dog_tn_samples.png"),
        help="输出图片路径",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/wheel_dog_tn_samples.csv"),
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("scripts/wheel_dog_tn_samples_summary.json"),
        help="输出摘要 JSON 路径",
    )
    parser.add_argument(
        "--progress", type=int, default=30, help="每隔多少记录步打印一次进度，0 表示关闭"
    )
    args = parser.parse_args()
    if args.vx_points <= 0 or args.vy_points <= 0:
        parser.error("--vx-points 和 --vy-points 必须为正整数")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps 不能为负数")
    if args.record_steps <= 0:
        parser.error("--record-steps 必须为正整数")
    if args.progress < 0:
        parser.error("--progress 不能为负数")

    rows, meta = _collect_samples(args)
    _write_csv(rows, args.output_csv)
    _write_summary(rows, meta, args.output_summary)
    _plot(rows, meta, args.output_png)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved summary: {args.output_summary}")
    print(f"Saved plot: {args.output_png}")


if __name__ == "__main__":
    main()
