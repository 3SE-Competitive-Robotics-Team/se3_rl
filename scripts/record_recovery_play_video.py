"""录制 Recovery/Loco 策略的 Play 视频，并同步导出动作与接触轨迹。

同一个 env 实例跑多个场景：每个场景改 reset 姿态与指令后 reset，逐步渲染帧并
记录 6D 动作和轮/腿/机身接触。产出每场景一个 MP4、动作 NPY、接触 NPZ 与一份
JSON（含一阶 Δa、二阶 jerk、接触时序和峰值力统计）。

headless 渲染需要 ``MUJOCO_GL=egl``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import se3_train  # noqa: F401
from se3_train.mdp.rewards import _contact_diagnostic_stats

DEFAULT_TASK_NAME = "SE3-WheelLegged-Recovery-Loco-Grouped-MLP"
POSE_WEIGHTS: dict[str, tuple[float, float, float, float, float]] = {
    "standing": (1.0, 0.0, 0.0, 0.0, 0.0),
    "left_side": (0.0, 1.0, 0.0, 0.0, 0.0),
    "right_side": (0.0, 0.0, 1.0, 0.0, 0.0),
    "prone": (0.0, 0.0, 0.0, 1.0, 0.0),
    "supine": (0.0, 0.0, 0.0, 0.0, 1.0),
}
JOINT_NAMES = ("LF", "LB", "RF", "RB", "L_wheel", "R_wheel")


# 评测必须复现 checkpoint 训练时的 action 零点语义。该 flag 由 CLI 显式指定，
# 因为 ONNX metadata 目前不记录它（见 backlog B18）。None = 沿用任务注册的默认值。
_ACTION_DEFAULT_OVERRIDE: bool | None = None


def _apply_action_default_override(cfg) -> None:
    if _ACTION_DEFAULT_OVERRIDE is None:
        return
    term = cfg.actions["delayed_action"]
    term.height_conditioned_action_default = bool(_ACTION_DEFAULT_OVERRIDE)
    print(
        f"[cfg] height_conditioned_action_default = {term.height_conditioned_action_default}",
        flush=True,
    )


@dataclass(frozen=True)
class Scene:
    name: str
    pose: str = "standing"
    vx: float = 0.0
    yaw: float = 0.0
    height: float = 0.26
    seconds: float = 5.0
    standing_cmd: bool = False


def _default_scenes() -> list[Scene]:
    return [
        Scene("prone_standup", pose="prone", seconds=4.0, standing_cmd=True),
        Scene("supine_standup", pose="supine", seconds=4.0, standing_cmd=True),
        Scene("left_side_standup", pose="left_side", seconds=4.0, standing_cmd=True),
        Scene("zero_hold", pose="standing", seconds=8.0, standing_cmd=True),
        Scene("vx_1.5", pose="standing", vx=1.5, seconds=6.0),
        Scene("yaw_2.5", pose="standing", yaw=2.5, seconds=6.0),
        Scene("height_0.39", pose="standing", height=0.39, seconds=6.0, standing_cmd=True),
        Scene("height_0.195", pose="standing", height=0.195, seconds=6.0, standing_cmd=True),
    ]


def _build_env_cfg(task_name: str, num_envs: int, width: int, height: int):
    cfg = load_env_cfg(task_name, play=True)
    _apply_action_default_override(cfg)
    cfg.scene.num_envs = int(num_envs)
    cfg.auto_reset = False
    cfg.episode_length_s = 1.0e6
    cfg.viewer.width = int(width)
    cfg.viewer.height = int(height)

    root_params = cfg.events["reset_root_state"].params
    root_params.pop("pose_weights_by_group", None)
    root_params.pop("source_curriculum_stages_by_group", None)
    root_params.update(
        {
            "pos_xy_range": (0.0, 0.0),
            "height_offset_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
            "roll_jitter_range": (0.0, 0.0),
            "pitch_jitter_range": (0.0, 0.0),
            "lin_vel_range": (0.0, 0.0),
            "ang_vel_range": (0.0, 0.0),
            "clearance_range": (0.003, 0.003),
            "pose_weights": POSE_WEIGHTS["standing"],
            "source_curriculum_stages": [],
            "standard_curriculum_stages": [],
            "use_iterations": False,
            "offset_iter": 0,
        }
    )
    joint_params = cfg.events["reset_joints"].params
    joint_params.update(
        {
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "align_root_height_to_wheels": True,
            "curriculum_stages": [],
            "use_iterations": False,
            "offset_iter": 0,
        }
    )
    joint_params.pop("randomization_group_names", None)
    if "wheel_joint_vel_range" in joint_params:
        joint_params["wheel_joint_vel_range"] = (0.0, 0.0)
    return cfg


def _term_cfg(manager, name: str):
    """兼容 mjlab 各 manager 的取 cfg 方式：EventManager 用 get_term_cfg，
    CommandManager 用 get_term(...).cfg。"""
    getter = getattr(manager, "get_term_cfg", None)
    if callable(getter):
        return getter(name)
    getter = getattr(manager, "get_term", None)
    if callable(getter):
        term = getter(name)
        return getattr(term, "cfg", term)
    terms = getattr(manager, "_terms", None)
    if isinstance(terms, dict) and name in terms:
        term = terms[name]
        return getattr(term, "cfg", term)
    raise RuntimeError(f"无法取出 term cfg: {name}")


def _apply_scene(base_env: ManagerBasedRlEnv, scene: Scene) -> None:
    reset_cfg = _term_cfg(base_env.event_manager, "reset_root_state")
    reset_cfg.params["pose_weights"] = POSE_WEIGHTS[scene.pose]

    cmd = _term_cfg(base_env.command_manager, "velocity_height")
    cmd.lin_vel_x_range = (scene.vx, scene.vx)
    cmd.ang_vel_yaw_range = (scene.yaw, scene.yaw)
    cmd.height_range = (scene.height, scene.height)
    cmd.standing_height_range = (scene.height, scene.height)
    cmd.standing_ratio = 1.0 if scene.standing_cmd else 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)


def _jitter_stats(actions: np.ndarray, dt: float) -> dict:
    """actions: (T, 6) 确定性动作序列。返回一阶/二阶抖动的分关节统计。"""
    d1 = np.diff(actions, axis=0)
    d2 = np.diff(actions, n=2, axis=0)
    out = {
        "steps": int(actions.shape[0]),
        "dt": dt,
        "action_abs_mean": actions.__abs__().mean(axis=0).tolist(),
        "action_abs_max": actions.__abs__().max(axis=0).tolist(),
        "delta_rms": np.sqrt((d1**2).mean(axis=0)).tolist(),
        "delta_abs_max": np.abs(d1).max(axis=0).tolist(),
        "jerk_rms": np.sqrt((d2**2).mean(axis=0)).tolist(),
        "jerk_abs_max": np.abs(d2).max(axis=0).tolist(),
        "sum_delta_sq_mean": float((d1**2).sum(axis=1).mean()),
        "joint_names": list(JOINT_NAMES),
    }
    # 物理量纲下的变化率（每秒），便于与执行器带宽对比
    out["delta_rms_per_s"] = (np.array(out["delta_rms"]) / dt).tolist()
    return out


def _contact_snapshot(base_env: ManagerBasedRlEnv, env_index: int = 0) -> dict[str, float]:
    """读取一个 env 的三路接触率与单步峰值力。"""
    wheel_ratio, wheel_max_force, _ = _contact_diagnostic_stats(base_env, "wheel_sensor", 1.0)
    leg_ratio, leg_max_force, _ = _contact_diagnostic_stats(base_env, "leg_contact_sensor", 1.0)
    base_ratio, base_max_force, _ = _contact_diagnostic_stats(base_env, "collision_sensor", 1.0)
    return {
        "wheel_contact_ratio": float(wheel_ratio[env_index].item()),
        "leg_contact_ratio": float(leg_ratio[env_index].item()),
        "base_contact_ratio": float(base_ratio[env_index].item()),
        "wheel_max_force_n": float(wheel_max_force[env_index].item()),
        "leg_max_force_n": float(leg_max_force[env_index].item()),
        "base_max_force_n": float(base_max_force[env_index].item()),
    }


def _contact_timing_stats(trace: dict[str, np.ndarray], dt: float) -> dict[str, float | int | None]:
    """从逐步接触轨迹派生首触地、前一秒占比和全程峰值力。"""
    steps = int(trace["wheel_contact_ratio"].shape[0])
    first_second_steps = min(steps, max(1, math.ceil(1.0 / dt)))
    wheel_full_contact = trace["wheel_contact_ratio"] >= 0.999
    first_contact_indices = np.flatnonzero(wheel_full_contact)
    first_contact_step = int(first_contact_indices[0]) if first_contact_indices.size > 0 else None

    def first_second_fraction(key: str, threshold: float) -> float:
        values = trace[key][:first_second_steps]
        return float((values > threshold).mean()) if values.size else 0.0

    def peak(key: str) -> float:
        values = trace[key]
        return float(values.max()) if values.size else 0.0

    return {
        "wheel_first_contact_step": first_contact_step,
        "wheel_first_contact_time_s": (
            float((first_contact_step + 1) * dt) if first_contact_step is not None else None
        ),
        "wheel_contact_frac_first_1s": first_second_fraction("wheel_contact_ratio", 0.999 - 1.0e-9),
        "leg_contact_frac_first_1s": first_second_fraction("leg_contact_ratio", 0.0),
        "base_contact_frac_first_1s": first_second_fraction("base_contact_ratio", 0.0),
        "max_wheel_force_n": peak("wheel_max_force_n"),
        "max_leg_force_n": peak("leg_max_force_n"),
        "max_base_force_n": peak("base_max_force_n"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK_NAME)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scenes", default=None, help="逗号分隔场景名，缺省全部")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--height-conditioned-action-default",
        choices=("true", "false"),
        default=None,
        help="覆盖 action 零点语义，必须与 checkpoint 训练时一致；缺省沿用任务默认。",
    )
    args = parser.parse_args()
    global _ACTION_DEFAULT_OVERRIDE
    if args.height_conditioned_action_default is not None:
        _ACTION_DEFAULT_OVERRIDE = args.height_conditioned_action_default == "true"

    configure_torch_backends()
    import imageio.v2 as imageio

    scenes = _default_scenes()
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = [s for s in scenes if s.name in want]
        if not scenes:
            raise SystemExit(f"没有匹配的场景: {sorted(want)}")

    env_cfg = _build_env_cfg(args.task, args.num_envs, args.width, args.height)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base_env)

    agent_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device
    )
    policy = runner.get_inference_policy(device=args.device)
    env.reset()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_dt = float(base_env.step_dt)
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    summary: dict[str, dict] = {}

    for scene in scenes:
        _apply_scene(base_env, scene)
        base_env.reset(env_ids=env_ids)
        reset_fn = getattr(policy, "reset", None)
        if reset_fn is not None:
            reset_fn()

        n_steps = max(1, math.ceil(scene.seconds / step_dt))
        frames: list[np.ndarray] = []
        act_log: list[np.ndarray] = []
        contact_log: dict[str, list[float]] = {
            name: []
            for name in (
                "wheel_contact_ratio",
                "leg_contact_ratio",
                "base_contact_ratio",
                "wheel_max_force_n",
                "leg_max_force_n",
                "base_max_force_n",
            )
        }
        for _ in range(n_steps):
            with torch.no_grad():
                obs = env.get_observations()
                actions = policy(obs)
                env.step(actions)
            act_log.append(actions[0].detach().float().cpu().numpy())
            contact = _contact_snapshot(base_env)
            for name, value in contact.items():
                contact_log[name].append(value)
            frame = base_env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        mp4 = args.out_dir / f"{scene.name}.mp4"
        if frames:
            imageio.mimwrite(mp4, frames, fps=args.fps, quality=8, macro_block_size=1)
        acts = np.stack(act_log)
        contact_trace = {
            name: np.asarray(values, dtype=np.float32) for name, values in contact_log.items()
        }
        stats = _jitter_stats(acts, step_dt)
        stats.update(_contact_timing_stats(contact_trace, step_dt))
        stats["scene"] = asdict(scene)
        stats["frames"] = len(frames)
        stats["contact_trace_env_index"] = 0
        summary[scene.name] = stats
        np.save(args.out_dir / f"{scene.name}_actions.npy", acts)
        contact_path = args.out_dir / f"{scene.name}_contacts.npz"
        np.savez_compressed(contact_path, **contact_trace)
        stats["contact_trace_file"] = contact_path.name
        print(
            f"[done] {scene.name}: {len(frames)} frames -> {mp4.name}; "
            f"sum_delta_sq_mean={stats['sum_delta_sq_mean']:.3e}; "
            f"wheel_first_contact_step={stats['wheel_first_contact_step']}",
            flush=True,
        )

    env.close()
    (args.out_dir / "jitter_stats.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "scenes": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir / 'jitter_stats.json'}")


if __name__ == "__main__":
    main()
