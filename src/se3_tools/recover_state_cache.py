"""起身轨迹状态库生成器：recover 组反向课程的出生状态（2026-11-01）。

用一个认可的起身策略（默认 mzxd7dmq Recover-Catch-Impact02 model_2500）在训练同款 recover 组环境里闭环回放：
倒地出生列全体当作 recover 组，按类别（默认侧躺 + 倒置）从倒地状态库出生，学生路径（CTS 角色组置 0，即部署路径）、
确定性动作。只收本局第一次就站稳（recover 组判据：倾角 < 15° 且离出生地面 ≥ 0.17 m 连续 stable_hold_s）的轨迹，
记下从出生到"开始站起"（站稳前那段连续直立的起点）之间每 `stride` 个控制步的完整状态，并标注离开始站起还剩的时间
`remaining_s`。反向课程按 remaining_s 由近到远放开出生状态（见 tasks/rough/recover.py 的 reset_from_recover_trajectories）。

格式与 `se3_tools.fall_state_cache` 相同（同名字段语义一致：root_pos 的 z 是离出生地面高度、xy 相对地块原点，
速度在世界系），可直接被训练端的 _FallStateCache 读取；另加 remaining_s / elapsed_s / spawn_class / traj_id。
状态只含物理量，不含动作历史、动作延迟与执行器滞后的内部状态——这些在 reset 时按常规清零。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "assets" / "recovery_states"
FORMAT_VERSION = 1
_DEFAULT_TASK = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-"
    "Recover-Catch-Impact02"
)


def _git_commit() -> str:
    """当前 checkout 的提交号；读不到时返回 unknown。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(x: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in np.quantile(x, (0.1, 0.5, 0.9, 1.0))]


def generate(args: argparse.Namespace) -> Path:
    os.environ.setdefault("WANDB_MODE", "disabled")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import se3_train  # noqa: F401
    from se3_tools.fall_state_cache import TILT_CLASS_NAMES, _tilt_class
    from se3_train.robot_cfg import _MJCF_PATH
    from se3_train.tasks.rough import events as ev
    from se3_train.tasks.rough import recover as rec
    from se3_train.tasks.rough.columns import column_mask

    device = args.device
    classes = tuple(int(c) for c in args.classes)
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = int(args.num_envs)
    params = env_cfg.events["reset_from_fall_states"].params
    params["class_weights"] = tuple(float(i in classes) for i in range(len(TILT_CLASS_NAMES)))
    if getattr(env_cfg.actions["delayed_action"], "teacher", None) is not None:
        env_cfg.actions["delayed_action"].teacher = None
    agent_cfg = load_rl_cfg(args.task)
    agent_cfg.logger = "tensorboard"
    raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    group = column_mask(raw, tuple(params["terrain_type_names"]))
    if group is None:
        raise RuntimeError("找不到倒地出生列")
    setattr(raw, rec.RECOVER_GROUP_ATTR, group.clone())
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    robot = raw.scene["robot"]
    names = tuple(str(name) for name in robot.joint_names)
    n, steps, dt = int(args.num_envs), int(args.steps), float(raw.step_dt)
    recover_params = env_cfg.terminations["recover_failed"].params
    hold = round(float(recover_params["stable_hold_s"]) / dt)
    if steps * dt >= float(recover_params["episode_s"]):
        raise ValueError("--steps 须短于 recover 组一局，否则站稳的 env 会在记录窗口里重置")

    joints = len(names)
    width = 3 + 4 + 3 + 3 + 2 * joints
    trajectories: list[dict[str, np.ndarray]] = []
    attempts = dict.fromkeys(classes, 0)
    successes = dict.fromkeys(classes, 0)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="se3-recover-cache-") as tmp:
        runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
            env, asdict(agent_cfg), tmp, device
        )
        runner.load(str(args.ckpt), map_location=device)
        policy = runner.get_inference_policy(device)
        role = getattr(policy, "role_group", None)
        torch.manual_seed(int(args.seed))
        for round_id in range(int(args.rounds)):
            env.reset()
            obs = env.get_observations()
            fall = getattr(raw, ev.FALL_SPAWN_MASK_ATTR).clone()
            spawn_class = getattr(raw, ev._FALL_CLASS_ATTR).clone()
            tracked = fall & torch.isin(spawn_class, torch.tensor(classes, device=device))
            buf = torch.zeros(steps + 1, n, width, device=device)
            alive = torch.ones(n, dtype=torch.bool, device=device)
            stable_step = torch.full((n,), -1, dtype=torch.long, device=device)

            def snapshot() -> torch.Tensor:
                ground = getattr(raw, ev._FALL_GROUND_Z_ATTR)
                pose = robot.data.root_link_pose_w
                rel = pose[:, :3] - raw.scene.env_origins
                rel[:, 2] = pose[:, 2] - ground
                vel = robot.data.root_link_vel_w
                return torch.cat(
                    (rel, pose[:, 3:7], vel, robot.data.joint_pos, robot.data.joint_vel), dim=1
                )

            buf[0] = snapshot()
            for t in range(1, steps + 1):
                if role is not None and role in obs.keys():  # noqa: SIM118
                    obs[role][:] = 0.0
                with torch.no_grad():
                    obs, _reward, dones, _extras = env.step(policy(obs))
                done = dones.bool().reshape(-1)
                state = rec.recover_state(raw)
                first = alive & (stable_step < 0) & state.just_stable
                stable_step = torch.where(first, torch.full_like(stable_step, t), stable_step)
                alive &= ~done
                buf[t] = snapshot()
            ok = tracked & (stable_step > 0)
            for c in classes:
                attempts[c] += int((tracked & (spawn_class == c)).sum())
                successes[c] += int((ok & (spawn_class == c)).sum())
            # 开始站起 = 站稳那一步往回数 hold 步（up_n 从 1 数到 hold 时站稳）
            up_start = stable_step - hold + 1
            for e in ok.nonzero().flatten().tolist():
                rows = torch.arange(0, int(up_start[e]) + 1, int(args.stride), device=device)
                states = buf[rows, e]
                trajectories.append(
                    {
                        "states": states.cpu().numpy(),
                        "elapsed_s": (rows.float() * dt).cpu().numpy(),
                        "remaining_s": ((up_start[e] - rows).float() * dt).cpu().numpy(),
                        "spawn_class": int(spawn_class[e]),
                        "traj_id": round_id * n + e,
                    }
                )
            print(
                f"round={round_id} tracked={int(tracked.sum())} stood={int(ok.sum())} "
                f"trajectories={len(trajectories)} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    raw.close()

    # 每类各取一半状态，按整条轨迹随机抽（不拆轨迹，保证每个出生状态都有完整的后续可走）
    rng = np.random.default_rng(int(args.seed))
    per_class = int(args.num_states) // len(classes)
    chosen: list[dict[str, np.ndarray]] = []
    for c in classes:
        pool = [tr for tr in trajectories if tr["spawn_class"] == c]
        total = 0
        for i in rng.permutation(len(pool)):
            if total >= per_class:
                break
            chosen.append(pool[i])
            total += len(pool[i]["states"])
        if total < per_class:
            raise RuntimeError(
                f"类别 {TILT_CLASS_NAMES[c]} 只收到 {total}/{per_class} 个状态，调大 --rounds"
            )

    states = np.concatenate([tr["states"] for tr in chosen], axis=0)
    o = 0
    root_pos, o = states[:, o : o + 3], o + 3
    root_quat, o = states[:, o : o + 4], o + 4
    root_lin_vel, o = states[:, o : o + 3], o + 3
    root_ang_vel, o = states[:, o : o + 3], o + 3
    joint_pos, o = states[:, o : o + joints], o + joints
    joint_vel = states[:, o : o + joints]
    # 机身系重力 z 分量 = −(世界 z 轴在机身系的投影)；四元数 wxyz
    _w, x, y, _z = root_quat.T
    pg_z = -(1.0 - 2.0 * (x * x + y * y))
    tilt = np.rad2deg(np.arccos(np.clip(-pg_z, -1.0, 1.0)))
    arrays = {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "tilt_deg": tilt.astype(np.float32),
        "tilt_class": _tilt_class(torch.as_tensor(tilt)).numpy(),
        "remaining_s": np.concatenate([tr["remaining_s"] for tr in chosen]).astype(np.float32),
        "elapsed_s": np.concatenate([tr["elapsed_s"] for tr in chosen]).astype(np.float32),
        "spawn_class": np.concatenate(
            [np.full(len(tr["states"]), tr["spawn_class"]) for tr in chosen]
        ),
        "traj_id": np.concatenate([np.full(len(tr["states"]), tr["traj_id"]) for tr in chosen]),
    }
    counts = {
        TILT_CLASS_NAMES[c]: {
            "attempted": attempts[c],
            "stood": successes[c],
            "trajectories": sum(tr["spawn_class"] == c for tr in chosen),
            "states": int((arrays["spawn_class"] == c).sum()),
            "remaining_s_p10_p50_p90_max": _quantiles(
                arrays["remaining_s"][arrays["spawn_class"] == c]
            ),
        }
        for c in classes
    }
    ckpt = Path(args.ckpt)
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator": "se3_tools.recover_state_cache",
        "git_commit": _git_commit(),
        "mjcf": str(_MJCF_PATH.relative_to(_PROJECT_ROOT)),
        "mjcf_sha256": _file_sha256(_MJCF_PATH),
        "task": args.task,
        "policy_ckpt": ckpt.name,
        "policy_ckpt_sha256": _file_sha256(ckpt),
        "policy_label": args.policy_label,
        "step_dt": dt,
        "stride_steps": int(args.stride),
        "stable_hold_steps": hold,
        "num_states": len(states),
        "tilt_class_names": list(TILT_CLASS_NAMES),
        "counts": counts,
        "args": {k: str(v) for k, v in vars(args).items() if k not in ("output", "device")},
    }
    output = args.output or _DEFAULT_OUTPUT_DIR / (
        f"recover_traj_{metadata['mjcf_sha256'][:8]}_{args.policy_label}_{len(states) // 1000}k.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        joint_names=np.asarray(names),
        **arrays,
    )
    print(
        f"已保存 {len(states)} 个起身轨迹状态到 {output}；{json.dumps(counts, ensure_ascii=False)}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="闭环回放起身策略，生成 recover 组反向课程的出生状态库"
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--policy-label", default="mzxd7dmq2500")
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-states", type=int, default=60000)
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
