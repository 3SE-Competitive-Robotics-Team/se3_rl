"""在训练模拟器（MJLab / MuJoCo-Warp，CPU）里用确定性策略做站立/定速 rollout，导出与 sim2x 同格式的逐 tick CSV。

用途：区分"评测端看到的腿部摆动是策略本身的性质"还是"原生 MuJoCo 与训练模拟器之间的 sim2sim gap"。
同一 checkpoint 在两个模拟器里各跑一次，用 scripts/analyze_sim2x_sway.py 比频率与幅值。

用法：
    uv run python scripts/eval_policy_mjlab_stand.py --task <task_id> --checkpoint model_N.pt \
        --out-prefix <dir>/mjlab_d4 [--num-envs 6] [--seconds 14] [--lin-vel-x 0] [--height 0.22]

每个 env 一份 CSV（<out-prefix>_env{i}.csv），列与 record_sim2x_video.py 的 --csv 对齐。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=14.0)
    parser.add_argument("--lin-vel-x", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=0.22)
    parser.add_argument(
        "--leg-action-scale",
        type=float,
        default=0.25,
        help="腿 action → 目标角的 scale，仅用于 CSV 的 leg_target 列",
    )
    parser.add_argument("--wheel-action-scale", type=float, default=15.0)
    parser.add_argument(
        "--stochastic", action="store_true", help="按训练时的高斯 σ 采样动作，而不是确定性均值"
    )
    args = parser.parse_args()

    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import se3_train  # noqa: F401
    from se3_train.mdp.joint_indices import policy_joint_ids

    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.logger = "tensorboard"
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    with tempfile.TemporaryDirectory(prefix="se3-mjlab-eval-") as tmp:
        runner = runner_cls(env, asdict(agent_cfg), tmp, "cpu")
        runner.load(str(args.checkpoint), map_location="cpu")
        policy = runner.get_inference_policy("cpu")

        robot = raw_env.scene["robot"]
        joint_ids = list(policy_joint_ids(robot))
        term = raw_env.command_manager.get_term("velocity_height")
        dt = float(raw_env.step_dt)
        steps = round(args.seconds / dt)
        obs, _ = env.reset()

        def pin_commands() -> None:
            # 契约顺序：[lin_vel_x, ang_vel_yaw, pitch, roll, height, jump_flag, jump_target_height, jump_phase]，
            # 与 sim2x adapter 的默认 [0,0,0,0,0.22,0,0,0] 对齐；其余字段一律钉零。
            cmd = term.command
            cmd[:, :] = 0.0
            cmd[:, 0] = args.lin_vel_x
            cmd[:, 1] = args.yaw_rate
            cmd[:, 4] = args.height

        pin_commands()
        obs = env.get_observations()
        print(
            f"[MJLab] task={args.task} envs={args.num_envs} dt={dt:.4f} command dim={term.command.shape[1]} command[0]={term.command[0].tolist()}"
        )
        rows = [[] for _ in range(args.num_envs)]
        resets = np.zeros(args.num_envs, dtype=int)
        for k in range(steps):
            with torch.no_grad():
                actions = policy(obs, stochastic_output=args.stochastic)
            obs, _rew, dones, _extras = env.step(actions)
            pin_commands()
            resets += dones.cpu().numpy().astype(int).reshape(-1)
            t = (k + 1) * dt
            g = robot.data.projected_gravity_b.cpu().numpy()
            qpos = robot.data.joint_pos[:, joint_ids].cpu().numpy()
            qvel = robot.data.joint_vel[:, joint_ids].cpu().numpy()
            pos = robot.data.root_link_pos_w.cpu().numpy()
            act = actions.detach().cpu().numpy()
            for i in range(args.num_envs):
                pitch = math.degrees(math.asin(float(np.clip(g[i, 0], -1.0, 1.0))))
                tilt = math.degrees(math.acos(float(np.clip(-g[i, 2], -1.0, 1.0))))
                row = {
                    "t": t,
                    "phase": "stand" if args.lin_vel_x == 0.0 and args.yaw_rate == 0.0 else "cmd",
                    "phase_index": 0.0,
                    "in_window": 1.0,
                    "cmd_vx": args.lin_vel_x,
                    "cmd_yaw": args.yaw_rate,
                    "tilt_deg": tilt,
                    "base_z": float(pos[i, 2]),
                    "wheel_l": float(qvel[i, 4]),
                    "wheel_r": float(qvel[i, 5]),
                    "pitch_deg": pitch,
                    "roll_deg": math.degrees(math.atan2(float(g[i, 1]), float(-g[i, 2]))),
                }
                row.update({f"q{j}": float(qpos[i, j]) for j in range(4)})
                row.update(
                    {f"leg_target{j}": float(act[i, j] * args.leg_action_scale) for j in range(4)}
                )
                row.update(
                    {
                        f"wheel_target{j}": float(act[i, 4 + j] * args.wheel_action_scale)
                        for j in range(2)
                    }
                )
                row.update({f"action{j}": float(act[i, j]) for j in range(6)})
                rows[i].append(row)
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        for i in range(args.num_envs):
            path = args.out_prefix.with_name(f"{args.out_prefix.name}_env{i}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[i][0].keys()))
                w.writeheader()
                w.writerows(rows[i])
        print(
            f"[MJLab] 写出 {args.num_envs} 个 CSV 到 {args.out_prefix.parent}，各 env 的 reset 次数: {resets.tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
