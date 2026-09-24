"""用真实策略在给定地形与 njmax / nconmax 下滚动若干步，逐步记录约束数 / 接触数峰值与 mjwarp 的 overflow 标志。

回答"把 njmax / nconmax 缩小会不会溢出"：先用现值跑出真实峰值，再用缩小值跑一遍看 `d.overflow` 有没有置位。
动作默认按训练方式从策略分布采样（等价 rollout），`--deterministic` 用均值动作；不传 `--checkpoint` 则零动作加噪声。
只建环境和 runner，不训练、不上传。

  uv run python scripts/check_sim_overflow.py --device cuda:0 --num-envs 8192 --terrain rough \\
      --checkpoint logs/rsl_rl/SE3-WheelLegged-Rough/<run>/model_2200.pt --steps 2000 --init-level-max 9
  uv run python scripts/check_sim_overflow.py ... --njmax 256 --nconmax 64
本机 CPU 冒烟：
  uv run python scripts/check_sim_overflow.py --device cpu --num-envs 2 --steps 3 --terrain rough --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict

os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg

ROUGH_TASK_ID = "SE3-WheelLegged-Rough"
FLAT_TASK_ID = "SE3-WheelLegged-Flat"
OVERFLOW_BITS = {1: "NEFC", 2: "NJMAX_NNZ", 4: "BROADPHASE", 8: "NARROWPHASE", 16: "CCD"}


def _make_cfg(terrain: str):
    if terrain == "flat":
        return flat_env_cfg()
    if terrain == "rough":
        return rough_env_cfg()
    raise ValueError(terrain)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wp.synchronize()


def _decode_overflow(flags: torch.Tensor) -> dict[str, int]:
    return {name: int(((flags & bit) != 0).sum()) for bit, name in OVERFLOW_BITS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--terrain", default="rough", choices=("flat", "rough"))
    parser.add_argument("--checkpoint", default=None, help="RSL-RL model_*.pt；不传则零动作加噪声")
    parser.add_argument(
        "--deterministic", action="store_true", help="用策略均值动作，默认按训练方式采样"
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--init-level-max",
        type=int,
        default=None,
        help="覆盖 max_init_terrain_level，9 = 全部难度行均匀起步",
    )
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cfg = _make_cfg(args.terrain)
    cfg.scene.num_envs = args.num_envs
    if args.init_level_max is not None and cfg.scene.terrain is not None:
        cfg.scene.terrain.max_init_terrain_level = args.init_level_max
    if args.njmax is not None:
        cfg.sim.njmax = args.njmax
    if args.nconmax is not None:
        cfg.sim.nconmax = args.nconmax
    torch.manual_seed(args.seed)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    d = env.sim.wp_data
    nworld = d.nworld
    print(
        f"[setup] terrain={args.terrain} num_envs={args.num_envs} njmax={d.njmax} "
        f"naconmax={d.naconmax} (nconmax/world={d.naconmax // max(nworld, 1)}) "
        f"init_level_max={args.init_level_max} checkpoint={args.checkpoint}",
        flush=True,
    )

    if args.checkpoint:
        from se3_train.tasks import register_all_tasks

        with contextlib.suppress(ValueError):  # 导入 se3_train.tasks 时已经注册过
            register_all_tasks()
        task_id = FLAT_TASK_ID if args.terrain == "flat" else ROUGH_TASK_ID
        agent_cfg = load_rl_cfg(task_id)
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
        runner.load(args.checkpoint, map_location=args.device)
        obs = venv.get_observations()
        policy = runner.get_inference_policy(device=args.device) if args.deterministic else None

        def step_fn():
            nonlocal obs
            with torch.inference_mode():
                actions = policy(obs) if policy is not None else runner.alg.act(obs)
            obs, _, dones, _ = venv.step(actions)
            return dones

    else:
        act_dim = env.action_manager.total_action_dim
        gen = torch.Generator(device=args.device).manual_seed(args.seed)

        def step_fn():
            action = args.action_noise * torch.randn(
                args.num_envs, act_dim, device=args.device, generator=gen
            )
            env.step(action)
            return env.reset_buf

    nefc_t = wp.to_torch(d.nefc)
    overflow_t = wp.to_torch(d.overflow)
    worldid_t = wp.to_torch(d.contact.worldid)
    niter_t = wp.to_torch(d.solver_niter)
    overflow_acc = torch.zeros(nworld, dtype=torch.int64, device=args.device)
    thresholds = (128, 192, 256, 384, 512)
    steps_over = dict.fromkeys(thresholds, 0)
    peak = {"nefc": 0, "ncon_world": 0, "nacon_total": 0, "ncollision": 0, "niter": 0}
    nefc_step_max_hist: list[int] = []
    dones_total = 0
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        dones = step_fn()
        dones_total += int(dones.sum())
        nefc_max = int(nefc_t.max())
        nacon = int(d.nacon.numpy()[0])
        ncollision = int(d.ncollision.numpy()[0])
        if nacon > 0:
            ids = worldid_t[: min(nacon, d.naconmax)].to(torch.int64)
            ncon_world = int(torch.bincount(ids, minlength=nworld).max())
        else:
            ncon_world = 0
        peak["nefc"] = max(peak["nefc"], nefc_max)
        peak["ncon_world"] = max(peak["ncon_world"], ncon_world)
        peak["nacon_total"] = max(peak["nacon_total"], nacon)
        peak["ncollision"] = max(peak["ncollision"], ncollision)
        peak["niter"] = max(peak["niter"], int(niter_t.max()))
        nefc_step_max_hist.append(nefc_max)
        for t in thresholds:
            steps_over[t] += int(nefc_max > t)
        overflow_acc |= overflow_t.to(torch.int64)
        if step % args.report_every == 0 or step == args.steps:
            _sync(args.device)
            elapsed = time.perf_counter() - t0
            ov = _decode_overflow(overflow_acc)
            print(
                f"[step {step}] nefc max so far {peak['nefc']} (this step {nefc_max}) | "
                f"ncon/world max {peak['ncon_world']} | nacon total max "
                f"{peak['nacon_total']}/{d.naconmax} | ncollision max {peak['ncollision']} | "
                f"niter max {peak['niter']} | overflow worlds {int((overflow_acc != 0).sum())} {ov} | "
                f"dones {dones_total} | {step / elapsed:.1f} steps/s",
                flush=True,
            )

    levels = getattr(env.scene.terrain, "terrain_levels", None)
    level_hist = (
        torch.bincount(levels.to(torch.int64), minlength=10).tolist()
        if levels is not None
        else None
    )
    hist = torch.tensor(nefc_step_max_hist, dtype=torch.float32)
    quantiles = {f"p{int(p * 100)}": float(torch.quantile(hist, p)) for p in (0.5, 0.9, 0.99)}
    result = {
        "terrain": args.terrain,
        "num_envs": args.num_envs,
        "checkpoint": args.checkpoint,
        "deterministic": bool(args.deterministic),
        "steps": args.steps,
        "njmax": int(d.njmax),
        "naconmax": int(d.naconmax),
        "nconmax_per_world": int(d.naconmax // max(nworld, 1)),
        "init_level_max": args.init_level_max,
        "peak": peak,
        "nefc_step_max_quantiles": quantiles,
        "steps_with_nefc_max_over": steps_over,
        "overflow_worlds": int((overflow_acc != 0).sum()),
        "overflow_by_type": _decode_overflow(overflow_acc),
        "dones_total": dones_total,
        "terrain_level_hist": level_hist,
    }
    print("[result]", json.dumps(result, ensure_ascii=False), flush=True)
    if args.json_out:
        with open(args.json_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    env.close()


if __name__ == "__main__":
    main()
