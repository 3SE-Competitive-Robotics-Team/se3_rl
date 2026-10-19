"""逐项计时 env 各 manager 的每一项（奖励 / 终止 / 观测 / 指令 / 事件），并统计每次调用的主机同步次数。

配合 scripts/bench_rough_sim.py 用：bench 量的是 env.step 与各 manager 的总毫秒，这里再往下拆一层，
看是哪一项、有没有 .item() / .nonzero() 这类主机同步在拖慢整条流水线。只建环境，不训练、不上传。

  uv run python scripts/profile_env_terms.py --device cuda:0 --num-envs 8192 --terrain rough
  uv run python scripts/profile_env_terms.py --device cpu --num-envs 2 --steps 2 --warmup 1 --terrain rough
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from torch.profiler import ProfilerActivity, profile

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg


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


def _time_ms(fn, n: int, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n * 1000.0


def _count_syncs(fn, n: int, device: str) -> dict:
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        for _ in range(n):
            fn()
        _sync(device)
    ka = prof.key_averages()

    def cnt(keys: tuple[str, ...]) -> float:
        return sum(e.count for e in ka if any(e.key == k or e.key.startswith(k) for k in keys)) / n

    return {
        "kernel_launches": cnt(("cudaLaunchKernel", "cuLaunchKernel")),
        "stream_sync": cnt(
            ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
        ),
        "memcpy": cnt(("cudaMemcpyAsync", "cudaMemcpy")),
        "item": cnt(("aten::item", "aten::_local_scalar_dense")),
        "nonzero": cnt(("aten::nonzero",)),
        "cpu_ms_per_call": sum(e.self_cpu_time_total for e in ka) / n / 1000.0,
    }


def _term_fn(env: ManagerBasedRlEnv, tcfg):
    """把 manager 项包成无参调用（在循环里直接写 lambda 会被 ruff B023 拦）。"""
    return lambda: tcfg.func(env, **tcfg.params)


def _bound(fn, *fargs):
    return lambda: fn(*fargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--terrain", default="rough", choices=("flat", "rough"))
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cfg = _make_cfg(args.terrain)
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg, device=args.device)
    act_dim = env.action_manager.total_action_dim
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    def noisy_action() -> torch.Tensor:
        return args.action_noise * torch.randn(
            args.num_envs, act_dim, device=args.device, generator=gen
        )

    for _ in range(args.warmup):
        env.step(noisy_action())
    dt = env.step_dt
    n = args.steps
    rows: list[tuple[str, str, float]] = []

    rm = env.reward_manager
    for name, tcfg in zip(rm._term_names, rm._term_cfgs, strict=False):
        rows.append(("reward", name, _time_ms(_term_fn(env, tcfg), n, args.device)))
    tm = env.termination_manager
    for name, tcfg in zip(tm._term_names, tm._term_cfgs, strict=False):
        rows.append(("termination", name, _time_ms(_term_fn(env, tcfg), n, args.device)))
    om = env.observation_manager
    for group, names in om._group_obs_term_names.items():
        for name, tcfg in zip(names, om._group_obs_term_cfgs[group], strict=False):
            rows.append(
                (
                    f"obs/{group}",
                    name,
                    _time_ms(_term_fn(env, tcfg), n, args.device),
                )
            )
    cm = env.command_manager
    for name, term in cm._terms.items():
        rows.append(
            ("command", f"{name}.compute", _time_ms(_bound(term.compute, dt), n, args.device))
        )
        for sub in ("_update_command", "_update_metrics"):
            fn = getattr(term, sub, None)
            if callable(fn):
                rows.append(("command", f"{name}.{sub}", _time_ms(fn, n, args.device)))
    if "interval" in env.event_manager.available_modes:
        rows.append(
            (
                "event",
                "interval",
                _time_ms(lambda: env.event_manager.apply(mode="interval", dt=dt), n, args.device),
            )
        )

    totals = {
        "reward.compute": _time_ms(lambda: rm.compute(dt=dt), n, args.device),
        "termination.compute": _time_ms(tm.compute, n, args.device),
        "observation.compute": _time_ms(lambda: om.compute(update_history=True), n, args.device),
        "command.compute": _time_ms(lambda: cm.compute(dt=dt), n, args.device),
    }
    syncs = {
        "reward.compute": _count_syncs(lambda: rm.compute(dt=dt), 8, args.device),
        "termination.compute": _count_syncs(tm.compute, 8, args.device),
        "observation.compute": _count_syncs(
            lambda: om.compute(update_history=True), 8, args.device
        ),
        "command.compute": _count_syncs(lambda: cm.compute(dt=dt), 8, args.device),
    }

    rows.sort(key=lambda r: -r[2])
    print(f"[terms] terrain={args.terrain} num_envs={args.num_envs} device={args.device}")
    print("manager | term | ms/call")
    for kind, name, ms in rows:
        print(f"{kind} | {name} | {ms:.3f}")
    print("[totals]", json.dumps({k: round(v, 3) for k, v in totals.items()}))
    print("[syncs per call]", json.dumps(syncs))
    if args.json_out:
        with open(args.json_out, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "terrain": args.terrain,
                        "num_envs": args.num_envs,
                        "device": args.device,
                        "terms": [{"manager": k, "term": t, "ms": round(v, 4)} for k, t, v in rows],
                        "totals": totals,
                        "syncs": syncs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    env.close()


if __name__ == "__main__":
    main()
