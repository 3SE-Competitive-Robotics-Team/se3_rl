"""Rough 地形仿真耗时对照基准：同一机器人、同一 env 数，只换地形列数 / 宽相算法，量 env.step 的毫秒数。

不走训练队列、不建 runner、不上传 W&B，只建 ManagerBasedRlEnv 跑零动作加小噪声。
输出每个变体的：env.step 毫秒（含传感器、奖励、终止、重置）、sim.step 物理毫秒（4 个子步合计）、
sim.sense 射线毫秒、按 24 步/轮折算的 collection 秒数，以及 nconmax / njmax 定尺寸用的
每世界最大接触数与约束数、宽相配对数。

Pod 单卡（与训练同规模）：
  uv run python scripts/bench_rough_sim.py --device cuda:0 --num-envs 8192 --steps 96
变体名可加 `+aabb` 后缀：宽相过滤加上 AABB（默认 plane|sphere|obb），几何与接触结果不变：
  uv run python scripts/bench_rough_sim.py --variants rough-default,rough-default+aabb
全局覆盖 nconmax / njmax / 求解器迭代上限（对所有变体生效，用来量官方文档说的"越小越快"）：
  uv run python scripts/bench_rough_sim.py --variants rough-default --njmax 300 --nconmax 64 --iterations 10 --ls-iterations 20
只比地形列数：
  uv run python scripts/bench_rough_sim.py --variants rough-default
看物理各阶段（宽相 / 窄相 / 求解器）各占多少毫秒（仅 CUDA，直接调 mjwarp.step 不走 CUDA graph）：
  uv run python scripts/bench_rough_sim.py --variants rough-default --event-trace
本机 CPU 冒烟（只验证脚本能跑）：
  uv run python scripts/bench_rough_sim.py --device cpu --num-envs 2 --steps 2 --warmup 1 --variants rough-default
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mujoco_warp._src.warp_util import EventTracer

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.terrains import stair_only_terrains_cfg

TERRAINS = ("flat", "rough", "3col")
BROADPHASES = ("default", "nxn", "sap_tile", "sap_segmented")


def _make_cfg(terrain: str):
    if terrain == "flat":
        return flat_env_cfg()
    if terrain == "rough":
        return rough_env_cfg()
    if terrain == "3col":
        return rough_env_cfg(terrain_generator=stair_only_terrains_cfg())
    raise ValueError(f"unknown terrain {terrain!r}, choose from {TERRAINS}")


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


def _contact_stats(env: ManagerBasedRlEnv) -> dict:
    d = env.sim.wp_data
    nacon = int(d.nacon.numpy()[0])
    worldid = d.contact.worldid.numpy()[:nacon]
    per_world = (
        np.bincount(worldid, minlength=d.nworld) if nacon > 0 else np.zeros(d.nworld, dtype=int)
    )
    nefc = d.nefc.numpy()
    niter = getattr(d, "solver_niter", None)
    niter = niter.numpy() if niter is not None else np.zeros(1, dtype=int)
    overflow = getattr(d, "overflow", None)
    overflow_worlds = int((overflow.numpy() != 0).sum()) if overflow is not None else -1
    return {
        "overflow_worlds": overflow_worlds,
        "solver_niter_max": int(niter.max()),
        "solver_niter_mean": float(niter.mean()),
        "opt_iterations": int(env.sim.mj_model.opt.iterations),
        "opt_ls_iterations": int(env.sim.mj_model.opt.ls_iterations),
        "ncon_total": nacon,
        "ncon_per_world_max": int(per_world.max()),
        "ncon_per_world_mean": float(per_world.mean()),
        "nefc_per_world_max": int(nefc.max()),
        "nefc_per_world_mean": float(nefc.mean()),
        "naconmax": int(d.naconmax),
        "njmax": int(d.njmax),
    }


def _flatten_trace(trace: dict, prefix: str = "", out: dict | None = None) -> dict:
    """把 EventTracer.trace() 的嵌套 {name: (elapsed_ms_tuple, sub)} 拍平成 {path: mean_ms}。"""
    out = {} if out is None else out
    for name, (events, sub) in trace.items():
        path = f"{prefix}{name}"
        out[path] = sum(events) / len(events) if events else 0.0
        _flatten_trace(sub, path + "/", out)
    return out


def _event_trace(env: ManagerBasedRlEnv, nsteps: int) -> dict | None:
    """直接调 mjwarp.step（不经 mjlab 的 CUDA graph）取物理各阶段的平均毫秒，只在 CUDA 上可用。"""
    try:
        with EventTracer(enabled=True) as tracer:
            for _ in range(nsteps):
                mjwarp.step(env.sim.wp_model, env.sim.wp_data)
            wp.synchronize()
            flat = _flatten_trace(tracer.trace())
    except Exception as exc:
        print(f"[event-trace] 跳过：{exc!r}", flush=True)
        return None
    return {k: round(v, 3) for k, v in flat.items()}


def _profile_managers(env: ManagerBasedRlEnv, action: torch.Tensor, n: int, device: str) -> dict:
    """把 env.step 拆成各 manager 单独计时（每次调用前后同步），定位物理之外的每步开销。"""
    dt = env.step_dt
    calls = {
        "action.process+apply": lambda: (
            env.action_manager.process_action(action),
            env.action_manager.apply_action(),
        ),
        "scene.write_data_to_sim": env.scene.write_data_to_sim,
        "sim.step_x1": env.sim.step,
        "scene.update": lambda: env.scene.update(dt=env.physics_dt),
        "termination.compute": env.termination_manager.compute,
        "reward.compute": lambda: env.reward_manager.compute(dt=dt),
        "sim.forward": env.sim.forward,
        "sim.sense": env.sim.sense,
        "command.compute": lambda: env.command_manager.compute(dt=dt),
        "event.apply_interval": lambda: env.event_manager.apply(mode="interval", dt=dt),
        "observation.compute": lambda: env.observation_manager.compute(update_history=True),
        "metrics.compute": env.metrics_manager.compute,
    }
    out: dict = {}
    for name, fn in calls.items():
        try:
            out[name] = round(_time_ms(fn, n, device), 3)
        except Exception as exc:  # 诊断路径，单项失败不中断
            out[name] = f"error: {exc!r}"
    return out


def run_variant(name: str, args: argparse.Namespace) -> dict:
    terrain, broadphase = name.split("-", 1)
    use_aabb = broadphase.endswith("+aabb")
    broadphase = broadphase.removesuffix("+aabb")
    cfg = _make_cfg(terrain)
    cfg.scene.num_envs = args.num_envs
    if broadphase != "default":
        cfg.sim.broadphase = broadphase
    if use_aabb:
        cfg.sim.broadphase_filter = ("plane", "sphere", "aabb", "obb")
    if args.nconmax is not None:
        cfg.sim.nconmax = args.nconmax
    if args.njmax is not None:
        cfg.sim.njmax = args.njmax
    if args.iterations is not None:
        cfg.sim.mujoco.iterations = args.iterations
    if args.ls_iterations is not None:
        cfg.sim.mujoco.ls_iterations = args.ls_iterations
    t_build = time.perf_counter()
    env = ManagerBasedRlEnv(cfg, device=args.device)
    t_build = time.perf_counter() - t_build
    m = env.sim.wp_model
    act_dim = env.action_manager.total_action_dim
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    def noisy_action() -> torch.Tensor:
        return args.action_noise * torch.randn(
            args.num_envs, act_dim, device=args.device, generator=gen
        )

    for _ in range(args.warmup):
        env.step(noisy_action())

    def env_step() -> None:
        env.step(noisy_action())

    step_ms = _time_ms(env_step, args.steps, args.device)
    physics_ms = _time_ms(env.sim.step, args.steps * cfg.decimation, args.device) * cfg.decimation
    sense_ms = _time_ms(env.sim.sense, args.steps, args.device)
    trace = None
    if args.event_trace and args.device.startswith("cuda"):
        trace = _event_trace(env, args.steps * cfg.decimation)
    profile = None
    if args.profile_managers:
        profile = _profile_managers(env, noisy_action(), args.steps, args.device)
    env.step(noisy_action())
    stats = _contact_stats(env)
    result = {
        "variant": name,
        "device": args.device,
        "num_envs": args.num_envs,
        "ngeom": int(m.ngeom),
        "nxn_pairs_filtered": int(m.nxn_geom_pair_filtered.shape[0]),
        "broadphase_effective": int(m.opt.broadphase),
        "broadphase_filter": int(m.opt.broadphase_filter),
        "use_cuda_graph": bool(getattr(env.sim, "use_cuda_graph", False)),
        "graph_conditional": bool(getattr(m.opt, "graph_conditional", False)),
        "cuda_driver": str(wp.get_cuda_driver_version())
        if args.device.startswith("cuda")
        else None,
        "build_s": round(t_build, 1),
        "env_step_ms": round(step_ms, 2),
        "physics_ms_per_env_step": round(physics_ms, 2),
        "sense_ms": round(sense_ms, 2),
        "collection_s_per_iter_24steps": round(step_ms * 24 / 1000.0, 3),
        **stats,
    }
    if profile is not None:
        result["manager_profile_ms"] = profile
        print(f"[managers] {name}: 每步各 manager 平均毫秒", flush=True)
        for k, v in profile.items():
            print(f"  {k}: {v}", flush=True)
    if trace is not None:
        result["physics_trace_ms_per_substep"] = trace
        print(f"[event-trace] {name}: 每个物理子步各阶段平均毫秒（按嵌套路径）", flush=True)
        for path, ms in trace.items():
            if ms >= 0.02:
                print(f"  {'  ' * path.count('/')}{path.rsplit('/', 1)[-1]}: {ms:.3f}", flush=True)
    env.close()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument(
        "--steps", type=int, default=96, help="计时的 env 步数（24 步 = 一轮 collection）"
    )
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument(
        "--variants",
        default="rough-default,rough-sap_tile",
        help=f"逗号分隔的 <terrain>-<broadphase>；terrain∈{TERRAINS}，broadphase∈{BROADPHASES}",
    )
    parser.add_argument("--json-out", default=None, help="把结果追加写成 JSON 行")
    parser.add_argument("--event-trace", action="store_true", help="打印物理各阶段耗时（仅 CUDA）")
    parser.add_argument(
        "--profile-managers", action="store_true", help="逐 manager 计时，定位物理之外的每步开销"
    )
    parser.add_argument(
        "--nconmax", type=int, default=None, help="覆盖 SimulationCfg.nconmax（每世界）"
    )
    parser.add_argument(
        "--njmax", type=int, default=None, help="覆盖 SimulationCfg.njmax（每世界）"
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="覆盖求解器迭代上限 opt.iterations"
    )
    parser.add_argument(
        "--ls-iterations", type=int, default=None, help="覆盖线搜索迭代上限 opt.ls_iterations"
    )
    args = parser.parse_args()

    results = []
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        result = run_variant(name, args)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.json_out:
            with open(args.json_out, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        "\nvariant | ngeom | pairs | bp | env.step ms | physics ms | sense ms | collect s/iter | ncon max | nefc max"
    )
    for r in results:
        print(
            f"{r['variant']} | {r['ngeom']} | {r['nxn_pairs_filtered']} | "
            f"{r['broadphase_effective']}/{r['broadphase_filter']} | "
            f"{r['env_step_ms']} | {r['physics_ms_per_env_step']} | {r['sense_ms']} | "
            f"{r['collection_s_per_iter_24steps']} | {r['ncon_per_world_max']} | {r['nefc_per_world_max']} | "
            f"{r['solver_niter_max']}/{r['solver_niter_mean']:.1f} | {r['overflow_worlds']}"
        )


if __name__ == "__main__":
    main()
