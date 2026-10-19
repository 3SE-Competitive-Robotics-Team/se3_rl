"""`se3-bench`：Rough benchmark 命令行。

    se3-bench run --onnx <model_N.onnx> [--suite quick|full] [--workers N]
    se3-bench compare <a.json> <b.json>

结果默认写到 ONNX 旁的 `bench/<model>_<suite>.json`（诊断图在 `bench/figures/<model>_<suite>/`）。
给了 `--wandb-project` 就额外记到同 group 的 `<run>-bench` 评测 run（见 report.log_wandb）。
训练中由 se3_train.bench_hook 以独立进程调用本命令（`--lock` 让两次评测排队，不重叠）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from se3_tools.bench import metrics, report, spec


def _default_workers() -> int:
    return max(1, min(16, (os.cpu_count() or 2) // 2))


class _Lock:
    """评测互斥：已有评测在跑时排队等待（wait=False 时直接跳过）。"""

    def __init__(self, path: Path, *, wait: bool) -> None:
        self.path = path
        self.wait = wait
        self.held = False

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return True
            except FileExistsError:
                if not _lock_owner_alive(self.path):
                    self.path.unlink(missing_ok=True)
                    continue
                if not self.wait:
                    return False
                time.sleep(30)

    def __exit__(self, *exc) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)


def _lock_owner_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return False
    if os.name == "nt":
        # Windows 上 os.kill(pid, 0) 会发 CTRL_C_EVENT，不能用来探活；本机手动评测不加锁，按仍在跑处理。
        return True
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _iteration_from_name(onnx: Path) -> int:
    stem = onnx.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def cmd_run(args: argparse.Namespace) -> int:
    onnx = Path(args.onnx).resolve()
    out_dir = Path(args.out).resolve() if args.out else onnx.parent / "bench"
    lock_path = out_dir / ".lock"
    lock = _Lock(lock_path, wait=True) if args.lock else None
    if lock is not None and not lock.__enter__():
        print(f"[se3-bench] 已有评测在跑（{lock_path}），跳过本次 {args.suite}", flush=True)
        return 0
    try:
        return _run(args, onnx, out_dir)
    finally:
        if lock is not None:
            lock.__exit__()


def _run(args: argparse.Namespace, onnx: Path, out_dir: Path) -> int:
    from se3_tools.bench.scenes import build_stair_grid, robot_mjcf_for

    started = time.time()
    rules = spec.rough_stair_rules()
    cases = spec.build_cases(args.suite, rules)
    if args.only:
        cases = [c for c in cases if c.suite in set(args.only)]
    rows = spec.stair_rows(cases)
    scene_path = None
    spawns: dict[str, list[float]] = {}
    if rows:
        grid = build_stair_grid(robot_mjcf_for(str(onnx)), rows)
        scene_path = grid.path
        spawns = {f"{c}:{r}": list(p) for (c, r), p in grid.spawns.items()}
    tag = f"{onnx.stem}_{args.suite}"
    jobs = [
        dict(
            case=c.as_dict(),
            onnx=str(onnx),
            scene_path=scene_path,
            spawns=spawns,
            solver_hz=args.solver_hz,
        )
        for c in cases
    ]
    print(
        f"[se3-bench] {spec.BENCH_VERSION} {args.suite}：{len(jobs)} 条，{args.workers} 进程，{onnx}",
        flush=True,
    )
    from se3_tools.bench.cases import run_case

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            r = future.result()
            results.append(r)
            if done % 10 == 0 or done == len(jobs):
                print(f"[se3-bench] {done}/{len(jobs)}", flush=True)
    order = {c.name: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(r["name"], 0))
    iteration = args.iteration if args.iteration is not None else _iteration_from_name(onnx)
    payload = {
        "bench_version": spec.BENCH_VERSION,
        "suite": args.suite,
        "onnx": str(onnx),
        "iteration": iteration,
        "solver_hz": args.solver_hz,
        "git": report.git_state(),
        "elapsed_s": round(time.time() - started, 1),
        "headline": metrics.headline(results),
        "results": results,
    }
    if not args.no_figures:
        from se3_tools.bench.figures import render_all

        payload["figures"] = render_all(payload, out_dir / "figures" / tag)
    json_path = out_dir / f"{tag}.json"
    report.write_json(json_path, payload)
    print(f"[se3-bench] 用时 {payload['elapsed_s']} s → {json_path}", flush=True)
    for key, value in payload["headline"].items():
        print(f"  {key:36s} {value:.4g}", flush=True)
    if args.wandb_project:
        try:
            url = report.log_wandb(
                payload,
                project=args.wandb_project,
                run_name=args.wandb_run_name or onnx.parent.parent.name,
                group=args.wandb_group or onnx.parent.parent.name,
                id_file=out_dir / "wandb_run_id.txt",
            )
            if url:
                print(f"[se3-bench] W&B {url}", flush=True)
        except Exception as error:
            print(f"[se3-bench] W&B 上报失败：{error}", flush=True)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    if a["bench_version"] != b["bench_version"]:
        print(
            f"[警告] bench 版本不同：{a['bench_version']} vs {b['bench_version']}，数字不直接可比"
        )
    keys = sorted(set(a["headline"]) | set(b["headline"]))
    print(f"{'指标':36s} {'A':>10s} {'B':>10s} {'B−A':>10s}")
    for k in keys:
        va, vb = a["headline"].get(k), b["headline"].get(k)
        diff = "" if va is None or vb is None else f"{vb - va:+10.4g}"
        fa = "—" if va is None else f"{va:10.4g}"
        fb = "—" if vb is None else f"{vb:10.4g}"
        print(f"{k:36s} {fa:>10s} {fb:>10s} {diff:>10s}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="se3-bench", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="评测一个 ONNX")
    run.add_argument("--onnx", required=True)
    run.add_argument("--suite", choices=("quick", "full"), default="quick")
    run.add_argument("--workers", type=int, default=_default_workers())
    run.add_argument("--out", default=None, help="结果目录，默认 ONNX 旁的 bench/")
    run.add_argument("--only", nargs="+", choices=list("ABCDE"), default=None, help="只跑这几项")
    run.add_argument(
        "--solver-hz", type=float, default=None, help="物理求解频率；默认与契约 physics_dt 一致"
    )
    run.add_argument("--iteration", type=int, default=None)
    run.add_argument("--no-figures", action="store_true", help="不出诊断图")
    run.add_argument("--lock", action="store_true", help="训练旁路调用时加锁，避免两次评测重叠")
    run.add_argument("--wandb-project", default=None)
    run.add_argument("--wandb-run-name", default=None)
    run.add_argument("--wandb-group", default=None)
    run.set_defaults(func=cmd_run)
    compare = sub.add_parser("compare", help="对比两份结果 JSON 的头条指标")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
