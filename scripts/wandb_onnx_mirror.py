"""W&B → ONNX 镜像转换器：把训练 run 上传到 W&B 的 model_N.pt 在本机导出为 ONNX，供 Viser browser 查看。

设计（2026-09-05，4 卡机 gpufree 上常驻）：
- 训练端只向 W&B 上传 .pt（不传 ONNX，省 W&B 存储）；本脚本每轮枚举 entity 下全部 project 与 run，
  只认 `train_cfg.experiment_name` 以 --experiment-prefix 开头的 run，新 project 自然覆盖。
- 输出目录形态与训练日志一致：<mirror-root>/<experiment_name>/<run 名>/onnx/model_N.onnx，
  下载的 .pt 保留在 <run 名>/ 下作为来源；增量以 onnx 文件是否存在判断。
- 导出必须在 run 自己的 commit 上做：用 `git worktree` 从 --repo 按 run.commit 开一份代码（含子模块），
  以 --python（主 venv 解释器）加 PYTHONPATH 调用与本脚本同目录的 export_onnx_from_checkpoint.py，
  不改 --repo 的 HEAD，与训练队列互不干扰。commit 不在本机时记"等 commit"跳过，不阻塞其他 run。

用法：
    python scripts/wandb_onnx_mirror.py --entity luzhongjin365-se3 --mirror-root /root/gpufree-data/se3-mirror
        --repo /workspace/se3_rl --python /workspace/se3_rl/.venv/bin/python [--once] [--interval 300]
        [--project-filter REGEX] [--run-name-filter REGEX]
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
SUBMODULE_PATH = "submodules/se3-sim2x"


@dataclass
class MirrorConfig:
    """一次运行的全部参数与跨轮状态。"""

    entity: str
    mirror_root: Path
    repo: Path
    python: Path
    export_script: Path
    experiment_prefix: str
    run_name_filter: re.Pattern[str] | None
    project_filter: re.Pattern[str] | None
    interval_s: float
    once: bool
    state: dict = field(default_factory=dict)


def log(message: str) -> None:
    """带本地时区时间戳输出；常驻运行时由调用方把 stdout 重定向到 mirror.log。"""
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """在指定仓库执行 git 子命令。"""
    return subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=check
    )


def commit_available(repo: Path, commit: str) -> bool:
    """commit 对象是否已在主仓库里（由本地 git push 送达）。"""
    return git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode == 0


def ensure_worktree(cfg: MirrorConfig, commit: str) -> Path:
    """为 commit 准备独立 worktree（含子模块），已存在则直接复用。"""
    worktree = cfg.mirror_root / ".worktrees" / commit[:12]
    if not (worktree / ".git").exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        git(cfg.repo, "worktree", "prune")
        git(cfg.repo, "worktree", "add", "--detach", str(worktree), commit)
        log(f"[worktree] 新建 {worktree} @ {commit[:12]}")
    head = git(worktree, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise RuntimeError(f"worktree {worktree} HEAD={head[:12]} 与 run commit {commit[:12]} 不符")
    # 子模块对象来自主仓库的 module git dir（本地推送的 bundle/main 分支），file 协议需显式放行。
    result = git(
        worktree, "-c", "protocol.file.allow=always", "submodule", "update", "--init", check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"子模块同步失败：{result.stderr.strip()[-400:]}")
    return worktree


def export_checkpoints(
    cfg: MirrorConfig, worktree: Path, task: str, run_dir: Path, checkpoints: list[Path]
) -> None:
    """在 run 的 commit 上以 CPU 导出一批 checkpoint。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(worktree / "src"), str(worktree / SUBMODULE_PATH / "src")]
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["WANDB_MODE"] = "disabled"
    env.setdefault("OMP_NUM_THREADS", "4")
    command = [
        str(cfg.python),
        str(cfg.export_script),
        "--task",
        task,
        "--output-dir",
        str(run_dir / "onnx"),
    ]
    for checkpoint in checkpoints:
        command += ["--checkpoint", str(checkpoint)]
    log(f"[export] {task} {run_dir.name}: {len(checkpoints)} 个 checkpoint @ {worktree.name}")
    result = subprocess.run(command, cwd=str(worktree), env=env, text=True, capture_output=True)
    tail = "\n".join(result.stdout.splitlines()[-6:] + result.stderr.splitlines()[-12:])
    if result.returncode != 0:
        raise RuntimeError(f"导出失败（exit {result.returncode}）：\n{tail}")
    missing = [c.stem for c in checkpoints if not (run_dir / "onnx" / f"{c.stem}.onnx").is_file()]
    if missing:
        raise RuntimeError(f"导出结束但缺少 ONNX：{missing[:5]}")
    log(f"[export] 完成 {run_dir.name}: {len(checkpoints)} 个 ONNX")


def process_run(cfg: MirrorConfig, run) -> None:
    """处理单个 run：筛选、增量下载 .pt、按 commit 导出 ONNX。"""
    config = run.config or {}
    train_cfg = config.get("train_cfg") if isinstance(config.get("train_cfg"), dict) else {}
    experiment = train_cfg.get("experiment_name")
    if not isinstance(experiment, str) or not experiment.startswith(cfg.experiment_prefix):
        return
    if cfg.run_name_filter is not None and not cfg.run_name_filter.search(run.name):
        return
    run_state = cfg.state.setdefault("runs", {}).setdefault(run.id, {})
    if run_state.get("complete"):
        return

    pt_files = {}
    for wandb_file in run.files():
        match = CHECKPOINT_RE.match(wandb_file.name)
        if match:
            pt_files[int(match.group(1))] = wandb_file
    if not pt_files:
        return
    run_dir = cfg.mirror_root / experiment / run.name
    onnx_dir = run_dir / "onnx"
    pending = sorted(it for it in pt_files if not (onnx_dir / f"model_{it}.onnx").is_file())
    if not pending:
        if run.state != "running":
            run_state["complete"] = True
            log(f"[done] {run.name}: {len(pt_files)} 个 ONNX 齐全，run 已 {run.state}")
        return

    commit = run.commit
    if not commit or not commit_available(cfg.repo, commit):
        if run_state.get("waiting_commit") != commit:
            log(f"[wait] {run.name}: commit {str(commit)[:12]} 不在 {cfg.repo}，等本地 push 后再转")
            run_state["waiting_commit"] = commit
        return
    run_state.pop("waiting_commit", None)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Path] = []
    for it in pending:
        target = run_dir / f"model_{it}.pt"
        if not target.is_file() or target.stat().st_size != pt_files[it].size:
            pt_files[it].download(root=str(run_dir), replace=True)
        checkpoints.append(target)
    log(
        f"[download] {run.name}: {len(checkpoints)} 个 .pt 就绪（run {run.state}，commit {commit[:12]}）"
    )

    worktree = ensure_worktree(cfg, commit)
    export_checkpoints(cfg, worktree, experiment, run_dir, checkpoints)
    (run_dir / "mirror_meta.json").write_text(
        json.dumps(
            {
                "wandb_run_id": run.id,
                "wandb_project": run.project,
                "wandb_url": run.url,
                "commit": commit,
                "experiment_name": experiment,
                "updated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def one_pass(cfg: MirrorConfig, api) -> None:
    """枚举 entity 下全部 project/run 并逐个处理；单个失败不影响其他。"""
    for project in api.projects(cfg.entity):
        if cfg.project_filter is not None and not cfg.project_filter.search(project.name):
            continue
        try:
            runs = list(api.runs(f"{cfg.entity}/{project.name}", order="-created_at"))
        except Exception as error:
            log(f"[error] 枚举 project {project.name} 失败：{error}")
            continue
        for run in runs:
            try:
                process_run(cfg, run)
            except Exception as error:
                log(f"[error] {project.name}/{run.name}: {error}")
                log(traceback.format_exc().strip().splitlines()[-1])


def save_state(cfg: MirrorConfig) -> None:
    """原子写回跨轮状态。"""
    path = cfg.mirror_root / "state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg.state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entity", required=True)
    parser.add_argument("--mirror-root", required=True, type=Path)
    parser.add_argument(
        "--repo", required=True, type=Path, help="含全部 run commit 的主仓库（只读，不改 HEAD）"
    )
    parser.add_argument("--python", required=True, type=Path, help="主 venv 的解释器")
    parser.add_argument("--experiment-prefix", default="SE3-")
    parser.add_argument("--project-filter", default=None, help="只处理名称匹配该正则的 project")
    parser.add_argument("--run-name-filter", default=None, help="只处理名称匹配该正则的 run")
    parser.add_argument("--interval", type=float, default=300.0, help="轮询间隔秒")
    parser.add_argument("--once", action="store_true", help="只跑一轮")
    args = parser.parse_args()

    export_script = Path(__file__).resolve().with_name("export_onnx_from_checkpoint.py")
    if not export_script.is_file():
        print(f"缺少 {export_script}", file=sys.stderr)
        return 2
    args.mirror_root.mkdir(parents=True, exist_ok=True)
    state_path = args.mirror_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    cfg = MirrorConfig(
        entity=args.entity,
        mirror_root=args.mirror_root,
        repo=args.repo,
        python=args.python,
        export_script=export_script,
        experiment_prefix=args.experiment_prefix,
        run_name_filter=re.compile(args.run_name_filter) if args.run_name_filter else None,
        project_filter=re.compile(args.project_filter) if args.project_filter else None,
        interval_s=args.interval,
        once=args.once,
        state=state,
    )

    lock_file = open(args.mirror_root / "mirror.lock", "w")  # noqa: SIM115  进程生命周期内持有
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("另一个镜像转换器已在运行", file=sys.stderr)
        return 3

    import wandb

    api = wandb.Api(timeout=120)
    log(f"[start] entity={cfg.entity} mirror={cfg.mirror_root} repo={cfg.repo} once={cfg.once}")
    while True:
        started = time.time()
        try:
            one_pass(cfg, api)
        except Exception as error:
            log(f"[error] 本轮失败：{error}")
        save_state(cfg)
        if cfg.once:
            return 0
        elapsed = time.time() - started
        time.sleep(max(30.0, cfg.interval_s - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
