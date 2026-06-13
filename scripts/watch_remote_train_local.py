"""Mirror a remote checkpoint locally and open the real training viewer.

The remote PPO run stays headless. This script periodically copies the latest
stable ``model_*.pt`` from the pod into ``logs/remote_watch/<run>/`` and starts
local ``se3-play`` with one environment using the training-scene task alias.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_EXCLUDED_RUN_DIRS = {"base_model", "wandb_checkpoints"}
# These aliases must register play_env_cfg as the real training env, because
# mjlab play always loads task configs with play=True.
_TRAIN_VIEW_TASKS = {
    "SE3-WheelLegged-Stair-GRU": "SE3-WheelLegged-Stair-GRU-TrainView",
    "SE3-WheelLegged-Stair-GRU-WarmStart": "SE3-WheelLegged-Stair-GRU-TrainView",
    "SE3-WheelLegged-Stair-NoCTBC-GRU": "SE3-WheelLegged-Stair-NoCTBC-GRU-TrainView",
    "SE3-WheelLegged-Stair-NoCTBC-GRU-WarmStart": ("SE3-WheelLegged-Stair-NoCTBC-GRU-TrainView"),
    "SE3-WheelLegged-Recovery-Stand-GRU": "SE3-WheelLegged-Recovery-Stand-GRU-TrainView",
    "SE3-WheelLegged-Recovery-GRU": "SE3-WheelLegged-Recovery-GRU-TrainView",
    "SE3-WheelLegged-Mixed-GRU": "SE3-WheelLegged-Mixed-GRU-TrainView",
    "SE3-WheelLegged-Mixed-GRU-WarmStart": "SE3-WheelLegged-Mixed-GRU-TrainView",
    "SE3-WheelLegged-Stair-Teacher-GRU-WarmStart": ("SE3-WheelLegged-Stair-Teacher-GRU-TrainView"),
    "SE3-WheelLegged-Universal-Final-GRU-WarmStart": (
        "SE3-WheelLegged-Universal-Final-GRU-TrainView"
    ),
}
_RECOVERY_STAND_TASK = "SE3-WheelLegged-Recovery-Stand-GRU"
_TRAIN_VIEW_ITER_ENV = "SE3_TRAIN_VIEW_ITER"
_TRAIN_VIEW_TERRAIN_LEVEL_ENV = "SE3_TRAIN_VIEW_TERRAIN_LEVEL"
_TRAIN_VIEW_COMMAND_HEIGHT_ENV = "SE3_TRAIN_VIEW_COMMAND_HEIGHT"
_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class RemoteCheckpoint:
    name: str
    iteration: int
    size: int
    mtime: str


def _run(cmd: list[str], *, capture: bool = False, timeout: int | None = None) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=capture,
        text=capture,
        timeout=timeout,
    )
    return result.stdout if capture else ""


def _remote_shell(args: argparse.Namespace, command: str, *, timeout: int = 60) -> str:
    remote = (
        f"kubectl exec -n {shlex.quote(args.namespace)} {shlex.quote(args.pod)} "
        f"-- bash -lc {shlex.quote(command)}"
    )
    return _run(["ssh", args.host, remote], capture=True, timeout=timeout)


def _resolve_run_dir(args: argparse.Namespace) -> str:
    if args.run_dir:
        return args.run_dir
    command = (
        f"cd {shlex.quote(args.remote_project)} && "
        f"find {shlex.quote(args.remote_log_dir)} -mindepth 1 -maxdepth 1 -type d "
        + " ".join(f"! -name {shlex.quote(name)}" for name in _EXCLUDED_RUN_DIRS)
        + r" -printf '%T@ %f\n' | sort -nr | head -1 | awk '{print $2}'"
    )
    run_dir = _remote_shell(args, command).strip().splitlines()[-1]
    if not run_dir:
        raise RuntimeError("No remote run directory found.")
    return run_dir


def _list_checkpoints(args: argparse.Namespace, run_dir: str) -> dict[str, RemoteCheckpoint]:
    command = (
        f"cd {shlex.quote(args.remote_project)} && "
        f"find {shlex.quote(args.remote_log_dir + '/' + run_dir)} "
        "-maxdepth 1 -name 'model_*.pt' -printf '%f\\t%s\\t%T@\\n'"
    )
    output = _remote_shell(args, command).strip()
    checkpoints: dict[str, RemoteCheckpoint] = {}
    for line in output.splitlines():
        parts = line.rstrip().split("\t")
        if len(parts) != 3:
            continue
        name, size, mtime = parts
        match = _MODEL_RE.match(name)
        if not match:
            continue
        try:
            checkpoints[name] = RemoteCheckpoint(
                name=name,
                iteration=int(match.group(1)),
                size=int(size),
                mtime=mtime,
            )
        except ValueError:
            continue
    return checkpoints


def _latest_checkpoint(args: argparse.Namespace, run_dir: str) -> RemoteCheckpoint | None:
    first = _list_checkpoints(args, run_dir)
    if not first:
        return None
    if args.stability_seconds <= 0:
        return max(first.values(), key=lambda ckpt: ckpt.iteration)

    time.sleep(args.stability_seconds)
    second = _list_checkpoints(args, run_dir)
    stable = [
        updated
        for name, initial in first.items()
        if (updated := second.get(name)) is not None
        and initial.size == updated.size
        and initial.mtime == updated.mtime
        and updated.size > 0
    ]
    if not stable:
        return None
    return max(stable, key=lambda ckpt: ckpt.iteration)


def _copy_checkpoint(args: argparse.Namespace, run_dir: str, ckpt: RemoteCheckpoint) -> Path:
    local_run_dir = args.local_log_dir / run_dir
    local_run_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_run_dir / ckpt.name
    tmp_path = local_run_dir / f"{ckpt.name}.tmp"
    if local_path.exists() and local_path.stat().st_size == ckpt.size:
        return local_path

    remote_path = f"{args.remote_project}/{args.remote_log_dir}/{run_dir}/{ckpt.name}"
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir)
    host_tmp = f"/tmp/se3_remote_watch_{safe_run}_{ckpt.name}"
    copy_cmd = (
        f"kubectl cp -n {shlex.quote(args.namespace)} "
        f"{shlex.quote(args.pod + ':' + remote_path)} {shlex.quote(host_tmp)}"
    )
    if tmp_path.exists():
        tmp_path.unlink()
    _run(["ssh", args.host, f"rm -f {shlex.quote(host_tmp)}"], timeout=30)
    try:
        _run(["ssh", args.host, copy_cmd], timeout=180)
        _run(["scp", f"{args.host}:{host_tmp}", str(tmp_path)], timeout=180)
        copied_size = tmp_path.stat().st_size
        if copied_size != ckpt.size:
            raise RuntimeError(
                f"Copied checkpoint size mismatch for {ckpt.name}: "
                f"local={copied_size}, remote={ckpt.size}"
            )
        tmp_path.replace(local_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        _run(["ssh", args.host, f"rm -f {shlex.quote(host_tmp)}"], timeout=30)
    return local_path


def _viewer_task(args: argparse.Namespace) -> str:
    if args.viewer_task:
        return args.viewer_task
    if args.train_env:
        return _TRAIN_VIEW_TASKS.get(args.task, args.task)
    return args.task


def _checkpoint_iteration_from_path(checkpoint: Path) -> int:
    match = _MODEL_RE.match(checkpoint.name)
    return int(match.group(1)) if match else -1


def _launch_viewer(args: argparse.Namespace, checkpoint: Path) -> subprocess.Popen:
    viewer_task = _viewer_task(args)
    env = os.environ.copy()
    if viewer_task.endswith("-TrainView"):
        view_iter = str(_checkpoint_iteration_from_path(checkpoint))
        env[_TRAIN_VIEW_ITER_ENV] = view_iter
        print(f"[local-watch] {_TRAIN_VIEW_ITER_ENV}={view_iter}")
    if args.terrain_level is not None:
        env[_TRAIN_VIEW_TERRAIN_LEVEL_ENV] = str(args.terrain_level)
        print(f"[local-watch] {_TRAIN_VIEW_TERRAIN_LEVEL_ENV}={args.terrain_level}")
    if args.command_height is not None:
        env[_TRAIN_VIEW_COMMAND_HEIGHT_ENV] = str(args.command_height)
        print(f"[local-watch] {_TRAIN_VIEW_COMMAND_HEIGHT_ENV}={args.command_height}")
    cmd = [
        sys.executable,
        "-m",
        "se3_train.play",
        viewer_task,
        "--checkpoint-file",
        str(checkpoint),
        "--num-envs",
        str(args.num_envs),
        "--viewer",
        args.viewer,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.no_terminations:
        cmd.extend(["--no-terminations", "True"])
    print("[local-watch] launching viewer:")
    print("[local-watch] " + " ".join(cmd))
    return subprocess.Popen(cmd, env=env, start_new_session=(os.name != "nt"))


def _stop_viewer(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"[local-watch] stopping previous viewer pid={proc.pid}")
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy remote checkpoints and view the real training scene locally."
    )
    parser.add_argument("--host", default="target-via-phone")
    parser.add_argument("--namespace", default="gczx-project06")
    parser.add_argument("--pod", default="gpu-8-b6994457c-kv2rj")
    parser.add_argument(
        "--remote-project",
        default="/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg",
    )
    parser.add_argument("--remote-log-dir", default="logs/rsl_rl/se3_wheel_leg")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--local-log-dir", type=Path, default=Path("logs/remote_watch"))
    parser.add_argument("--task", default=_RECOVERY_STAND_TASK)
    parser.add_argument(
        "--viewer-task",
        default=None,
        help="Explicit viewer task override. Defaults to the training-view alias for --task.",
    )
    parser.add_argument(
        "--train-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the training-scene task alias. Kept enabled by default.",
    )
    parser.add_argument("--interval-iters", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--terrain-level",
        type=int,
        choices=range(10),
        default=None,
        metavar="0..9",
        help="固定 Stair TrainView 的地形 row: 0 约 5 cm, 9 约 20 cm。",
    )
    parser.add_argument(
        "--command-height",
        type=float,
        default=None,
        help="固定 Stair TrainView 的 height command；例如 0.37 用于检查最高站姿。",
    )
    parser.add_argument(
        "--stability-seconds",
        type=float,
        default=2.0,
        help="Require remote checkpoint size and mtime to stay unchanged for this long.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable terminations in the viewer. Default keeps training terminations enabled.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and copy the latest checkpoint, then exit without opening a viewer.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args)
    print(f"[local-watch] remote run: {run_dir}")
    print(f"[local-watch] viewer task: {_viewer_task(args)}")

    last_iter = -1
    viewer_proc: subprocess.Popen | None = None
    try:
        while True:
            latest = _latest_checkpoint(args, run_dir)
            if latest is None:
                print("[local-watch] no stable checkpoint yet; waiting...")
                time.sleep(args.poll_seconds)
                continue
            iteration = latest.iteration
            should_launch = last_iter < 0 or iteration - last_iter >= args.interval_iters
            if viewer_proc is not None and viewer_proc.poll() is not None:
                print(f"[local-watch] viewer exited with code {viewer_proc.returncode}")
                viewer_proc = None
                should_launch = True
            if should_launch:
                local_ckpt = _copy_checkpoint(args, run_dir, latest)
                print(f"[local-watch] checkpoint model_{iteration}.pt -> {local_ckpt}")
                if args.dry_run:
                    raise SystemExit(0)
                _stop_viewer(viewer_proc)
                viewer_proc = _launch_viewer(args, local_ckpt)
                last_iter = iteration
                if args.once:
                    raise SystemExit(viewer_proc.wait())
            time.sleep(args.poll_seconds)
    finally:
        _stop_viewer(viewer_proc)


if __name__ == "__main__":
    main()
