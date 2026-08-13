"""Mirror a remote checkpoint locally and open the real training viewer.

The remote PPO run stays headless. This script periodically copies the latest
stable ``model_*.pt`` from the pod into ``logs/remote_watch/<run>/`` and starts
local ``se3-play`` with one environment using the training-scene task alias.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_EXCLUDED_RUN_DIRS = {"base_model"}
# Kept for compatibility only. The default path now launches the original task
# name and asks that task to use its training env as play_env_cfg.
_LEGACY_TRAIN_VIEW_TASKS = {
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
_WATCH_USE_TRAIN_ENV_ENV = "SE3_WATCH_USE_TRAIN_ENV"
_WATCH_ITER_ENV = "SE3_WATCH_ITER"
_WATCH_TERRAIN_LEVEL_ENV = "SE3_WATCH_TERRAIN_LEVEL"
_WATCH_COMMAND_HEIGHT_ENV = "SE3_WATCH_COMMAND_HEIGHT"
_TRAIN_VIEW_ITER_ENV = "SE3_TRAIN_VIEW_ITER"
_TRAIN_VIEW_TERRAIN_LEVEL_ENV = "SE3_TRAIN_VIEW_TERRAIN_LEVEL"
_TRAIN_VIEW_COMMAND_HEIGHT_ENV = "SE3_TRAIN_VIEW_COMMAND_HEIGHT"
_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")
_SSH_STABILITY_OPTS = [
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=6",
    "-o",
    "TCPKeepAlive=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ConnectionAttempts=3",
]


@dataclass(frozen=True)
class RemoteCheckpoint:
    name: str
    iteration: int
    size: int
    mtime: str


def _run(cmd: list[str], *, capture: bool = False, timeout: int | None = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=capture,
            text=capture,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        if capture and exc.stdout:
            sys.stderr.write(exc.stdout)
        if capture and exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
    return result.stdout if capture else ""


def _run_with_retries(
    cmd: list[str],
    *,
    capture: bool = False,
    timeout: int | None = None,
    attempts: int = 3,
) -> str:
    last_exc: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _run(cmd, capture=capture, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_exc = exc
            if attempt == attempts:
                raise
            time.sleep(min(5 * attempt, 15))
    assert last_exc is not None
    raise last_exc


def _ssh_options_for_shell() -> str:
    return " ".join(shlex.quote(part) for part in _SSH_STABILITY_OPTS)


def _host_ssh_args(args: argparse.Namespace, command: str) -> list[str]:
    encoded = base64.b64encode(command.encode()).decode()
    remote_command = f"echo {shlex.quote(encoded)} | base64 -d | bash"
    if args.entry_host and args.inner_host:
        inner = (
            f"ssh {_ssh_options_for_shell()} "
            f"{shlex.quote(args.inner_host)} {shlex.quote(remote_command)}"
        )
        return ["ssh", *_SSH_STABILITY_OPTS, args.entry_host, inner]
    return ["ssh", *_SSH_STABILITY_OPTS, args.host, remote_command]


def _cleanup_entry_tmp(args: argparse.Namespace, entry_tmp: str) -> None:
    if not args.entry_host:
        return
    command = f"Remove-Item -Force -ErrorAction SilentlyContinue '{entry_tmp}'"
    with contextlib.suppress(subprocess.CalledProcessError):
        _run(["ssh", *_SSH_STABILITY_OPTS, args.entry_host, command], capture=True, timeout=120)


def _copy_host_file_to_local(
    args: argparse.Namespace,
    *,
    host_path: str,
    local_path: Path,
    entry_tmp: str,
) -> None:
    if not (args.entry_host and args.inner_host):
        _run_with_retries(
            ["scp", *_SSH_STABILITY_OPTS, f"{args.host}:{host_path}", str(local_path)],
            timeout=300,
        )
        return

    _cleanup_entry_tmp(args, entry_tmp)
    try:
        _run_with_retries(
            [
                "ssh",
                *_SSH_STABILITY_OPTS,
                args.entry_host,
                (
                    f"scp {_ssh_options_for_shell()} "
                    f"{shlex.quote(args.inner_host + ':' + host_path)} "
                    f"{shlex.quote(entry_tmp)}"
                ),
            ],
            timeout=300,
        )
        _run_with_retries(
            ["scp", *_SSH_STABILITY_OPTS, f"{args.entry_host}:{entry_tmp}", str(local_path)],
            timeout=300,
        )
    finally:
        _cleanup_entry_tmp(args, entry_tmp)


def _remote_shell(args: argparse.Namespace, command: str, *, timeout: int = 180) -> str:
    remote = (
        f"kubectl exec -n {shlex.quote(args.namespace)} {shlex.quote(args.pod)} "
        f"-- bash -lc {shlex.quote(command)}"
    )
    return _run_with_retries(_host_ssh_args(args, remote), capture=True, timeout=timeout)


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
    tmp_path = local_run_dir / f"{ckpt.name}.{os.getpid()}.tmp"
    if local_path.exists() and local_path.stat().st_size == ckpt.size:
        return local_path

    remote_path = f"{args.remote_project}/{args.remote_log_dir}/{run_dir}/{ckpt.name}"
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir)
    host_tmp = f"/tmp/se3_remote_watch_{safe_run}_{ckpt.name}.{os.getpid()}"
    entry_tmp = f"/tmp/se3_remote_watch_entry_{safe_run}_{ckpt.name}.{os.getpid()}"
    copy_cmd = (
        f"kubectl cp -n {shlex.quote(args.namespace)} "
        f"{shlex.quote(args.pod + ':' + remote_path)} {shlex.quote(host_tmp)}"
    )
    with contextlib.suppress(OSError):
        if tmp_path.exists():
            tmp_path.unlink()
    _run_with_retries(_host_ssh_args(args, f"rm -f {shlex.quote(host_tmp)}"), timeout=120)
    try:
        _run_with_retries(_host_ssh_args(args, copy_cmd), timeout=180)
        _copy_host_file_to_local(
            args,
            host_path=host_tmp,
            local_path=tmp_path,
            entry_tmp=entry_tmp,
        )
        copied_size = tmp_path.stat().st_size
        if copied_size != ckpt.size:
            raise RuntimeError(
                f"Copied checkpoint size mismatch for {ckpt.name}: "
                f"local={copied_size}, remote={ckpt.size}"
            )
        tmp_path.replace(local_path)
    finally:
        with contextlib.suppress(OSError):
            if tmp_path.exists():
                tmp_path.unlink()
        _run_with_retries(_host_ssh_args(args, f"rm -f {shlex.quote(host_tmp)}"), timeout=120)
    return local_path


def _latest_local_checkpoint(args: argparse.Namespace, run_dir: str) -> Path | None:
    local_run_dir = args.local_log_dir / run_dir
    if not local_run_dir.exists():
        return None
    checkpoints = [
        checkpoint
        for checkpoint in local_run_dir.glob("model_*.pt")
        if _checkpoint_iteration_from_path(checkpoint) >= 0
        and checkpoint.is_file()
        and checkpoint.stat().st_size > 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_iteration_from_path)


def _viewer_task(args: argparse.Namespace) -> str:
    if args.viewer_task:
        return args.viewer_task
    if args.legacy_train_view_alias:
        return _LEGACY_TRAIN_VIEW_TASKS.get(args.task, args.task)
    return args.task


def _checkpoint_iteration_from_path(checkpoint: Path) -> int:
    match = _MODEL_RE.match(checkpoint.name)
    return int(match.group(1)) if match else -1


def _launch_viewer(args: argparse.Namespace, checkpoint: Path) -> subprocess.Popen:
    viewer_task = _viewer_task(args)
    env = os.environ.copy()
    if args.train_env and not args.legacy_train_view_alias:
        env[_WATCH_USE_TRAIN_ENV_ENV] = "1"
        print(f"[local-watch] {_WATCH_USE_TRAIN_ENV_ENV}=1")
    if args.train_env or viewer_task.endswith("-TrainView"):
        view_iter = str(_checkpoint_iteration_from_path(checkpoint))
        env[_WATCH_ITER_ENV] = view_iter
        env[_TRAIN_VIEW_ITER_ENV] = view_iter
        print(f"[local-watch] {_WATCH_ITER_ENV}={view_iter}")
    if args.terrain_level is not None:
        env[_WATCH_TERRAIN_LEVEL_ENV] = str(args.terrain_level)
        env[_TRAIN_VIEW_TERRAIN_LEVEL_ENV] = str(args.terrain_level)
        print(f"[local-watch] {_WATCH_TERRAIN_LEVEL_ENV}={args.terrain_level}")
    if args.command_height is not None:
        env[_WATCH_COMMAND_HEIGHT_ENV] = str(args.command_height)
        env[_TRAIN_VIEW_COMMAND_HEIGHT_ENV] = str(args.command_height)
        print(
            f"[local-watch] {_WATCH_COMMAND_HEIGHT_ENV}={args.command_height} "
            "(fixed height override)"
        )
    elif args.train_env:
        print("[local-watch] command height follows the training curriculum")
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
    parser.add_argument("--host", default="a800")
    parser.add_argument(
        "--entry-host",
        default=None,
        help="两跳 SSH 的入口机；设置后需同时设置 --inner-host。",
    )
    parser.add_argument(
        "--inner-host",
        default=None,
        help="两跳 SSH 的训练机别名，例如从入口机再 ssh 到 a800。",
    )
    parser.add_argument("--namespace", default="gczx-project06")
    parser.add_argument("--pod", default="abbtask-79cdb78487-mgx44")
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
        help="Use the original task name with its training env as play_env_cfg.",
    )
    parser.add_argument(
        "--legacy-train-view-alias",
        action="store_true",
        help="Use the old *-TrainView task alias instead of launching the original task.",
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
        help=(
            "固定 Stair watch 的 height command；省略时按训练课程随机采样。"
            "例如 0.37 只用于检查最高站姿。"
        ),
    )
    parser.add_argument(
        "--stability-seconds",
        type=float,
        default=10.0,
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
    if (args.entry_host is None) != (args.inner_host is None):
        parser.error("--entry-host 和 --inner-host 必须同时设置")

    run_dir = _resolve_run_dir(args)
    print(f"[local-watch] remote run: {run_dir}")
    if args.entry_host and args.inner_host:
        print(f"[local-watch] ssh route: {args.entry_host} -> {args.inner_host}")
    print(f"[local-watch] viewer task: {_viewer_task(args)}")

    last_iter = -1
    viewer_proc: subprocess.Popen | None = None
    try:
        while True:
            try:
                latest = _latest_checkpoint(args, run_dir)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                local_ckpt = None if args.dry_run else _latest_local_checkpoint(args, run_dir)
                if local_ckpt is not None:
                    local_iter = _checkpoint_iteration_from_path(local_ckpt)
                    should_launch_local = (
                        last_iter < 0 or local_iter - last_iter >= args.interval_iters
                    )
                    if viewer_proc is not None and viewer_proc.poll() is not None:
                        print(f"[local-watch] viewer exited with code {viewer_proc.returncode}")
                        viewer_proc = None
                        should_launch_local = True
                    if should_launch_local:
                        print(
                            "[local-watch] remote checkpoint query failed; "
                            f"using local checkpoint model_{local_iter}.pt"
                        )
                        _stop_viewer(viewer_proc)
                        viewer_proc = _launch_viewer(args, local_ckpt)
                        last_iter = local_iter
                print(
                    "[local-watch] remote checkpoint query failed; "
                    f"retrying in {args.poll_seconds}s: {exc}",
                    file=sys.stderr,
                )
                time.sleep(args.poll_seconds)
                continue
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
                try:
                    local_ckpt = _copy_checkpoint(args, run_dir, latest)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    RuntimeError,
                ) as exc:
                    print(
                        "[local-watch] checkpoint copy failed; "
                        f"retrying in {args.poll_seconds}s: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(args.poll_seconds)
                    continue
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
