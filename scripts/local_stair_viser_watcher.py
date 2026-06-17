"""本机 Stair Viser 值守：同步远端 checkpoint，并在本机运行 native Viser。"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_RE = re.compile(r"^model_(\d+)\.pt$")
SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=NUL",
]


def short_command(cmd: list[str], *, limit: int = 360) -> str:
    text = subprocess.list2cmdline(cmd)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


@dataclass(frozen=True)
class CheckpointInfo:
    name: str
    iteration: int
    size: int
    mtime: str
    download_url: str | None = None


def run(
    cmd: list[str],
    *,
    capture: bool = True,
    timeout: float = 120.0,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=capture,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and result.returncode != 0:
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            f"command failed code={result.returncode}: {short_command(cmd)}\n"
            f"stdout={stdout}\nstderr={stderr}"
        )
    return result


def run_retry(
    cmd: list[str],
    *,
    capture: bool = True,
    timeout: float = 120.0,
    attempts: int = 3,
) -> subprocess.CompletedProcess[str]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run(cmd, capture=capture, timeout=timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(5.0 * attempt, 15.0))
    assert last_error is not None
    raise last_error


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def powershell_encoded_command(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    result = run(
        ["git", "credential", "fill"],
        capture=True,
        timeout=20.0,
        check=False,
        input_text="protocol=https\nhost=github.com\n\n",
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return fields.get("password") or None


def github_request(args: argparse.Namespace, url: str, *, accept: str) -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "se3-local-viser-watch",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=float(args.github_timeout_s)) as response:
        return response.read()


def github_json(args: argparse.Namespace, url: str) -> object:
    return json.loads(
        github_request(args, url, accept="application/vnd.github+json").decode("utf-8")
    )


def laptop_cmd(args: argparse.Namespace, *command: str) -> list[str]:
    return ["ssh", *SSH_OPTS, args.laptop_host, *command]


def laptop_powershell_cmd(args: argparse.Namespace, script: str) -> list[str]:
    return laptop_cmd(
        args,
        "powershell",
        "-NoProfile",
        "-EncodedCommand",
        powershell_encoded_command(script),
    )


def a800_cmd(args: argparse.Namespace, remote_command: str) -> list[str]:
    ps_args = ["-T", *SSH_OPTS, args.a800_host, remote_command]
    ps_array = ", ".join(ps_quote(part) for part in ps_args)
    script = f"""
$ErrorActionPreference = 'Stop'
$sshArgs = @({ps_array})
& ssh @sshArgs
exit $LASTEXITCODE
"""
    return laptop_powershell_cmd(args, script)


def remote_pod_shell(args: argparse.Namespace, shell_script: str, *, timeout: float = 120.0) -> str:
    payload = base64.b64encode(shell_script.encode("utf-8")).decode("ascii")
    remote = (
        f"printf %s {shlex.quote(payload)} | base64 -d | "
        f"kubectl -n {shlex.quote(args.namespace)} exec -i {shlex.quote(args.pod)} "
        "-- bash -s"
    )
    result = run_retry(a800_cmd(args, remote), timeout=timeout, attempts=args.ssh_attempts)
    return result.stdout


def latest_remote_checkpoint(args: argparse.Namespace) -> CheckpointInfo | None:
    script = (
        f"cd {shlex.quote(args.remote_project)}\n"
        f"for f in {shlex.quote(args.remote_log_dir + '/' + args.run_dir)}/model_*.pt; do\n"
        '  [ -f "$f" ] || continue\n'
        '  base="$(basename "$f")"\n'
        '  [[ "$base" =~ ^model_[0-9]+\\.pt$ ]] || continue\n'
        '  printf \'%s\\t%s\\t%s\\n\' "$base" "$(stat -c%s "$f")" "$(stat -c%Y "$f")"\n'
        "done | sort -V | tail -1\n"
    )
    output = remote_pod_shell(args, script, timeout=120.0).strip()
    if not output:
        return None
    parts = output.split("\t")
    if len(parts) != 3:
        raise RuntimeError(f"unexpected checkpoint line: {output}")
    match = MODEL_RE.match(parts[0])
    if match is None:
        raise RuntimeError(f"unexpected checkpoint name: {parts[0]}")
    return CheckpointInfo(
        name=parts[0],
        iteration=int(match.group(1)),
        size=int(parts[1]),
        mtime=parts[2],
    )


def stable_remote_checkpoint(args: argparse.Namespace) -> CheckpointInfo | None:
    first = latest_remote_checkpoint(args)
    if first is None or args.stability_seconds <= 0:
        return first
    time.sleep(args.stability_seconds)
    second = latest_remote_checkpoint(args)
    if second is None:
        return None
    if first.name == second.name and first.size == second.size and first.mtime == second.mtime:
        return second
    return None


def latest_github_checkpoint(args: argparse.Namespace) -> CheckpointInfo | None:
    url = (
        "https://api.github.com/repos/"
        f"{args.github_release_repo}/releases/tags/{args.github_release_tag}"
    )
    release = github_json(args, url)
    if not isinstance(release, dict):
        raise RuntimeError("unexpected GitHub release response")
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError("unexpected GitHub release assets response")
    candidates: list[CheckpointInfo] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", ""))
        match = MODEL_RE.match(name)
        if match is None:
            continue
        size = int(asset.get("size", 0) or 0)
        if size <= 0:
            continue
        api_url = str(asset.get("url", ""))
        if not api_url:
            continue
        candidates.append(
            CheckpointInfo(
                name=name,
                iteration=int(match.group(1)),
                size=size,
                mtime=str(asset.get("updated_at", "")),
                download_url=api_url,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda ckpt: ckpt.iteration)


def stable_github_checkpoint(args: argparse.Namespace) -> CheckpointInfo | None:
    first = latest_github_checkpoint(args)
    if first is None or args.stability_seconds <= 0:
        return first
    time.sleep(args.stability_seconds)
    second = latest_github_checkpoint(args)
    if second is None:
        return None
    if first.name == second.name and first.size == second.size and first.mtime == second.mtime:
        return second
    return None


def latest_local_checkpoint(args: argparse.Namespace) -> CheckpointInfo | None:
    local_run_dir = args.local_run_root / args.run_dir
    if not local_run_dir.exists():
        return None
    checkpoints: list[CheckpointInfo] = []
    for path in local_run_dir.glob("model_*.pt"):
        match = MODEL_RE.match(path.name)
        if match is None or not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size <= 0:
            continue
        checkpoints.append(
            CheckpointInfo(
                name=path.name,
                iteration=int(match.group(1)),
                size=stat.st_size,
                mtime=str(stat.st_mtime),
            )
        )
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda ckpt: ckpt.iteration)


def select_checkpoint(args: argparse.Namespace) -> tuple[CheckpointInfo | None, str]:
    if args.source == "local":
        return latest_local_checkpoint(args), "local"
    if args.source == "github-release":
        try:
            github = stable_github_checkpoint(args)
        except Exception as exc:
            print(
                "[local-viser-watch] GitHub release query failed; "
                f"falling back to local checkpoints: {exc}",
                file=sys.stderr,
            )
            return latest_local_checkpoint(args), "local"
        if github is not None:
            return github, "github-release"
        local = latest_local_checkpoint(args)
        if local is not None:
            print("[local-viser-watch] GitHub release has no checkpoint; using local checkpoint")
        return local, "local"
    try:
        remote = stable_remote_checkpoint(args)
    except Exception as exc:
        print(
            f"[local-viser-watch] remote query failed; falling back to local checkpoints: {exc}",
            file=sys.stderr,
        )
        return latest_local_checkpoint(args), "local"
    if remote is not None:
        return remote, "remote"
    local = latest_local_checkpoint(args)
    if local is not None:
        print("[local-viser-watch] remote has no stable checkpoint; using local checkpoint")
    return local, "local"


def terrain_level(args: argparse.Namespace) -> int:
    if args.terrain_level >= 0:
        return max(0, min(9, args.terrain_level))
    script = (
        f"if [ -f {shlex.quote(args.train_log)} ]; then\n"
        f"  grep -E 'Curriculum/terrain_levels/(move_up_height_mean|level_mean)' "
        f"{shlex.quote(args.train_log)} | tail -20\n"
        "fi\n"
    )
    try:
        output = remote_pod_shell(args, script, timeout=60.0)
    except Exception as exc:
        print(f"[local-viser-watch] terrain query failed, fallback level=3: {exc}", file=sys.stderr)
        return 3
    for line in reversed(output.splitlines()):
        match = re.search(r"level_mean:\s*([-+0-9.eE]+)", line)
        if match:
            return max(0, min(9, round(float(match.group(1)))))
    return 3


def ensure_laptop_checkpoint(args: argparse.Namespace, ckpt: CheckpointInfo) -> str:
    laptop_run_dir = args.laptop_run_root.replace("\\", "/").rstrip("/") + "/" + args.run_dir
    laptop_path = f"{laptop_run_dir}/{ckpt.name}"
    remote_path = f"{args.remote_project}/{args.remote_log_dir}/{args.run_dir}/{ckpt.name}"
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_dir)
    host_tmp = f"/tmp/se3_local_viser_{safe_run}_{ckpt.name}"

    check_script = (
        f"$p = {ps_quote(laptop_path)}; "
        f"if ((Test-Path $p) -and ((Get-Item $p).Length -eq {ckpt.size})) "
        "{ Write-Output 'READY' }"
    )
    ready = run_retry(
        laptop_powershell_cmd(args, check_script),
        timeout=60.0,
        attempts=args.ssh_attempts,
    ).stdout
    if "READY" in ready:
        return laptop_path

    mkdir_script = (
        f"New-Item -ItemType Directory -Force -Path {ps_quote(laptop_run_dir)} | Out-Null"
    )
    run_retry(
        laptop_powershell_cmd(args, mkdir_script),
        timeout=60.0,
        attempts=args.ssh_attempts,
    )

    copy_to_host = (
        f"rm -f {shlex.quote(host_tmp)} && "
        f"timeout 240s kubectl cp -n {shlex.quote(args.namespace)} "
        f"{shlex.quote(args.pod + ':' + remote_path)} {shlex.quote(host_tmp)} && "
        f"stat -c%s {shlex.quote(host_tmp)}"
    )
    host_size = (
        run_retry(
            a800_cmd(args, copy_to_host),
            timeout=300.0,
            attempts=args.ssh_attempts,
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    if int(host_size) != ckpt.size:
        raise RuntimeError(f"A800 temp size mismatch: {host_size} != {ckpt.size}")

    laptop_tmp = f"{laptop_path}.tmp"
    run_retry(
        laptop_cmd(
            args,
            "scp",
            "-C",
            *SSH_OPTS,
            f"{args.a800_host}:{host_tmp}",
            laptop_tmp,
        ),
        timeout=420.0,
        attempts=args.ssh_attempts,
    )
    verify_script = (
        f"$tmp = {ps_quote(laptop_tmp)}; $dst = {ps_quote(laptop_path)}; "
        f"if ((Get-Item $tmp).Length -ne {ckpt.size}) "
        "{ throw 'laptop tmp size mismatch' }; "
        "Move-Item -LiteralPath $tmp -Destination $dst -Force"
    )
    run_retry(
        laptop_powershell_cmd(args, verify_script),
        timeout=60.0,
        attempts=args.ssh_attempts,
    )
    run(a800_cmd(args, f"rm -f {shlex.quote(host_tmp)}"), timeout=60.0, check=False)
    return laptop_path


def copy_laptop_to_local(args: argparse.Namespace, ckpt: CheckpointInfo, laptop_path: str) -> Path:
    local_run_dir = args.local_run_root / args.run_dir
    local_run_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_run_dir / ckpt.name
    if local_path.exists() and local_path.stat().st_size == ckpt.size:
        return local_path.resolve()
    tmp_path = local_run_dir / f"{ckpt.name}.{os.getpid()}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        run_retry(
            [
                "scp",
                "-C",
                *SSH_OPTS,
                f"{args.laptop_host}:{laptop_path}",
                str(tmp_path),
            ],
            timeout=600.0,
            attempts=args.ssh_attempts,
        )
        if tmp_path.stat().st_size != ckpt.size:
            raise RuntimeError(f"local tmp size mismatch: {tmp_path.stat().st_size} != {ckpt.size}")
        tmp_path.replace(local_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return local_path.resolve()


def download_github_checkpoint(args: argparse.Namespace, ckpt: CheckpointInfo) -> Path:
    if not ckpt.download_url:
        raise RuntimeError(f"GitHub checkpoint {ckpt.name} has no download URL")
    local_run_dir = args.local_run_root / args.run_dir
    local_run_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_run_dir / ckpt.name
    if local_path.exists() and local_path.stat().st_size == ckpt.size:
        return local_path.resolve()
    tmp_path = local_run_dir / f"{ckpt.name}.{os.getpid()}.github.tmp"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        data = github_request(args, ckpt.download_url, accept="application/octet-stream")
        tmp_path.write_bytes(data)
        if tmp_path.stat().st_size != ckpt.size:
            raise RuntimeError(
                f"GitHub asset size mismatch for {ckpt.name}: "
                f"{tmp_path.stat().st_size} != {ckpt.size}"
            )
        tmp_path.replace(local_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return local_path.resolve()


def local_checkpoint_path(args: argparse.Namespace, ckpt: CheckpointInfo) -> Path:
    return (args.local_run_root / args.run_dir / ckpt.name).resolve()


def copy_checkpoint_to_local(
    args: argparse.Namespace, ckpt: CheckpointInfo, *, source: str
) -> tuple[Path, CheckpointInfo]:
    if source == "local":
        return local_checkpoint_path(args, ckpt), ckpt
    if source == "github-release":
        try:
            return download_github_checkpoint(args, ckpt), ckpt
        except Exception as exc:
            print(
                "[local-viser-watch] GitHub release download failed; "
                f"using latest local checkpoint if available: {exc}",
                file=sys.stderr,
            )
            local = latest_local_checkpoint(args)
            if local is None:
                raise
            return local_checkpoint_path(args, local), local
    try:
        laptop_path = ensure_laptop_checkpoint(args, ckpt)
        return copy_laptop_to_local(args, ckpt, laptop_path), ckpt
    except Exception as exc:
        print(
            "[local-viser-watch] remote copy failed; using latest local checkpoint if available: "
            f"{exc}",
            file=sys.stderr,
        )
        local = latest_local_checkpoint(args)
        if local is None:
            raise
        return local_checkpoint_path(args, local), local


def local_viewer_pids() -> list[int]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'se3-sim2sim|se3_sim2sim\\.cli' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = run(["powershell", "-NoProfile", "-Command", command], timeout=20.0, check=False)
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit() and int(line) != os.getpid():
            pids.append(int(line))
    return pids


def stop_viewer(proc: subprocess.Popen[str] | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    for old in local_viewer_pids():
        if os.name == "nt":
            run(["taskkill", "/PID", str(old), "/T", "/F"], timeout=30.0, check=False)
        else:
            with contextlib.suppress(OSError):
                os.kill(old, signal.SIGTERM)


def start_viewer(
    args: argparse.Namespace, checkpoint: Path, iteration: int, level: int
) -> subprocess.Popen[str]:
    log_dir = args.viewer_log_root / args.run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "viser.out.log").open("w", encoding="utf-8", errors="replace")
    stderr = (log_dir / "viser.err.log").open("w", encoding="utf-8", errors="replace")
    command = [
        "uv",
        "run",
        "se3-sim2sim",
        "--checkpoint",
        str(checkpoint),
        "--model-variant",
        "closedchain",
        "--sim-dt",
        str(args.sim_dt),
        "--control-decimation",
        str(args.control_decimation),
        "--viewer",
        "viser",
        "--device",
        args.device,
        "--print-every",
        "0",
        "--stair-terrain",
        "--stair-terrain-level",
        str(level),
        "--stair-ctbc",
        "--command",
        str(args.command_vx),
        "0",
        "0",
        "0",
        str(args.command_height),
        "0",
        "0",
        "0",
    ]
    if args.fixed_ctbc_iter:
        command.extend(["--stair-ctbc-iter", str(iteration)])
    print("[local-viser-watch] launching viewer:")
    print("[local-viser-watch] " + subprocess.list2cmdline(command))
    return subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=Path.cwd())


def wait_for_viser(timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = run(
            [
                "curl.exe",
                "-o",
                "NUL",
                "-s",
                "-w",
                "%{http_code}",
                "--max-time",
                "5",
                "http://127.0.0.1:8080/",
            ],
            timeout=10.0,
            check=False,
        )
        if result.stdout.strip() == "200":
            return True
        time.sleep(2.0)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laptop-host", default="laptop-imgpi2nm-shanghai")
    parser.add_argument("--a800-host", default="a800")
    parser.add_argument("--namespace", default="gczx-project06")
    parser.add_argument("--pod", default="abbtask-79cdb78487-mgx44")
    parser.add_argument(
        "--remote-project",
        default="/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_run_f4ebc01_20260617_3k",
    )
    parser.add_argument("--remote-log-dir", default="logs/rsl_rl/se3_wheel_leg")
    parser.add_argument(
        "--run-dir",
        default="2026-06-17_10-13-55_stair3k_ctbcslow_m4999_8gpu4096_f4ebc01_20260617_3k",
    )
    parser.add_argument(
        "--train-log",
        default="/tmp/train_stair3k_ctbcslow_m4999_8gpu4096_f4ebc01_20260617_3k.log",
    )
    parser.add_argument("--laptop-run-root", default="E:/se3_stair_viewer/logs/remote_watch")
    parser.add_argument("--local-run-root", type=Path, default=Path("logs/remote_watch"))
    parser.add_argument("--viewer-log-root", type=Path, default=Path("logs/local_viser"))
    parser.add_argument("--interval-iters", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stability-seconds", type=float, default=2.0)
    parser.add_argument("--terrain-level", type=int, default=-1)
    parser.add_argument(
        "--source",
        choices=("github-release", "remote", "local"),
        default="github-release",
    )
    parser.add_argument(
        "--github-release-repo",
        default="3SE-Competitive-Robotics-Team/se3_checkpoint_exchange",
    )
    parser.add_argument(
        "--github-release-tag",
        default="run-20260617-101355-stair3k-ctbcslow-f4ebc01",
    )
    parser.add_argument("--github-timeout-s", type=float, default=120.0)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--control-decimation", type=int, default=4)
    parser.add_argument(
        "--fixed-ctbc-iter",
        action="store_true",
        default=True,
        help="按 checkpoint iteration 固定 CTBC 退火阶段，默认开启。",
    )
    parser.add_argument(
        "--live-ctbc-iter",
        dest="fixed_ctbc_iter",
        action="store_false",
        help="让 Viser 从本地 rollout step 重新计算 CTBC 阶段，仅用于调试前馈。",
    )
    parser.add_argument("--command-vx", type=float, default=1.2)
    parser.add_argument("--command-height", type=float, default=0.32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ssh-attempts", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    last_synced_iter = -1
    last_viewer_iter = -1
    viewer_proc: subprocess.Popen[str] | None = None
    while True:
        try:
            ckpt, source = select_checkpoint(args)
            if ckpt is None:
                print("[local-viser-watch] no stable checkpoint yet")
            else:
                local_path = local_checkpoint_path(args, ckpt)
                if source != "local" and ckpt.iteration > last_synced_iter:
                    print(
                        f"[local-viser-watch] syncing {ckpt.name} from {source} "
                        f"iter={ckpt.iteration} size={ckpt.size}"
                    )
                    local_path, ckpt = copy_checkpoint_to_local(args, ckpt, source=source)
                    last_synced_iter = ckpt.iteration
                    print(f"[local-viser-watch] local checkpoint {local_path}")
                elif source == "local":
                    last_synced_iter = max(last_synced_iter, ckpt.iteration)
                    print(
                        f"[local-viser-watch] local latest {ckpt.name} "
                        f"iter={ckpt.iteration} size={ckpt.size}"
                    )

                viewer_exited = viewer_proc is not None and viewer_proc.poll() is not None
                should_restart = not args.no_viewer and (
                    last_viewer_iter < 0
                    or ckpt.iteration - last_viewer_iter >= args.interval_iters
                    or viewer_exited
                )
                if should_restart:
                    level = (
                        terrain_level(args)
                        if source == "remote"
                        else max(0, min(9, args.terrain_level if args.terrain_level >= 0 else 3))
                    )
                    stop_viewer(viewer_proc)
                    viewer_proc = start_viewer(args, local_path, ckpt.iteration, level)
                    if wait_for_viser():
                        print("[local-viser-watch] viser ready http://127.0.0.1:8080/")
                    else:
                        print(
                            "[local-viser-watch] viser did not become ready in time",
                            file=sys.stderr,
                        )
                    last_viewer_iter = ckpt.iteration
        except Exception as exc:
            print(f"[local-viser-watch] error: {exc}", file=sys.stderr)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
