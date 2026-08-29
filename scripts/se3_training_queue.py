"""单机单卡训练队列。

队列只负责串行调度已经存在的 ``se3-train`` 任务。训练源码仍由 git 管理，
每个任务在提交时固定 commit；队列状态、日志和训练产物统一写入外部持久目录。
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,
    name TEXT NOT NULL,
    task TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    num_envs INTEGER NOT NULL,
    iterations INTEGER NOT NULL,
    save_interval INTEGER NOT NULL,
    smoke INTEGER NOT NULL,
    wandb_mode TEXT NOT NULL,
    extra_args_json TEXT NOT NULL,
    extra_env_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    pid INTEGER,
    pgid INTEGER,
    log_path TEXT,
    run_dir TEXT,
    exit_code INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_id ON jobs(status, id);
"""

_ACTIVE_STATUSES = ("starting", "running", "cancelling")
_SENSITIVE_ENV_RE = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Settings:
    """队列运行目录。"""

    root: Path
    repo: Path
    uv: Path

    @property
    def db_path(self) -> Path:
        return self.root / "queue.db"

    @property
    def checkpoint_root(self) -> Path:
        return self.root / "checkpoints"

    @property
    def job_log_root(self) -> Path:
        return self.root / "logs" / "jobs"

    @property
    def worker_log_path(self) -> Path:
        return self.root / "logs" / "queue-worker.log"

    @property
    def wandb_root(self) -> Path:
        return self.root / "wandb"

    @property
    def smoke_root(self) -> Path:
        return self.root / "smoke"

    @property
    def lock_path(self) -> Path:
        return self.root / "worker.lock"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized[:80] or "job"


def _settings(args: argparse.Namespace) -> Settings:
    root_value = args.root or os.environ.get("SE3_QUEUE_ROOT")
    repo_value = args.repo or os.environ.get("SE3_QUEUE_REPO")
    uv_value = args.uv or os.environ.get("SE3_QUEUE_UV")
    missing = [
        name
        for name, value in (
            ("SE3_QUEUE_ROOT", root_value),
            ("SE3_QUEUE_REPO", repo_value),
            ("SE3_QUEUE_UV", uv_value),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"缺少队列配置: {', '.join(missing)}")
    root = Path(root_value).expanduser()
    repo = Path(repo_value).expanduser()
    uv = Path(uv_value).expanduser()
    return Settings(root=root.resolve(), repo=repo.resolve(), uv=uv.resolve())


def _prepare(settings: Settings) -> None:
    for path in (
        settings.root,
        settings.checkpoint_root,
        settings.job_log_root,
        settings.wandb_root,
        settings.smoke_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _connect(settings: Settings) -> sqlite3.Connection:
    _prepare(settings)
    connection = sqlite3.connect(settings.db_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(_SCHEMA)
    return connection


def _worker_log(settings: Settings, message: str) -> None:
    _prepare(settings)
    line = f"{_now()} {message}\n"
    with settings.worker_log_path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
    print(line, end="", flush=True)


def _git_output(settings: Settings, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(settings.repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"环境变量必须使用 NAME=VALUE: {value}")
        name, item = value.split("=", 1)
        if not _ENV_NAME_RE.fullmatch(name):
            raise SystemExit(f"环境变量名无效: {name}")
        if _SENSITIVE_ENV_RE.search(name):
            raise SystemExit(f"拒绝把敏感环境变量写入队列数据库: {name}")
        result[name] = item
    return result


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(zip(row.keys(), row, strict=True))


def _job(settings: Settings, job_id: int) -> sqlite3.Row:
    with _connect(settings) as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise SystemExit(f"任务不存在: {job_id}")
    return row


def _update_job(settings: Settings, job_id: int, **fields: object) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = [*fields.values(), job_id]
    with _connect(settings) as connection:
        connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)


def _cmd_submit(args: argparse.Namespace, settings: Settings) -> None:
    commit = args.commit or _git_output(settings, "rev-parse", "HEAD")
    if _git_output(settings, "cat-file", "-t", commit) != "commit":
        raise SystemExit(f"不是有效 git commit: {commit}")
    extra_env = _parse_env(args.env)
    created_at = _now()
    with _connect(settings) as connection:
        cursor = connection.execute(
            """
            INSERT INTO jobs (
                status, name, task, git_commit, num_envs, iterations,
                save_interval, smoke, wandb_mode, extra_args_json,
                extra_env_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "queued",
                args.name or args.task,
                args.task,
                commit,
                args.envs,
                args.iterations,
                args.save_interval,
                int(args.smoke),
                args.wandb_mode,
                json.dumps(args.extra_arg, ensure_ascii=False),
                json.dumps(extra_env, ensure_ascii=False),
                created_at,
            ),
        )
        job_id = int(cursor.lastrowid)
    print(f"已提交任务 {job_id}: {args.task} @ {commit[:12]}")


def _cmd_list(args: argparse.Namespace, settings: Settings) -> None:
    with _connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (args.limit,)
        ).fetchall()
    if not rows:
        print("队列为空")
        return
    print(f"{'ID':>5}  {'STATUS':<11} {'PID':>7}  {'TASK':<52} CREATED")
    for row in rows:
        pid = "-" if row["pid"] is None else str(row["pid"])
        print(
            f"{row['id']:>5}  {row['status']:<11} {pid:>7}  "
            f"{row['task'][:52]:<52} {row['created_at']}"
        )


def _cmd_show(args: argparse.Namespace, settings: Settings) -> None:
    row = _job(settings, args.job_id)
    print(json.dumps(_row_dict(row), ensure_ascii=False, indent=2))


def _cmd_tail(args: argparse.Namespace, settings: Settings) -> None:
    row = _job(settings, args.job_id)
    if not row["log_path"]:
        raise SystemExit("任务还没有日志")
    path = Path(row["log_path"])
    if not path.exists():
        raise SystemExit(f"日志不存在: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-args.lines :]))


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _external_training_pids() -> list[int]:
    result: list[int] = []
    proc_root = Path("/proc")
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"se3-train" in command:
            result.append(int(path.name))
    return sorted(result)


def _claim_next(settings: Settings) -> sqlite3.Row | None:
    with _connect(settings) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        connection.execute(
            "UPDATE jobs SET status = 'starting', started_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        connection.commit()
    return _job(settings, int(row["id"]))


def _clean_environment(settings: Settings, row: sqlite3.Row) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("SE3_LOGGER", None)
    environment.pop("SE3_SMOKE", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "WANDB_DIR": str(settings.wandb_root),
            "WANDB_MODE": row["wandb_mode"],
        }
    )
    environment.update(json.loads(row["extra_env_json"]))
    return environment


def _run_name(row: sqlite3.Row) -> str:
    return f"queue-{row['id']:06d}-{_slug(row['name'])}"


def _smoke_command(settings: Settings, row: sqlite3.Row) -> list[str]:
    return [
        str(settings.uv),
        "run",
        "--frozen",
        "se3-train",
        row["task"],
        "--log-root",
        str(settings.smoke_root),
        "--env.scene.num-envs",
        "1",
        "--gpu-ids",
        "None",
    ]


def _train_command(settings: Settings, row: sqlite3.Row) -> list[str]:
    command = [
        str(settings.uv),
        "run",
        "--frozen",
        "se3-train",
        row["task"],
        "--log-root",
        str(settings.checkpoint_root),
        "--env.scene.num-envs",
        str(row["num_envs"]),
        "--agent.max-iterations",
        str(row["iterations"]),
        "--agent.save-interval",
        str(row["save_interval"]),
        "--agent.run-name",
        _run_name(row),
    ]
    command.extend(json.loads(row["extra_args_json"]))
    return command


def _find_run_dir(settings: Settings, row: sqlite3.Row) -> Path | None:
    experiment_root = settings.checkpoint_root / row["task"]
    if not experiment_root.is_dir():
        return None
    candidates = [
        path for path in experiment_root.iterdir() if path.is_dir() and _run_name(row) in path.name
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _stream_command(
    settings: Settings,
    row: sqlite3.Row,
    command: list[str],
    environment: dict[str, str],
    log_stream: Any,
) -> int:
    print(f"[{_now()}] command={json.dumps(command, ensure_ascii=False)}", file=log_stream)
    log_stream.flush()
    process = subprocess.Popen(
        command,
        cwd=settings.repo,
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    while process.poll() is None:
        run_dir = _find_run_dir(settings, row)
        if run_dir is not None:
            _update_job(settings, int(row["id"]), run_dir=str(run_dir))
        time.sleep(2)
    return int(process.returncode)


def _execute_job(settings: Settings, job_id: int) -> int:
    row = _job(settings, job_id)
    log_path = settings.job_log_root / f"job-{job_id:06d}.log"
    _update_job(settings, job_id, log_path=str(log_path), status="running")
    environment = _clean_environment(settings, row)
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
            print(f"[{_now()}] 开始任务 {job_id}: {row['task']}", file=log_stream)
            current_commit = _git_output(settings, "rev-parse", "HEAD")
            if current_commit != row["git_commit"]:
                raise RuntimeError(
                    f"远端 commit 不一致: current={current_commit}, expected={row['git_commit']}"
                )
            dirty = _git_output(settings, "status", "--porcelain", "--untracked-files=no")
            if dirty:
                raise RuntimeError("远端仓库存在已跟踪文件改动，拒绝启动")

            if row["smoke"]:
                smoke_env = dict(environment)
                smoke_env.update(
                    {"SE3_SMOKE": "1", "SE3_LOGGER": "tensorboard", "WANDB_MODE": "offline"}
                )
                smoke_result = _stream_command(
                    settings, row, _smoke_command(settings, row), smoke_env, log_stream
                )
                if smoke_result != 0:
                    raise RuntimeError(f"smoke 失败，退出码 {smoke_result}")

            result = _stream_command(
                settings, row, _train_command(settings, row), environment, log_stream
            )
            cancel_requested = bool(_job(settings, job_id)["cancel_requested"])
            status = "cancelled" if cancel_requested else ("succeeded" if result == 0 else "failed")
            _update_job(
                settings,
                job_id,
                status=status,
                finished_at=_now(),
                exit_code=result,
                message=None if result == 0 else f"训练退出码 {result}",
            )
            return result
    except KeyboardInterrupt:
        _update_job(
            settings,
            job_id,
            status="cancelled",
            finished_at=_now(),
            exit_code=130,
            message="收到取消信号",
        )
        return 130
    except Exception as exc:
        _update_job(
            settings,
            job_id,
            status="failed",
            finished_at=_now(),
            exit_code=1,
            message=str(exc),
        )
        with log_path.open("a", encoding="utf-8") as log_stream:
            print(f"[{_now()}] ERROR: {exc}", file=log_stream)
        return 1


def _recover_active_job(settings: Settings) -> bool:
    with _connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs WHERE status IN ('starting', 'running', 'cancelling') ORDER BY id"
        ).fetchall()
    for row in rows:
        if _process_alive(row["pid"]):
            return True
        if row["status"] == "starting" and row["pid"] is None:
            _update_job(
                settings,
                int(row["id"]),
                status="queued",
                started_at=None,
                message="worker 启动前中断，已重新排队",
            )
            continue
        _update_job(
            settings,
            int(row["id"]),
            status="failed",
            finished_at=_now(),
            message="执行器消失且没有写入最终状态",
        )
    return False


def _cmd_worker(args: argparse.Namespace, settings: Settings) -> None:
    _prepare(settings)
    with settings.lock_path.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有 queue worker 在运行")
            return
        _worker_log(settings, "queue worker 启动")
        last_wait_reason = ""
        while True:
            if _recover_active_job(settings):
                reason = "等待已启动的队列任务结束"
                if reason != last_wait_reason:
                    _worker_log(settings, reason)
                    last_wait_reason = reason
                time.sleep(args.poll_seconds)
                continue

            external_pids = _external_training_pids()
            if external_pids:
                reason = f"等待外部 se3-train 结束: pids={external_pids}"
                if reason != last_wait_reason:
                    _worker_log(settings, reason)
                    last_wait_reason = reason
                time.sleep(args.poll_seconds)
                continue

            row = _claim_next(settings)
            if row is None:
                last_wait_reason = ""
                time.sleep(args.poll_seconds)
                continue

            job_id = int(row["id"])
            log_path = settings.job_log_root / f"job-{job_id:06d}.log"
            executor = [
                str(settings.uv),
                "run",
                "--no-project",
                "python",
                str(Path(__file__).resolve()),
                "--root",
                str(settings.root),
                "--repo",
                str(settings.repo),
                "--uv",
                str(settings.uv),
                "_execute",
                str(job_id),
            ]
            with log_path.open("a", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    executor,
                    cwd=settings.repo,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            _update_job(
                settings,
                job_id,
                status="running",
                pid=process.pid,
                pgid=process.pid,
                log_path=str(log_path),
            )
            _worker_log(settings, f"启动任务 {job_id}, executor_pid={process.pid}")
            process.wait()
            last_wait_reason = ""


def _cmd_cancel(args: argparse.Namespace, settings: Settings) -> None:
    row = _job(settings, args.job_id)
    if row["status"] == "queued":
        _update_job(
            settings,
            args.job_id,
            status="cancelled",
            finished_at=_now(),
            cancel_requested=1,
            message="排队阶段取消",
        )
        print(f"已取消排队任务 {args.job_id}")
        return
    if row["status"] not in _ACTIVE_STATUSES:
        raise SystemExit(f"任务状态 {row['status']} 不能取消")
    pgid = row["pgid"]
    if not pgid:
        raise SystemExit("任务尚未记录进程组，请稍后重试")
    _update_job(settings, args.job_id, status="cancelling", cancel_requested=1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(int(pgid), signal.SIGINT)
    print(f"已向任务 {args.job_id} 的进程组发送 SIGINT")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单机单卡 se3-train 任务队列")
    parser.add_argument("--root", help="队列持久目录；也可用 SE3_QUEUE_ROOT")
    parser.add_argument("--repo", help="训练仓库；也可用 SE3_QUEUE_REPO")
    parser.add_argument("--uv", help="uv 路径；也可用 SE3_QUEUE_UV")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="提交训练任务")
    submit.add_argument("task")
    submit.add_argument("--name")
    submit.add_argument("--commit")
    submit.add_argument("--envs", type=int, default=8192)
    submit.add_argument("--iterations", type=int, default=5000)
    submit.add_argument("--save-interval", type=int, default=200)
    submit.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    submit.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    submit.add_argument("--extra-arg", action="append", default=[])
    submit.add_argument("--env", action="append", default=[])

    listing = subparsers.add_parser("list", help="查看任务队列")
    listing.add_argument("--limit", type=int, default=50)

    show = subparsers.add_parser("show", help="查看一个任务")
    show.add_argument("job_id", type=int)

    tail = subparsers.add_parser("tail", help="查看任务日志")
    tail.add_argument("job_id", type=int)
    tail.add_argument("--lines", type=int, default=100)

    cancel = subparsers.add_parser("cancel", help="取消任务")
    cancel.add_argument("job_id", type=int)

    worker = subparsers.add_parser("worker", help="运行单 GPU worker")
    worker.add_argument("--poll-seconds", type=float, default=10.0)

    execute = subparsers.add_parser("_execute")
    execute.add_argument("job_id", type=int)
    return parser


def main() -> None:
    """执行队列命令。"""

    parser = _build_parser()
    args = parser.parse_args()
    settings = _settings(args)
    _prepare(settings)
    handlers = {
        "submit": _cmd_submit,
        "list": _cmd_list,
        "show": _cmd_show,
        "tail": _cmd_tail,
        "cancel": _cmd_cancel,
        "worker": _cmd_worker,
    }
    if args.command == "_execute":
        raise SystemExit(_execute_job(settings, args.job_id))
    handlers[args.command](args, settings)


if __name__ == "__main__":
    main()
