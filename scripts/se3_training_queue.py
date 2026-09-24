"""单机 GPU-aware 训练队列。

任务独占申请 1、2 或 4 张 GPU。worker 在空闲 GPU 上并发装箱；当队首任务暂时放不下时
停止启动后续任务，让现有任务自然排空，避免多卡任务长期饥饿。训练源码仍由 git 管理，
每个任务在提交时固定 commit；队列状态、日志和训练产物统一写入外部持久目录。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - worker 只在 Linux 运行
    fcntl = None  # type: ignore[assignment]

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
    gpu_count INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 0,
    assigned_gpu_ids_json TEXT,
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
_GPU_COUNT_CHOICES = (1, 2, 4)
_QUEUE_ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "SE3_LOGGER",
    "SE3_SMOKE",
    "VIRTUAL_ENV",
    "WANDB_DIR",
    "WANDB_MODE",
}
_SENSITIVE_ENV_RE = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Settings:
    """队列运行目录与可选 GPU 设备范围。"""

    root: Path
    repo: Path
    uv: Path
    gpu_ids: tuple[int, ...] = ()
    training_log_root: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.root / "queue.db"

    @property
    def checkpoint_root(self) -> Path:
        return self.training_log_root or self.root / "checkpoints"

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


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    """解析逗号分隔的物理 GPU 编号，并拒绝重复或负数。"""
    try:
        gpu_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit(f"GPU 编号必须是逗号分隔的非负整数: {value}") from error
    if not gpu_ids:
        raise SystemExit("GPU 设备列表不得为空")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise SystemExit(f"GPU 编号不得为负数: {value}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit(f"GPU 编号不得重复: {value}")
    return gpu_ids


def _settings(args: argparse.Namespace) -> Settings:
    root_value = args.root or os.environ.get("SE3_QUEUE_ROOT")
    repo_value = args.repo or os.environ.get("SE3_QUEUE_REPO")
    uv_value = args.uv or os.environ.get("SE3_QUEUE_UV")
    log_root_value = args.log_root or os.environ.get("SE3_QUEUE_LOG_ROOT")
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
    training_log_root = Path(log_root_value).expanduser().resolve() if log_root_value else None
    devices_value = args.devices or os.environ.get("SE3_QUEUE_GPU_IDS")
    gpu_ids = _parse_gpu_ids(devices_value) if devices_value else ()
    return Settings(
        root=root.resolve(),
        repo=repo.resolve(),
        uv=uv.resolve(),
        gpu_ids=gpu_ids,
        training_log_root=training_log_root,
    )


def _prepare(settings: Settings) -> None:
    for path in (
        settings.root,
        settings.checkpoint_root,
        settings.job_log_root,
        settings.wandb_root,
        settings.smoke_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    """打开一次短事务，并确保连接在离开作用域时关闭。"""
    _prepare(settings)
    connection = sqlite3.connect(settings.db_path, timeout=30.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(_SCHEMA)
        _migrate_schema(connection)
        connection.commit()
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
    finally:
        connection.close()


def _migrate_schema(connection: sqlite3.Connection) -> None:
    """向后兼容旧单卡队列数据库，不破坏已有任务。"""
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(jobs)").fetchall()}
    migrations = {
        "gpu_count": "ALTER TABLE jobs ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1",
        "priority": "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0",
        "assigned_gpu_ids_json": "ALTER TABLE jobs ADD COLUMN assigned_gpu_ids_json TEXT",
    }
    for name, statement in migrations.items():
        if name not in columns:
            connection.execute(statement)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_schedule ON jobs(status, priority DESC, id)"
    )


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


def _tracked_training_changes(settings: Settings) -> tuple[str, ...]:
    """返回影响训练仓库的改动；独立 sim2x checkout 不阻塞训练。"""
    output = _git_output(settings, "diff-index", "--name-only", "HEAD", "--")
    return tuple(path for path in output.splitlines() if path and path != "submodules/se3-sim2x")


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
        if name in _QUEUE_ENV_NAMES:
            raise SystemExit(f"环境变量由队列管理，禁止覆盖: {name}")
        result[name] = item
    return result


def _validate_extra_args(values: list[str]) -> None:
    """拒绝绕过队列 GPU 隔离的训练参数。"""
    for value in values:
        if value == "--gpu-ids" or value.startswith("--gpu-ids="):
            raise SystemExit("--gpu-ids 由队列根据分配结果生成，不能通过 --extra-arg 覆盖")


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(zip(row.keys(), row, strict=True))


def _assigned_gpu_ids(row: sqlite3.Row) -> tuple[int, ...]:
    """读取数据库中的物理 GPU 租约。"""
    encoded = row["assigned_gpu_ids_json"]
    if not encoded:
        return ()
    values = json.loads(encoded)
    if not isinstance(values, list) or any(not isinstance(value, int) for value in values):
        raise RuntimeError(f"任务 {row['id']} 的 GPU 租约损坏: {encoded}")
    return tuple(values)


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
    for name, value in (
        ("--envs", args.envs),
        ("--iterations", args.iterations),
        ("--save-interval", args.save_interval),
    ):
        if value <= 0:
            raise SystemExit(f"{name} 必须为正整数")
    if settings.gpu_ids and args.gpus > len(settings.gpu_ids):
        raise SystemExit(f"任务请求 {args.gpus} GPU，但当前配置只管理 {settings.gpu_ids}")
    commit = args.commit or _git_output(settings, "rev-parse", "HEAD")
    if _git_output(settings, "cat-file", "-t", commit) != "commit":
        raise SystemExit(f"不是有效 git commit: {commit}")
    extra_env = _parse_env(args.env)
    _validate_extra_args(args.extra_arg)
    created_at = _now()
    with _connect(settings) as connection:
        cursor = connection.execute(
            """
            INSERT INTO jobs (
                status, name, task, git_commit, num_envs, iterations,
                save_interval, smoke, wandb_mode, extra_args_json,
                extra_env_json, gpu_count, priority, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                args.gpus,
                args.priority,
                created_at,
            ),
        )
        job_id = int(cursor.lastrowid)
    print(
        f"已提交任务 {job_id}: {args.task} @ {commit[:12]} "
        f"gpus={args.gpus} priority={args.priority}"
    )


def _cmd_list(args: argparse.Namespace, settings: Settings) -> None:
    with _connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (args.limit,)
        ).fetchall()
    if not rows:
        print("队列为空")
        return
    print(f"{'ID':>5}  {'STATUS':<11} {'REQ':>3} {'GPU':<9} {'PID':>7}  {'TASK':<46} CREATED")
    for row in rows:
        pid = "-" if row["pid"] is None else str(row["pid"])
        assigned = ",".join(str(value) for value in _assigned_gpu_ids(row)) or "-"
        print(
            f"{row['id']:>5}  {row['status']:<11} {row['gpu_count']:>3} "
            f"{assigned:<9} {pid:>7}  {row['task'][:46]:<46} {row['created_at']}"
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


def _cmd_resources(_args: argparse.Namespace, settings: Settings) -> None:
    """显示 worker 当前可调度资源与活动租约。"""
    managed_gpu_ids = settings.gpu_ids or _discover_gpu_ids()
    active_rows = _recover_active_jobs(settings)
    leased_gpu_ids = _leased_gpu_ids(active_rows, managed_gpu_ids)
    active_pgids = {int(row["pgid"]) for row in active_rows if row["pgid"] is not None}
    external_pids = _external_training_pids(active_pgids)
    hardware_busy_gpu_ids = _busy_gpu_ids(managed_gpu_ids)
    schedulable_busy_gpu_ids = (
        set(managed_gpu_ids) if external_pids else leased_gpu_ids | hardware_busy_gpu_ids
    )
    print(
        json.dumps(
            {
                "managed_gpu_ids": managed_gpu_ids,
                "free_gpu_ids": tuple(
                    gpu_id for gpu_id in managed_gpu_ids if gpu_id not in schedulable_busy_gpu_ids
                ),
                "leased_gpu_ids": tuple(sorted(leased_gpu_ids)),
                "hardware_busy_gpu_ids": tuple(sorted(hardware_busy_gpu_ids)),
                "external_training_pids": external_pids,
                "leases": {str(row["id"]): _assigned_gpu_ids(row) for row in active_rows},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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


def _external_training_pids(excluded_pgids: set[int] | None = None) -> list[int]:
    """查找不属于队列租约的手工 ``se3-train`` 进程。"""
    excluded_pgids = excluded_pgids or set()
    result: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return result
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"se3-train" not in command:
            continue
        pid = int(path.name)
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if pgid not in excluded_pgids:
            result.append(pid)
    return sorted(result)


def _nvidia_smi_output(*arguments: str) -> str:
    """执行只读的 ``nvidia-smi`` 查询，并统一错误信息。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi 查询失败: {error}") from error
    return result.stdout


def _parse_gpu_inventory(output: str) -> dict[str, int]:
    """解析 ``index, uuid`` 查询结果。"""
    inventory: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            raise RuntimeError(f"无法解析 GPU inventory: {line}")
        inventory[parts[1]] = int(parts[0])
    return inventory


def _discover_gpu_ids() -> tuple[int, ...]:
    """从 NVIDIA 驱动发现当前容器可见的物理 GPU 编号。"""
    output = _nvidia_smi_output(
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    )
    inventory = _parse_gpu_inventory(output)
    if not inventory:
        raise RuntimeError("当前容器没有可见 GPU")
    return tuple(sorted(inventory.values()))


def _busy_gpu_ids(managed_gpu_ids: tuple[int, ...]) -> set[int]:
    """返回存在 NVIDIA compute process 的受管 GPU。"""
    inventory = _parse_gpu_inventory(
        _nvidia_smi_output(
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        )
    )
    app_output = _nvidia_smi_output(
        "--query-compute-apps=gpu_uuid",
        "--format=csv,noheader,nounits",
    )
    busy = {
        inventory[line.strip()] for line in app_output.splitlines() if line.strip() in inventory
    }
    return busy.intersection(managed_gpu_ids)


def _allocate_gpu_ids(
    managed_gpu_ids: tuple[int, ...],
    free_gpu_ids: set[int],
    gpu_count: int,
) -> tuple[int, ...] | None:
    """优先分配设备列表中的连续区间，降低双卡任务碎片。"""
    if gpu_count > len(free_gpu_ids):
        return None
    for start in range(len(managed_gpu_ids) - gpu_count + 1):
        candidate = managed_gpu_ids[start : start + gpu_count]
        if set(candidate).issubset(free_gpu_ids):
            return candidate
    ordered_free = tuple(gpu_id for gpu_id in managed_gpu_ids if gpu_id in free_gpu_ids)
    return ordered_free[:gpu_count]


def _next_queued_job(settings: Settings) -> sqlite3.Row | None:
    """返回具有最高优先级的最老任务；资源不足时不得越过它。"""
    with _connect(settings) as connection:
        return connection.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY priority DESC, id LIMIT 1"
        ).fetchone()


def _claim_job(
    settings: Settings,
    job_id: int,
    assigned_gpu_ids: tuple[int, ...],
) -> sqlite3.Row | None:
    """原子领取一个排队任务并写入 GPU 租约。"""
    with _connect(settings) as connection:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            "UPDATE jobs SET status = 'starting', started_at = ?, "
            "assigned_gpu_ids_json = ?, message = NULL "
            "WHERE id = ? AND status = 'queued'",
            (_now(), json.dumps(assigned_gpu_ids), job_id),
        )
        if cursor.rowcount != 1:
            connection.commit()
            return None
        connection.commit()
    return _job(settings, job_id)


def _clean_environment(settings: Settings, row: sqlite3.Row) -> dict[str, str]:
    assigned_gpu_ids = _assigned_gpu_ids(row)
    if len(assigned_gpu_ids) != int(row["gpu_count"]):
        raise RuntimeError(
            f"任务 {row['id']} 的 GPU 租约数量错误: "
            f"requested={row['gpu_count']}, assigned={assigned_gpu_ids}"
        )
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("SE3_LOGGER", None)
    environment.pop("SE3_SMOKE", None)
    environment.update(json.loads(row["extra_env_json"]))
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in assigned_gpu_ids),
            "PYTHONUNBUFFERED": "1",
            "WANDB_DIR": str(settings.wandb_root),
            "WANDB_MODE": row["wandb_mode"],
        }
    )
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
        "--gpu-ids",
        "all",
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
            print(
                f"[{_now()}] 开始任务 {job_id}: {row['task']} "
                f"physical_gpus={_assigned_gpu_ids(row)}",
                file=log_stream,
            )
            current_commit = _git_output(settings, "rev-parse", "HEAD")
            if current_commit != row["git_commit"]:
                raise RuntimeError(
                    f"远端 commit 不一致: current={current_commit}, expected={row['git_commit']}"
                )
            dirty_paths = _tracked_training_changes(settings)
            if dirty_paths:
                raise RuntimeError(f"远端训练源码存在改动，拒绝启动: {dirty_paths}")

            if row["smoke"]:
                smoke_env = dict(environment)
                smoke_env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": "",
                        "SE3_SMOKE": "1",
                        "SE3_LOGGER": "tensorboard",
                        "WANDB_MODE": "offline",
                    }
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


def _seconds_since(timestamp: str | None) -> float:
    """计算 UTC 时间戳距今秒数；缺失值视为无限久。"""
    if not timestamp:
        return float("inf")
    return max(0.0, (datetime.now(UTC) - datetime.fromisoformat(timestamp)).total_seconds())


def _recover_active_jobs(settings: Settings) -> list[sqlite3.Row]:
    """恢复 worker 状态，并返回仍占用 GPU 租约的任务。"""
    with _connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs WHERE status IN ('starting', 'running', 'cancelling') ORDER BY id"
        ).fetchall()
    live_rows: list[sqlite3.Row] = []
    for row in rows:
        if _process_alive(row["pid"]):
            live_rows.append(row)
            continue
        if row["status"] == "starting" and row["pid"] is None:
            if _seconds_since(row["started_at"]) < 30.0:
                live_rows.append(row)
                continue
            _update_job(
                settings,
                int(row["id"]),
                status="queued",
                started_at=None,
                assigned_gpu_ids_json=None,
                message="worker 启动前中断，已重新排队",
            )
            continue
        cancelled = bool(row["cancel_requested"]) or row["status"] == "cancelling"
        _update_job(
            settings,
            int(row["id"]),
            status="cancelled" if cancelled else "failed",
            finished_at=_now(),
            exit_code=130 if cancelled else 1,
            message="取消后执行器已退出" if cancelled else "执行器消失且没有写入最终状态",
        )
    return live_rows


def _leased_gpu_ids(
    rows: list[sqlite3.Row],
    managed_gpu_ids: tuple[int, ...],
) -> set[int]:
    """校验活动租约互斥且位于 worker 管理范围。"""
    managed = set(managed_gpu_ids)
    owners: dict[int, int] = {}
    for row in rows:
        assigned = _assigned_gpu_ids(row)
        if len(assigned) != int(row["gpu_count"]):
            raise RuntimeError(
                f"任务 {row['id']} 的活动租约数量错误: "
                f"requested={row['gpu_count']}, assigned={assigned}"
            )
        for gpu_id in assigned:
            if gpu_id not in managed:
                raise RuntimeError(f"任务 {row['id']} 租用了未受管 GPU {gpu_id}")
            if gpu_id in owners:
                raise RuntimeError(f"GPU {gpu_id} 被任务 {owners[gpu_id]} 与 {row['id']} 重复租用")
            owners[gpu_id] = int(row["id"])
    return set(owners)


def _launch_job(settings: Settings, row: sqlite3.Row) -> subprocess.Popen[bytes]:
    """启动独立 executor；worker 不等待，从而允许多个任务并发。"""
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
        "--log-root",
        str(settings.checkpoint_root),
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
    try:
        _update_job(
            settings,
            job_id,
            status="running",
            pid=process.pid,
            pgid=process.pid,
            log_path=str(log_path),
        )
    except Exception:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        raise
    return process


def _cmd_worker(args: argparse.Namespace, settings: Settings) -> None:
    _prepare(settings)
    if fcntl is None:
        raise SystemExit("queue worker 只支持 Linux")
    if args.poll_seconds <= 0.0:
        raise SystemExit("--poll-seconds 必须为正数")
    managed_gpu_ids = settings.gpu_ids or _discover_gpu_ids()
    with settings.lock_path.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有 queue worker 在运行")
            return
        _worker_log(settings, f"GPU-aware worker 启动 devices={managed_gpu_ids}")
        last_wait_reason = ""
        children: dict[int, subprocess.Popen[bytes]] = {}
        while True:
            for pid, process in tuple(children.items()):
                if process.poll() is not None:
                    children.pop(pid)

            try:
                active_rows = _recover_active_jobs(settings)
                leased_gpu_ids = _leased_gpu_ids(active_rows, managed_gpu_ids)
            except RuntimeError as error:
                reason = f"停止调度：{error}"
                if reason != last_wait_reason:
                    _worker_log(settings, reason)
                    last_wait_reason = reason
                time.sleep(args.poll_seconds)
                continue

            active_pgids = {int(row["pgid"]) for row in active_rows if row["pgid"] is not None}
            external_pids = _external_training_pids(active_pgids)
            if external_pids:
                hardware_busy_gpu_ids = set(managed_gpu_ids)
            else:
                try:
                    hardware_busy_gpu_ids = _busy_gpu_ids(managed_gpu_ids)
                except RuntimeError as error:
                    reason = f"停止调度：{error}"
                    if reason != last_wait_reason:
                        _worker_log(settings, reason)
                        last_wait_reason = reason
                    time.sleep(args.poll_seconds)
                    continue

            free_gpu_ids = set(managed_gpu_ids).difference(leased_gpu_ids | hardware_busy_gpu_ids)
            launched = False
            while True:
                row = _next_queued_job(settings)
                if row is None:
                    break
                gpu_count = int(row["gpu_count"])
                if gpu_count not in _GPU_COUNT_CHOICES:
                    _update_job(
                        settings,
                        int(row["id"]),
                        status="failed",
                        finished_at=_now(),
                        exit_code=2,
                        message=f"不支持的 GPU 数量: {gpu_count}",
                    )
                    continue
                if gpu_count > len(managed_gpu_ids):
                    reason = (
                        f"队首任务 {row['id']} 请求 {gpu_count} GPU，"
                        f"worker 仅管理 {managed_gpu_ids}"
                    )
                    break
                assigned_gpu_ids = _allocate_gpu_ids(
                    managed_gpu_ids,
                    free_gpu_ids,
                    gpu_count,
                )
                if assigned_gpu_ids is None:
                    reason = (
                        f"预留队首任务 {row['id']} 的 {gpu_count} GPU；"
                        f"free={tuple(sorted(free_gpu_ids))} "
                        f"leased={tuple(sorted(leased_gpu_ids))} "
                        f"hardware_busy={tuple(sorted(hardware_busy_gpu_ids))}"
                    )
                    break
                claimed = _claim_job(settings, int(row["id"]), assigned_gpu_ids)
                if claimed is None:
                    continue
                try:
                    process = _launch_job(settings, claimed)
                except Exception as error:
                    _update_job(
                        settings,
                        int(row["id"]),
                        status="failed",
                        finished_at=_now(),
                        exit_code=1,
                        message=f"executor 启动失败: {error}",
                    )
                    continue
                children[process.pid] = process
                leased_gpu_ids.update(assigned_gpu_ids)
                free_gpu_ids.difference_update(assigned_gpu_ids)
                launched = True
                _worker_log(
                    settings,
                    f"启动任务 {row['id']}, executor_pid={process.pid}, "
                    f"physical_gpus={assigned_gpu_ids}",
                )

            if launched:
                last_wait_reason = ""
            elif row is not None and reason != last_wait_reason:
                _worker_log(settings, reason)
                last_wait_reason = reason
            elif row is None:
                last_wait_reason = ""
            time.sleep(args.poll_seconds)


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
    parser = argparse.ArgumentParser(description="单机 1/2/4 GPU-aware se3-train 任务队列")
    parser.add_argument("--root", help="队列持久目录；也可用 SE3_QUEUE_ROOT")
    parser.add_argument("--repo", help="训练仓库；也可用 SE3_QUEUE_REPO")
    parser.add_argument("--uv", help="uv 路径；也可用 SE3_QUEUE_UV")
    parser.add_argument(
        "--log-root",
        help="训练产物根目录；也可用 SE3_QUEUE_LOG_ROOT，默认 <root>/checkpoints",
    )
    parser.add_argument(
        "--devices",
        help="worker 管理的物理 GPU，例如 0,1,2,3；也可用 SE3_QUEUE_GPU_IDS",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="提交训练任务")
    submit.add_argument("task")
    submit.add_argument("--name")
    submit.add_argument("--commit")
    submit.add_argument("--gpus", type=int, choices=_GPU_COUNT_CHOICES, default=1)
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--envs", type=int, default=8192, help="每张 GPU 的环境数")
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

    subparsers.add_parser("resources", help="查看 GPU 占用与队列租约")

    cancel = subparsers.add_parser("cancel", help="取消任务")
    cancel.add_argument("job_id", type=int)

    worker = subparsers.add_parser("worker", help="运行 GPU-aware worker")
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
        "resources": _cmd_resources,
        "cancel": _cmd_cancel,
        "worker": _cmd_worker,
    }
    if args.command == "_execute":
        raise SystemExit(_execute_job(settings, args.job_id))
    handlers[args.command](args, settings)


if __name__ == "__main__":
    main()
