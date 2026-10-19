"""导出已注册训练任务的 reward 与 termination 最终生效配置。"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "web" / "task-atlas" / "public" / "task-data.json"


def _git_output(*args: str) -> str:
    """读取 Git 元数据；导出不应因非 Git 环境而失败。"""

    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _github_repository_url() -> str | None:
    """将 origin 地址规范化为可用于源码跳转的 GitHub HTTPS 地址。"""

    remote = _git_output("remote", "get-url", "origin")
    if not remote:
        return None
    if remote.startswith("git@github.com:"):
        remote = f"https://github.com/{remote.removeprefix('git@github.com:')}"
    if not remote.startswith("https://github.com/"):
        return None
    return remote.removesuffix(".git")


def _relative_source(path: str | Path | None) -> str | None:
    """把函数源文件转换为仓库相对 POSIX 路径。"""

    if path is None:
        return None
    source_path = Path(path).resolve()
    try:
        return source_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return source_path.as_posix()


def _callable_name(value: Any) -> str:
    """返回稳定、可搜索的可调用对象全名。"""

    module = getattr(value, "__module__", value.__class__.__module__)
    qualname = getattr(value, "__qualname__", value.__class__.__qualname__)
    return f"{module}.{qualname}"


def _callable_source(value: Any) -> dict[str, str | int] | None:
    """定位 reward/termination 实现的源文件与首行。"""

    target = value if inspect.isfunction(value) or inspect.isclass(value) else value.__class__
    try:
        source_file = inspect.getsourcefile(target)
        _, line = inspect.getsourcelines(target)
    except (OSError, TypeError):
        return None
    path = _relative_source(source_file)
    return {"path": path, "line": line} if path is not None else None


def _number_text(value: float) -> str:
    """用紧凑格式显示浮点数，同时保留小权重的科学计数法。"""

    if value == 0:
        return "0"
    absolute = abs(value)
    if absolute < 1.0e-4 or absolute >= 1.0e5:
        return f"{value:.4g}"
    return f"{value:.8g}"


def _display_value(value: Any, depth: int = 0) -> str:
    """把参数压缩成适合网页阅读的一行稳定文本。"""

    if depth >= 5:
        return "…"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number_text(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return value.as_posix()
    if isinstance(value, Enum):
        return str(value.value)
    if callable(value):
        return _callable_name(value)
    if isinstance(value, Mapping):
        items = [f"{key}: {_display_value(item, depth + 1)}" for key, item in value.items()]
        return "{" + ", ".join(items) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        opening, closing = ("(", ")") if isinstance(value, tuple) else ("[", "]")
        return opening + ", ".join(_display_value(item, depth + 1) for item in value) + closing
    text = repr(value)
    text = re.sub(r" at 0x[0-9A-Fa-f]+", "", text)
    return text.replace(str(PROJECT_ROOT), ".")


def _serialize_params(params: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """保持参数声明顺序并生成可搜索文本。"""

    return [
        {"name": str(name), "value": _display_value(value)}
        for name, value in (params or {}).items()
    ]


def _term_groups(term: Any) -> list[str]:
    """读取 Filtered term 声明的环境组作用域。"""

    params = getattr(term, "params", None) or {}
    group_names = params.get("group_names")
    if not isinstance(group_names, Sequence) or isinstance(group_names, (str, bytes)):
        return []
    return [str(name) for name in group_names]


def _is_gate_parameter(name: str) -> bool:
    """判断参数是否描述门控、触发阈值或持续条件。"""

    normalized = name.lower()
    direct_names = {
        "group_names",
        "terminate",
        "max_steps",
        "min_steps",
        "min_delta",
        "max_delta",
    }
    direct_tokens = (
        "gate",
        "threshold",
        "limit",
        "probability",
        "phase",
        "window",
        "sustain",
        "grace",
        "cool_down",
        "warmup",
        "start_s",
        "end_s",
        "tolerance",
        "_tol",
    )
    bounded_tokens = (
        "height",
        "angle",
        "velocity",
        "vel",
        "speed",
        "force",
        "delta",
        "ratio",
        "duration",
        "error",
        "clearance",
        "progress",
        "vz",
    )
    return (
        normalized in direct_names
        or normalized.endswith("_only")
        or normalized.endswith("_prob")
        or normalized.startswith("safe_")
        or any(token in normalized for token in direct_tokens)
        or (
            normalized.startswith(("min_", "max_"))
            and any(token in normalized for token in bounded_tokens)
        )
    )


def _serialize_gates(params: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """递归提取 term 配置中的显式门控与触发参数。"""

    gates: list[dict[str, str]] = []

    def collect(values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if isinstance(value, Mapping):
                collect(value)
            elif _is_gate_parameter(str(name)):
                gates.append({"name": str(name), "value": _display_value(value)})

    collect(params or {})
    return gates


def _effective_term_callable(term: Any) -> Any:
    """Filtered wrapper 展示其内部真正执行的 term 函数。"""

    params = getattr(term, "params", None) or {}
    wrapped_term = params.get("wrapped_term")
    if isinstance(wrapped_term, Mapping) and callable(wrapped_term.get("func")):
        return wrapped_term["func"]
    return term.func


def _serialize_reward(name: str, term: Any) -> dict[str, Any]:
    """序列化一个最终生效的 reward term。"""

    effective_func = _effective_term_callable(term)
    function = _callable_name(effective_func)
    weight = float(term.weight)
    return {
        "name": name,
        "function": function,
        "functionShort": function.rsplit(".", maxsplit=1)[-1],
        "weight": weight,
        "weightText": _number_text(weight),
        "polarity": "positive" if weight > 0 else "negative" if weight < 0 else "neutral",
        "groups": _term_groups(term),
        "gates": _serialize_gates(term.params),
        "params": _serialize_params(term.params),
        "source": _callable_source(effective_func),
    }


def _serialize_termination(name: str, term: Any) -> dict[str, Any]:
    """序列化一个最终生效的 termination term。"""

    effective_func = _effective_term_callable(term)
    function = _callable_name(effective_func)
    return {
        "name": name,
        "function": function,
        "functionShort": function.rsplit(".", maxsplit=1)[-1],
        "timeOut": bool(term.time_out),
        "groups": _term_groups(term),
        "gates": _serialize_gates(term.params),
        "params": _serialize_params(term.params),
        "source": _callable_source(effective_func),
    }


def _serialize_env_groups(cfg: Any) -> list[dict[str, str | float]]:
    """导出 AssignEnvGroups 的最终分组比例。"""

    event = (getattr(cfg, "events", None) or {}).get("assign_env_groups")
    params = getattr(event, "params", None) or {}
    groups = params.get("env_groups")
    if not isinstance(groups, Mapping):
        return []
    return [{"name": str(name), "ratio": float(ratio)} for name, ratio in groups.items()]


def _task_metadata(task_id: str) -> dict[str, str]:
    """从稳定的任务 ID 推导网页分组与短名称。"""

    robot = "WheelDog" if "WheelDog" in task_id else "SerialLeg"
    if "Jump-PreTrain" in task_id:
        family, label = "Jump", "Jump PreTrain"
    elif "Jump-FineTune" in task_id:
        family, label = "Jump", "Jump FineTune"
    elif "Recovery-Expert" in task_id:
        family, label = "Recovery", "Recovery Expert"
    elif "Recovery-Loco-Grouped" in task_id:
        family, label = "Recovery", "Grouped: loco + recover"
    elif "Recovery" in task_id:
        family = "Recovery"
        label = task_id.removeprefix("SE3-WheelLegged-Recovery-")
        label = re.sub(r"-(GRU|MLP)$", "", label).replace("-", " ")
    elif "BlindClimb" in task_id:
        family, label = "WheelDog", "Blind Climb"
    elif "WheelDog-Flat" in task_id:
        family, label = "WheelDog", "WheelDog Flat"
    elif "Stair" in task_id:
        family, label = "Stair", "Stair TrainView" if "TrainView" in task_id else "Stair"
    elif "Rough" in task_id:
        family, label = "Locomotion", "Rough"
    else:
        family, label = "Locomotion", "Flat"

    if task_id.endswith("History-MLP"):
        model = "History-MLP"
    elif task_id.endswith("MLP"):
        model = "MLP"
    elif "GRU" in task_id:
        model = "GRU"
    else:
        model = "Policy"
    return {"robot": robot, "family": family, "label": label, "model": model}


def _capture_registered_tasks() -> list[dict[str, Any]]:
    """拦截正式注册入口，直接拿到每个 task 的最终环境配置。"""

    import mjlab.tasks.registry as registry

    captured: list[dict[str, Any]] = []
    original_register: Callable[..., Any] = registry.register_mjlab_task

    def capture(**kwargs: Any) -> None:
        caller = inspect.currentframe().f_back
        module_name = caller.f_globals.get("__name__", "") if caller is not None else ""
        module_file = caller.f_globals.get("__file__") if caller is not None else None
        captured.append(
            {
                "task_id": kwargs["task_id"],
                "env_cfg": kwargs["env_cfg"],
                "play_env_cfg": kwargs["play_env_cfg"],
                "registration_module": module_name,
                "registration_source": _relative_source(module_file),
            }
        )

    registry.register_mjlab_task = capture
    try:
        import se3_train  # noqa: F401
    finally:
        registry.register_mjlab_task = original_register
    return captured


def _task_payload(item: dict[str, Any]) -> dict[str, Any]:
    """生成单个 task 的网页数据及配置指纹。"""

    task_id = item["task_id"]
    cfg = item["env_cfg"]
    rewards = [_serialize_reward(name, term) for name, term in cfg.rewards.items()]
    terminations = [_serialize_termination(name, term) for name, term in cfg.terminations.items()]
    fingerprint_payload = json.dumps(
        {"rewards": rewards, "terminations": terminations},
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    return {
        "id": task_id,
        **_task_metadata(task_id),
        "registrationModule": item["registration_module"],
        "registrationSource": item["registration_source"],
        "rewardCount": len(rewards),
        "terminationCount": len(terminations),
        "trainEnvGroups": _serialize_env_groups(cfg),
        "playEnvGroups": _serialize_env_groups(item["play_env_cfg"]),
        "fingerprint": hashlib.sha256(fingerprint_payload).hexdigest()[:12],
        "rewards": rewards,
        "terminations": terminations,
    }


def export(output: Path) -> None:
    """构建完整 Task Atlas JSON，并原子式写入目标文件。"""

    tasks = [_task_payload(item) for item in _capture_registered_tasks()]
    branch = _git_output("branch", "--show-current") or "HEAD"
    revision = _git_output("rev-parse", "--short", "HEAD") or "unknown"
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "sourceBranch": branch,
        "sourceRevision": revision,
        "repositoryUrl": _github_repository_url(),
        "taskCount": len(tasks),
        "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        f"已导出 {len(tasks)} 个任务、"
        f"{sum(task['rewardCount'] for task in tasks)} 个 reward、"
        f"{sum(task['terminationCount'] for task in tasks)} 个 termination -> {output}"
    )


def main() -> None:
    """解析命令行参数并执行导出。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.output.resolve())


if __name__ == "__main__":
    main()
