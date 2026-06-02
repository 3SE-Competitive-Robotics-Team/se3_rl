"""MJLab 训练运行时资源探测与吞吐记录。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class CpuRuntimeInfo:
    """容器内可见 CPU 与 cgroup 配额。"""

    visible_count: int
    affinity_count: int | None
    quota_count: float | None

    @property
    def effective_count(self) -> float:
        """返回最接近训练进程真实可用 CPU 的数量。"""
        candidates = [float(self.visible_count)]
        if self.affinity_count is not None:
            candidates.append(float(self.affinity_count))
        if self.quota_count is not None:
            candidates.append(float(self.quota_count))
        return max(1.0, min(candidates))


@dataclass(frozen=True)
class GpuRuntimeInfo:
    """CUDA 设备选择状态。"""

    cuda_visible_devices: str
    local_rank: str | None
    rank: str | None


@dataclass(frozen=True)
class TrainingRuntimeInfo:
    """训练运行时资源快照。"""

    cpu: CpuRuntimeInfo
    gpu: GpuRuntimeInfo

    def as_log_dict(self) -> dict[str, float]:
        """转换为可以写入训练日志的数值指标。"""
        log = {
            "Runtime/cpu_visible_count": float(self.cpu.visible_count),
            "Runtime/cpu_effective_count": float(self.cpu.effective_count),
        }
        if self.cpu.affinity_count is not None:
            log["Runtime/cpu_affinity_count"] = float(self.cpu.affinity_count)
        if self.cpu.quota_count is not None:
            log["Runtime/cpu_quota_count"] = float(self.cpu.quota_count)
        return log


@dataclass
class IterationTiming:
    """单次 PPO 迭代耗时。"""

    collect_s: float
    returns_s: float
    learn_s: float
    total_s: float
    steps_per_second: float

    def as_log_dict(self) -> dict[str, float]:
        """转换为训练日志字段。"""
        return {
            "Perf/collect_s": self.collect_s,
            "Perf/returns_s": self.returns_s,
            "Perf/learn_s": self.learn_s,
            "Perf/iteration_s": self.total_s,
            "Perf/steps_per_second": self.steps_per_second,
        }


def read_cgroup_cpu_quota(path: Path = Path("/sys/fs/cgroup/cpu.max")) -> float | None:
    """读取 cgroup v2 CPU 配额，非 Linux 或无限额时返回 None。"""
    try:
        raw = path.read_text(encoding="utf-8").strip().split()
    except OSError:
        return None
    if len(raw) != 2 or raw[0] == "max":
        return None
    quota_us = float(raw[0])
    period_us = float(raw[1])
    if quota_us <= 0.0 or period_us <= 0.0:
        return None
    return quota_us / period_us


def detect_cpu_runtime() -> CpuRuntimeInfo:
    """探测当前进程的 CPU 可用度。"""
    visible_count = os.cpu_count() or 1
    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except OSError:
            affinity_count = None
    return CpuRuntimeInfo(
        visible_count=visible_count,
        affinity_count=affinity_count,
        quota_count=read_cgroup_cpu_quota(),
    )


def detect_gpu_runtime() -> GpuRuntimeInfo:
    """探测 CUDA 设备选择环境变量。"""
    return GpuRuntimeInfo(
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        local_rank=os.environ.get("LOCAL_RANK"),
        rank=os.environ.get("RANK"),
    )


def detect_training_runtime() -> TrainingRuntimeInfo:
    """采集训练启动时的资源快照。"""
    return TrainingRuntimeInfo(cpu=detect_cpu_runtime(), gpu=detect_gpu_runtime())


def format_runtime_summary(info: TrainingRuntimeInfo) -> str:
    """生成人类可读的训练运行时摘要。"""
    quota = "unlimited" if info.cpu.quota_count is None else f"{info.cpu.quota_count:.2f}"
    affinity = "unknown" if info.cpu.affinity_count is None else str(info.cpu.affinity_count)
    cuda_visible = info.gpu.cuda_visible_devices or "<cpu>"
    return (
        "[SE3 Runtime] "
        f"cpu_visible={info.cpu.visible_count}, "
        f"cpu_affinity={affinity}, "
        f"cpu_quota={quota}, "
        f"cpu_effective={info.cpu.effective_count:.2f}, "
        f"cuda_visible={cuda_visible}"
    )


class IterationTimer:
    """PPO 迭代分段计时器。"""

    def __init__(self, num_envs: int, num_steps_per_env: int) -> None:
        self._num_envs = int(num_envs)
        self._num_steps_per_env = int(num_steps_per_env)
        self._t0 = perf_counter()
        self._collect_end = self._t0
        self._returns_end = self._t0

    def mark_collect_done(self) -> None:
        """标记采样结束。"""
        self._collect_end = perf_counter()

    def mark_returns_done(self) -> None:
        """标记 return 计算结束。"""
        self._returns_end = perf_counter()

    def finish(self) -> IterationTiming:
        """结束计时并返回分段耗时。"""
        end = perf_counter()
        collect_s = self._collect_end - self._t0
        returns_s = self._returns_end - self._collect_end
        learn_s = end - self._returns_end
        total_s = end - self._t0
        steps = self._num_envs * self._num_steps_per_env
        steps_per_second = steps / total_s if total_s > 0.0 else 0.0
        return IterationTiming(
            collect_s=collect_s,
            returns_s=returns_s,
            learn_s=learn_s,
            total_s=total_s,
            steps_per_second=steps_per_second,
        )


def update_extras_log(extras: object, values: dict[str, float]) -> None:
    """把 runtime 指标合并进 MJLab/RSL-RL 的 extras['log']。"""
    if not isinstance(extras, dict):
        return
    log = extras.setdefault("log", {})
    if isinstance(log, dict):
        log.update(values)
