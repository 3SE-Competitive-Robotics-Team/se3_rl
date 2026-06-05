"""Flow Matching 蒸馏任务注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from se3_shared import TaskMode


@dataclass(frozen=True)
class DistillTaskSpec:
    """单个 teacher 蒸馏任务配置。"""

    name: str
    mode: TaskMode
    task_id: str
    teacher_path: Path | None
    action_policy: str = "default"


TASK_SPECS: dict[str, DistillTaskSpec] = {
    "wheel": DistillTaskSpec(
        name="wheel",
        mode=TaskMode.WHEEL,
        task_id="SE3-WheelLegged-FlowMatch-Wheel-GRU",
        teacher_path=Path("assets/base_model/model_4999_wheel_expert.pt"),
    ),
    "gait": DistillTaskSpec(
        name="gait",
        mode=TaskMode.GAIT,
        task_id="SE3-WheelLegged-FlowMatch-Gait-FineTune-GRU",
        teacher_path=Path("assets/base_model/model_1999_gait_finetune_expert.pt"),
    ),
    "wheel_leg": DistillTaskSpec(
        name="wheel_leg",
        mode=TaskMode.WHEEL_LEG,
        task_id="SE3-WheelLegged-FlowMatch-WheelLeg-GRU",
        teacher_path=None,
    ),
    "gait_wheel": DistillTaskSpec(
        name="gait_wheel",
        mode=TaskMode.GAIT_WHEEL,
        task_id="SE3-WheelLegged-FlowMatch-GaitWheel-GRU",
        teacher_path=None,
    ),
    "jump": DistillTaskSpec(
        name="jump",
        mode=TaskMode.JUMP,
        task_id="SE3-WheelLegged-FlowMatch-Jump-GRU",
        teacher_path=None,
    ),
}


def parse_task_names(raw: str) -> list[str]:
    """解析逗号分隔 task 列表。"""
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("tasks 不能为空")
    for name in names:
        if name not in TASK_SPECS:
            raise ValueError(f"未知 task：{name}，可选：{', '.join(TASK_SPECS)}")
    return names


def task_spec(name: str) -> DistillTaskSpec:
    """读取已配置 teacher 的 task spec。"""
    spec = TASK_SPECS[name]
    if spec.teacher_path is None:
        raise ValueError(f"{name} 暂未配置 teacher checkpoint")
    return spec


__all__ = ["TASK_SPECS", "DistillTaskSpec", "parse_task_names", "task_spec"]
