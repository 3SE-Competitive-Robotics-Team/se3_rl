"""台阶 teacher/student PPO 的配置对象。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from mjlab.rl import RslRlPpoAlgorithmCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAIR_TEACHER_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "rsl_rl"
    / "se3_wheel_leg"
    / "2026-06-01_12-35-40_stair_user_state_ff"
    / "model_4499.pt"
)


@dataclass(frozen=True)
class TeacherCheckpointConfig:
    """冻结 teacher checkpoint 的加载配置。"""

    name: str = "stair"
    path: Path | str = DEFAULT_STAIR_TEACHER_CHECKPOINT
    strict: bool = True
    enabled: bool = True
    actor_obs_dim: int | None = None

    def resolved_path(self) -> Path:
        """解析相对仓库根目录的 checkpoint 路径。"""
        path = Path(self.path)
        if path.is_absolute():
            return path
        return REPO_ROOT / path


@dataclass(frozen=True)
class MaskConfig:
    """根据观测和地形元数据生成样本级 teacher mask。"""

    actor_group: str = "actor"
    projected_gravity_key: str = "projected_gravity"
    projected_gravity_z_index: int = 5
    recovery_pg_z_threshold: float = -0.25
    stair_type_names: tuple[str, ...] = ("stage_stairs", "inv_pyramid_stairs")
    trust_non_curriculum_terrain_types: bool = False


@dataclass(frozen=True)
class TeacherLossConfig:
    """masked teacher loss 的线性退火配置。"""

    initial_coef: float = 0.2
    end_iter: int = 1200
    loss_type: str = "mse"

    def coef_at(self, iteration: int) -> float:
        """按 PPO update iteration 线性退火 teacher loss 系数。"""
        if self.end_iter <= 0:
            return self.initial_coef
        if iteration >= self.end_iter:
            return 0.0
        ratio = max(0.0, 1.0 - float(iteration) / float(self.end_iter))
        return self.initial_coef * ratio


@dataclass(frozen=True)
class StairTeacherStudentConfig:
    """只使用 stair teacher 的 PPO student 配置。"""

    enabled: bool = True
    mask: MaskConfig = field(default_factory=MaskConfig)
    teacher_loss: TeacherLossConfig = field(default_factory=TeacherLossConfig)
    stair_teacher: TeacherCheckpointConfig = field(default_factory=TeacherCheckpointConfig)

    @property
    def teachers(self) -> tuple[TeacherCheckpointConfig, ...]:
        """返回启用的 teacher 列表。"""
        if not self.enabled or not self.stair_teacher.enabled:
            return ()
        return (self.stair_teacher,)

    def to_dict(self) -> dict[str, Any]:
        """转为 MJLab/RSL-RL 可序列化配置。"""
        data = asdict(self)
        data["stair_teacher"]["path"] = str(data["stair_teacher"]["path"])
        return data


@dataclass
class StairTeacherPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """让 MJLab asdict 保留 teacher_student 字段的 PPO 配置。"""

    class_name: str = "se3_train.teacher_student.algorithm:StairTeacherPPO"
    teacher_student: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        base: RslRlPpoAlgorithmCfg,
        teacher_student: StairTeacherStudentConfig,
    ) -> StairTeacherPpoAlgorithmCfg:
        """从普通 PPO 配置复制超参，只替换算法实现。"""
        inherited_field_names = {
            item.name for item in fields(RslRlPpoAlgorithmCfg) if item.name != "class_name"
        }
        values = {name: getattr(base, name) for name in inherited_field_names}
        return cls(**values, teacher_student=teacher_student.to_dict())


def stair_teacher_checkpoint_from_env(
    default: Path | str = DEFAULT_STAIR_TEACHER_CHECKPOINT,
) -> Path | str:
    """读取 stair teacher checkpoint 环境变量。"""
    return os.environ.get("SE3_STAIR_TEACHER_CHECKPOINT", str(default))
