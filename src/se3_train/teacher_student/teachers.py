"""冻结 stair teacher actor 的加载和推理封装。"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .config import StairTeacherStudentConfig, TeacherCheckpointConfig


@dataclass(frozen=True)
class ShapeMismatch:
    """checkpoint 张量形状与目标 actor 不一致的记录。"""

    key: str
    expected: tuple[int, ...]
    actual: tuple[int, ...]


@dataclass(frozen=True)
class ActorCheckpointReport:
    """单个 actor checkpoint 的结构检查报告。"""

    name: str
    checkpoint: Path
    expected_key_count: int
    loaded_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    obs_normalizer_keys: tuple[str, ...]
    distribution_keys: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """检查结果是否可 strict 加载。"""
        return not self.missing_keys and not self.unexpected_keys and not self.shape_mismatches

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 友好的检查报告。"""
        data = asdict(self)
        data["checkpoint"] = str(self.checkpoint)
        return data


@dataclass
class FrozenTeacher:
    """冻结的 teacher actor，保留自己的 obs normalizer 和 GRU hidden state。"""

    name: str
    actor: MLPModel
    checkpoint: Path
    report: ActorCheckpointReport
    actor_obs_dim: int
    actor_group: str = "actor"

    def act(self, obs: TensorDict) -> torch.Tensor:
        """用确定性 actor 均值生成 teacher action。"""
        with torch.inference_mode():
            return self.actor(self._adapt_obs(obs), stochastic_output=False).detach()

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """按 env done mask 重置 recurrent hidden state。"""
        self.actor.reset(dones)

    def _adapt_obs(self, obs: TensorDict) -> TensorDict:
        """把 student obs 裁剪到 teacher 训练时的 actor 维度。"""
        actor_obs = obs[self.actor_group]
        if actor_obs.shape[-1] == self.actor_obs_dim:
            return obs
        if actor_obs.shape[-1] < self.actor_obs_dim:
            raise ValueError(
                f"teacher {self.name} 需要 {self.actor_obs_dim}D obs，实际只有 {actor_obs.shape[-1]}D"
            )
        data = {key: value for key, value in obs.items()}
        data[self.actor_group] = actor_obs[..., : self.actor_obs_dim]
        return TensorDict(data, batch_size=obs.batch_size, device=obs.device)


@dataclass
class TeacherBank:
    """按名称管理冻结 teacher。"""

    teachers: dict[str, FrozenTeacher]

    def __contains__(self, name: str) -> bool:
        """判断 teacher 是否存在。"""
        return name in self.teachers

    def __getitem__(self, name: str) -> FrozenTeacher:
        """返回指定名称的 teacher。"""
        return self.teachers[name]

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """同步重置所有 teacher 的 recurrent hidden state。"""
        for teacher in self.teachers.values():
            teacher.reset(dones)

    def reports(self) -> dict[str, dict[str, Any]]:
        """返回所有 teacher 的结构检查报告。"""
        return {name: teacher.report.to_dict() for name, teacher in self.teachers.items()}


def load_teacher_bank(
    config: StairTeacherStudentConfig,
    *,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    actor_class: type[MLPModel],
    actor_kwargs: dict[str, Any],
    action_dim: int,
    device: str | torch.device,
) -> TeacherBank | None:
    """按配置加载已启用的冻结 teacher。"""
    teachers: dict[str, FrozenTeacher] = {}
    for teacher_cfg in config.teachers:
        teachers[teacher_cfg.name] = load_frozen_teacher(
            teacher_cfg,
            obs=obs,
            obs_groups=obs_groups,
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
            action_dim=action_dim,
            device=device,
        )
    if not teachers:
        return None
    return TeacherBank(teachers=teachers)


def load_frozen_teacher(
    teacher_cfg: TeacherCheckpointConfig,
    *,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    actor_class: type[MLPModel],
    actor_kwargs: dict[str, Any],
    action_dim: int,
    device: str | torch.device,
) -> FrozenTeacher:
    """加载单个冻结 teacher actor，并执行结构检查。"""
    checkpoint = teacher_cfg.resolved_path()
    actor_obs_dim = _resolve_actor_obs_dim(obs, teacher_cfg.actor_obs_dim)
    teacher_obs = _teacher_obs_spec(obs, actor_obs_dim)
    actor = actor_class(
        teacher_obs,
        obs_groups,
        "actor",
        action_dim,
        **copy.deepcopy(actor_kwargs),
    ).to(device)
    actor_state = load_actor_state_dict(checkpoint, map_location=device)
    report = inspect_actor_state_dict(teacher_cfg.name, checkpoint, actor, actor_state)
    if teacher_cfg.strict and not report.compatible:
        raise ValueError(_format_report_error(report))
    actor.load_state_dict(actor_state, strict=teacher_cfg.strict)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return FrozenTeacher(
        name=teacher_cfg.name,
        actor=actor,
        checkpoint=checkpoint,
        report=report,
        actor_obs_dim=actor_obs_dim,
    )


def load_actor_state_dict(
    checkpoint: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """读取 checkpoint 并抽取当前 RSL-RL actor_state_dict。"""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint 不存在: {checkpoint}")
    loaded = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if "actor_state_dict" in loaded:
        actor_state = dict(loaded["actor_state_dict"])
    elif "model_state_dict" in loaded:
        actor_state = _migrate_legacy_model_state_dict(loaded["model_state_dict"])
    else:
        keys = ", ".join(sorted(loaded.keys()))
        raise KeyError(f"checkpoint 缺少 actor_state_dict/model_state_dict，可用 keys: {keys}")
    return _migrate_distribution_keys(actor_state)


def inspect_actor_state_dict(
    name: str,
    checkpoint: Path,
    actor: MLPModel,
    actor_state: dict[str, torch.Tensor],
) -> ActorCheckpointReport:
    """比较 checkpoint actor 与目标 actor 的 key 和 shape。"""
    expected_state = actor.state_dict()
    expected_keys = set(expected_state)
    loaded_keys = set(actor_state)
    common_keys = expected_keys & loaded_keys
    shape_mismatches = tuple(
        ShapeMismatch(
            key=key,
            expected=tuple(int(dim) for dim in expected_state[key].shape),
            actual=tuple(int(dim) for dim in actor_state[key].shape),
        )
        for key in sorted(common_keys)
        if expected_state[key].shape != actor_state[key].shape
    )
    obs_normalizer_keys = tuple(
        sorted(key for key in loaded_keys if key.startswith("obs_normalizer."))
    )
    distribution_keys = tuple(sorted(key for key in loaded_keys if key.startswith("distribution.")))
    return ActorCheckpointReport(
        name=name,
        checkpoint=checkpoint,
        expected_key_count=len(expected_keys),
        loaded_key_count=len(loaded_keys),
        missing_keys=tuple(sorted(expected_keys - loaded_keys)),
        unexpected_keys=tuple(sorted(loaded_keys - expected_keys)),
        shape_mismatches=shape_mismatches,
        obs_normalizer_keys=obs_normalizer_keys,
        distribution_keys=distribution_keys,
    )


def _migrate_legacy_model_state_dict(
    model_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """迁移旧版 actor.* / actor_obs_normalizer.* checkpoint key。"""
    actor_state: dict[str, torch.Tensor] = {}
    for key, value in model_state_dict.items():
        if key.startswith("actor."):
            actor_state[key.replace("actor.", "mlp.")] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
        elif key in {"std", "log_std"}:
            actor_state[key] = value
    return actor_state


def _migrate_distribution_keys(
    actor_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """迁移 RSL-RL 4.x 的 std/log_std 到 5.x distribution key。"""
    migrated = dict(actor_state)
    if "std" in migrated:
        migrated["distribution.std_param"] = migrated.pop("std")
    if "log_std" in migrated:
        migrated["distribution.log_std_param"] = migrated.pop("log_std")
    return migrated


def _format_report_error(report: ActorCheckpointReport) -> str:
    """把结构检查报告压缩成异常信息。"""
    parts = [f"teacher {report.name} checkpoint 与 actor 结构不兼容: {report.checkpoint}"]
    if report.missing_keys:
        parts.append(f"missing={list(report.missing_keys[:8])}")
    if report.unexpected_keys:
        parts.append(f"unexpected={list(report.unexpected_keys[:8])}")
    if report.shape_mismatches:
        mismatch = report.shape_mismatches[0]
        parts.append(
            f"shape={mismatch.key}: expected {mismatch.expected}, actual {mismatch.actual}"
        )
    return "; ".join(parts)


def _resolve_actor_obs_dim(obs: TensorDict, actor_obs_dim: int | None) -> int:
    """解析 teacher actor obs 维度；None 表示沿用当前 student actor 维度。"""
    current_dim = int(obs["actor"].shape[-1])
    return current_dim if actor_obs_dim is None else int(actor_obs_dim)


def _teacher_obs_spec(obs: TensorDict, actor_obs_dim: int) -> TensorDict:
    """构造用于实例化 teacher actor 的 obs spec。"""
    data = {key: value for key, value in obs.items()}
    current_actor = obs["actor"]
    if int(current_actor.shape[-1]) != int(actor_obs_dim):
        data["actor"] = torch.zeros(
            *current_actor.shape[:-1],
            int(actor_obs_dim),
            device=current_actor.device,
            dtype=current_actor.dtype,
        )
    return TensorDict(data, batch_size=obs.batch_size, device=obs.device)
