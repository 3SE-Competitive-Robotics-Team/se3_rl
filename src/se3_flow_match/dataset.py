"""Teacher rollout 数据集读取。"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FlowPolicyConfig


class FlowSequenceSample(TypedDict):
    """单条 teacher 序列样本。"""

    obs: torch.Tensor
    actions: torch.Tensor
    dones: torch.Tensor
    modes: torch.Tensor
    teacher_names: list[str]


class TeacherFlowDataset(Dataset[FlowSequenceSample]):
    """读取并校验 Flow Matching teacher 序列数据。"""

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        dones: torch.Tensor | None = None,
        modes: torch.Tensor | None = None,
        teacher_names: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        config: FlowPolicyConfig | None = None,
    ) -> None:
        """构造 teacher 数据集，要求 obs/actions 都是 [N, T, D]。"""
        obs = _as_float_tensor(obs, "obs")
        actions = _as_float_tensor(actions, "actions")
        if obs.ndim != 3:
            raise ValueError(f"obs 必须是 [N, T, obs_dim]，实际形状为 {tuple(obs.shape)}")
        if actions.ndim != 3:
            raise ValueError(
                f"actions 必须是 [N, T, action_dim]，实际形状为 {tuple(actions.shape)}"
            )
        if obs.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "obs/actions 的 [N, T] 必须一致："
                f"obs={tuple(obs.shape)}, actions={tuple(actions.shape)}"
            )

        if config is not None:
            if obs.shape[-1] != config.obs_dim:
                raise ValueError(f"obs 末维必须为 {config.obs_dim}，实际为 {obs.shape[-1]}")
            if actions.shape[-1] != config.action_dim:
                raise ValueError(
                    f"actions 末维必须为 {config.action_dim}，实际为 {actions.shape[-1]}"
                )

        batch_shape = obs.shape[:2]
        if dones is None:
            dones = torch.zeros(batch_shape, dtype=torch.bool)
        else:
            dones = torch.as_tensor(dones, dtype=torch.bool)
        if modes is None:
            modes = torch.zeros(batch_shape, dtype=torch.long)
        else:
            modes = torch.as_tensor(modes, dtype=torch.long)

        if dones.shape != batch_shape:
            raise ValueError(f"dones 必须是 {tuple(batch_shape)}，实际为 {tuple(dones.shape)}")
        if modes.shape != batch_shape:
            raise ValueError(f"modes 必须是 {tuple(batch_shape)}，实际为 {tuple(modes.shape)}")

        self.obs = obs.contiguous()
        self.actions = actions.contiguous()
        self.dones = dones.contiguous()
        self.modes = modes.contiguous()
        self.teacher_names = teacher_names or ["unknown"] * int(obs.shape[0])
        if len(self.teacher_names) != int(obs.shape[0]):
            raise ValueError(
                f"teacher_names 长度必须为 {int(obs.shape[0])}，实际为 {len(self.teacher_names)}"
            )
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """返回序列条数。"""
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> FlowSequenceSample:
        """返回一条完整时间序列。"""
        return {
            "obs": self.obs[index],
            "actions": self.actions[index],
            "dones": self.dones[index],
            "modes": self.modes[index],
            "teacher_names": [self.teacher_names[index]],
        }

    @classmethod
    def from_file(
        cls, path: str | Path, *, config: FlowPolicyConfig | None = None
    ) -> TeacherFlowDataset:
        """从 .pt 或 .npz 文件读取 teacher 数据。"""
        return load_teacher_dataset(path, config=config)


def load_teacher_dataset(
    path: str | Path, *, config: FlowPolicyConfig | None = None
) -> TeacherFlowDataset:
    """读取 teacher 数据文件并返回 Dataset。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"teacher 数据文件不存在：{path}")
    if path.suffix == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=False)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            raw = {key: data[key] for key in data.files}
    else:
        raise ValueError(f"只支持 .pt 或 .npz teacher 数据，实际为：{path.suffix}")
    if not isinstance(raw, dict):
        raise TypeError("teacher 数据必须是 dict，且包含 obs/actions")

    try:
        obs = raw["obs"]
        actions = raw["actions"]
    except KeyError as exc:
        raise KeyError("teacher 数据必须包含 obs 和 actions 字段") from exc

    return TeacherFlowDataset(
        torch.as_tensor(obs),
        torch.as_tensor(actions),
        dones=torch.as_tensor(raw["dones"]) if "dones" in raw else None,
        modes=torch.as_tensor(raw["modes"]) if "modes" in raw else None,
        teacher_names=_teacher_names_from_raw(raw, int(torch.as_tensor(obs).shape[0])),
        metadata=raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None,
        config=config,
    )


def _as_float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    """转成 float32 tensor，并拒绝非有限值。"""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return tensor


def _teacher_names_from_raw(raw: dict, count: int) -> list[str] | None:
    """从原始 payload 中恢复每条序列的 teacher 名称。"""
    names = raw.get("teacher_names")
    if names is None:
        return None
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if not isinstance(names, list):
        raise TypeError("teacher_names 必须是 list 或 numpy array")
    result = [str(name) for name in names]
    if len(result) != count:
        raise ValueError(f"teacher_names 长度必须为 {count}，实际为 {len(result)}")
    return result
