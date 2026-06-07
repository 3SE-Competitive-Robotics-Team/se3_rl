"""se3_flow_match checkpoint 工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import FlowPolicyConfig
from .model import FlowVelocityField

_CHECKPOINT_FORMAT = "se3_flow_match"


def save_flow_checkpoint(
    path: str | Path,
    model: FlowVelocityField,
    config: FlowPolicyConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """保存 Flow Matching checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _CHECKPOINT_FORMAT,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_flow_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[FlowVelocityField, FlowPolicyConfig, dict[str, Any]]:
    """加载 Flow Matching checkpoint。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Flow checkpoint 必须是 dict")
    fmt = payload.get("format")
    if fmt != _CHECKPOINT_FORMAT:
        raise ValueError(f"不支持的 Flow checkpoint 格式：{payload.get('format')!r}")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError("Flow checkpoint 缺少 config 字典")
    config = FlowPolicyConfig.from_dict(raw_config)
    model = FlowVelocityField(config).to(device)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("Flow checkpoint 缺少 model_state_dict")
    model.load_state_dict(state_dict)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return model, config, metadata
