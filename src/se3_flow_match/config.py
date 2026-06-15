"""Flow policy 配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from se3_shared import ObservationConfig

_OBS_CFG = ObservationConfig()


@dataclass(frozen=True)
class FlowPolicyConfig:
    """GRU Flow Matching 策略配置。"""

    obs_dim: int = _OBS_CFG.num_obs
    action_dim: int = _OBS_CFG.num_actions
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 1
    activation: str = "elu"

    def __post_init__(self) -> None:
        """校验配置值，避免训练时才暴露形状错误。"""
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim 必须为正数，实际为 {self.obs_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim 必须为正数，实际为 {self.action_dim}")
        if self.rnn_hidden_dim <= 0:
            raise ValueError(f"rnn_hidden_dim 必须为正数，实际为 {self.rnn_hidden_dim}")
        if self.rnn_num_layers <= 0:
            raise ValueError(f"rnn_num_layers 必须为正数，实际为 {self.rnn_num_layers}")

    def to_dict(self) -> dict[str, Any]:
        """转成 checkpoint 可序列化字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FlowPolicyConfig:
        """从 checkpoint 字典恢复配置。"""
        return cls(
            obs_dim=int(raw.get("obs_dim", _OBS_CFG.num_obs)),
            action_dim=int(raw.get("action_dim", _OBS_CFG.num_actions)),
            rnn_hidden_dim=int(raw.get("rnn_hidden_dim", 512)),
            rnn_num_layers=int(raw.get("rnn_num_layers", 1)),
            activation=str(raw.get("activation", "elu")),
        )
