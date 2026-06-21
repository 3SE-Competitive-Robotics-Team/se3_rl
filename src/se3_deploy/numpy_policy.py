"""NumPy 版 actor 推理，用于 NX 轻量部署。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NumpyPolicyRuntime:
    """从 `.npz` 权重执行确定性 GRU/MLP actor 推理。"""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        payload = np.load(self.checkpoint_path, allow_pickle=False)
        self.num_obs = int(payload["num_obs"])
        self.num_actions = int(payload["num_actions"])
        self.iteration = str(int(payload["iteration"]))
        self.activation = str(payload["activation"].item())
        self.eps = float(payload["normalizer_eps"])
        self.obs_mean = payload["obs_mean"].astype(np.float32).reshape(self.num_obs)
        self.obs_std = payload["obs_std"].astype(np.float32).reshape(self.num_obs)

        self.rnn_type = str(payload["rnn_type"].item())
        if self.rnn_type != "gru":
            raise ValueError(f"only GRU npz policy is supported, got {self.rnn_type}")
        self.rnn_num_layers = int(payload["rnn_num_layers"])
        if self.rnn_num_layers != 1:
            raise ValueError(f"only 1-layer GRU is supported, got {self.rnn_num_layers}")
        self.rnn_hidden_dim = int(payload["rnn_hidden_dim"])
        self.weight_ih = payload["rnn_weight_ih_l0"].astype(np.float32)
        self.weight_hh = payload["rnn_weight_hh_l0"].astype(np.float32)
        self.bias_ih = payload["rnn_bias_ih_l0"].astype(np.float32)
        self.bias_hh = payload["rnn_bias_hh_l0"].astype(np.float32)

        self.mlp_weights: list[np.ndarray] = []
        self.mlp_biases: list[np.ndarray] = []
        num_linear = int(payload["mlp_num_linear"])
        for idx in range(num_linear):
            self.mlp_weights.append(payload[f"mlp_{idx}_weight"].astype(np.float32))
            self.mlp_biases.append(payload[f"mlp_{idx}_bias"].astype(np.float32))
        self._hidden = np.zeros(self.rnn_hidden_dim, dtype=np.float32)

    @property
    def policy_type(self) -> str:
        return f"numpy-gru(hidden={self.rnn_hidden_dim}, layers={self.rnn_num_layers})"

    def reset(self) -> None:
        self._hidden.fill(0.0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.shape != (self.num_obs,):
            raise ValueError(f"obs shape mismatch: expected {(self.num_obs,)}, got {x.shape}")
        x = (x - self.obs_mean) / (self.obs_std + self.eps)
        x = self._gru_step(x)
        for idx, (weight, bias) in enumerate(zip(self.mlp_weights, self.mlp_biases, strict=True)):
            x = weight @ x + bias
            if idx < len(self.mlp_weights) - 1:
                x = activate(x, self.activation)
        return x.astype(np.float32, copy=False)

    def _gru_step(self, x: np.ndarray) -> np.ndarray:
        gate_x = self.weight_ih @ x + self.bias_ih
        gate_h = self.weight_hh @ self._hidden + self.bias_hh
        x_r, x_z, x_n = np.split(gate_x, 3)
        h_r, h_z, h_n = np.split(gate_h, 3)
        reset = sigmoid(x_r + h_r)
        update = sigmoid(x_z + h_z)
        new = np.tanh(x_n + reset * h_n)
        self._hidden = ((1.0 - update) * new + update * self._hidden).astype(np.float32)
        return self._hidden


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def activate(x: np.ndarray, name: str) -> np.ndarray:
    normalized = name.lower()
    if normalized == "elu":
        return np.where(x > 0.0, x, np.expm1(x)).astype(np.float32)
    if normalized == "relu":
        return np.maximum(x, 0.0)
    if normalized == "tanh":
        return np.tanh(x)
    if normalized == "sigmoid":
        return sigmoid(x)
    raise ValueError(f"unsupported activation for numpy policy: {name}")


def npz_metadata(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    return {
        "iteration": int(payload["iteration"]),
        "num_obs": int(payload["num_obs"]),
        "num_actions": int(payload["num_actions"]),
        "rnn_type": str(payload["rnn_type"].item()),
        "rnn_hidden_dim": int(payload["rnn_hidden_dim"]),
        "rnn_num_layers": int(payload["rnn_num_layers"]),
        "activation": str(payload["activation"].item()),
    }
