"""ONNX Runtime 版 recovery actor 推理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


class OnnxPolicyRuntime:
    """用显式 hidden 输入/输出执行 GRU actor。"""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.checkpoint_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        self.iteration = meta.get("iteration", "unknown")
        self.num_obs = _metadata_int(meta, "num_obs", self._input_dim("obs"))
        self.num_actions = _metadata_int(meta, "num_actions", self._output_dim("action"))
        self.rnn_hidden_dim = _metadata_int(meta, "rnn_hidden_dim", self._hidden_dim())
        self.rnn_num_layers = _metadata_int(meta, "rnn_num_layers", self._hidden_layers())
        self.rnn_type = meta.get("rnn_type", "gru")
        if self.rnn_type != "gru":
            raise ValueError(f"only GRU ONNX policy is supported, got {self.rnn_type}")
        self.activation = meta.get("activation", "unknown")
        self._hidden = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_dim), dtype=np.float32)

    @property
    def policy_type(self) -> str:
        return f"onnx-gru(hidden={self.rnn_hidden_dim}, layers={self.rnn_num_layers})"

    def reset(self) -> None:
        self._hidden.fill(0.0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, self.num_obs)
        action, hidden = self.session.run(
            ["action", "hidden_out"],
            {"obs": x, "hidden_in": self._hidden},
        )
        self._hidden = np.asarray(hidden, dtype=np.float32).reshape(
            self.rnn_num_layers, 1, self.rnn_hidden_dim
        )
        return np.asarray(action, dtype=np.float32).reshape(self.num_actions)

    def _input_dim(self, name: str) -> int:
        for item in self.session.get_inputs():
            if item.name == name:
                return _shape_dim(item.shape[-1])
        raise ValueError(f"ONNX input not found: {name}")

    def _output_dim(self, name: str) -> int:
        for item in self.session.get_outputs():
            if item.name == name:
                return _shape_dim(item.shape[-1])
        raise ValueError(f"ONNX output not found: {name}")

    def _hidden_dim(self) -> int:
        for item in self.session.get_inputs():
            if item.name == "hidden_in":
                return _shape_dim(item.shape[-1])
        raise ValueError("ONNX input not found: hidden_in")

    def _hidden_layers(self) -> int:
        for item in self.session.get_inputs():
            if item.name == "hidden_in":
                return _shape_dim(item.shape[0])
        raise ValueError("ONNX input not found: hidden_in")


def _shape_dim(value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError(f"expected static ONNX dimension, got {value!r}")
    return value


def _metadata_int(meta: dict[str, str], key: str, default: int) -> int:
    value = meta.get(key)
    if value is None or value == "":
        return int(default)
    return int(value)


def onnx_metadata(path: Path) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    meta = session.get_modelmeta().custom_metadata_map
    return {
        "iteration": meta.get("iteration", "unknown"),
        "num_obs": _metadata_int(meta, "num_obs", 0),
        "num_actions": _metadata_int(meta, "num_actions", 0),
        "rnn_type": meta.get("rnn_type", "unknown"),
        "rnn_hidden_dim": _metadata_int(meta, "rnn_hidden_dim", 0),
        "rnn_num_layers": _metadata_int(meta, "rnn_num_layers", 0),
        "activation": meta.get("activation", "unknown"),
    }
