"""MJLab checkpoint 的确定性推理加载器，支持 MLP 和 GRU 策略。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .runtime_spec import PolicyArchitectureSpec, RuntimeSpec

TensorStateDict = dict[str, torch.Tensor]


class _EmpiricalNormalizer(nn.Module):
    def __init__(self, num_obs: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(1, int(num_obs)))
        self.register_buffer("_std", torch.ones(1, int(num_obs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class _DeterministicActor(nn.Module):
    def __init__(self, spec: PolicyArchitectureSpec) -> None:
        super().__init__()
        self.obs_normalizer = _EmpiricalNormalizer(spec.num_obs)
        self.mlp = self._build_mlp(spec)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))

    @staticmethod
    def _build_mlp(spec: PolicyArchitectureSpec) -> nn.Sequential:
        dims = [
            int(spec.num_obs),
            *[int(dim) for dim in spec.actor_hidden_dims],
            int(spec.num_actions),
        ]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(_activation(spec.activation))
        return nn.Sequential(*layers)


class _RNNWrapper(nn.Module):
    """匹配 rsl_rl 的权重命名: rnn.rnn.weight_ih_l0 → self.rnn.weight_ih_l0。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )


class _DeterministicGRUActor(nn.Module):
    """GRU 推理：obs_normalizer → GRU → MLP head。

    MLP head 输入维度为 rnn_hidden_dim（非 num_obs）。
    """

    def __init__(self, spec: PolicyArchitectureSpec) -> None:
        super().__init__()
        self.obs_normalizer = _EmpiricalNormalizer(spec.num_obs)
        self.rnn = _RNNWrapper(
            input_size=spec.num_obs,
            hidden_size=spec.rnn_hidden_dim,
            num_layers=spec.rnn_num_layers,
        )
        self.mlp = self._build_mlp(spec)
        self._hidden_dim = spec.rnn_hidden_dim
        self._num_layers = spec.rnn_num_layers
        self._hidden: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = self.obs_normalizer(obs)
        # (batch=1, seq=1, obs_dim)
        rnn_input = normalized.unsqueeze(1)
        rnn_out, self._hidden = self.rnn.rnn(rnn_input, self._hidden)
        # rnn_out: (1, 1, hidden_dim) → squeeze to (1, hidden_dim)
        return self.mlp(rnn_out.squeeze(1))

    def reset_hidden(self, device: torch.device) -> None:
        self._hidden = torch.zeros(self._num_layers, 1, self._hidden_dim, device=device)

    @staticmethod
    def _build_mlp(spec: PolicyArchitectureSpec) -> nn.Sequential:
        dims = [
            int(spec.rnn_hidden_dim),
            *[int(dim) for dim in spec.actor_hidden_dims],
            int(spec.num_actions),
        ]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(_activation(spec.activation))
        return nn.Sequential(*layers)


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "leaky_relu":
        return nn.LeakyReLU()
    if normalized == "selu":
        return nn.SELU()
    if normalized == "mish":
        return nn.Mish()
    raise ValueError(f"unsupported policy activation: {name}")


class PolicyRuntime:
    def __init__(self, *, checkpoint: Path, device: str, runtime: RuntimeSpec) -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        self.runtime = runtime
        self._numpy_policy = None
        self._onnx_policy = None
        if self.checkpoint_path.suffix == ".npz":
            from se3_deploy.numpy_policy import NumpyPolicyRuntime

            self._numpy_policy = NumpyPolicyRuntime(self.checkpoint_path)
            self.iteration = self._numpy_policy.iteration
            self.spec = replace(
                self.runtime.policy,
                num_obs=int(self._numpy_policy.num_obs),
                num_actions=int(self._numpy_policy.num_actions),
                activation=self._numpy_policy.activation,
                rnn_type=self._numpy_policy.rnn_type,
                rnn_hidden_dim=int(self._numpy_policy.rnn_hidden_dim),
                rnn_num_layers=int(self._numpy_policy.rnn_num_layers),
            )
            self._validate_runtime(self.spec)
            self.model = None
            return
        if self.checkpoint_path.suffix == ".onnx":
            from se3_deploy.onnx_policy import OnnxPolicyRuntime

            self._onnx_policy = OnnxPolicyRuntime(self.checkpoint_path)
            self.iteration = self._onnx_policy.iteration
            self.spec = replace(
                self.runtime.policy,
                num_obs=int(self._onnx_policy.num_obs),
                num_actions=int(self._onnx_policy.num_actions),
                activation=self._onnx_policy.activation,
                rnn_type=self._onnx_policy.rnn_type,
                rnn_hidden_dim=int(self._onnx_policy.rnn_hidden_dim),
                rnn_num_layers=int(self._onnx_policy.rnn_num_layers),
            )
            self._validate_runtime(self.spec)
            self.model = None
            return
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor_state_dict, self.critic_state_dict = self._extract_state_dicts(payload)
        self.iteration = payload.get("iter", "unknown")
        inferred = self._infer_spec(self.actor_state_dict, self.critic_state_dict)
        self.spec = self._resolve_spec(inferred, self.actor_state_dict, self.critic_state_dict)
        self._validate_runtime(self.spec)
        if self.spec.is_sequence:
            raise NotImplementedError(
                "SE3 workflow currently supports ActorCritic checkpoints, not sequence checkpoints"
            )
        if self.spec.is_recurrent:
            self.model: _DeterministicActor | _DeterministicGRUActor = _DeterministicGRUActor(
                self.spec
            ).to(self.device)
        else:
            self.model = _DeterministicActor(self.spec).to(self.device)
        self._load_actor_state(self.actor_state_dict)
        self.model.eval()
        if self.spec.is_recurrent:
            self.reset()

    @property
    def policy_type(self) -> str:
        if self._numpy_policy is not None:
            return self._numpy_policy.policy_type
        if self._onnx_policy is not None:
            return self._onnx_policy.policy_type
        if self.spec.is_recurrent:
            return f"gru(hidden={self.spec.rnn_hidden_dim}, layers={self.spec.rnn_num_layers})"
        return "mlp"

    def reset(self) -> None:
        if self._numpy_policy is not None:
            self._numpy_policy.reset()
            return
        if self._onnx_policy is not None:
            self._onnx_policy.reset()
            return
        if isinstance(self.model, _DeterministicGRUActor):
            self.model.reset_hidden(self.device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._numpy_policy is not None:
            return self._numpy_policy.act(obs)
        if self._onnx_policy is not None:
            return self._onnx_policy.act(obs)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape != (self.spec.num_obs,):
            raise ValueError(
                f"obs shape mismatch: expected {(self.spec.num_obs,)}, got {obs.shape}"
            )
        with torch.no_grad():
            tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.model(tensor).cpu().numpy().reshape(-1)
        if action.shape != (self.spec.num_actions,):
            raise RuntimeError(
                f"action shape mismatch: expected {(self.spec.num_actions,)}, got {action.shape}"
            )
        return action.astype(np.float32, copy=False)

    @staticmethod
    def _extract_state_dicts(payload: object) -> tuple[TensorStateDict, TensorStateDict | None]:
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload must be a dictionary")

        actor = payload.get("actor_state_dict")
        critic = payload.get("critic_state_dict")
        if isinstance(actor, dict):
            return (
                PolicyRuntime._normalize_actor_keys(
                    PolicyRuntime._as_tensor_state_dict(actor, "actor_state_dict")
                ),
                PolicyRuntime._normalize_actor_keys(
                    PolicyRuntime._as_tensor_state_dict(critic, "critic_state_dict")
                )
                if isinstance(critic, dict)
                else None,
            )

        model = payload.get("model_state_dict")
        if isinstance(model, dict):
            model_state = PolicyRuntime._as_tensor_state_dict(model, "model_state_dict")
            actor_state = PolicyRuntime._strip_prefix(model_state, "actor")
            critic_state = PolicyRuntime._strip_prefix(model_state, "critic")
            if actor_state:
                return (
                    PolicyRuntime._normalize_actor_keys(actor_state),
                    PolicyRuntime._normalize_actor_keys(critic_state) if critic_state else None,
                )

        raise KeyError(
            "checkpoint must contain actor_state_dict or model_state_dict with actor.* keys"
        )

    @staticmethod
    def _as_tensor_state_dict(raw: object, name: str) -> TensorStateDict:
        if not isinstance(raw, dict):
            raise TypeError(f"{name} must be a state_dict")
        state: TensorStateDict = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string key: {key!r}")
            if not isinstance(value, torch.Tensor):
                continue
            state[key] = value
        if not state:
            raise ValueError(f"{name} does not contain tensor weights")
        return state

    @staticmethod
    def _strip_prefix(state_dict: TensorStateDict, prefix: str) -> TensorStateDict:
        marker = prefix + "."
        return {
            key.removeprefix(marker): value
            for key, value in state_dict.items()
            if key.startswith(marker)
        }

    @staticmethod
    def _normalize_actor_keys(state_dict: TensorStateDict) -> TensorStateDict:
        if any(key.startswith("mlp.") for key in state_dict):
            return state_dict
        normalized: TensorStateDict = {}
        for key, value in state_dict.items():
            first = key.split(".", 1)[0]
            normalized[f"mlp.{key}" if first.isdigit() else key] = value
        return normalized

    @staticmethod
    def _detect_rnn(state_dict: TensorStateDict) -> tuple[str | None, int, int]:
        """从 state_dict 中检测 RNN 类型和参数。

        返回 (rnn_type, hidden_dim, num_layers)。无 RNN 时返回 (None, 0, 0)。
        """
        rnn_keys = [k for k in state_dict if k.startswith("rnn.")]
        if not rnn_keys:
            return None, 0, 0

        # weight_ih_l0 形状: (gate_size * hidden_dim, input_size)
        # GRU gate_size=3, LSTM gate_size=4
        ih_key = "rnn.rnn.weight_ih_l0"
        if ih_key not in state_dict:
            return None, 0, 0

        gate_hidden = state_dict[ih_key].shape[0]
        # 检测层数
        num_layers = 0
        while f"rnn.rnn.weight_ih_l{num_layers}" in state_dict:
            num_layers += 1

        # GRU: gate_size=3, LSTM: gate_size=4
        if gate_hidden % 3 == 0:
            hidden_dim = gate_hidden // 3
            rnn_type = "gru"
        elif gate_hidden % 4 == 0:
            hidden_dim = gate_hidden // 4
            rnn_type = "lstm"
        else:
            raise ValueError(
                f"cannot infer RNN type from weight_ih shape: {state_dict[ih_key].shape}"
            )

        return rnn_type, hidden_dim, num_layers

    @staticmethod
    def _linear_shapes(state_dict: TensorStateDict, prefix: str) -> list[tuple[int, int]]:
        layers: list[tuple[int, tuple[int, int]]] = []
        for key, tensor in state_dict.items():
            if not key.startswith(prefix + ".") or not key.endswith(".weight") or tensor.ndim != 2:
                continue
            parts = key.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                continue
            layers.append((int(parts[1]), (int(tensor.shape[0]), int(tensor.shape[1]))))
        return [shape for _, shape in sorted(layers)]

    @staticmethod
    def _expected_shapes(dims: Sequence[int]) -> list[tuple[int, int]]:
        return [(int(dims[i + 1]), int(dims[i])) for i in range(len(dims) - 1)]

    def _infer_spec(
        self,
        actor_state_dict: TensorStateDict,
        critic_state_dict: TensorStateDict | None,
    ) -> PolicyArchitectureSpec:
        rnn_type, rnn_hidden_dim, rnn_num_layers = self._detect_rnn(actor_state_dict)

        actor = self._linear_shapes(actor_state_dict, "mlp")
        critic = (
            self._linear_shapes(critic_state_dict, "mlp") if critic_state_dict is not None else []
        )
        if not actor:
            raise ValueError("checkpoint does not contain actor MLP linear layers")
        critic_hidden_dims = (
            tuple(int(out) for out, _ in critic[:-1])
            if critic
            else self.runtime.policy.critic_hidden_dims
        )
        num_critic_obs = int(critic[0][1]) if critic else int(self.runtime.policy.num_obs)

        # 对于 GRU，MLP 第一层输入维度是 rnn_hidden_dim，num_obs 从 normalizer 推断
        if rnn_type is not None:
            normalizer_mean = actor_state_dict.get("obs_normalizer._mean")
            num_obs = (
                int(normalizer_mean.shape[1]) if normalizer_mean is not None else int(actor[0][1])
            )
        else:
            num_obs = int(actor[0][1])

        return PolicyArchitectureSpec(
            policy_class_name="ActorCritic",
            num_obs=num_obs,
            num_actions=int(actor[-1][0]),
            actor_hidden_dims=tuple(int(out) for out, _ in actor[:-1]),
            critic_hidden_dims=critic_hidden_dims,
            activation=self.runtime.policy.activation,
            init_noise_std=self.runtime.policy.init_noise_std,
            num_critic_obs=num_critic_obs,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
        )

    def _resolve_spec(
        self,
        spec: PolicyArchitectureSpec,
        actor_state_dict: TensorStateDict,
        critic_state_dict: TensorStateDict | None,
    ) -> PolicyArchitectureSpec:
        critic = (
            self._linear_shapes(critic_state_dict, "mlp") if critic_state_dict is not None else []
        )
        resolved = replace(
            spec, num_critic_obs=int(critic[0][1]) if critic else spec.num_critic_obs
        )

        # MLP shape 验证：GRU 时第一层输入是 rnn_hidden_dim，MLP 时是 num_obs
        mlp_input_dim = resolved.rnn_hidden_dim if resolved.is_recurrent else resolved.num_obs

        actor_expected = self._expected_shapes(
            [mlp_input_dim, *resolved.actor_hidden_dims, resolved.num_actions]
        )
        actor_actual = self._linear_shapes(actor_state_dict, "mlp")
        if actor_actual != actor_expected:
            raise ValueError(f"actor shape mismatch: expected {actor_expected}, got {actor_actual}")
        if critic:
            critic_expected = self._expected_shapes(
                [resolved.num_critic_obs, *resolved.critic_hidden_dims, 1]
            )
            if critic != critic_expected:
                raise ValueError(f"critic shape mismatch: expected {critic_expected}, got {critic}")
        return resolved

    def _load_actor_state(self, actor_state_dict: TensorStateDict) -> None:
        model_state = self.model.state_dict()
        loadable: TensorStateDict = {}
        for key, value in actor_state_dict.items():
            if key not in model_state:
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                raise ValueError(
                    f"actor tensor shape mismatch for {key}: expected {tuple(model_state[key].shape)}, "
                    f"got {tuple(value.shape)}"
                )
            loadable[key] = value

        required_prefixes = ("mlp.", "rnn.") if self.spec.is_recurrent else ("mlp.",)
        missing_required = sorted(
            key
            for key in model_state
            if any(key.startswith(p) for p in required_prefixes) and key not in loadable
        )
        if missing_required:
            raise ValueError(f"actor checkpoint is missing weights: {missing_required}")

        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        required_missing = sorted(
            key for key in missing if any(key.startswith(p) for p in required_prefixes)
        )
        if required_missing:
            raise ValueError(f"actor checkpoint is missing required weights: {required_missing}")
        if unexpected:
            raise ValueError(f"unexpected actor weights after filtering: {unexpected}")

        ignored = set(actor_state_dict) - set(loadable)
        allowed_ignored = {
            "obs_normalizer._var",
            "obs_normalizer.count",
            "distribution.std_param",
            "distribution.log_std_param",
            "std",
            "log_std",
        }
        unsupported = sorted(ignored - allowed_ignored)
        if unsupported:
            raise ValueError(f"unsupported actor checkpoint keys: {unsupported}")

    def _validate_runtime(self, spec: PolicyArchitectureSpec) -> None:
        if int(spec.num_obs) != int(self.runtime.policy.num_obs):
            raise ValueError(
                f"runtime num_obs mismatch: runtime={self.runtime.policy.num_obs}, checkpoint={spec.num_obs}"
            )
        if int(spec.num_actions) != int(self.runtime.policy.num_actions):
            raise ValueError(
                f"runtime num_actions mismatch: runtime={self.runtime.policy.num_actions}, checkpoint={spec.num_actions}"
            )
