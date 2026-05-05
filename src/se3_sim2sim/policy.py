"""MJLab checkpoint 的确定性推理加载器。"""

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
    """复现 rsl_rl 的 EmpiricalNormalization 推理行为。"""

    def __init__(self, num_obs: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(1, int(num_obs)))
        self.register_buffer("_std", torch.ones(1, int(num_obs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class _DeterministicActor(nn.Module):
    """只保留 sim2sim 推理需要的 actor 均值网络。"""

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
        self.model = _DeterministicActor(self.spec).to(self.device)
        self._load_actor_state(self.actor_state_dict)
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
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
        return PolicyArchitectureSpec(
            policy_class_name="ActorCritic",
            num_obs=int(actor[0][1]),
            num_actions=int(actor[-1][0]),
            actor_hidden_dims=tuple(int(out) for out, _ in actor[:-1]),
            critic_hidden_dims=critic_hidden_dims,
            activation=self.runtime.policy.activation,
            init_noise_std=self.runtime.policy.init_noise_std,
            num_critic_obs=num_critic_obs,
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
        actor_expected = self._expected_shapes(
            [resolved.num_obs, *resolved.actor_hidden_dims, resolved.num_actions]
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

        missing_mlp = sorted(
            key for key in model_state if key.startswith("mlp.") and key not in loadable
        )
        if missing_mlp:
            raise ValueError(f"actor checkpoint is missing MLP weights: {missing_mlp}")

        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        required_missing = sorted(key for key in missing if key.startswith("mlp."))
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
