"""Training checkpoint loader for deterministic sim2sim inference."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from wheel_legged_gym.rsl_rl.modules.actor_critic import ActorCritic

from .runtime_spec import PolicyArchitectureSpec, RuntimeSpec


class PolicyRuntime:
    def __init__(self, *, checkpoint: Path, device: str, runtime: RuntimeSpec) -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        self.runtime = runtime
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        if "model_state_dict" not in payload:
            raise KeyError(f"{self.checkpoint_path} is missing model_state_dict")
        self.state_dict = payload["model_state_dict"]
        self.iteration = payload.get("iter", "unknown")
        inferred = self._infer_spec(self.state_dict)
        self.spec = self._resolve_spec(inferred, self.state_dict)
        self._validate_runtime(self.spec)
        if self.spec.is_sequence:
            raise NotImplementedError("SE3 workflow currently supports ActorCritic checkpoints, not sequence checkpoints")
        self.model = ActorCritic(
            num_actor_obs=self.spec.num_obs,
            num_critic_obs=self.spec.num_critic_obs,
            num_actions=self.spec.num_actions,
            actor_hidden_dims=list(self.spec.actor_hidden_dims),
            critic_hidden_dims=list(self.spec.critic_hidden_dims),
            activation=self.spec.activation,
            init_noise_std=self.spec.init_noise_std,
        ).to(self.device)
        self.model.load_state_dict(self.state_dict, strict=True)
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape != (self.spec.num_obs,):
            raise ValueError(f"obs shape mismatch: expected {(self.spec.num_obs,)}, got {obs.shape}")
        with torch.no_grad():
            tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.model.act_inference(tensor).cpu().numpy().reshape(-1)
        if action.shape != (self.spec.num_actions,):
            raise RuntimeError(f"action shape mismatch: expected {(self.spec.num_actions,)}, got {action.shape}")
        return action.astype(np.float32, copy=False)

    @staticmethod
    def _linear_shapes(state_dict: dict[str, torch.Tensor], prefix: str) -> list[tuple[int, int]]:
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

    def _infer_spec(self, state_dict: dict[str, torch.Tensor]) -> PolicyArchitectureSpec:
        actor = self._linear_shapes(state_dict, "actor")
        critic = self._linear_shapes(state_dict, "critic")
        if not actor or not critic:
            raise ValueError("checkpoint does not contain actor/critic linear layers")
        return PolicyArchitectureSpec(
            policy_class_name="ActorCritic",
            num_obs=int(actor[0][1]),
            num_actions=int(actor[-1][0]),
            actor_hidden_dims=tuple(int(out) for out, _ in actor[:-1]),
            critic_hidden_dims=tuple(int(out) for out, _ in critic[:-1]),
            activation=self.runtime.policy.activation,
            init_noise_std=self.runtime.policy.init_noise_std,
            num_critic_obs=int(critic[0][1]),
        )

    def _resolve_spec(
        self,
        spec: PolicyArchitectureSpec,
        state_dict: dict[str, torch.Tensor],
    ) -> PolicyArchitectureSpec:
        critic = self._linear_shapes(state_dict, "critic")
        num_critic_obs = int(critic[0][1])
        resolved = replace(spec, num_critic_obs=num_critic_obs)
        actor_expected = self._expected_shapes([resolved.num_obs, *resolved.actor_hidden_dims, resolved.num_actions])
        critic_expected = self._expected_shapes([resolved.num_critic_obs, *resolved.critic_hidden_dims, 1])
        actor_actual = self._linear_shapes(state_dict, "actor")
        critic_actual = self._linear_shapes(state_dict, "critic")
        if actor_actual != actor_expected:
            raise ValueError(f"actor shape mismatch: expected {actor_expected}, got {actor_actual}")
        if critic_actual != critic_expected:
            raise ValueError(f"critic shape mismatch: expected {critic_expected}, got {critic_actual}")
        return resolved

    def _validate_runtime(self, spec: PolicyArchitectureSpec) -> None:
        if int(spec.num_obs) != int(self.runtime.policy.num_obs):
            raise ValueError(f"runtime num_obs mismatch: runtime={self.runtime.policy.num_obs}, checkpoint={spec.num_obs}")
        if int(spec.num_actions) != int(self.runtime.policy.num_actions):
            raise ValueError(
                f"runtime num_actions mismatch: runtime={self.runtime.policy.num_actions}, checkpoint={spec.num_actions}"
            )
