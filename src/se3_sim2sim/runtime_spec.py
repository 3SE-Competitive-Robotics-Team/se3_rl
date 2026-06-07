"""sim2sim 端共享的运行契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from se3_shared import (
    LEGACY_LOCOMOTION_CONTRACT,
    JointGroup,
    ObservationConfig,
    ObservationTermSpec,
    TaskContract,
    task_contract_for_num_obs,
)

_OBS_CFG = ObservationConfig()
_JOINT_NAMES = JointGroup.joint_names()


@dataclass(frozen=True, slots=True)
class PolicyArchitectureSpec:
    policy_class_name: str = "ActorCritic"
    contract: TaskContract = LEGACY_LOCOMOTION_CONTRACT
    num_obs: int = LEGACY_LOCOMOTION_CONTRACT.num_obs
    num_actions: int = _OBS_CFG.num_actions
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0
    num_critic_obs: int | None = None
    rnn_type: str | None = None
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 1

    @property
    def is_sequence(self) -> bool:
        return self.policy_class_name == "ActorCriticSequence"

    @property
    def is_recurrent(self) -> bool:
        return self.rnn_type is not None

    @property
    def is_task_mode(self) -> bool:
        return self.contract.is_task_mode

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["contract"] = self.contract.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    task: str = "wheel_legged_joint_pos"
    spec_name: str = "se3/wheel_legged_joint_pos"
    policy: PolicyArchitectureSpec = PolicyArchitectureSpec()
    joint_names: tuple[str, ...] = _JOINT_NAMES
    actuator_names: tuple[str, ...] = _JOINT_NAMES
    observation_terms: tuple[ObservationTermSpec, ...] = (
        LEGACY_LOCOMOTION_CONTRACT.observation.terms
    )
    clip_observations: float = 100.0

    def with_num_obs(self, num_obs: int) -> RuntimeSpec:
        """按 checkpoint 观测维度匹配任务契约。"""
        contract = task_contract_for_num_obs(int(num_obs))
        new_policy = replace(self.policy, contract=contract, num_obs=contract.num_obs)
        return replace(
            self,
            policy=new_policy,
            observation_terms=contract.observation.terms,
        )

    @property
    def observation_slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        cursor = 0
        for term in self.observation_terms:
            out[term.name] = slice(cursor, cursor + term.size)
            cursor += term.size
        return out

    @property
    def observation_component_dims(self) -> tuple[int, ...]:
        return tuple(term.size for term in self.observation_terms)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["observation_slices"] = {
            name: [sl.start, sl.stop] for name, sl in self.observation_slices.items()
        }
        return payload


def as_float64(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)
