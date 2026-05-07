"""Single runtime contract shared by robot, policy, and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PolicyArchitectureSpec:
    policy_class_name: str = "ActorCritic"
    num_obs: int = 27
    num_actions: int = 6
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0
    num_critic_obs: int | None = None

    @property
    def is_sequence(self) -> bool:
        return self.policy_class_name == "ActorCriticSequence"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationTermSpec:
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    task: str = "wheel_legged_joint_pos"
    spec_name: str = "3se/wheel_legged_joint_pos"
    policy: PolicyArchitectureSpec = PolicyArchitectureSpec()
    joint_names: tuple[str, ...] = (
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    )
    actuator_names: tuple[str, ...] = (
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    )
    observation_terms: tuple[ObservationTermSpec, ...] = (
        ObservationTermSpec("ang_vel", 3),
        ObservationTermSpec("gravity", 3),
        ObservationTermSpec("commands", 3),
        ObservationTermSpec("leg_joint_pos", 4),
        ObservationTermSpec("leg_joint_vel", 4),
        ObservationTermSpec("wheel_pos", 2),
        ObservationTermSpec("wheel_vel", 2),
        ObservationTermSpec("actions", 6),
    )
    clip_observations: float = 100.0

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
