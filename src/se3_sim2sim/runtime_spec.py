"""Single runtime contract shared by robot, policy, and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from se3_shared import JointGroup, ObservationConfig

_OBS_CFG = ObservationConfig()
_JOINT_NAMES = JointGroup.joint_names()


@dataclass(frozen=True, slots=True)
class PolicyArchitectureSpec:
    policy_class_name: str = "ActorCritic"
    num_obs: int = _OBS_CFG.num_obs
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationTermSpec:
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    task: str = "wheel_legged_joint_pos"
    spec_name: str = "se3/wheel_legged_joint_pos"
    policy: PolicyArchitectureSpec = PolicyArchitectureSpec()
    joint_names: tuple[str, ...] = _JOINT_NAMES
    actuator_names: tuple[str, ...] = _JOINT_NAMES
    observation_terms: tuple[ObservationTermSpec, ...] = (
        ObservationTermSpec("ang_vel", 3),
        ObservationTermSpec("gravity", 3),
        ObservationTermSpec("commands", 5),
        ObservationTermSpec("leg_joint_pos", 6),
        ObservationTermSpec("leg_joint_vel", 4),
        ObservationTermSpec("wheel_pos_zero", 2),
        ObservationTermSpec("wheel_vel", 2),
        ObservationTermSpec("last_actions", 6),
        ObservationTermSpec("jump_commands", 3),  # [jump_flag, jump_target_height, jump_phase]
    )
    observation_history_length: int = 1
    clip_observations: float = 100.0

    @classmethod
    def for_policy_num_obs(cls, num_obs: int, *, task: str) -> RuntimeSpec:
        """根据 checkpoint 输入维度生成单帧或整数帧历史运行时契约。"""

        num_obs = int(num_obs)
        if num_obs < 1 or num_obs % _OBS_CFG.num_obs != 0:
            raise ValueError(
                f"checkpoint observation 维度必须是基础 34D 契约的整数倍: num_obs={num_obs}"
            )
        history_length = num_obs // _OBS_CFG.num_obs
        return cls(
            task=task,
            policy=replace(PolicyArchitectureSpec(), num_obs=num_obs),
            observation_history_length=history_length,
        )

    def __post_init__(self) -> None:
        if self.observation_history_length < 1:
            raise ValueError("observation_history_length 必须至少为 1")
        expected = self.base_num_obs * self.observation_history_length
        if self.policy.num_obs != expected:
            raise ValueError(
                "policy observation 维度与历史契约不一致: "
                f"num_obs={self.policy.num_obs}, expected={expected}"
            )

    @property
    def base_num_obs(self) -> int:
        return sum(term.size for term in self.observation_terms)

    @property
    def observation_slices(self) -> dict[str, slice]:
        """返回各观测项最新一帧的切片。"""

        out: dict[str, slice] = {}
        cursor = 0
        for term in self.observation_terms:
            current = cursor + (self.observation_history_length - 1) * term.size
            out[term.name] = slice(current, current + term.size)
            cursor += term.size * self.observation_history_length
        return out

    @property
    def observation_history_slices(self) -> dict[str, slice]:
        """返回各观测项包含全部历史帧的连续切片。"""

        out: dict[str, slice] = {}
        cursor = 0
        for term in self.observation_terms:
            size = term.size * self.observation_history_length
            out[term.name] = slice(cursor, cursor + size)
            cursor += size
        return out

    @property
    def observation_component_dims(self) -> tuple[int, ...]:
        return tuple(term.size for term in self.observation_terms)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["observation_slices"] = {
            name: [sl.start, sl.stop] for name, sl in self.observation_slices.items()
        }
        payload["observation_history_slices"] = {
            name: [sl.start, sl.stop] for name, sl in self.observation_history_slices.items()
        }
        return payload


def as_float64(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)
