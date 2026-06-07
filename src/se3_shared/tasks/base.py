"""任务级运行契约。

每个任务种类声明自己的 actor 观测布局和动作维度。训练、蒸馏和 sim2sim
只通过契约对齐维度，不把不同任务的观测定义塞进同一个全局配置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ObservationTermSpec:
    """actor 观测中的一个连续字段。"""

    name: str
    size: int


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    """actor 观测布局。"""

    terms: tuple[ObservationTermSpec, ...]

    @property
    def num_obs(self) -> int:
        """总观测维度。"""
        return sum(term.size for term in self.terms)

    @property
    def component_dims(self) -> tuple[int, ...]:
        """各字段维度。"""
        return tuple(term.size for term in self.terms)

    @property
    def slices(self) -> dict[str, slice]:
        """各字段在扁平 actor obs 中的切片。"""
        out: dict[str, slice] = {}
        cursor = 0
        for term in self.terms:
            out[term.name] = slice(cursor, cursor + term.size)
            cursor += term.size
        return out

    def require_slice(self, name: str) -> slice:
        """读取字段切片，不存在时直接报错。"""
        try:
            return self.slices[name]
        except KeyError as exc:
            raise KeyError(f"观测布局中不存在字段: {name}") from exc

    def to_dict(self) -> dict[str, object]:
        """转成可序列化字典。"""
        return {
            "terms": [asdict(term) for term in self.terms],
            "num_obs": self.num_obs,
            "slices": {name: [sl.start, sl.stop] for name, sl in self.slices.items()},
        }


@dataclass(frozen=True, slots=True)
class TaskContract:
    """任务种类的共享契约。"""

    name: str
    observation: ObservationLayout
    num_actions: int = 6
    is_task_mode: bool = False

    @property
    def num_obs(self) -> int:
        """actor 观测维度。"""
        return self.observation.num_obs

    def to_dict(self) -> dict[str, object]:
        """转成可序列化字典。"""
        return {
            "name": self.name,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
            "is_task_mode": self.is_task_mode,
            "observation": self.observation.to_dict(),
        }


COMMON_LOCOMOTION_TERMS: tuple[ObservationTermSpec, ...] = (
    ObservationTermSpec("base_ang_vel", 3),
    ObservationTermSpec("projected_gravity", 3),
    ObservationTermSpec("commands", 5),
    ObservationTermSpec("leg_joint_pos", 4),
    ObservationTermSpec("leg_joint_vel", 4),
    ObservationTermSpec("wheel_pos", 2),
    ObservationTermSpec("wheel_vel", 2),
    ObservationTermSpec("last_actions", 6),
)


__all__ = [
    "COMMON_LOCOMOTION_TERMS",
    "ObservationLayout",
    "ObservationTermSpec",
    "TaskContract",
]
