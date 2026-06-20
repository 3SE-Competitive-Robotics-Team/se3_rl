"""WheelDog 机器人物理与训练控制配置。"""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_RESOURCES = Path(__file__).resolve().parents[4] / "assets"
_MJCF_PATH = _RESOURCES / "robots" / "minidog" / "mjcf" / "minidog_16dof_20kg.xml"

DOG_BASE_HEIGHT = 0.32
"""默认 base_link 高度，对应 M20 风格前后反向屈膝的深蹲姿态。"""

DOG_JOINT_NAMES: tuple[str, ...] = (
    "fl_abad_joint",
    "fl_hip_joint",
    "fl_knee_joint",
    "fl_wheel_joint",
    "fr_abad_joint",
    "fr_hip_joint",
    "fr_knee_joint",
    "fr_wheel_joint",
    "hl_abad_joint",
    "hl_hip_joint",
    "hl_knee_joint",
    "hl_wheel_joint",
    "hr_abad_joint",
    "hr_hip_joint",
    "hr_knee_joint",
    "hr_wheel_joint",
)

DOG_LEG_JOINT_NAMES: tuple[str, ...] = tuple(
    name for name in DOG_JOINT_NAMES if "wheel" not in name
)
DOG_WHEEL_JOINT_NAMES: tuple[str, ...] = tuple(name for name in DOG_JOINT_NAMES if "wheel" in name)
DOG_LEG_JOINT_IDS: tuple[int, ...] = tuple(
    idx for idx, name in enumerate(DOG_JOINT_NAMES) if "wheel" not in name
)
DOG_WHEEL_JOINT_IDS: tuple[int, ...] = tuple(
    idx for idx, name in enumerate(DOG_JOINT_NAMES) if "wheel" in name
)
DOG_ABAD_JOINT_IDS: tuple[int, ...] = (0, 4, 8, 12)
DOG_HIP_JOINT_IDS: tuple[int, ...] = (1, 5, 9, 13)
DOG_KNEE_JOINT_IDS: tuple[int, ...] = (2, 6, 10, 14)

DOG_WHEEL_BODY_NAMES: tuple[str, ...] = (
    "fl_wheel_link",
    "fr_wheel_link",
    "hl_wheel_link",
    "hr_wheel_link",
)

DOG_DEFAULT_JOINT_POS: tuple[float, ...] = (
    0.0,
    -0.97,
    1.8,
    0.0,
    0.0,
    -0.97,
    1.8,
    0.0,
    0.0,
    0.97,
    -1.8,
    0.0,
    0.0,
    0.97,
    -1.8,
    0.0,
)
"""M20 风格默认姿态：前后膝盖都向机身内侧折，四轮贴地。"""


def _load_wheel_dog_spec() -> mujoco.MjSpec:
    """加载最简 WheelDog MJCF，并移除训练场景不需要的 floor/XML actuator。"""
    spec = mujoco.MjSpec.from_file(str(_MJCF_PATH))
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
            break
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def get_wheel_dog_cfg() -> EntityCfg:
    """构造 WheelDog 训练实体配置。"""
    return EntityCfg(
        spec_fn=_load_wheel_dog_spec,
        articulation=EntityArticulationInfoCfg(
            soft_joint_pos_limit_factor=0.9,
            actuators=(
                DcMotorActuatorCfg(
                    target_names_expr=DOG_LEG_JOINT_NAMES,
                    stiffness=80.0,
                    damping=2.0,
                    saturation_effort=50.0,
                    effort_limit=50.0,
                    velocity_limit=22.4,
                ),
                DcMotorActuatorCfg(
                    target_names_expr=DOG_WHEEL_JOINT_NAMES,
                    stiffness=0.0,
                    damping=0.6,
                    armature=0.0024,
                    saturation_effort=12.0,
                    effort_limit=12.0,
                    velocity_limit=79.3,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, DOG_BASE_HEIGHT),
            joint_pos={
                name: DOG_DEFAULT_JOINT_POS[idx] for idx, name in enumerate(DOG_JOINT_NAMES)
            },
            joint_vel={".*": 0.0},
        ),
    )


__all__ = [
    "DOG_ABAD_JOINT_IDS",
    "DOG_BASE_HEIGHT",
    "DOG_DEFAULT_JOINT_POS",
    "DOG_HIP_JOINT_IDS",
    "DOG_JOINT_NAMES",
    "DOG_KNEE_JOINT_IDS",
    "DOG_LEG_JOINT_IDS",
    "DOG_LEG_JOINT_NAMES",
    "DOG_WHEEL_BODY_NAMES",
    "DOG_WHEEL_JOINT_IDS",
    "DOG_WHEEL_JOINT_NAMES",
    "get_wheel_dog_cfg",
]
