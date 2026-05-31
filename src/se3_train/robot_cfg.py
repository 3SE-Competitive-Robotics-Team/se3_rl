import os
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_HEXROLL, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_DEFAULT_MJCF_PATH = _MJCF_DIR / "serialleg_fidelity_cylinder_wheels.xml"
_CLOSEDCHAIN_SPRING_MJCF_PATH = (
    _MJCF_DIR / "serialleg_fidelity_cylinder_wheels_closedchain_spring.xml"
)
_MJCF_ENV_VAR = "SE3_ROBOT_MJCF"
_MJCF_VARIANT_ENV_VAR = "SE3_ROBOT_MJCF_VARIANT"

_ROBOT_CFG = SharedRobotConfig()

_ALL_JOINT_NAMES = JointGroup.joint_names()
_WHEEL_JOINT_NAMES = ("l_wheel_Joint", "r_wheel_Joint")
_LEG_JOINT_NAMES = tuple(n for n in _ALL_JOINT_NAMES if n not in set(_WHEEL_JOINT_NAMES))


def _resolve_mjcf_path() -> Path:
    """解析训练使用的 MJCF 路径，默认保持原模型不变。"""
    override = os.environ.get(_MJCF_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{_MJCF_ENV_VAR} 指向的 MJCF 不存在: {path}")
        return path

    variant = os.environ.get(_MJCF_VARIANT_ENV_VAR, "default").strip().lower()
    if variant in {"default", "openchain", "no-spring", "no_spring"}:
        return _DEFAULT_MJCF_PATH
    if variant in {"closedchain", "closedchain_spring", "spring", "fourbar"}:
        return _CLOSEDCHAIN_SPRING_MJCF_PATH
    raise ValueError(
        f"{_MJCF_VARIANT_ENV_VAR}={variant!r} 不支持；可选 default/closedchain_spring，"
        f"或用 {_MJCF_ENV_VAR} 指定 MJCF 路径。"
    )


def get_serialleg_cfg() -> EntityCfg:
    mjcf_path = _resolve_mjcf_path()
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(mjcf_path)),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                DcMotorActuatorCfg(
                    target_names_expr=_LEG_JOINT_NAMES,
                    stiffness=_ROBOT_CFG.leg_kp,
                    damping=_ROBOT_CFG.leg_kd,
                    saturation_effort=DM8009P.stall_torque,
                    velocity_limit=DM8009P.no_load_speed,
                    effort_limit=DM8009P.rated_torque,
                ),
                DcMotorActuatorCfg(
                    target_names_expr=_WHEEL_JOINT_NAMES,
                    stiffness=0.0,
                    damping=_ROBOT_CFG.wheel_kd,
                    saturation_effort=M3508_HEXROLL.stall_torque,
                    velocity_limit=M3508_HEXROLL.no_load_speed,
                    effort_limit=M3508_HEXROLL.rated_torque,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, _ROBOT_CFG.default_base_height),
            joint_pos={
                name: _ROBOT_CFG.default_dof_pos[i] for i, name in enumerate(_ALL_JOINT_NAMES)
            },
            joint_vel={".*": 0.0},
        ),
    )
