import os
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_HEXROLL, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_OPENCHAIN_MJCF_PATH = _MJCF_DIR / "serialleg_fidelity_cylinder_wheels.xml"
_CLOSEDCHAIN_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train.xml"
_CLOSEDCHAIN_SPRING_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train_spring.xml"
_MJCF_ENV_VAR = "SE3_ROBOT_MJCF"
_MJCF_VARIANT_ENV_VAR = "SE3_ROBOT_MJCF_VARIANT"

_ROBOT_CFG = SharedRobotConfig()

_WHEEL_JOINT_NAMES = JointGroup.WHEEL_NAMES


def _resolve_mjcf_path() -> Path:
    """解析训练使用的 MJCF 路径，默认使用闭链四连杆无气弹簧常力模型。"""
    override = os.environ.get(_MJCF_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{_MJCF_ENV_VAR} 指向的 MJCF 不存在: {path}")
        return path

    variant = os.environ.get(_MJCF_VARIANT_ENV_VAR, "closedchain").strip().lower()
    if variant in {"default", "closedchain", "fourbar", "no-spring", "no_spring"}:
        return _CLOSEDCHAIN_MJCF_PATH
    if variant in {"closedchain_spring", "spring", "gas_spring", "gas-spring"}:
        return _CLOSEDCHAIN_SPRING_MJCF_PATH
    if variant in {"openchain"}:
        return _OPENCHAIN_MJCF_PATH
    raise ValueError(
        f"{_MJCF_VARIANT_ENV_VAR}={variant!r} 不支持；可选 closedchain/closedchain_spring/openchain，"
        f"或用 {_MJCF_ENV_VAR} 指定 MJCF 路径。"
    )


def _leg_joint_names_for(mjcf_path: Path) -> tuple[str, ...]:
    """根据模型变体选择腿部电机目标。"""
    if mjcf_path.name == _OPENCHAIN_MJCF_PATH.name:
        return JointGroup.OPENCHAIN_LEG_NAMES
    return JointGroup.POLICY_LEG_NAMES


def get_serialleg_cfg() -> EntityCfg:
    mjcf_path = _resolve_mjcf_path()
    leg_joint_names = _leg_joint_names_for(mjcf_path)
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(mjcf_path)),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                DcMotorActuatorCfg(
                    target_names_expr=leg_joint_names,
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
            joint_pos=_ROBOT_CFG.default_model_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )
