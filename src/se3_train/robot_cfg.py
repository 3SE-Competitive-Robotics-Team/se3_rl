import os
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_C620_14, M3508_HEXROLL, FourbarRobotConfig, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.torque_speed_actuator import TorqueSpeedCurveActuatorCfg

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_OPENCHAIN_MJCF_PATH = _MJCF_DIR / "serialleg_fidelity_cylinder_wheels.xml"
_CLOSEDCHAIN_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train_obb_trim.xml"
_FOURBAR_SURROGATE_MJCF_PATH = _MJCF_DIR / "serialleg_fourbar_surrogate_train.xml"
_MJCF_ENV_VAR = "SE3_ROBOT_MJCF"
_MJCF_VARIANT_ENV_VAR = "SE3_ROBOT_MJCF_VARIANT"

_ROBOT_CFG = SharedRobotConfig()
_FOURBAR_ROBOT_CFG = FourbarRobotConfig()

_WHEEL_JOINT_NAMES = JointGroup.WHEEL_NAMES
_WHEEL_LOCK_RANGE = (-1.0e-4, 1.0e-4)
_OPENCHAIN_LEG_JOINT_NAMES = ("lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint")
_OPENCHAIN_JOINT_NAMES = (
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
)
_OPENCHAIN_DEFAULT_DOF_POS = (0.4610, 0.4742, 0.0, 0.4610, 0.4742, 0.0)

OPENCHAIN_MJCF_PATH = _OPENCHAIN_MJCF_PATH
FOURBAR_SURROGATE_MJCF_PATH = _FOURBAR_SURROGATE_MJCF_PATH
CLOSEDCHAIN_MJCF_PATH = _CLOSEDCHAIN_MJCF_PATH


def _resolve_mjcf_path(default_variant: str = "openchain") -> Path:
    """解析训练使用的 MJCF 路径；无覆盖时保持 main 既有开链默认。"""
    override = os.environ.get(_MJCF_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{_MJCF_ENV_VAR} 指向的 MJCF 不存在: {path}")
        return path

    variant = os.environ.get(_MJCF_VARIANT_ENV_VAR, default_variant).strip().lower()
    if variant in {"default", "openchain"}:
        return _OPENCHAIN_MJCF_PATH
    if variant in {
        "fourbar",
        "fourbar-surrogate",
        "fourbar_surrogate",
        "surrogate",
        "equivalent-openchain",
    }:
        return _FOURBAR_SURROGATE_MJCF_PATH
    if variant in {"closedchain", "closedchain-obb", "closedchain_obb", "no-spring", "no_spring"}:
        return _CLOSEDCHAIN_MJCF_PATH
    raise ValueError(
        f"{_MJCF_VARIANT_ENV_VAR}={variant!r} 不支持；可选 fourbar-surrogate/closedchain/openchain，"
        f"或用 {_MJCF_ENV_VAR} 指定 MJCF 路径。"
    )


def _leg_joint_names_for(mjcf_path: Path) -> tuple[str, ...]:
    """根据模型变体选择腿部电机目标。"""
    if mjcf_path.name == _OPENCHAIN_MJCF_PATH.name:
        return _OPENCHAIN_LEG_JOINT_NAMES
    if mjcf_path.name == _FOURBAR_SURROGATE_MJCF_PATH.name:
        return JointGroup.OPENCHAIN_LEG_NAMES
    return JointGroup.POLICY_LEG_NAMES


def _is_fourbar_surrogate_path(mjcf_path: Path) -> bool:
    """判断当前 MJCF 是否属于解析四连杆等效开树模型。"""
    return mjcf_path.name == _FOURBAR_SURROGATE_MJCF_PATH.name


def _load_serialleg_spec(
    mjcf_path: Path,
    *,
    lock_wheels: bool = False,
    strip_floor: bool = True,
) -> mujoco.MjSpec:
    """加载 SerialLeg MJCF，并按任务需要锁定轮轴。"""
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    if strip_floor:
        for geom in list(spec.worldbody.geoms):
            if geom.name == "floor":
                spec.delete(geom)
                break
    if lock_wheels:
        for joint in spec.joints:
            if joint.name in _WHEEL_JOINT_NAMES:
                joint.limited = True
                joint.range[:] = _WHEEL_LOCK_RANGE
                joint.damping[0] = 10.0
                joint.frictionloss = 1.0
    return spec


def get_serialleg_cfg(
    *,
    mjcf_path: Path | None = None,
    default_mjcf_variant: str = "openchain",
    lock_wheels: bool = False,
    wheel_kd_override: float | None = None,
) -> EntityCfg:
    """构造训练实体；默认沿用全局 MJCF，也允许按任务显式覆盖。"""
    mjcf_path = _resolve_mjcf_path(default_mjcf_variant) if mjcf_path is None else Path(mjcf_path)
    leg_joint_names = _leg_joint_names_for(mjcf_path)
    is_fourbar_surrogate = _is_fourbar_surrogate_path(mjcf_path)
    is_openchain = mjcf_path.name == _OPENCHAIN_MJCF_PATH.name
    if is_fourbar_surrogate:
        leg_actuator_cfg = IdealPdActuatorCfg(
            target_names_expr=leg_joint_names,
            stiffness=0.0,
            damping=0.0,
            effort_limit=float("inf"),
        )
    elif is_openchain:
        leg_actuator_cfg = DcMotorActuatorCfg(
            target_names_expr=leg_joint_names,
            stiffness=40.0,
            damping=2.0,
            saturation_effort=DM8009P.stall_torque,
            velocity_limit=DM8009P.no_load_speed,
            effort_limit=DM8009P.rated_torque,
        )
    else:
        leg_actuator_cfg = DcMotorActuatorCfg(
            target_names_expr=leg_joint_names,
            stiffness=_FOURBAR_ROBOT_CFG.leg_kp,
            damping=_FOURBAR_ROBOT_CFG.leg_kd,
            saturation_effort=DM8009P.stall_torque,
            velocity_limit=DM8009P.no_load_speed,
            effort_limit=DM8009P.rated_torque,
        )
    wheel_kd = (
        (_ROBOT_CFG.wheel_kd if is_openchain else _FOURBAR_ROBOT_CFG.wheel_kd)
        if wheel_kd_override is None
        else float(wheel_kd_override)
    )
    if is_openchain:
        wheel_actuator_cfg = DcMotorActuatorCfg(
            target_names_expr=_WHEEL_JOINT_NAMES,
            stiffness=0.0,
            damping=wheel_kd,
            saturation_effort=M3508_HEXROLL.stall_torque,
            velocity_limit=M3508_HEXROLL.no_load_speed,
            effort_limit=M3508_HEXROLL.rated_torque,
        )
        init_joint_pos = dict(zip(_OPENCHAIN_JOINT_NAMES, _OPENCHAIN_DEFAULT_DOF_POS, strict=True))
    else:
        wheel_actuator_cfg = TorqueSpeedCurveActuatorCfg(
            target_names_expr=_WHEEL_JOINT_NAMES,
            stiffness=0.0,
            damping=wheel_kd,
            effort_limit=M3508_C620_14.rated_torque,
            torque_speed_curve=M3508_C620_14.torque_speed_curve,
        )
        init_joint_pos = _FOURBAR_ROBOT_CFG.default_model_joint_pos
    return EntityCfg(
        spec_fn=lambda: _load_serialleg_spec(
            mjcf_path,
            lock_wheels=lock_wheels,
            strip_floor=is_openchain,
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                leg_actuator_cfg,
                wheel_actuator_cfg,
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(
                0.0,
                0.0,
                _ROBOT_CFG.default_base_height
                if is_openchain
                else _FOURBAR_ROBOT_CFG.default_base_height,
            ),
            joint_pos=init_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )
