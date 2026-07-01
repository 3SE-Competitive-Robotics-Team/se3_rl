import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_C620_14, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_OPENCHAIN_MJCF_PATH = _MJCF_DIR / "serialleg_fidelity_cylinder_wheels.xml"
_CLOSEDCHAIN_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train_obb_trim.xml"
_FOURBAR_SURROGATE_MJCF_PATH = _MJCF_DIR / "serialleg_fourbar_surrogate_train.xml"
_FOURBAR_SURROGATE_STAIR_MJCF_PATH = (
    _MJCF_DIR / "serialleg_fourbar_surrogate_stair_visualbase_coacd_train.xml"
)
_MJCF_ENV_VAR = "SE3_ROBOT_MJCF"
_MJCF_VARIANT_ENV_VAR = "SE3_ROBOT_MJCF_VARIANT"

_ROBOT_CFG = SharedRobotConfig()

_WHEEL_JOINT_NAMES = JointGroup.WHEEL_NAMES


def _resolve_mjcf_path() -> Path:
    """解析训练使用的 MJCF 路径，默认使用四连杆 surrogate 模型。"""
    override = os.environ.get(_MJCF_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{_MJCF_ENV_VAR} 指向的 MJCF 不存在: {path}")
        return path

    variant = os.environ.get(_MJCF_VARIANT_ENV_VAR, "fourbar-surrogate").strip().lower()
    if variant in {
        "default",
        "fourbar",
        "fourbar-surrogate",
        "fourbar_surrogate",
        "surrogate",
        "equivalent-openchain",
    }:
        return _FOURBAR_SURROGATE_MJCF_PATH
    if variant in {
        "stair",
        "stair-surrogate",
        "stair_surrogate",
        "fourbar-surrogate-stair",
        "fourbar_surrogate_stair",
        "stair-visualbase",
        "stair_visualbase",
    }:
        return _FOURBAR_SURROGATE_STAIR_MJCF_PATH
    if variant in {"closedchain", "closedchain-obb", "closedchain_obb", "no-spring", "no_spring"}:
        return _CLOSEDCHAIN_MJCF_PATH
    if variant in {"openchain"}:
        return _OPENCHAIN_MJCF_PATH
    raise ValueError(
        f"{_MJCF_VARIANT_ENV_VAR}={variant!r} 不支持；可选 fourbar-surrogate/closedchain/openchain，"
        f"stair-surrogate 或用 {_MJCF_ENV_VAR} 指定 MJCF 路径。"
    )


def _leg_joint_names_for(mjcf_path: Path) -> tuple[str, ...]:
    """根据模型变体选择腿部电机目标。"""
    if mjcf_path.name in {
        _OPENCHAIN_MJCF_PATH.name,
        _FOURBAR_SURROGATE_MJCF_PATH.name,
        _FOURBAR_SURROGATE_STAIR_MJCF_PATH.name,
    }:
        return JointGroup.OPENCHAIN_LEG_NAMES
    return JointGroup.POLICY_LEG_NAMES


def _is_fourbar_surrogate_path(mjcf_path: Path) -> bool:
    """判断当前 MJCF 是否属于解析四连杆等效开树模型。"""
    return mjcf_path.name in {
        _FOURBAR_SURROGATE_MJCF_PATH.name,
        _FOURBAR_SURROGATE_STAIR_MJCF_PATH.name,
    }


def _serialleg_spec_for_training(mjcf_path: Path) -> mujoco.MjSpec:
    """Load SerialLeg MJCF without its standalone world floor.

    MJLab scenes provide terrain separately. Keeping the MJCF's global plane
    covers generated stair pits at z=0 and makes robots collide with a flat
    barrier instead of the stair terrain.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
            break
    return spec


def get_serialleg_cfg(
    *, mjcf_path: Path | None = None, wheel_kd_override: float | None = None
) -> EntityCfg:
    """构造训练实体；默认沿用全局 MJCF，也允许按任务显式覆盖。

    执行器使用 MuJoCo 原生 builtin 元素：
    - 腿部：``<position>`` 执行器，由 MuJoCo 在隐式积分步内完成关节级 PD 控制。
    - 轮子：``<velocity>`` 执行器，由 MuJoCo 完成速度阻尼控制。

    策略 action term 只需设置关节位置/速度目标，PD 计算和动作延迟
    均由执行器层处理。
    """
    mjcf_path = _resolve_mjcf_path() if mjcf_path is None else Path(mjcf_path)
    leg_joint_names = _leg_joint_names_for(mjcf_path)

    # 动作延迟：从共享配置转为物理步级 lag。
    delay_cfg = _ROBOT_CFG.action_delay
    if delay_cfg.enabled:
        min_lag, max_lag = delay_cfg.step_bounds(_ROBOT_CFG.sim_dt)
    else:
        min_lag, max_lag = 0, 0

    leg_actuator_cfg = BuiltinPositionActuatorCfg(
        target_names_expr=leg_joint_names,
        stiffness=_ROBOT_CFG.leg_kp,
        damping=_ROBOT_CFG.leg_kd,
        effort_limit=DM8009P.rated_torque,
        delay_min_lag=min_lag,
        delay_max_lag=max_lag,
    )
    wheel_kd = _ROBOT_CFG.wheel_kd if wheel_kd_override is None else float(wheel_kd_override)
    wheel_actuator_cfg = BuiltinVelocityActuatorCfg(
        target_names_expr=_WHEEL_JOINT_NAMES,
        damping=wheel_kd,
        effort_limit=M3508_C620_14.rated_torque,
        delay_min_lag=min_lag,
        delay_max_lag=max_lag,
    )
    return EntityCfg(
        spec_fn=lambda: _serialleg_spec_for_training(mjcf_path),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                leg_actuator_cfg,
                wheel_actuator_cfg,
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, _ROBOT_CFG.default_base_height),
            joint_pos=_ROBOT_CFG.default_model_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )
