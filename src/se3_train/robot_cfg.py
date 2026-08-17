from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_C620_14, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.torque_speed_actuator import TorqueSpeedCurveActuatorCfg

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train_obb_trim.xml"

_ROBOT_CFG = SharedRobotConfig()

_WHEEL_JOINT_NAMES = JointGroup.WHEEL_NAMES


def _serialleg_spec_for_training() -> mujoco.MjSpec:
    """加载不含独立世界地面的 SerialLeg MJCF。

    MJLab 场景单独提供地形。保留 MJCF 的全局平面会覆盖 z=0 的生成台阶坑，
    使机器人与平面障碍碰撞，而不是与台阶地形碰撞。
    """
    spec = mujoco.MjSpec.from_file(str(_MJCF_PATH))
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
            break
    return spec


def get_serialleg_closedchain_cfg(*, wheel_kd_override: float | None = None) -> EntityCfg:
    """构造固定使用正式 OBB 闭链 MJCF 的 SerialLeg 训练实体。"""
    leg_actuator_cfg = DcMotorActuatorCfg(
        target_names_expr=JointGroup.POLICY_LEG_NAMES,
        stiffness=_ROBOT_CFG.leg_kp,
        damping=_ROBOT_CFG.leg_kd,
        saturation_effort=DM8009P.stall_torque,
        velocity_limit=DM8009P.no_load_speed,
        effort_limit=DM8009P.rated_torque,
    )
    wheel_kd = _ROBOT_CFG.wheel_kd if wheel_kd_override is None else float(wheel_kd_override)
    return EntityCfg(
        spec_fn=_serialleg_spec_for_training,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                leg_actuator_cfg,
                TorqueSpeedCurveActuatorCfg(
                    target_names_expr=_WHEEL_JOINT_NAMES,
                    stiffness=0.0,
                    damping=wheel_kd,
                    effort_limit=M3508_C620_14.rated_torque,
                    torque_speed_curve=M3508_C620_14.torque_speed_curve,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, _ROBOT_CFG.default_base_height),
            joint_pos=_ROBOT_CFG.default_model_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )
