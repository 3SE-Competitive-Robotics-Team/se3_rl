from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_PATH = _RESOURCES / "robots" / "serialleg" / "mjcf" / "serialleg_fidelity_cylinder_wheels.xml"

_ROBOT_CFG = SharedRobotConfig()

_ALL_JOINT_NAMES = JointGroup.joint_names()
_LEG_JOINT_NAMES = tuple(
    name
    for name in _ALL_JOINT_NAMES
    if name not in {_ALL_JOINT_NAMES[i] for i in JointGroup.WHEELS}
)
_WHEEL_JOINT_NAMES = tuple(_ALL_JOINT_NAMES[i] for i in JointGroup.WHEELS)


def get_serialleg_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_MJCF_PATH)),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=_LEG_JOINT_NAMES,
                    stiffness=_ROBOT_CFG.leg_kp,
                    damping=_ROBOT_CFG.leg_kd,
                    effort_limit=_ROBOT_CFG.torque_limits[JointGroup.LEGS[0]],
                ),
                BuiltinVelocityActuatorCfg(
                    target_names_expr=_WHEEL_JOINT_NAMES,
                    damping=_ROBOT_CFG.wheel_kd,
                    effort_limit=_ROBOT_CFG.torque_limits[JointGroup.WHEELS[0]],
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.301),
            joint_pos={
                name: _ROBOT_CFG.default_dof_pos[i] for i, name in enumerate(_ALL_JOINT_NAMES)
            },
            joint_vel={".*": 0.0},
        ),
    )
