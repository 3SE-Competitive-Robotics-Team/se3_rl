from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_HEXROLL, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_PATH = _RESOURCES / "robots" / "serialleg" / "mjcf" / "serialleg_fidelity_cylinder_wheels.xml"

_ROBOT_CFG = SharedRobotConfig()

_ALL_JOINT_NAMES = JointGroup.joint_names()
_WHEEL_JOINT_NAMES = ("l_wheel_Joint", "r_wheel_Joint")
_LEG_JOINT_NAMES = tuple(n for n in _ALL_JOINT_NAMES if n not in set(_WHEEL_JOINT_NAMES))


def get_serialleg_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_MJCF_PATH)),
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
            pos=(0.0, 0.0, 0.301),
            joint_pos={
                name: _ROBOT_CFG.default_dof_pos[i] for i, name in enumerate(_ALL_JOINT_NAMES)
            },
            joint_vel={".*": 0.0},
        ),
    )
