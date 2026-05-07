from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_PATH = _RESOURCES / "robots" / "serialleg" / "mjcf" / "serialleg_fidelity_cylinder_wheels.xml"


def get_serialleg_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_MJCF_PATH)),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=("lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"),
                    stiffness=40.0,
                    damping=2.0,
                    effort_limit=30.0,
                ),
                BuiltinVelocityActuatorCfg(
                    target_names_expr=("l_wheel_Joint", "r_wheel_Joint"),
                    damping=0.5,
                    effort_limit=3.3,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.301),
            joint_pos={
                "lf0_Joint": 0.6171,
                "lf1_Joint": 0.2070,
                "l_wheel_Joint": 0.0,
                "rf0_Joint": 0.6171,
                "rf1_Joint": 0.2070,
                "r_wheel_Joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
