from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_PATH = (
    _RESOURCES / "robots" / "serialleg" / "mjcf" / "serialleg_fidelity_cylinder_wheels.xml"
)


def get_serialleg_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_MJCF_PATH)),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={
                "lf0_Joint": 0.5412,
                "lf1_Joint": 0.3398,
                "l_wheel_Joint": 0.0,
                "rf0_Joint": 0.5412,
                "rf1_Joint": 0.3398,
                "r_wheel_Joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )
