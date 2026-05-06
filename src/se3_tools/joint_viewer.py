"""展示机器人默认站立姿态。

base 固定在空中, 关节设为参考实现的默认角度, 静态展示。

用法:
    uv run se3-joint-viewer
"""

from pathlib import Path

import mujoco
import mujoco.viewer

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml"

DEFAULT_JOINT_ANGLES = {
    "lf0_Joint": 0.5412,
    "lf1_Joint": 0.3398,
    "l_wheel_Joint": 0.0,
    "rf0_Joint": 0.5412,
    "rf1_Joint": 0.3398,
    "r_wheel_Joint": 0.0,
}

BASE_HEIGHT = 0.28


def _create_fixed_base_mjcf(source_path: str) -> str:
    source = Path(source_path).resolve()
    xml = source.read_text()

    xml = xml.replace("<freejoint />", "")
    xml = xml.replace("<actuator>", "<!-- actuator removed -->\n  <!-- <actuator>")
    xml = xml.replace("</actuator>", "</actuator> -->")

    xml = xml.replace('damping="0" />', 'damping="50" />')
    xml = xml.replace('damping="0"', 'damping="50"')

    xml = xml.replace(
        '<body name="base_link" pos="0 0 0.28">',
        f'<body name="base_link" pos="0 0 {BASE_HEIGHT}">',
    )

    mocap_body = f"""
    <body name="mocap_target" mocap="true" pos="0 0 {BASE_HEIGHT}">
      <geom type="box" size="0.005 0.005 0.005" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>
"""
    xml = xml.replace("</worldbody>", mocap_body + "  </worldbody>")

    equality = """
  <equality>
    <weld body1="base_link" body2="mocap_target" solref="0.002 1" solimp="0.99 0.99 0.001"/>
  </equality>
"""
    xml = xml.replace("</mujoco>", equality + "</mujoco>")

    tmp_path = source.parent / "_joint_viewer_tmp.xml"
    tmp_path.write_text(xml)
    return str(tmp_path)


def main() -> None:
    fixed_mjcf = _create_fixed_base_mjcf(MJCF_PATH)
    model = mujoco.MjModel.from_xml_path(fixed_mjcf)
    data = mujoco.MjData(model)
    Path(fixed_mjcf).unlink(missing_ok=True)

    for jnt_name, angle in DEFAULT_JOINT_ANGLES.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr] = angle

    mujoco.mj_forward(model, data)

    print(f"默认站立姿态: base_z={BASE_HEIGHT}m")
    print(f"关节角: {DEFAULT_JOINT_ANGLES}")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
