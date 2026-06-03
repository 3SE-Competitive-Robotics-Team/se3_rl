"""展示机器人默认站立姿态。

base 固定在空中, 关节设为参考实现的默认角度, 静态展示。

用法:
    uv run se3-joint-viewer
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer

from se3_shared import JointGroup, RobotConfig

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train.xml"
OPENCHAIN_MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml"
FOURBAR_SURROGATE_MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_fourbar_surrogate_train.xml"
CLOSEDCHAIN_SPRING_MJCF_PATH = (
    "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_spring.xml"
)

_ROBOT_CFG = RobotConfig()
DEFAULT_JOINT_ANGLES = _ROBOT_CFG.default_model_joint_pos

BASE_HEIGHT = _ROBOT_CFG.default_base_height
VIEWER_JOINT_DAMPING = 6.0
VIEWER_POSITION_KP = 40.0
VIEWER_FULL_ROTATION_CTRL_RANGE = (-6.283185307179586, 6.283185307179586)

LEG_CTRL_RANGES = {
    # 前主动杆是整腿摆动的共模参考，物理上可整周转；只有两主动杆夹角受 tendon 限位。
    # 这里给整周范围，避免 viewer 把整腿摆动卡在一个小区间。
    "lf0_Joint": (-3.14159265, 3.14159265),
    "rf0_Joint": (-3.14159265, 3.14159265),
}
ACTIVE_ROD_ANGLE_CTRL_RANGES = {
    "l_active_rod_angle": _ROBOT_CFG.active_rod_angle_limits,
    "r_active_rod_angle": _ROBOT_CFG.active_rod_angle_limits,
}
OPENCHAIN_LEG_CTRL_RANGES = {
    "lf0_Joint": (-1.0, 1.2),
    "lf1_Joint": (-0.6, 0.8),
    "rf0_Joint": (-1.0, 1.2),
    "rf1_Joint": (-0.6, 0.8),
}

WHEEL_VEL_CTRL_RANGES = {name: (-30.0, 30.0) for name in JointGroup.WHEEL_NAMES}


def _filter_geom_view(xml: str, geom_view: str) -> str:
    if geom_view == "both":
        return xml

    root = ET.fromstring(xml)
    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    for geom in list(root.iter("geom")):
        group = geom.get("group")
        name = geom.get("name", "")
        if name == "floor":
            continue
        if (geom_view == "visual" and group == "0") or (geom_view == "collision" and group == "1"):
            parent_by_child[geom].remove(geom)
    return ET.tostring(root, encoding="unicode")


def _viewer_actuator_xml(xml: str, *, position_kp: float = VIEWER_POSITION_KP) -> str:
    lines = ["  <actuator>", *_viewer_actuator_inner_lines(xml, position_kp=position_kp)]
    lines.append("  </actuator>")
    return "\n".join(lines)


def _joint_ctrl_ranges_from_xml(
    xml: str,
    fallback: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    root = ET.fromstring(xml)
    joints = {joint.get("name"): joint for joint in root.iter("joint") if joint.get("name")}
    ranges: dict[str, tuple[float, float]] = {}
    for joint_name, fallback_range in fallback.items():
        joint = joints.get(joint_name)
        if joint is None:
            ranges[joint_name] = fallback_range
            continue
        if joint.get("limited", "").lower() == "false":
            ranges[joint_name] = VIEWER_FULL_ROTATION_CTRL_RANGE
            continue
        range_text = joint.get("range")
        if range_text:
            low, high = (float(value) for value in range_text.split())
            ranges[joint_name] = (low, high)
        else:
            ranges[joint_name] = fallback_range
    return ranges


def _viewer_actuator_inner_lines(xml: str, *, position_kp: float = VIEWER_POSITION_KP) -> list[str]:
    lines = []
    closedchain = 'name="l_active_rod_angle"' in xml and 'name="r_active_rod_angle"' in xml
    fallback_leg_ranges = LEG_CTRL_RANGES if closedchain else OPENCHAIN_LEG_CTRL_RANGES
    leg_ranges = _joint_ctrl_ranges_from_xml(xml, fallback_leg_ranges)
    for joint_name, (ctrl_min, ctrl_max) in leg_ranges.items():
        lines.append(
            f'    <position name="{joint_name}_viewer_pos" joint="{joint_name}" '
            f'kp="{position_kp:g}" ctrlrange="{ctrl_min} {ctrl_max}" forcerange="-40 40" />'
        )
    if closedchain:
        for tendon_name, (ctrl_min, ctrl_max) in ACTIVE_ROD_ANGLE_CTRL_RANGES.items():
            lines.append(
                f'    <position name="{tendon_name}_viewer_pos" tendon="{tendon_name}" '
                f'kp="{position_kp:g}" ctrlrange="{ctrl_min} {ctrl_max}" forcerange="-40 40" />'
            )
    for joint_name, (ctrl_min, ctrl_max) in WHEEL_VEL_CTRL_RANGES.items():
        lines.append(
            f'    <velocity name="{joint_name}_viewer_vel" joint="{joint_name}" '
            f'kv="0.5" ctrlrange="{ctrl_min} {ctrl_max}" forcerange="-2 2" />'
        )
    return lines


def _drop_actuator_blocks(xml: str) -> str:
    while "<actuator>" in xml:
        start = xml.find("<actuator>")
        end = xml.find("</actuator>", start)
        if end < 0:
            return xml
        end += len("</actuator>")
        xml = xml[:start] + "  <!-- joint_viewer 临时移除原 actuator。 -->\n" + xml[end:]
    return xml


def _drop_keyframe_blocks(xml: str) -> str:
    while "<keyframe>" in xml:
        start = xml.find("<keyframe>")
        end = xml.find("</keyframe>", start)
        if end < 0:
            return xml
        end += len("</keyframe>")
        xml = xml[:start] + "  <!-- joint_viewer 固定 base 后临时移除 keyframe。 -->\n" + xml[end:]
    return xml


def _add_viewer_actuators(xml: str, *, position_kp: float = VIEWER_POSITION_KP) -> str:
    if "<actuator>" in xml:
        return xml.replace(
            "</actuator>",
            "\n".join(_viewer_actuator_inner_lines(xml, position_kp=position_kp)) + "\n  </actuator>",
            1,
        )
    return xml.replace("</mujoco>", _viewer_actuator_xml(xml, position_kp=position_kp) + "\n</mujoco>")


def _set_floor_contact(xml: str, *, enabled: bool) -> str:
    if enabled:
        return xml
    lines: list[str] = []
    for line in xml.splitlines(keepends=True):
        if '<geom name="floor"' in line:
            line = line.replace('contype="2"', 'contype="0"')
            line = line.replace('conaffinity="1"', 'conaffinity="0"')
        lines.append(line)
    return "".join(lines)


def _create_fixed_base_mjcf(
    source_path: str | Path,
    *,
    base_height: float = BASE_HEIGHT,
    floor_contact: bool = False,
    drop_model_actuators: bool = False,
    geom_view: str = "both",
    viewer_position_kp: float = VIEWER_POSITION_KP,
    viewer_joint_damping: float = VIEWER_JOINT_DAMPING,
) -> str:
    source = Path(source_path).resolve()
    xml = source.read_text(encoding="utf-8")

    xml = _filter_geom_view(xml, geom_view)
    xml = _set_floor_contact(xml, enabled=floor_contact)
    xml = xml.replace("<freejoint />", "")
    xml = _drop_keyframe_blocks(xml)
    if drop_model_actuators:
        xml = _drop_actuator_blocks(xml)

    xml = xml.replace('damping="0" />', f'damping="{viewer_joint_damping}" />')
    xml = xml.replace('damping="0"', f'damping="{viewer_joint_damping}"')

    xml = xml.replace(
        '<body name="base_link" pos="0 0 0.301">',
        f'<body name="base_link" pos="0 0 {base_height}">',
    )
    xml = xml.replace(
        '<body name="base_link" pos="0 0 0.30">',
        f'<body name="base_link" pos="0 0 {base_height}">',
    )
    xml = xml.replace(
        '<body name="base_link" pos="0 0 0.22">',
        f'<body name="base_link" pos="0 0 {base_height}">',
    )

    mocap_body = f"""
    <body name="mocap_target" mocap="true" pos="0 0 {base_height}">
      <geom type="box" size="0.005 0.005 0.005" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>
"""
    xml = xml.replace("</worldbody>", mocap_body + "  </worldbody>")

    weld = '    <weld body1="base_link" body2="mocap_target" solref="0.002 1" solimp="0.99 0.99 0.001"/>\n'
    if "<equality>" in xml:
        xml = xml.replace("<equality>", "<equality>\n" + weld, 1)
    else:
        xml = xml.replace(
            "</mujoco>",
            "  <equality>\n" + weld + "  </equality>\n</mujoco>",
        )

    xml = _add_viewer_actuators(xml, position_kp=viewer_position_kp)

    tmp_path = source.parent / "_joint_viewer_tmp.xml"
    tmp_path.write_text(xml, encoding="utf-8")
    return str(tmp_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(MJCF_PATH))
    parser.add_argument("--openchain", action="store_true")
    parser.add_argument("--fourbar-surrogate", action="store_true")
    parser.add_argument("--closedchain-spring", action="store_true")
    parser.add_argument("--base-height", type=float, default=BASE_HEIGHT)
    parser.add_argument("--floor-contact", action="store_true")
    parser.add_argument("--drop-model-actuators", action="store_true")
    parser.add_argument("--viewer-kp", type=float, default=VIEWER_POSITION_KP)
    parser.add_argument("--viewer-damping", type=float, default=VIEWER_JOINT_DAMPING)
    parser.add_argument(
        "--geom-view",
        choices=("visual", "collision", "both"),
        default="both",
        help="选择 viewer 初始几何显示：visual 只看视觉模型，collision 只看碰撞体，both 两者都看。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_height = float(args.base_height)
    if args.openchain:
        model_path = Path(OPENCHAIN_MJCF_PATH)
    elif args.fourbar_surrogate:
        model_path = Path(FOURBAR_SURROGATE_MJCF_PATH)
    elif args.closedchain_spring:
        model_path = Path(CLOSEDCHAIN_SPRING_MJCF_PATH)
    else:
        model_path = args.model
    fixed_mjcf = _create_fixed_base_mjcf(
        model_path,
        base_height=base_height,
        floor_contact=bool(args.floor_contact),
        drop_model_actuators=bool(args.drop_model_actuators),
        geom_view=str(args.geom_view),
        viewer_position_kp=float(args.viewer_kp),
        viewer_joint_damping=float(args.viewer_damping),
    )
    model = mujoco.MjModel.from_xml_path(fixed_mjcf)
    data = mujoco.MjData(model)
    Path(fixed_mjcf).unlink(missing_ok=True)

    for jnt_name, angle in DEFAULT_JOINT_ANGLES.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id < 0:
            continue
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr] = angle

    closedchain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "l_active_rod_angle") >= 0
    active_ctrl_ranges = LEG_CTRL_RANGES if closedchain else OPENCHAIN_LEG_CTRL_RANGES
    for jnt_name in active_ctrl_ranges:
        act_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{jnt_name}_viewer_pos",
        )
        if act_id < 0:
            continue
        data.ctrl[act_id] = DEFAULT_JOINT_ANGLES[jnt_name]
    if closedchain:
        for tendon_name, angle in zip(
            ACTIVE_ROD_ANGLE_CTRL_RANGES,
            _ROBOT_CFG.default_active_rod_angles,
            strict=True,
        ):
            act_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{tendon_name}_viewer_pos",
            )
            if act_id >= 0:
                data.ctrl[act_id] = angle

    mujoco.mj_forward(model, data)

    print(f"默认站立姿态: base_z={base_height}m, floor_contact={bool(args.floor_contact)}")
    print(f"关节角: {DEFAULT_JOINT_ANGLES}")

    print(
        "MuJoCo Control panel: drag LF/RF sliders, active-rod-angle sliders, and wheel velocity sliders."
    )

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
