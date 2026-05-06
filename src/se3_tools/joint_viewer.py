"""交互式关节调试 Viewer。

使用 MuJoCo viewer + actuator ctrl 滑块。
base_link 通过 mocap body 固定在空中,可自由拖动调整姿态,
只观察腿部和轮子对力矩的响应。

用法:
    uv run se3-joint-viewer
    uv run se3-joint-viewer --height 0.35
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml"


def _create_fixed_base_mjcf(source_path: str, height: float) -> str:
    """生成一个 base 被 weld 到 mocap body 的临时 MJCF。"""
    source = Path(source_path).resolve()
    xml = source.read_text()

    xml = xml.replace("<freejoint />", "")

    mocap_body = f"""
    <body name="mocap_target" mocap="true" pos="0 0 {height}">
      <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 0.3" contype="0" conaffinity="0"/>
    </body>
"""
    xml = xml.replace("</worldbody>", mocap_body + "  </worldbody>")

    equality = """
  <equality>
    <weld body1="base_link" body2="mocap_target" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </equality>
"""
    xml = xml.replace("</mujoco>", equality + "</mujoco>")

    tmp_path = source.parent / "_joint_viewer_tmp.xml"
    tmp_path.write_text(xml)
    return str(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式关节调试 Viewer (base 固定)")
    parser.add_argument("--mjcf", default=MJCF_PATH, help="MJCF 模型路径")
    parser.add_argument("--height", type=float, default=0.50, help="固定高度")
    parser.add_argument("--free", action="store_true", help="不固定 base,自由落体模式")
    args = parser.parse_args()

    if args.free:
        model = mujoco.MjModel.from_xml_path(args.mjcf)
        data = mujoco.MjData(model)
        data.qpos[2] = args.height
        fixed_mjcf = None
    else:
        fixed_mjcf = _create_fixed_base_mjcf(args.mjcf, args.height)
        model = mujoco.MjModel.from_xml_path(fixed_mjcf)
        data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("=" * 50)
    print("关节调试 Viewer")
    print("=" * 50)
    if not args.free:
        print(f"模式: base 固定在 z={args.height}m (可鼠标拖动 mocap body)")
        print("  - 双击红色方块可拖动 base 位置/姿态")
    else:
        print("模式: 自由落体 (--free)")
    print()
    print("操作:")
    print("  Ctrl+A: 显示 actuator 滑块面板")
    print("  拖动滑块: 施加力矩")
    print("  双击红色 mocap 方块: 拖动 base 位置/旋转")
    print("  空格: 暂停/继续")
    print("  Backspace: 重置")
    print()

    mujoco.viewer.launch(model, data)

    if fixed_mjcf:
        Path(fixed_mjcf).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
