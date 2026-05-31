"""把 SW 外轮廓碰撞网格 manifest 接入闭链 MJCF。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_BODIES = (
    "base_link",
    "lf0_Link",
    "rf0_Link",
    "lf1_Link",
    "rf1_Link",
    "l_wheel_Link",
    "r_wheel_Link",
    "l_drive_bar_Link",
    "r_drive_bar_Link",
    "l_coupler_Link",
    "r_coupler_Link",
)

NON_CONTACT_BODIES = (
    "l_drive_bar_Link",
    "r_drive_bar_Link",
    "l_coupler_Link",
    "r_coupler_Link",
)

PRIMITIVE_COLLISION_GEOMS: dict[str, tuple[dict[str, str], ...]] = {
    "base_link": (
        {
            "name": "base_link_collision_bottom_plate",
            "type": "box",
            "size": "0.245 0.130 0.006",
            "pos": "0 0 -0.090",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
        {
            "name": "base_link_collision_left_rail",
            "type": "box",
            "size": "0.1605 0.010 0.020",
            "pos": "0 0.100 -0.044",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
        {
            "name": "base_link_collision_right_rail",
            "type": "box",
            "size": "0.1605 0.010 0.020",
            "pos": "0 -0.100 -0.044",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
        {
            "name": "base_link_collision_front_armor",
            "type": "box",
            "size": "0.043 0.084 0.064",
            "pos": "0.232 0 0.006",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
        {
            "name": "base_link_collision_rear_armor",
            "type": "box",
            "size": "0.043 0.084 0.064",
            "pos": "-0.232 0 0.006",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
        {
            "name": "base_link_collision_center_deck",
            "type": "box",
            "size": "0.110 0.105 0.026",
            "pos": "0 0 0.035",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
        {
            "name": "base_link_collision_top_plate",
            "type": "box",
            "size": "0.130 0.100 0.006",
            "pos": "0 0 0.082",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
        },
    ),
    "l_wheel_Link": (
        {
            "name": "l_wheel_Link_collision_0",
            "type": "cylinder",
            "size": "0.059 0.016",
            "pos": "0 -0.0066 0",
            "euler": "1.5708 0 0",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
            "friction": "0.8 0.005 0.0001",
        },
    ),
    "r_wheel_Link": (
        {
            "name": "r_wheel_Link_collision_0",
            "type": "cylinder",
            "size": "0.059 0.016",
            "pos": "0 0.0066 0",
            "euler": "1.5708 0 0",
            "group": "0",
            "contype": "1",
            "conaffinity": "2",
            "friction": "0.8 0.005 0.0001",
        },
    ),
}


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"找不到 body: {name}")


def _is_collision_geom(element: ET.Element) -> bool:
    if element.tag != "geom":
        return False
    name = element.get("name", "")
    mesh = element.get("mesh", "")
    return (
        "_sw_env_col_" in name
        or mesh.startswith("swcol_")
        or (
            element.get("group") == "0"
            and element.get("contype") == "1"
            and element.get("conaffinity") == "2"
        )
    )


def _remove_old_collision_assets(asset: ET.Element) -> None:
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name", "")
        if name.startswith("swcol_") or "collision_mesh" in name:
            asset.remove(mesh)


def _insert_index_before_child_body(body: ET.Element) -> int:
    for index, child in enumerate(list(body)):
        if child.tag == "body":
            return index
    return len(list(body))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("assets/robots/serialleg/mjcf/serialleg_closed_chain_v2_spring.xml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/sw_collision/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree = ET.parse(args.mjcf)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("MJCF 缺少 asset 节点")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    generated_by_target = {item["target"]: item["generated"] for item in manifest}
    missing = [target for target in TARGET_BODIES if target not in generated_by_target]
    if missing:
        raise RuntimeError(f"manifest 缺少目标: {missing}")

    _remove_old_collision_assets(asset)

    for target in TARGET_BODIES:
        if target in PRIMITIVE_COLLISION_GEOMS:
            continue
        for index, item in enumerate(generated_by_target[target]):
            mesh_name = f"swcol_{target}_{index:02d}"
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": item["file"]})

    for target in TARGET_BODIES:
        body = _body_by_name(root, target)
        for child in list(body):
            if _is_collision_geom(child):
                body.remove(child)

        insert_index = _insert_index_before_child_body(body)
        primitive_geoms = PRIMITIVE_COLLISION_GEOMS.get(target)
        if primitive_geoms is not None:
            for index, attributes in enumerate(primitive_geoms):
                body.insert(insert_index + index, ET.Element("geom", dict(attributes)))
            continue

        for index, _item in enumerate(generated_by_target[target]):
            mesh_name = f"swcol_{target}_{index:02d}"
            contype = "0" if target in NON_CONTACT_BODIES else "1"
            conaffinity = "0" if target in NON_CONTACT_BODIES else "2"
            geom = ET.Element(
                "geom",
                {
                    "name": f"{target}_sw_env_col_{index:02d}",
                    "type": "mesh",
                    "group": "0",
                    "contype": contype,
                    "conaffinity": conaffinity,
                    "mesh": mesh_name,
                },
            )
            body.insert(insert_index + index, geom)

    ET.indent(tree, space="  ")
    tree.write(args.mjcf, encoding="utf-8", xml_declaration=False)
    mesh_collision_count = sum(
        len(items)
        for target, items in generated_by_target.items()
        if target not in PRIMITIVE_COLLISION_GEOMS
    )
    primitive_collision_count = sum(len(items) for items in PRIMITIVE_COLLISION_GEOMS.values())
    print(f"已更新: {args.mjcf}")
    print(f"碰撞 mesh 数: {mesh_collision_count}")
    print(f"primitive 碰撞数: {primitive_collision_count}")


if __name__ == "__main__":
    main()
