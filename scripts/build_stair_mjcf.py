"""生成台阶训练专用 MJCF。

该脚本从当前四连杆等效开树训练模型派生 stair 版本：
- 保留关节、惯量、四连杆 surrogate 标记、默认站姿和轮子 cylinder 碰撞。
- 仅将机身、大腿、小腿的训练碰撞替换为 SW 外轮廓 mesh 碰撞。
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "assets/robots/serialleg/mjcf/serialleg_fourbar_surrogate_train.xml"
DEFAULT_MANIFEST = ROOT / "assets/robots/serialleg/meshes/sw_collision_v3/manifest.json"
DEFAULT_OUTPUT = ROOT / "assets/robots/serialleg/mjcf/serialleg_fourbar_surrogate_stair_train.xml"

STAIR_COLLISION_TARGETS = ("base_link", "lf0_Link", "rf0_Link", "lf1_Link", "rf1_Link")
COLLISION_ATTRS = {
    "type": "mesh",
    "group": "0",
    "contype": "1",
    "conaffinity": "2",
}


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"找不到 body: {name}")


def _is_train_collision_geom(element: ET.Element) -> bool:
    if element.tag != "geom":
        return False
    return (
        element.get("group") == "0"
        and element.get("contype") == "1"
        and element.get("conaffinity") == "2"
    )


def _remove_generated_assets(asset: ET.Element) -> None:
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name", "")
        if name.startswith("staircol_"):
            asset.remove(mesh)


def _insert_index_before_child_body(body: ET.Element) -> int:
    for index, child in enumerate(list(body)):
        if child.tag == "body":
            return index
    return len(list(body))


def _manifest_by_target(manifest_path: Path) -> dict[str, list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(item["target"]): list(item["generated"]) for item in manifest}


def build_stair_mjcf(source: Path, manifest_path: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "serialleg_fourbar_surrogate_stair_train")

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("MJCF 缺少 asset 节点")

    generated_by_target = _manifest_by_target(manifest_path)
    missing = [target for target in STAIR_COLLISION_TARGETS if target not in generated_by_target]
    if missing:
        raise RuntimeError(f"manifest 缺少目标: {missing}")

    _remove_generated_assets(asset)
    for target in STAIR_COLLISION_TARGETS:
        for index, item in enumerate(generated_by_target[target]):
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": f"staircol_{target}_{index:02d}",
                    "file": str(item["file"]),
                },
            )

    for target in STAIR_COLLISION_TARGETS:
        body = _body_by_name(root, target)
        for child in list(body):
            if _is_train_collision_geom(child):
                body.remove(child)

        insert_index = _insert_index_before_child_body(body)
        for index, _item in enumerate(generated_by_target[target]):
            body.insert(
                insert_index + index,
                ET.Element(
                    "geom",
                    {
                        "name": f"{target}_stair_col_{index:02d}",
                        "mesh": f"staircol_{target}_{index:02d}",
                        **COLLISION_ATTRS,
                    },
                ),
            )

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_stair_mjcf(
        source=args.source.resolve(),
        manifest_path=args.manifest.resolve(),
        output=args.output.resolve(),
    )
    print(args.output)


if __name__ == "__main__":
    main()
