"""把 SolidWorks 原始腿部 STL 转成 MJCF 可引用的视觉 mesh。

脚本只做单位缩放和刚体坐标系对齐，不做凸包、合并或简化；输出文件仍保留
原始 STL 的三角面片拓扑，用作 group=1 的视觉 mesh。碰撞几何继续由独立
的简化 collision geom 负责，避免螺栓、垫片等小件进入 contact。
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

LEG_TARGETS = (
    "lf0_Link",
    "l_drive_bar_Link",
    "l_coupler_Link",
    "lf1_Link",
    "rf0_Link",
    "r_drive_bar_Link",
    "r_coupler_Link",
    "rf1_Link",
)


def _safe_clean_dir(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise RuntimeError(f"拒绝清理输出根目录之外的路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} 不是可用的三角 mesh")
    return mesh.copy()


def _raw_stl_files(raw_target_dir: Path) -> list[Path]:
    files = {*raw_target_dir.glob("*.STL"), *raw_target_dir.glob("*.stl")}
    return sorted(path.resolve() for path in files)


def _manifest_by_target(manifest_path: Path) -> dict[str, dict[str, object]]:
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(item["target"]): item for item in items}


def _build_meshes(
    raw_root: Path,
    output_root: Path,
    transform_manifest: Path,
    targets: tuple[str, ...],
) -> list[dict[str, object]]:
    manifest_by_target = _manifest_by_target(transform_manifest)
    mesh_rel_root = output_root.name
    generated_manifest: list[dict[str, object]] = []

    for target in targets:
        if target not in manifest_by_target:
            raise KeyError(f"变换 manifest 缺少目标: {target}")
        raw_target_dir = raw_root / target
        files = _raw_stl_files(raw_target_dir)
        if not files:
            raise RuntimeError(f"{target} 没有 SW 原始 STL: {raw_target_dir}")

        transform = np.asarray(manifest_by_target[target]["transform"], dtype=np.float64)
        output_dir = output_root / target
        output_dir.mkdir(parents=True, exist_ok=True)

        generated: list[dict[str, object]] = []
        for index, source_path in enumerate(files):
            mesh = _load_mesh(source_path)
            mesh.apply_scale(0.001)
            mesh.apply_transform(transform)
            out_name = f"{target}_sw_raw_{index:02d}.stl"
            out_path = output_dir / out_name
            mesh.export(out_path)
            generated.append(
                {
                    "name": f"visual_swraw_{target}_{index:02d}",
                    "file": f"{mesh_rel_root}/{target}/{out_name}",
                    "source": source_path.name,
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "extents": np.asarray(mesh.extents, dtype=float).round(6).tolist(),
                }
            )

        generated_manifest.append({"target": target, "generated": generated})

    return generated_manifest


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"找不到 body: {name}")


def _is_old_leg_visual_mesh(mesh: ET.Element) -> bool:
    name = mesh.get("name", "")
    return name.startswith("visual_swraw_") or name in {
        "visual_lf_thigh",
        "visual_lf_calf_1",
        "visual_lf_calf_2",
        "visual_lf_calf_3",
        "visual_rf_thigh",
        "visual_rf_calf_1",
        "visual_rf_calf_2",
        "visual_rf_calf_3",
    }


def _is_old_leg_visual_geom(geom: ET.Element, target: str) -> bool:
    name = geom.get("name", "")
    return name.startswith(f"{target}_sw_raw_visual_") or name == f"{target}_visual"


def _insert_index_after_inertial(body: ET.Element) -> int:
    children = list(body)
    for index, child in enumerate(children):
        if child.tag == "inertial":
            return index + 1
    return 0


def _update_mjcf(mjcf_path: Path, visual_manifest: list[dict[str, object]]) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("MJCF 缺少 asset 节点")

    for mesh in list(asset.findall("mesh")):
        if _is_old_leg_visual_mesh(mesh):
            asset.remove(mesh)

    for item in visual_manifest:
        for mesh_item in item["generated"]:
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": str(mesh_item["name"]),
                    "file": str(mesh_item["file"]),
                },
            )

    for item in visual_manifest:
        target = str(item["target"])
        body = _body_by_name(root, target)
        for child in list(body):
            if child.tag == "geom" and _is_old_leg_visual_geom(child, target):
                body.remove(child)

        insert_index = _insert_index_after_inertial(body)
        for index, mesh_item in enumerate(item["generated"]):
            geom = ET.Element(
                "geom",
                {
                    "name": f"{target}_sw_raw_visual_{index:02d}",
                    "type": "mesh",
                    "mesh": str(mesh_item["name"]),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
            body.insert(insert_index + index, geom)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path(".tmp/sw_collision_raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/sw_original_legs"),
    )
    parser.add_argument(
        "--transform-manifest",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/sw_collision_v3/manifest.json"),
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"),
    )
    parser.add_argument("--target", action="append", choices=sorted(LEG_TARGETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    targets = tuple(args.target or LEG_TARGETS)

    _safe_clean_dir(output_root, output_root)
    visual_manifest = _build_meshes(
        raw_root=raw_root,
        output_root=output_root,
        transform_manifest=args.transform_manifest.resolve(),
        targets=targets,
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(visual_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _update_mjcf(args.mjcf.resolve(), visual_manifest)

    print(f"生成完成: {manifest_path}")
    for item in visual_manifest:
        print(f"  {item['target']}: {len(item['generated'])} visual meshes")


if __name__ == "__main__":
    main()
