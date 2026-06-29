"""为闭链细连杆生成 COACD 碰撞体并写回 MJCF。

脚本只处理 drive bar 和 coupler 四个闭链细连杆，不修改底盘、轮子和主腿连杆碰撞体。运行方式示例：

uv run --with coacd python scripts/build_leg_coacd_collision.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

LEG_TARGETS = (
    "l_drive_bar_Link",
    "l_coupler_Link",
    "r_drive_bar_Link",
    "r_coupler_Link",
)

COLLISION_ATTRS = {
    "type": "mesh",
    "group": "0",
    "contype": "1",
    "conaffinity": "2",
}


@dataclass(frozen=True, slots=True)
class CoacdRule:
    threshold: float
    max_convex_hull: int
    resolution: int
    mcts_nodes: int
    mcts_iterations: int
    max_ch_vertex: int


DEFAULT_RULE = CoacdRule(
    threshold=0.080,
    max_convex_hull=4,
    resolution=700,
    mcts_nodes=8,
    mcts_iterations=35,
    max_ch_vertex=80,
)

RULES: dict[str, CoacdRule] = {
    "l_drive_bar_Link": CoacdRule(0.060, 4, 700, 8, 35, 72),
    "r_drive_bar_Link": CoacdRule(0.060, 4, 700, 8, 35, 72),
    "l_coupler_Link": CoacdRule(0.055, 4, 700, 8, 35, 72),
    "r_coupler_Link": CoacdRule(0.055, 4, 700, 8, 35, 72),
}

MAX_COACD_INPUT_FACES = 1800


def _parse_vec(value: str | None, size: int, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    parts = [float(item) for item in value.split()]
    if len(parts) != size:
        raise ValueError(f"向量长度应为 {size}: {value}")
    return np.asarray(parts, dtype=np.float64)


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("quat 不能为零")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _euler_xyz_matrix(euler: np.ndarray) -> np.ndarray:
    cx, cy, cz = np.cos(euler)
    sx, sy, sz = np.sin(euler)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _geom_transform(geom: ET.Element) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if "quat" in geom.attrib:
        transform[:3, :3] = _quat_matrix(_parse_vec(geom.get("quat"), 4, (1, 0, 0, 0)))
    elif "euler" in geom.attrib:
        transform[:3, :3] = _euler_xyz_matrix(_parse_vec(geom.get("euler"), 3, (0, 0, 0)))
    transform[:3, 3] = _parse_vec(geom.get("pos"), 3, (0, 0, 0))
    return transform


def _mesh_dir(mjcf_path: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if not meshdir:
        return mjcf_path.parent.resolve()
    meshdir_path = Path(meshdir)
    if meshdir_path.is_absolute():
        return meshdir_path.resolve()
    return (mjcf_path.parent / meshdir_path).resolve()


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"找不到 body: {name}")


def _asset_mesh_map(asset: ET.Element, mesh_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        file_attr = mesh.get("file")
        if not name or not file_attr:
            continue
        file_path = Path(file_attr)
        result[name] = file_path if file_path.is_absolute() else (mesh_root / file_path).resolve()
    return result


def _load_visual_meshes(body: ET.Element, asset_meshes: dict[str, Path]) -> trimesh.Trimesh:
    meshes: list[trimesh.Trimesh] = []
    for geom in body.findall("geom"):
        if geom.get("group") != "1" or geom.get("type") != "mesh":
            continue
        mesh_name = geom.get("mesh")
        if not mesh_name or mesh_name not in asset_meshes:
            continue
        mesh = trimesh.load_mesh(asset_meshes[mesh_name], process=True, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"{asset_meshes[mesh_name]} 不是可用 mesh")
        mesh = mesh.copy()
        mesh.apply_transform(_geom_transform(geom))
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        meshes.append(mesh)
    if not meshes:
        raise RuntimeError(f"{body.get('name')} 没有可用 visual mesh")
    return trimesh.util.concatenate(meshes)


def _safe_clean_dir(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise RuntimeError(f"拒绝清理输出根目录之外的路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _run_coacd(mesh: trimesh.Trimesh, rule: CoacdRule, seed: int) -> list[trimesh.Trimesh]:
    try:
        import coacd
    except ImportError as exc:
        raise SystemExit("缺少 coacd；请用 `uv run --with coacd python ...` 运行。") from exc

    coacd_input = mesh.copy()
    if len(coacd_input.faces) > MAX_COACD_INPUT_FACES:
        coacd_input = coacd_input.simplify_quadric_decimation(face_count=MAX_COACD_INPUT_FACES)
        coacd_input.remove_unreferenced_vertices()
        coacd_input.merge_vertices()

    vertices = np.asarray(coacd_input.vertices, dtype=np.float64)
    faces = np.asarray(coacd_input.faces, dtype=np.int32)
    result = coacd.run_coacd(
        coacd.Mesh(vertices, faces),
        threshold=rule.threshold,
        max_convex_hull=rule.max_convex_hull,
        resolution=rule.resolution,
        mcts_nodes=rule.mcts_nodes,
        mcts_iterations=rule.mcts_iterations,
        max_ch_vertex=rule.max_ch_vertex,
        merge=True,
        seed=seed,
    )
    parts: list[trimesh.Trimesh] = []
    for part_vertices, part_faces in result:
        part = trimesh.Trimesh(
            vertices=np.asarray(part_vertices, dtype=np.float64),
            faces=np.asarray(part_faces, dtype=np.int64),
            process=True,
        )
        part.remove_unreferenced_vertices()
        part.merge_vertices()
        parts.append(part)
    if not parts:
        raise RuntimeError("COACD 没有生成任何凸分解结果")
    return parts


def _is_leg_collision_geom(geom: ET.Element) -> bool:
    if geom.tag != "geom":
        return False
    return (
        geom.get("group") == "0" and geom.get("contype") == "1" and geom.get("conaffinity") == "2"
    )


def _remove_old_leg_coacd_assets(asset: ET.Element) -> None:
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name", "")
        if name.startswith("legcoacd_"):
            asset.remove(mesh)


def _insert_index_before_child_body(body: ET.Element) -> int:
    for index, child in enumerate(list(body)):
        if child.tag == "body":
            return index
    return len(list(body))


def build_and_apply(
    mjcf_path: Path,
    output_root: Path,
    manifest_path: Path,
    seed: int,
    selected_targets: set[str] | None = None,
) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("MJCF 缺少 asset 节点")

    mesh_root = _mesh_dir(mjcf_path, root)
    output_root = output_root.resolve()
    _safe_clean_dir(output_root, mesh_root)
    asset_meshes = _asset_mesh_map(asset, mesh_root)
    _remove_old_leg_coacd_assets(asset)

    manifest: list[dict[str, object]] = []
    targets = [
        target for target in LEG_TARGETS if selected_targets is None or target in selected_targets
    ]
    if not targets:
        raise RuntimeError("没有匹配的腿部 target")

    for target_index, target in enumerate(targets):
        body = _body_by_name(root, target)
        visual_mesh = _load_visual_meshes(body, asset_meshes)
        rule = RULES.get(target, DEFAULT_RULE)
        print(f"{target}: 开始 COACD，visual faces={len(visual_mesh.faces)}", flush=True)
        parts = _run_coacd(visual_mesh, rule, seed + target_index)

        target_dir = output_root / target
        target_dir.mkdir(parents=True, exist_ok=True)
        generated: list[dict[str, object]] = []
        for part_index, part in enumerate(parts):
            out_name = f"{target}_coacd_{part_index:02d}.stl"
            out_path = target_dir / out_name
            part.export(out_path)
            mesh_name = f"legcoacd_{target}_{part_index:02d}"
            file_rel = f"{output_root.name}/{target}/{out_name}"
            ET.SubElement(asset, "mesh", {"name": mesh_name, "file": file_rel})
            generated.append(
                {
                    "name": mesh_name,
                    "file": file_rel,
                    "vertices": len(part.vertices),
                    "faces": len(part.faces),
                    "extents": np.asarray(part.extents, dtype=float).round(6).tolist(),
                }
            )

        for child in list(body):
            if _is_leg_collision_geom(child):
                body.remove(child)
        insert_index = _insert_index_before_child_body(body)
        for part_index, item in enumerate(generated):
            body.insert(
                insert_index + part_index,
                ET.Element(
                    "geom",
                    {
                        "name": f"{target}_coacd_col_{part_index:02d}",
                        "mesh": str(item["name"]),
                        **COLLISION_ATTRS,
                    },
                ),
            )

        manifest.append(
            {
                "target": target,
                "rule": asdict(rule),
                "generated": generated,
            }
        )
        print(f"{target}: {len(generated)} COACD parts", flush=True)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/leg_coacd_v1"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/leg_coacd_v1/manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", action="append", choices=LEG_TARGETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_and_apply(
        mjcf_path=args.mjcf.resolve(),
        output_root=args.output_root,
        manifest_path=args.manifest.resolve(),
        seed=args.seed,
        selected_targets=set(args.target) if args.target else None,
    )
    print(f"已更新: {args.mjcf}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
