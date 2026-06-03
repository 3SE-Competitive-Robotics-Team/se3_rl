"""生成 Unitree 风格的简洁 collision geom。

Unitree/MuJoCo Menagerie 的模型通常保留高细节 visual mesh，而 collision 使用
少量 primitive 近似。这个脚本按“非轮 body 只保留一个 box、轮子保留 cylinder”
的规则从当前 visual mesh 估算干净 collision。
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

TARGETS = (
    "base_link",
    "lf0_Link",
    "l_drive_bar_Link",
    "l_coupler_Link",
    "lf1_Link",
    "rf0_Link",
    "r_drive_bar_Link",
    "r_coupler_Link",
    "rf1_Link",
)
COLLISION_ATTRS = {
    "group": "0",
    "contype": "1",
    "conaffinity": "2",
}


@dataclass
class VisualMesh:
    name: str
    path: Path
    transform: np.ndarray


@dataclass(frozen=True)
class BoxRule:
    low_percentile: float
    high_percentile: float
    margin: float
    oriented: bool


BOX_RULES = {
    "base_link": BoxRule(5.0, 95.0, 0.0015, False),
}
DEFAULT_BOX_RULE = BoxRule(3.0, 97.0, 0.0015, True)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def _fmt_vec(values: np.ndarray) -> str:
    return " ".join(_fmt(float(value)) for value in values)


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


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _geom_transform(geom: ET.Element) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if "quat" in geom.attrib:
        transform[:3, :3] = _quat_matrix(_parse_vec(geom.get("quat"), 4, (1, 0, 0, 0)))
    elif "euler" in geom.attrib:
        transform[:3, :3] = _euler_xyz_matrix(_parse_vec(geom.get("euler"), 3, (0, 0, 0)))
    transform[:3, 3] = _parse_vec(geom.get("pos"), 3, (0, 0, 0))
    return transform


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"找不到 body: {name}")


def _mesh_dir(mjcf_path: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if not meshdir:
        return mjcf_path.parent.resolve()
    meshdir_path = Path(meshdir)
    if meshdir_path.is_absolute():
        return meshdir_path.resolve()
    return (mjcf_path.parent / meshdir_path).resolve()


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


def _visual_meshes_for_body(
    body: ET.Element,
    asset_meshes: dict[str, Path],
) -> list[VisualMesh]:
    visual_meshes: list[VisualMesh] = []
    for geom in body.findall("geom"):
        if geom.get("group") != "1" or geom.get("type") != "mesh":
            continue
        mesh_name = geom.get("mesh")
        if not mesh_name or mesh_name not in asset_meshes:
            continue
        visual_meshes.append(
            VisualMesh(
                name=mesh_name,
                path=asset_meshes[mesh_name],
                transform=_geom_transform(geom),
            )
        )
    return visual_meshes


def _load_vertices(item: VisualMesh) -> np.ndarray:
    mesh = trimesh.load_mesh(item.path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{item.path} 不是可用 mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not np.allclose(item.transform, np.eye(4)):
        hom = np.c_[vertices, np.ones(len(vertices), dtype=np.float64)]
        vertices = (hom @ item.transform.T)[:, :3]
    return vertices


def _vertices_for_visuals(visual_meshes: list[VisualMesh]) -> np.ndarray:
    if not visual_meshes:
        raise RuntimeError("没有 visual mesh 可用于估算 collision")
    return np.concatenate([_load_vertices(item) for item in visual_meshes], axis=0)


def _make_box(
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    quat: np.ndarray | None = None,
) -> ET.Element:
    attrs = {
        "name": name,
        "type": "box",
        "pos": _fmt_vec(center),
        "size": _fmt_vec(half_size),
        **COLLISION_ATTRS,
    }
    if quat is not None:
        attrs["quat"] = _fmt_vec(quat)
    return ET.Element(
        "geom",
        attrs,
    )


def _single_box_primitive(
    target: str,
    body: ET.Element,
    asset_meshes: dict[str, Path],
) -> list[ET.Element]:
    rule = BOX_RULES.get(target, DEFAULT_BOX_RULE)
    vertices = _vertices_for_visuals(_visual_meshes_for_body(body, asset_meshes))
    if rule.oriented:
        reference = np.median(vertices, axis=0)
        centered = vertices - reference
        _values, vectors = np.linalg.eigh(np.cov(centered.T))
        axes = vectors[:, np.argsort(_values)[::-1]]
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] *= -1.0
        local = centered @ axes
        bounds_min, bounds_max = np.percentile(
            local,
            [rule.low_percentile, rule.high_percentile],
            axis=0,
        )
        local_center = (bounds_min + bounds_max) * 0.5
        center = reference + local_center @ axes.T
        half = (bounds_max - bounds_min) * 0.5 + rule.margin
        quat = _matrix_to_quat(axes)
    else:
        bounds_min, bounds_max = np.percentile(
            vertices,
            [rule.low_percentile, rule.high_percentile],
            axis=0,
        )
        center = (bounds_min + bounds_max) * 0.5
        half = (bounds_max - bounds_min) * 0.5 + rule.margin
        quat = None
    half = np.maximum(half, np.array([0.008, 0.006, 0.008]))
    return [_make_box(f"{target}_ucol_main", center, half, quat)]


def _target_primitives(
    target: str,
    body: ET.Element,
    asset_meshes: dict[str, Path],
) -> list[ET.Element]:
    return _single_box_primitive(target, body, asset_meshes)


def _is_collision_geom(geom: ET.Element) -> bool:
    if geom.tag != "geom":
        return False
    name = geom.get("name", "")
    mesh = geom.get("mesh", "")
    return (
        "_ucol_" in name
        or "_vcol_" in name
        or "_sw_env_col_" in name
        or mesh.startswith("vcol_")
        or mesh.startswith("swcol_")
        or (
            geom.get("group") == "0"
            and geom.get("contype") == "1"
            and geom.get("conaffinity") == "2"
        )
    )


def _remove_generated_collision_assets(asset: ET.Element) -> None:
    for mesh in list(asset.findall("mesh")):
        name = mesh.get("name", "")
        if name.startswith("vcol_") or name.startswith("swcol_") or "collision_mesh" in name:
            asset.remove(mesh)


def _insert_index_before_child_body(body: ET.Element) -> int:
    for index, child in enumerate(list(body)):
        if child.tag == "body":
            return index
    return len(list(body))


def _apply_unitree_style_collision(mjcf_path: Path, manifest_path: Path | None) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("MJCF 缺少 asset 节点")

    mesh_root = _mesh_dir(mjcf_path, root)
    asset_meshes = _asset_mesh_map(asset, mesh_root)
    _remove_generated_collision_assets(asset)

    manifest: list[dict[str, object]] = []
    for target in TARGETS:
        body = _body_by_name(root, target)
        generated = _target_primitives(target, body, asset_meshes)
        for child in list(body):
            if _is_collision_geom(child):
                body.remove(child)
        insert_index = _insert_index_before_child_body(body)
        for offset, geom in enumerate(generated):
            body.insert(insert_index + offset, geom)
        manifest.append(
            {
                "target": target,
                "generated": [dict(geom.attrib) for geom in generated],
            }
        )

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    if manifest_path is not None:
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
        "--manifest",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/unitree_collision_v1/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_unitree_style_collision(args.mjcf.resolve(), args.manifest.resolve())
    print(f"已更新: {args.mjcf}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
