"""筛选 SolidWorks 导出的组件 STL，并生成更干净的外轮廓碰撞网格。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class BuildRule:
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    min_extent_m: float = 0.006
    max_parts: int | None = None


COMMON_EXCLUDE = (
    "~$",
    "标准件",
    "螺栓",
    "螺母",
    "垫片",
    "轴承",
    "轴套",
    "电机",
    "电调",
    "官方物资",
    "C620",
    "滑环",
    "气弹簧1",
    "气弹簧2",
    "弹簧",
    "合页",
    "减速箱",
    "客户模型",
    "MF",
    "GB",
    "HK",
    "RBU",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
)

RULES: dict[str, BuildRule] = {
    "base_link": BuildRule(
        include=(
            "板",
            "碳板",
            "pom",
            "POM",
            "铝方管",
            "装甲",
            "保护",
            "底板",
            "架板",
            "支撑",
            "腿组内侧",
            "腿组外侧",
        ),
        exclude=(
            *COMMON_EXCLUDE,
            "Left_calf",
            "Right_calf",
            "Left_thigh",
            "Right_thigh",
            "小腿传动杆",
            "异形连杆",
            "减速箱",
            "轮毂",
            "包胶",
            "导轮",
            "包胶导轮",
            "张紧轮",
            "继电器",
            "RFID",
            "外购件",
            "开发板",
            "达妙",
            "打印件",
        ),
        min_extent_m=0.018,
        max_parts=48,
    ),
    "lf0_Link": BuildRule(
        include=("大腿板", "腿限位块"),
        exclude=COMMON_EXCLUDE,
        min_extent_m=0.012,
        max_parts=4,
    ),
    "rf0_Link": BuildRule(
        include=("大腿板", "腿限位块"),
        exclude=COMMON_EXCLUDE,
        min_extent_m=0.012,
        max_parts=4,
    ),
    "lf1_Link": BuildRule(
        include=("小腿板", "电调保护板", "轴承座", "气弹簧安装件", "气弹簧紧固"),
        exclude=COMMON_EXCLUDE,
        min_extent_m=0.012,
        max_parts=8,
    ),
    "rf1_Link": BuildRule(
        include=("小腿板", "电调保护板", "轴承座", "气弹簧安装件", "气弹簧紧固"),
        exclude=COMMON_EXCLUDE,
        min_extent_m=0.012,
        max_parts=8,
    ),
    "l_wheel_Link": BuildRule(
        include=("包胶", "轮毂"),
        exclude=COMMON_EXCLUDE,
        min_extent_m=0.012,
        max_parts=4,
    ),
    "r_wheel_Link": BuildRule(
        include=("包胶", "轮毂"),
        exclude=COMMON_EXCLUDE,
        min_extent_m=0.012,
        max_parts=4,
    ),
    "l_drive_bar_Link": BuildRule(include=("小腿传动杆",), min_extent_m=0.006, max_parts=1),
    "r_drive_bar_Link": BuildRule(include=("小腿传动杆",), min_extent_m=0.006, max_parts=1),
    "l_coupler_Link": BuildRule(include=("异形连杆",), min_extent_m=0.006, max_parts=1),
    "r_coupler_Link": BuildRule(include=("异形连杆",), min_extent_m=0.006, max_parts=1),
}

REFERENCE_MESHES: dict[str, tuple[str, ...]] = {
    "base_link": tuple(f"base_link_{index}.obj" for index in range(12)),
    "lf0_Link": ("lf_thigh_link.STL",),
    "rf0_Link": ("rf_thigh_link.STL",),
    "lf1_Link": ("lf_calf_3_link.STL",),
    "rf1_Link": ("rf_calf_3_link.STL",),
    "l_wheel_Link": ("lf_wheel_link.STL",),
    "r_wheel_Link": ("rf_wheel_link.STL",),
    "l_drive_bar_Link": ("lf_calf_1_link.STL",),
    "r_drive_bar_Link": ("rf_calf_1_link.STL",),
    "l_coupler_Link": ("lf_calf_2_link.STL",),
    "r_coupler_Link": ("rf_calf_2_link.STL",),
}


def _safe_clean_dir(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise RuntimeError(f"拒绝清理输出根目录之外的路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _candidate_files(raw_target_dir: Path, rule: BuildRule) -> list[Path]:
    files = sorted(
        {path.resolve() for path in [*raw_target_dir.glob("*.STL"), *raw_target_dir.glob("*.stl")]}
    )
    candidates = [
        path
        for path in files
        if _matches_any(path.name, rule.include) and not _matches_any(path.name, rule.exclude)
    ]
    if not candidates and len(files) == 1:
        candidates = files
    return candidates


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices()
    return mesh


def _component_hulls(mesh: trimesh.Trimesh, min_extent_m: float) -> list[trimesh.Trimesh]:
    pieces = mesh.split(only_watertight=False)
    if len(pieces) == 0:
        pieces = [mesh]
    hulls: list[trimesh.Trimesh] = []
    for piece in pieces:
        if len(piece.vertices) < 4:
            continue
        extents = np.asarray(piece.extents, dtype=np.float64)
        if float(np.max(extents)) < min_extent_m:
            continue
        try:
            hull = piece.convex_hull
        except Exception:
            continue
        hull.remove_unreferenced_vertices()
        hull.merge_vertices()
        hulls.append(hull)
    return hulls


def _score_mesh(mesh: trimesh.Trimesh) -> float:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return float(np.prod(np.maximum(extents, 1e-6)))


def _reference_mesh(target: str, mesh_root: Path) -> trimesh.Trimesh:
    paths = [mesh_root / rel_path for rel_path in REFERENCE_MESHES[target]]
    meshes = [_load_mesh(path) for path in paths]
    return trimesh.util.concatenate(meshes)


def _sample_points(points: np.ndarray, max_count: int) -> np.ndarray:
    if len(points) <= max_count:
        return points
    indices = np.linspace(0, len(points) - 1, max_count, dtype=np.int64)
    return points[indices]


def _best_axis_transform(
    source: trimesh.Trimesh, reference: trimesh.Trimesh
) -> tuple[np.ndarray, float]:
    src_points = np.asarray(source.vertices, dtype=np.float64)
    ref_points = np.asarray(reference.vertices, dtype=np.float64)
    src_center = (src_points.min(axis=0) + src_points.max(axis=0)) * 0.5
    ref_center = (ref_points.min(axis=0) + ref_points.max(axis=0)) * 0.5
    src_centered = src_points - src_center
    ref_sample = _sample_points(ref_points, 2500)
    ref_tree = cKDTree(ref_sample)

    best_score = float("inf")
    best_matrix = np.eye(4)
    for perm in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3), dtype=np.float64)
            for axis, src_axis in enumerate(perm):
                rotation[axis, src_axis] = signs[axis]

            transformed = src_centered @ rotation.T + ref_center
            src_sample = _sample_points(transformed, 2500)
            src_tree = cKDTree(src_sample)
            src_to_ref = ref_tree.query(src_sample, k=1)[0]
            ref_to_src = src_tree.query(ref_sample, k=1)[0]
            extent_delta = np.linalg.norm(
                np.sort(np.ptp(transformed, axis=0)) - np.sort(np.ptp(ref_points, axis=0))
            )
            score = float(np.mean(src_to_ref) + np.mean(ref_to_src) + 0.25 * extent_delta)
            if score < best_score:
                best_score = score
                best_matrix = np.eye(4)
                best_matrix[:3, :3] = rotation
                best_matrix[:3, 3] = ref_center - rotation @ src_center
    return best_matrix, best_score


def _build_target(
    raw_root: Path, output_root: Path, target: str, rule: BuildRule
) -> dict[str, object]:
    raw_target_dir = raw_root / target
    files = _candidate_files(raw_target_dir, rule)
    if not files:
        raise RuntimeError(f"{target} 没有筛出可用 STL，请检查导出结果和筛选规则。")

    output_dir = output_root / target
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, object]] = []
    hulls_with_source: list[tuple[Path, trimesh.Trimesh]] = []
    for path in files:
        mesh = _load_mesh(path)
        for hull in _component_hulls(mesh, rule.min_extent_m):
            hulls_with_source.append((path, hull))

    hulls_with_source.sort(key=lambda item: _score_mesh(item[1]), reverse=True)
    if rule.max_parts is not None:
        hulls_with_source = hulls_with_source[: rule.max_parts]
    if not hulls_with_source:
        raise RuntimeError(f"{target} 筛选后没有满足尺寸阈值的外轮廓。")

    for _, hull in hulls_with_source:
        hull.apply_scale(0.001)

    combined = trimesh.util.concatenate([hull for _, hull in hulls_with_source])
    reference = _reference_mesh(target, output_root.parent)
    transform, align_score = _best_axis_transform(combined, reference)
    for _, hull in hulls_with_source:
        hull.apply_transform(transform)

    for index, (source_path, hull) in enumerate(hulls_with_source):
        out_name = f"{target}_sw_env_{index:02d}.stl"
        out_path = output_dir / out_name
        hull.export(out_path)
        mesh_rel_root = output_root.name
        generated.append(
            {
                "file": f"{mesh_rel_root}/{target}/{out_name}",
                "source": source_path.name,
                "vertices": len(hull.vertices),
                "faces": len(hull.faces),
                "extents": np.asarray(hull.extents, dtype=float).round(6).tolist(),
            }
        )

    return {
        "target": target,
        "align_score": align_score,
        "transform": np.asarray(transform, dtype=float).round(8).tolist(),
        "raw_files": [path.name for path in files],
        "generated": generated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path(".tmp/sw_collision_raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/sw_collision"),
    )
    parser.add_argument("--target", action="append", choices=sorted(RULES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    _safe_clean_dir(output_root, output_root)

    selected = set(args.target or [])
    targets = [name for name in RULES if not selected or name in selected]
    manifest = [_build_target(raw_root, output_root, name, RULES[name]) for name in targets]
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成完成: {manifest_path}")
    for item in manifest:
        print(f"  {item['target']}: {len(item['generated'])} collision meshes")


if __name__ == "__main__":
    main()
