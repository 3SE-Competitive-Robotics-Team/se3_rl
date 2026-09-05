"""视觉网格抽稀：把闭链 MJCF 引用的超精细视觉 STL 原地替换为低面数版本，并验证动力学逐位不变。

背景（2026-09-05）：`sw_original_base/` 的 13 块机身视觉网格合计约 234 万三角形（117 MB STL），
轮/小腿/大腿视觉网格再加 17 万。Viser 每次连接都要把整套网格推给浏览器：Pod 内本地都只能以
约 1 MB/s 产出（物理线程占着 GIL），再经 0.4 MB/s 的远程链路根本传不完，4040 页面永远在加载。
视觉 geom 全部 contype=conaffinity=0，11 个 body 都有显式 inertial，网格不进入动力学，
抽稀只影响外观；MJCF 本身不改，ONNX 元数据里的资产 hash（只覆盖 XML）保持不变，旧 checkpoint
在 Viser browser 中继续可用。原始网格保留在 git 历史里（抽稀提交的父 commit）。

用法（fast-simplification 只在本脚本临时使用，不进 pyproject）：
    uv run --with fast-simplification python scripts/decimate_visual_meshes.py [--dry-run] [--steps N]

流程：先用原网格编译并仿真一段固定随机控制序列作为基线，再抽稀写回，重新编译仿真并逐位比较
qpos/qvel/body 质量惯量与碰撞 geom 参数；任何不一致都以非零退出码报告。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = REPO_ROOT / "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
MESH_DIR = REPO_ROOT / "assets/robots/serialleg/meshes"

# 目标面数按零件尺寸给：机身块每块 3000 面（1.7%），轮 8000，小腿 6000，大腿 4000。
# 机身 13 块是同一网格的切片，边界必须锁住（preserve_border）才不会在接缝处开裂，
# 因此实际面数受边界顶点数托底，通常停在目标之上；这是外观代价最小的取舍。
TARGET_FACES: dict[str, int] = {
    **{f"sw_original_base/base_link_chunk_{i:02d}.stl": 3000 for i in range(13)},
    "lf_wheel_link.STL": 8000,
    "rf_wheel_link.STL": 8000,
    "lf_calf_3_link.STL": 6000,
    "rf_calf_3_link.STL": 6000,
    "lf_thigh_link.STL": 4000,
    "rf_thigh_link.STL": 4000,
}
# agg 越低越保形；扫描显示 5 时 99% 顶点偏差 <3 mm，再低面数降不下来。
_AGGRESSIVENESS = 5.0


def _simulate(xml_path: Path, steps: int, seed: int) -> dict[str, np.ndarray]:
    """编译 MJCF，用固定随机控制序列仿真，返回可逐位比较的轨迹与惯性量。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    limited = model.actuator_ctrllimited.astype(bool)
    lo = np.where(limited, lo, -1.0)
    hi = np.where(limited, hi, 1.0)
    qpos, qvel = [], []
    for _ in range(steps):
        data.ctrl[:] = rng.uniform(lo, hi)
        mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    return {
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "body_mass": model.body_mass.copy(),
        "body_inertia": model.body_inertia.copy(),
        "body_ipos": model.body_ipos.copy(),
        "collision_geom_size": model.geom_size[collision].copy(),
        "collision_geom_pos": model.geom_pos[collision].copy(),
        "nmeshvert": np.asarray([model.nmeshvert]),
    }


def _decimate(path: Path, target_faces: int, dry_run: bool) -> tuple[int, int, float, float]:
    """抽稀单个 STL，返回（原面数，新面数，顶点偏差最大值 mm，顶点偏差 99 分位 mm）。"""
    import fast_simplification
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(path, force="mesh")
    faces_before = len(mesh.faces)
    if faces_before <= target_faces:
        return faces_before, faces_before, 0.0, 0.0
    vertices, faces = fast_simplification.simplify(
        mesh.vertices.astype(np.float64),
        mesh.faces.astype(np.int32),
        target_count=target_faces,
        agg=_AGGRESSIVENESS,
        preserve_border=True,
    )
    decimated = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # 偏差用抽稀后顶点到原网格顶点集的最近距离（KD 树）度量，是到原表面距离的上界；
    # 最大值受切片网格自身少数离群顶点影响（无损模式同样出现），以 99 分位为准。
    distances, _ = cKDTree(np.asarray(mesh.vertices)).query(np.asarray(decimated.vertices))
    max_deviation_mm = float(np.max(distances) * 1000.0)
    p99_deviation_mm = float(np.percentile(distances, 99) * 1000.0)
    if not dry_run:
        decimated.export(path, file_type="stl")
    return faces_before, len(decimated.faces), max_deviation_mm, p99_deviation_mm


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="只报告抽稀结果，不写回文件")
    parser.add_argument("--steps", type=int, default=2000, help="动力学等价验证的仿真步数")
    parser.add_argument("--seed", type=int, default=0, help="随机控制序列种子")
    args = parser.parse_args()

    print(f"[基线] 编译并仿真 {args.steps} 步：{MJCF_PATH.relative_to(REPO_ROOT)}")
    baseline = _simulate(MJCF_PATH, args.steps, args.seed)

    total_before = total_after = 0
    bytes_before = bytes_after = 0
    for rel, target in TARGET_FACES.items():
        path = MESH_DIR / rel
        if not path.is_file():
            print(f"错误：网格不存在 {path}", file=sys.stderr)
            return 2
        size_before = path.stat().st_size
        before, after, deviation, p99 = _decimate(path, target, args.dry_run)
        size_after = path.stat().st_size
        total_before += before
        total_after += after
        bytes_before += size_before
        bytes_after += size_after
        print(
            f"  {rel:45s} {before:7d} -> {after:5d} 面  偏差 max {deviation:5.2f} / p99 {p99:4.2f} mm"
            f"  {size_before / 1e6:5.1f} -> {size_after / 1e6:4.2f} MB"
        )
    print(
        f"[汇总] 面数 {total_before} -> {total_after}（{total_after / total_before:.1%}），"
        f"文件 {bytes_before / 1e6:.1f} -> {bytes_after / 1e6:.1f} MB"
    )
    if args.dry_run:
        print("[dry-run] 未写回文件")
        return 0

    print("[验证] 用抽稀后的网格重新编译并仿真")
    after_sim = _simulate(MJCF_PATH, args.steps, args.seed)
    failed = False
    for key, value in baseline.items():
        if key == "nmeshvert":
            print(f"  nmeshvert {int(value[0])} -> {int(after_sim[key][0])}")
            continue
        if np.array_equal(value, after_sim[key]):
            print(f"  {key}: 逐位相同")
        else:
            failed = True
            print(
                f"  {key}: 不一致，最大差 {np.max(np.abs(value - after_sim[key])):.3e}",
                file=sys.stderr,
            )
    if failed:
        print("动力学不等价，请用 git checkout 恢复网格后排查", file=sys.stderr)
        return 1
    print("[完成] 动力学逐位等价，视觉网格已抽稀写回")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
