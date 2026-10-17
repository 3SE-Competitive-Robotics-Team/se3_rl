"""把 RMUC 2027 官方场地 CAD（三角化网格）导出成 sim2x 场景：碰撞用高度场，视觉用按颜色合并的 mesh。

数据来源：keilking66/RMUC2027-map Release v1.0.0 的 `field_meshes.npz` + `manifest.json`，是官方
`RMUC2027_V1.0.0.stp`（2026-10-15 规则版）按 2 mm 线性偏差三角化的结果，644 个分件、605 万三角面，
米制、Z 向上、保留 CAD 原点。许可 CC BY-NC-SA 4.0（署名 RoboMaster 高校赛务，转换 keilking66），
原始文件放 `assets/fields/rmuc2027/raw/`，不进 Git。

为什么碰撞用高度场而不是直接挂 mesh：MuJoCo 的 mesh 碰撞一律按凸包算，场地的台阶、凹槽、坡道
挂成 mesh 会被凸包填平；605 万三角面也远超可接受的接触检测开销。轮腿机器人与场地的接触基本是
2.5D 的（地面、台阶、坡、飞坡），所以把全部分件按「每格取最高点」栅格化成高度场：
- 分辨率默认 2 cm；格内取所有三角面采样点的最高 z，采样间距 ≤ 半格，保证细台阶沿不漏；
- 代价：桥下、洞口等悬空结构会被填成实心柱（高度场表达不了悬空），需要钻洞时要单独做凸分解；
- 官方 CAD 地面不是水平面：沿 y 向两侧各有约 1.5°（坡度 0.026）的屋脊形坡，脊线在场地中线
  y_cad = 1.6243 m，两侧边缘比脊线低约 0.12 m。这是 CAD 原样，没有改平。

视觉：同色分件合并成一份 mesh，用顶点聚类（默认 1 cm 网格）减面，contype/conaffinity 均为 0，
放 group 2（sim2x Viser 默认显示）；碰撞高度场放 group 3（默认隐藏，可在 GUI 打开对照）。

坐标：把 CAD 平移 (0, -1.6243434, +1.6403437)，使规则矩形中心落在世界原点、地面脊线 z=0，
x 沿长边（±14 m），y 沿短边（规则矩形 ±7.5 m）。官方 CAD 不含外围围挡，脚本按高度场边界加四面
0.5 m 高的墙（不是官方几何，只防机器人掉出高度场）。

出生点：每个 `--spawn X Y` 写一个 `se3_env_<i>_rmuc2027_r<i>` site（含地面 z，取出生点 ±0.2 m 内最高格）；
sim2x browser 只建一台机器人，在 GUI 的地形/难度下拉里按 r<i> 切换出生点。默认两处中场两侧的空地，
关于原点中心对称（场地本身中心对称）。

用法：
    uv run python scripts/make_rmuc2027_field_scene.py

输出：
    assets/robots/serialleg/meshes/rmuc2027/   高度场 .bin + 视觉 .stl（生成物，不进 Git）
    assets/robots/serialleg/mjcf/serialleg_rmuc2027.xml   场景（include 机器人 MJCF）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "assets/fields/rmuc2027/raw"
MJCF_DIR = REPO_ROOT / "assets/robots/serialleg/mjcf"
MESH_SUBDIR = "rmuc2027"
MESH_DIR = REPO_ROOT / "assets/robots/serialleg/meshes" / MESH_SUBDIR

SITE_PREFIX = "se3_env_"
DROP_FLOOR_NUMERIC = "se3_drop_robot_floor"
DROP_KNEE_GAS_SPRING_NUMERIC = "se3_drop_knee_gas_spring"

# 规则矩形中心（README：y∈[-5.8756566, 9.1243434]）与地面脊线高度（p00000 顶面最高点）。
FIELD_CENTER_CAD = np.array([0.0, 1.6243434, -1.6403437])
OFFSET = -FIELD_CENTER_CAD
DEFAULT_SPAWNS = ((-6.5, -1.6), (6.5, 1.6))
STL_MAX_FACES = 200_000
WALL_HEIGHT_M = 0.5
WALL_THICKNESS_M = 0.1
# 与机器人 MJCF 的 floor 一致的接触参数。
CONTACT_ATTRS = 'friction="0.8 0.005 0.0001" condim="3" contype="2" conaffinity="1"'


def _fmt(values, digits: int = 7) -> str:
    return " ".join(f"{float(v):.{digits}g}" for v in np.asarray(values).ravel())


def _check_sha256(npz_path: Path) -> None:
    """按 Release 附带的 SHA256SUMS.txt 校验 npz，防止拿错版本或下载不完整。"""
    sums = (npz_path.parent / "SHA256SUMS.txt").read_text(encoding="utf-8").split()
    expected = dict(zip(sums[1::2], sums[0::2], strict=True))[npz_path.name]
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{npz_path.name} SHA-256 不匹配：{digest} != {expected}")


def load_parts(raw_dir: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """读取全部分件，返回 [(世界坐标顶点, 面, rgba)]；npz 禁用 pickle。"""
    npz_path = raw_dir / "field_meshes.npz"
    _check_sha256(npz_path)
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["units"] != "meter" or manifest["up_axis"] != "Z":
        raise SystemExit(f"manifest 坐标约定变了：{manifest['units']} / {manifest['up_axis']}")
    data = np.load(npz_path, allow_pickle=False)
    parts = []
    for part in manifest["parts"]:
        mid = part["mesh_id"]
        vertices = data[f"{mid}_vertices"].astype(np.float64) + OFFSET
        parts.append((vertices, data[f"{mid}_faces"].astype(np.int64), np.asarray(part["rgba"])))
    return parts


def rasterize(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], cell: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把所有三角面按「每格最高 z」栅格化；返回 (高度 [ny, nx]，x 网格，y 网格)，行沿 +y。"""
    lo = np.min([v.min(axis=0) for v, _, _ in parts], axis=0)
    hi = np.max([v.max(axis=0) for v, _, _ in parts], axis=0)
    nx = int(np.floor((hi[0] - lo[0]) / cell)) + 1
    ny = int(np.floor((hi[1] - lo[1]) / cell)) + 1
    xs = lo[0] + cell * np.arange(nx)
    ys = lo[1] + cell * np.arange(ny)
    height = np.full((ny, nx), -np.inf)
    batch_points = 4_000_000
    for vertices, faces, _ in parts:
        tri = vertices[faces]
        edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
        # 采样间距 ≤ 半格：每条边 n 等分，三角内取 (n+1)(n+2)/2 个重心坐标点。
        splits = np.ceil(edge / (0.5 * cell)).astype(np.int64).clip(1, None)
        for n in np.unique(splits):
            sel = tri[splits == n]
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = (i + j) <= n
            a = (i[keep] / n)[None, :, None]
            b = (j[keep] / n)[None, :, None]
            step = max(1, batch_points // a.shape[1])
            for s in range(0, len(sel), step):
                q = sel[s : s + step]
                pts = (q[:, None, 0] * (1 - a - b) + q[:, None, 1] * a + q[:, None, 2] * b).reshape(
                    -1, 3
                )
                ix = np.clip(np.rint((pts[:, 0] - lo[0]) / cell).astype(np.int64), 0, nx - 1)
                iy = np.clip(np.rint((pts[:, 1] - lo[1]) / cell).astype(np.int64), 0, ny - 1)
                np.maximum.at(height, (iy, ix), pts[:, 2])
    empty = ~np.isfinite(height)
    if empty.any():
        # 边缘少量格子没有采样点：取最近有效格。
        _, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
        height = height[iy, ix]
    return height, xs, ys


def cluster_mesh(
    vertices: np.ndarray, faces: np.ndarray, grid: float
) -> tuple[np.ndarray, np.ndarray]:
    """顶点聚类减面：顶点吸附到 grid 网格后合并，删掉退化与重复面。"""
    keys = np.rint(vertices / grid).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(unique), 3))
    np.add.at(sums, inverse, vertices)
    merged = sums / np.bincount(inverse, minlength=len(unique))[:, None]
    new_faces = inverse[faces]
    ok = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[ok]
    _, first = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    new_faces = new_faces[np.sort(first)]
    used = np.unique(new_faces)
    remap = np.full(len(merged), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return merged[used], remap[new_faces]


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """写二进制 STL（MuJoCo 原生支持，不依赖额外库）。"""
    tri = vertices[faces].astype(np.float32)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    record = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["n"] = normal
    record["v"] = tri
    with path.open("wb") as f:
        f.write(b"RMUC2027 field visual mesh".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        f.write(record.tobytes())


def write_hfield(path: Path, height: np.ndarray) -> tuple[float, float]:
    """写 MuJoCo 自定义高度场二进制（int32 nrow, int32 ncol, float32 数据，行沿 +y）。

    MuJoCo 加载时把数据归一化到 [0, 1]，所以返回 (最低 z, 高差)，由 geom pos 与 size 还原。
    """
    z_min = float(height.min())
    z_range = float(height.max() - z_min)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", *height.shape))
        f.write(np.ascontiguousarray(height - z_min, dtype=np.float32).tobytes())
    return z_min, z_range


def build(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    parts = load_parts(args.raw_dir)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    for stale in MESH_DIR.glob("rmuc2027_visual_*.stl"):
        stale.unlink()

    height, xs, ys = rasterize(parts, args.cell)
    z_min, z_range = write_hfield(MESH_DIR / "field_hfield.bin", height)
    center = np.array([0.5 * (xs[0] + xs[-1]), 0.5 * (ys[0] + ys[-1])])
    radius = np.array([0.5 * (xs[-1] - xs[0]), 0.5 * (ys[-1] - ys[0])])
    print(
        f"高度场 {height.shape[1]}×{height.shape[0]}（{args.cell * 100:g} cm），z ∈ [{z_min:.3f}, {z_min + z_range:.3f}]"
    )

    groups: dict[tuple[float, ...], list[int]] = {}
    for index, (_, _, rgba) in enumerate(parts):
        groups.setdefault(tuple(np.round(rgba, 3)), []).append(index)
    mesh_lines: list[str] = []
    geom_lines: list[str] = []
    total_in = total_out = 0
    for k, (rgba, members) in enumerate(sorted(groups.items())):
        offsets = np.cumsum([0] + [len(parts[i][0]) for i in members])
        vertices = np.concatenate([parts[i][0] for i in members])
        faces = np.concatenate([parts[i][1] + offsets[n] for n, i in enumerate(members)])
        total_in += len(faces)
        vertices, faces = cluster_mesh(vertices, faces, args.visual_grid)
        total_out += len(faces)
        # MuJoCo 的 STL 解码器单文件上限 200000 面，超出按面序切块。
        for c, start in enumerate(range(0, len(faces), STL_MAX_FACES)):
            name = f"rmuc2027_visual_{k:02d}_{c}"
            write_stl(MESH_DIR / f"{name}.stl", vertices, faces[start : start + STL_MAX_FACES])
            # 只做视觉：maxhullvert 压低凸包计算，inertia=shell 避免对非封闭网格算体积。
            mesh_lines.append(
                f'    <mesh name="{name}" file="{MESH_SUBDIR}/{name}.stl" maxhullvert="4" inertia="shell" />'
            )
            geom_lines.append(
                f'    <geom name="{name}" type="mesh" mesh="{name}" rgba="{_fmt(rgba, 3)}"'
                ' contype="0" conaffinity="0" group="2" />'
            )
    print(
        f"视觉 mesh {len(geom_lines)} 份，三角面 {total_in} → {total_out}（聚类 {args.visual_grid * 100:g} cm）"
    )

    site_lines = []
    for i, spawn_xy in enumerate(np.asarray(args.spawn or DEFAULT_SPAWNS, dtype=float)):
        iy = int(np.rint((spawn_xy[1] - ys[0]) / args.cell))
        ix = int(np.rint((spawn_xy[0] - xs[0]) / args.cell))
        spawn_z = float(height[max(iy - 10, 0) : iy + 11, max(ix - 10, 0) : ix + 11].max())
        site_lines.append(
            f'    <site name="{SITE_PREFIX}{i:03d}_rmuc2027_r{i}" pos="{_fmt((*spawn_xy, spawn_z), 6)}"'
            ' size="0.001" rgba="0 0 0 0" />'
        )

    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    half_t = 0.5 * WALL_THICKNESS_M
    # 墙底取高度场最低点（就是场地边缘地面），墙顶高出它 WALL_HEIGHT_M。
    wall_center_z = z_min + 0.5 * WALL_HEIGHT_M
    walls = [
        ("north", (0.5 * (x0 + x1), y1 + half_t), (radius[0] + WALL_THICKNESS_M, half_t)),
        ("south", (0.5 * (x0 + x1), y0 - half_t), (radius[0] + WALL_THICKNESS_M, half_t)),
        ("east", (x1 + half_t, 0.5 * (y0 + y1)), (half_t, radius[1] + WALL_THICKNESS_M)),
        ("west", (x0 - half_t, 0.5 * (y0 + y1)), (half_t, radius[1] + WALL_THICKNESS_M)),
    ]
    wall_lines = [
        f'    <geom name="rmuc2027_wall_{name}" type="box" pos="{_fmt((*xy, wall_center_z), 6)}"'
        f' size="{_fmt((*half, 0.5 * WALL_HEIGHT_M), 6)}" rgba="0.75 0.75 0.78 0.35" group="2" {CONTACT_ATTRS} />'
        for name, xy, half in walls
    ]

    lines = [
        '<mujoco model="serialleg_rmuc2027">',
        "  <!-- 由 scripts/make_rmuc2027_field_scene.py 生成，请勿手改。",
        "       RMUC 2027 官方场地 CAD V1.0.0（CC BY-NC-SA 4.0，RoboMaster 高校赛务；网格转换 keilking66/RMUC2027-map）。",
        f"       CAD → 世界平移 {_fmt(OFFSET, 8)}：规则矩形中心在原点，地面脊线 z=0。",
        f"       碰撞 = {args.cell * 100:g} cm 最高点高度场（悬空结构被填实）；视觉 = 同色合并 + {args.visual_grid * 100:g} cm 顶点聚类。",
        "       四面围墙不是官方几何，只防掉出高度场。 -->",
        f'  <include file="{args.robot_mjcf}" />',
        f'  <statistic center="0 0 0" extent="{2.0 * float(np.hypot(*radius)):.3g}" />',
        "  <custom>",
        f'    <numeric name="{DROP_FLOOR_NUMERIC}" data="1" />',
    ]
    if args.drop_knee_gas_spring:
        lines.append(f'    <numeric name="{DROP_KNEE_GAS_SPRING_NUMERIC}" data="1" />')
    lines += [
        "  </custom>",
        "  <asset>",
        f'    <hfield name="rmuc2027_hfield" file="{MESH_SUBDIR}/field_hfield.bin"'
        f' size="{_fmt((*radius, max(z_range, 1e-3), 0.1), 8)}" />',
        *mesh_lines,
        "  </asset>",
        "  <worldbody>",
        *site_lines,
        f'    <geom name="rmuc2027_hfield" type="hfield" hfield="rmuc2027_hfield" pos="{_fmt((*center, z_min), 8)}"'
        f' rgba="0.55 0.57 0.6 1" group="3" {CONTACT_ATTRS} />',
        *wall_lines,
        *geom_lines,
        "  </worldbody>",
        "</mujoco>",
        "",
    ]
    return "\n".join(lines), height, xs, ys


def verify(scene_path: Path, height: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    """往返校验：在编译后的场景里对高度场竖直打射线，和栅格高度逐点比对（只看碰撞 group 3）。"""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mask = np.zeros(6, dtype=np.uint8)
    mask[3] = 1
    rng = np.random.default_rng(0)
    geom_id = np.zeros(1, dtype=np.int32)
    worst = 0.0
    for _ in range(2000):
        iy = int(rng.integers(1, len(ys) - 1))
        ix = int(rng.integers(1, len(xs) - 1))
        dist = mujoco.mj_ray(
            model,
            data,
            np.array([xs[ix], ys[iy], 10.0]),
            np.array([0.0, 0.0, -1.0]),
            mask,
            1,
            -1,
            geom_id,
        )
        if dist < 0:
            raise SystemExit(f"往返校验失败：({xs[ix]:.2f}, {ys[iy]:.2f}) 射线没打到高度场")
        worst = max(worst, abs((10.0 - dist) - height[iy, ix]))
    if worst > 2.0e-3:
        raise SystemExit(
            f"往返校验失败：高度最大偏差 {worst * 1000:.2f} mm，超过 2 mm（行序可能翻了）"
        )
    print(f"往返校验通过：2000 个格点，高度场最大偏差 {worst * 1000:.3f} mm；geom {model.ngeom}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help="field_meshes.npz 等原始文件目录"
    )
    parser.add_argument("--out", type=Path, default=MJCF_DIR / "serialleg_rmuc2027.xml")
    parser.add_argument("--cell", type=float, default=0.02, help="高度场分辨率 m")
    parser.add_argument("--visual-grid", type=float, default=0.01, help="视觉 mesh 顶点聚类网格 m")
    parser.add_argument(
        "--spawn",
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("X", "Y"),
        help="出生点世界 xy（规则矩形中心为原点），可重复；默认 (-6.5, -1.6) 与 (6.5, 1.6)",
    )
    parser.add_argument(
        "--robot-mjcf",
        default="serialleg_closed_chain_v3_train_obb_trim.xml",
        help="被 include 的机器人 MJCF 文件名（必须与输出同目录）",
    )
    parser.add_argument(
        "--drop-knee-gas-spring",
        action="store_true",
        help="写入 se3_drop_knee_gas_spring 标记，sim2x 删掉膝气弹簧（回放无弹簧训练的策略）",
    )
    args = parser.parse_args()
    xml, height, xs, ys = build(args)
    args.out.write_text(xml, encoding="utf-8")
    print(f"已写入 {args.out}")
    verify(args.out, height, xs, ys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
