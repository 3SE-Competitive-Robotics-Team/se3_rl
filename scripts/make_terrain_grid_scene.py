"""把训练的整张地形网格导出成 sim2x 场景：每块地形一个出生点，sim2x 按出生点数量建 env（每块一台机器人）。

与 `make_stair_scene.py` 同一思路：不重新实现几何，直接用 mjlab 的 `TerrainGenerator` 编译一遍训练的
`rough_terrains_cfg()`，再把编译后模型里的每个 geom 原样写进场景——box 逐个复制（含 friction / solref /
solimp / condim / priority，从**编译后**的模型读，避免 spec 阶段默认值未解析的坑），heightfield 用
MJCF 的内联 `elevation` 导出（MuJoCo 3.10 支持），高度数据按 MuJoCo 的存储顺序写回（XML 第一行对应
存储的最后一行，写出时要翻转行序）。

每块地形写一个 site：`se3_env_<index>_<column>_r<row>`，pos = 该块的出生点（`terrain_origins`，含地面 z）。
sim2x 的 browser 看到这些 site 就按块建 env，每个 env 出生在自己那块地上；场景里还写一个
`<numeric name="se3_drop_robot_floor">`，让 sim2x 编译时删掉机器人 MJCF 自带的 z=0 平面
（反金字塔的坑底在 z<0，平面会把坑填平）。

随机地形（random_rough / obstacles）每次构造生成器都重新随机，导出的是一次抽样；往返校验必须复用
导出时那份编译好的参考模型。

用法（M51 地形集，全部 10 列 × 10 行 = 100 台）：
    uv run python scripts/make_terrain_grid_scene.py --out assets/robots/serialleg/mjcf/serialleg_terrain_grid_m51.xml

只要部分行/列（例如 3、6、9 三个难度 → 30 台）：
    uv run python scripts/make_terrain_grid_scene.py --rows 3 6 9 --out ...

输出必须与机器人 MJCF 同目录：compiler 的 meshdir="../meshes" 按主模型所在目录解析。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from se3_train.tasks.rough.terrains import rough_terrains_cfg

SITE_PREFIX = "se3_env_"
DROP_FLOOR_NUMERIC = "se3_drop_robot_floor"


def _fmt(values, digits: int = 6) -> str:
    return " ".join(f"{float(v):.{digits}g}" for v in np.asarray(values).ravel())


def build(
    *,
    random_terrains: bool,
    rows: tuple[int, ...] | None,
    columns: tuple[str, ...] | None,
    robot_mjcf: str,
) -> tuple[str, list[tuple[str, np.ndarray]], mujoco.MjModel]:
    """返回场景 XML、出生点列表 [(site 名, 世界坐标)] 与编译后的参考地形模型。"""
    from mjlab.terrains.terrain_generator import TerrainGenerator

    cfg = rough_terrains_cfg(random_terrains=random_terrains)
    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    names = list(cfg.sub_terrains.keys())
    origins = np.asarray(generator.terrain_origins)  # [rows, cols, 3]
    num_rows = origins.shape[0]
    sel_rows = tuple(range(num_rows)) if rows is None else tuple(sorted(set(rows)))
    sel_cols = names if columns is None else [c for c in names if c in set(columns)]
    for r in sel_rows:
        if not 0 <= r < num_rows:
            raise SystemExit(f"--rows 超出范围 0..{num_rows - 1}：{r}")
    if columns is not None:
        unknown = sorted(set(columns) - set(names))
        if unknown:
            raise SystemExit(f"--columns 未知列名 {unknown}；可选 {names}")
    half = float(cfg.size[0]) * 0.5 - 1e-6
    keep = [(origins[r, names.index(c), :2], c, r) for c in sel_cols for r in sel_rows]

    def geom_in_selection(gid: int) -> bool:
        pos = np.asarray(data.geom_xpos[gid], dtype=float)
        return any(abs(pos[0] - xy[0]) <= half and abs(pos[1] - xy[1]) <= half for xy, _, _ in keep)

    grid_half = float(cfg.size[0]) * max(num_rows, len(names)) * 0.5

    def in_grid(pos) -> bool:
        return abs(pos[0]) <= grid_half and abs(pos[1]) <= grid_half

    extent = 2.0 * (grid_half + float(cfg.border_width))
    lines: list[str] = [
        '<mujoco model="serialleg_terrain_grid">',
        "  <!-- 由 scripts/make_terrain_grid_scene.py 生成，请勿手改。",
        "       几何取自 mjlab TerrainGenerator 编译训练 rough_terrains_cfg("
        f"random_terrains={random_terrains}) 后的整张网格，逐块与训练一致（随机地形为一次抽样）。",
        f"       列：{', '.join(sel_cols)}；行（难度等级）：{', '.join(str(r) for r in sel_rows)}。",
        f"       每块一个 site `{SITE_PREFIX}<index>_<列>_r<行>`（出生点，含地面 z），sim2x 按 site 建 env。 -->",
        f'  <include file="{robot_mjcf}" />',
        f'  <statistic center="0 0 0" extent="{extent:g}" />',
        "  <custom>",
        f'    <numeric name="{DROP_FLOOR_NUMERIC}" data="1" />',
        "  </custom>",
    ]

    used_hfields: list[int] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_HFIELD and geom_in_selection(gid):
            hid = int(model.geom_dataid[gid])
            if hid not in used_hfields:
                used_hfields.append(hid)
    if used_hfields:
        lines.append("  <asset>")
        for hid in used_hfields:
            nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
            adr = int(model.hfield_adr[hid])
            grid = np.asarray(model.hfield_data[adr : adr + nrow * ncol], dtype=float).reshape(
                nrow, ncol
            )
            lines.append(
                f'    <hfield name="se3_hf_{hid}" nrow="{nrow}" ncol="{ncol}"'
                f' size="{_fmt(model.hfield_size[hid])}" elevation="{_fmt(grid[::-1].ravel(), 5)}" />'
            )
        lines.append("  </asset>")

    lines.append("  <worldbody>")
    sites: list[tuple[str, np.ndarray]] = []
    for index, (_xy, col, row) in enumerate(keep):
        origin = origins[row, names.index(col)]
        name = f"{SITE_PREFIX}{index:03d}_{col}_r{row}"
        lines.append(f'    <site name="{name}" pos="{_fmt(origin)}" size="0.02" rgba="0 0 0 0" />')
        sites.append((name, np.asarray(origin, dtype=float)))

    n_box = n_hf = 0
    for gid in range(model.ngeom):
        gtype = int(model.geom_type[gid])
        if gtype not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_HFIELD):
            continue
        pos = np.asarray(data.geom_xpos[gid], dtype=float)
        # 网格内但不在选中块里的几何丢掉；网格外的边框保留。
        if in_grid(pos) and not geom_in_selection(gid):
            continue
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.asarray(data.geom_xmat[gid], dtype=float).ravel())
        common = (
            f' pos="{_fmt(pos)}" quat="{_fmt(quat)}" rgba="{_fmt(model.geom_rgba[gid], 4)}"'
            f' friction="{_fmt(model.geom_friction[gid])}" solref="{_fmt(model.geom_solref[gid])}"'
            f' solimp="{_fmt(model.geom_solimp[gid])}" condim="{int(model.geom_condim[gid])}"'
            f' priority="{int(model.geom_priority[gid])}" contype="{int(model.geom_contype[gid])}"'
            f' conaffinity="{int(model.geom_conaffinity[gid])}"'
        )
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            n_box += 1
            lines.append(
                f'    <geom name="se3_tg_box_{gid}" type="box" size="{_fmt(model.geom_size[gid])}"'
                f"{common} />"
            )
        else:
            n_hf += 1
            hid = int(model.geom_dataid[gid])
            lines.append(
                f'    <geom name="se3_tg_hf_{gid}" type="hfield" hfield="se3_hf_{hid}"{common} />'
            )
    lines += ["  </worldbody>", "</mujoco>", ""]
    print(
        f"选中 {len(sites)} 块：box {n_box} 个，heightfield {n_hf} 个（资产 {len(used_hfields)} 份）"
    )
    return "\n".join(lines), sites, model


def verify(out: Path, ref_model: mujoco.MjModel, sites: list[tuple[str, np.ndarray]]) -> None:
    """把导出的场景重新编译，在每块出生点周围用射线量地面高度，与导出时的参考地形逐点对比。"""
    ref_data = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, ref_data)
    scene_model = mujoco.MjModel.from_xml_path(str(out))
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)
    # 机器人自身几何放到 group 5 并屏蔽，射线只打地形。
    for gid in range(scene_model.ngeom):
        if not (scene_model.geom(gid).name or "").startswith("se3_tg_"):
            scene_model.geom_group[gid] = 5
    mask = np.ones(6, dtype=np.uint8)
    mask[5] = 0

    def probe(model, data, x, y, geomgroup):
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(
            model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]), geomgroup, 1, -1, geomid
        )
        return np.nan if dist < 0 else 5.0 - dist

    offsets = np.linspace(-4.0, 4.0, 17)
    per_site: list[tuple[float, str]] = []
    for site_name, origin in sites:
        site_worst = 0.0
        for dx in offsets:
            for dy in offsets:
                x, y = float(origin[0] + dx), float(origin[1] + dy)
                a = probe(ref_model, ref_data, x, y, None)
                b = probe(scene_model, scene_data, x, y, mask)
                if np.isnan(a) and np.isnan(b):
                    continue
                if np.isnan(a) != np.isnan(b):
                    raise SystemExit(
                        f"往返校验失败：({x:.2f},{y:.2f}) 一侧无命中 ref={a} scene={b}"
                    )
                site_worst = max(site_worst, abs(a - b))
        per_site.append((site_worst, site_name))
    worst = max(v for v, _ in per_site)
    for value, site_name in sorted(per_site, reverse=True)[:3]:
        print(f"  偏差最大的块：{site_name} {value * 1000:.2f} mm")
    if worst > 2.0e-3:
        raise SystemExit(f"往返校验失败：地面高度最大偏差 {worst * 1000:.2f} mm，超过 2 mm")
    print(
        f"往返校验通过：{len(sites)} 块 × {len(offsets) ** 2} 点，地面高度最大偏差 {worst * 1000:.2f} mm"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="输出 MJCF 路径（与机器人 MJCF 同目录）"
    )
    parser.add_argument(
        "--terrain-set",
        choices=("m50", "m51"),
        default="m51",
        help="m50 = 七列规则地形；m51 = 加随机粗糙/波浪/障碍三列（默认）",
    )
    parser.add_argument(
        "--rows", type=int, nargs="*", default=None, help="只导出这些行（难度等级），默认全部"
    )
    parser.add_argument("--columns", nargs="*", default=None, help="只导出这些列名，默认全部")
    parser.add_argument(
        "--robot-mjcf",
        default="serialleg_closed_chain_v3_train_obb_trim.xml",
        help="被 include 的机器人 MJCF 文件名（必须与输出同目录）",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过往返校验")
    args = parser.parse_args()
    random_terrains = args.terrain_set == "m51"
    xml, sites, ref_model = build(
        random_terrains=random_terrains,
        rows=None if args.rows is None else tuple(args.rows),
        columns=None if args.columns is None else tuple(args.columns),
        robot_mjcf=args.robot_mjcf,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(args.out))
    n_sites = sum(
        1 for i in range(model.nsite) if (model.site(i).name or "").startswith(SITE_PREFIX)
    )
    print(f"已写入 {args.out}：geom {model.ngeom}、hfield {model.nhfield}、出生点 {n_sites}")
    if not args.no_verify:
        verify(args.out, ref_model, sites)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
