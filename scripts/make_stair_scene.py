"""生成 sim2x 用的台阶场景：直接取训练地形 `stairs_up` 列的第 1–9 级，逐块一致。

不重新实现台阶几何——调用 mjlab 自己的 `TerrainGenerator` 编译一遍
`rough_terrains_cfg()`，再把 `stairs_up` 列每一行的 box 原样搬进场景。所以台阶高、踏面、
平台宽、边框、每级的盒子数与位置，都与训练时机器人真正踩到的那块地形相同。

难度取自生成器的行映射 `t = row / (num_rows - 1)`，即第 N 级 → difficulty N/9，
台阶高 `0.02 + (N/9) × 0.18`：第 1 级 4 cm、第 6 级 14 cm、第 9 级 20 cm。

每一级放进一个独立的 **mocap body** `se3_stair_level_<N>`，并整体平移使该块的出生点
（反金字塔的坑底，`terrain_origins`）落在世界原点——机器人就站在坑底，朝外开即开始爬。
默认只有第 1 级在原点，其余停在 y=1000；sim2x 的 Terrain 面板用一个滑块切换哪一级在原点
（见 submodules/se3-sim2x 的 viewer._setup_terrain_gui）。

**为什么必须是 mocap 而不是普通静态 body**：mjviser 把没有自由度的 body 的几何全部烘进一个
`fixed_bodies_frame`，那个 frame 的位姿只在建场景时算一次、之后每帧只跟随相机偏移——
改 `model.body_pos` 物理会动但画面不动。mocap body 走 `mocap_pos/mocap_quat` 这条独立通道，
mjviser 每帧都读，切级才能实时看到。mocap body 同样没有自由度、不参与动力学，接触照常。

用法：
    uv run python scripts/make_stair_scene.py \
        --out assets/robots/serialleg/mjcf/serialleg_stairs_1to9.xml

输出必须与机器人 MJCF 同目录：compiler 的 meshdir="../meshes" 按主模型所在目录解析。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from se3_train.tasks.rough.terrains import rough_terrains_cfg

PARKED_Y = 1000.0
"""未选中的台阶级停放位置（y 方向），远到不会被碰到也不会挡视线。"""


def _patch_geoms(
    model: mujoco.MjModel, data: mujoco.MjData, origin: np.ndarray, half: float
) -> list[dict]:
    """收集落在以 `origin` 为中心、边长 2*half 的方块内的 box geom，坐标相对 origin。

    从**编译后的模型**读而不是从 MjSpec 读：接触相关的属性（friction / solref / solimp /
    condim / priority）在 spec 阶段可能还没解析默认值，照抄 spec 会拿到错的数。
    2026-09-09 就栽在这：早先版本只搬了 pos/size/quat/rgba，friction 按 floor 硬写成 0.8
    （mjlab 生成的是 1.0）、solref/solimp 被机器人 MJCF 的 <default> 覆盖成 0.01/1.5，
    结果台阶比训练里滑且硬，评测得出"连 4 cm 都上不去"的假结论。
    """
    out: list[dict] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        pos = np.asarray(data.geom_xpos[gid], dtype=float)
        if abs(pos[0] - origin[0]) > half or abs(pos[1] - origin[1]) > half:
            continue
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.asarray(data.geom_xmat[gid], dtype=float).ravel())
        out.append(
            {
                "pos": pos - origin,
                "size": np.asarray(model.geom_size[gid], dtype=float).copy(),
                "quat": quat,
                "rgba": np.asarray(model.geom_rgba[gid], dtype=float).copy(),
                "friction": np.asarray(model.geom_friction[gid], dtype=float).copy(),
                "solref": np.asarray(model.geom_solref[gid], dtype=float).copy(),
                "solimp": np.asarray(model.geom_solimp[gid], dtype=float).copy(),
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
                "condim": int(model.geom_condim[gid]),
                "priority": int(model.geom_priority[gid]),
            }
        )
    return out


def build(levels: tuple[int, ...], robot_mjcf: str) -> str:
    cfg = rough_terrains_cfg()
    from mjlab.terrains.terrain_generator import TerrainGenerator

    generator = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    generator.compile(spec)
    terrain_model = spec.compile()
    terrain_data = mujoco.MjData(terrain_model)
    mujoco.mj_forward(terrain_model, terrain_data)

    names = list(cfg.sub_terrains.keys())
    col = names.index("stairs_up")
    origins = np.asarray(generator.terrain_origins)
    half = float(cfg.size[0]) * 0.5 - 1e-6

    step_low, step_high = cfg.sub_terrains["stairs_up"].step_height_range
    lines: list[str] = [
        '<mujoco model="serialleg_stairs_1to9">',
        "  <!-- 由 scripts/make_stair_scene.py 生成，请勿手改。",
        "       几何取自 mjlab TerrainGenerator 编译 rough_terrains_cfg() 后 stairs_up 列的对应行，",
        "       与训练时机器人踩到的那块地形逐块一致。每级平移使坑底出生点落在世界原点。",
        f"       台阶高 = {step_low} + (级/9) × {round(step_high - step_low, 4)} m。 -->",
        f'  <include file="{robot_mjcf}" />',
        # 未激活地块停在远处；相机统计只描述放在原点的当前地块。
        f'  <statistic center="0 0 0" extent="{max(cfg.size):g}" />',
        "  <worldbody>",
    ]
    for index, level in enumerate(levels):
        origin = origins[level, col]
        geoms = _patch_geoms(terrain_model, terrain_data, origin, half)
        step_height = step_low + (level / max(cfg.num_rows - 1, 1)) * (step_high - step_low)
        y = 0.0 if index == 0 else PARKED_Y
        lines.append(
            f"    <!-- 第 {level} 级：difficulty {level}/{cfg.num_rows - 1}"
            f"，台阶高 {step_height:.4f} m，{len(geoms)} 个 box -->"
        )
        lines.append(f'    <body name="se3_stair_level_{level}" mocap="true" pos="0 {y:g} 0">')
        for gi, g in enumerate(geoms):
            p, s, q, c = g["pos"], g["size"], g["quat"], g["rgba"]
            f, sr, si = g["friction"], g["solref"], g["solimp"]
            lines.append(
                f'      <geom name="stair_{level}_{gi}" type="box"'
                f' pos="{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}"'
                f' size="{s[0]:.6g} {s[1]:.6g} {s[2]:.6g}"'
                f' quat="{q[0]:.6g} {q[1]:.6g} {q[2]:.6g} {q[3]:.6g}"'
                f' rgba="{c[0]:.4g} {c[1]:.4g} {c[2]:.4g} {c[3]:.4g}"'
                f' friction="{f[0]:.6g} {f[1]:.6g} {f[2]:.6g}"'
                f' solref="{sr[0]:.6g} {sr[1]:.6g}"'
                f' solimp="{si[0]:.6g} {si[1]:.6g} {si[2]:.6g}"'
                f' condim="{g["condim"]}" priority="{g["priority"]}"'
                f' contype="{g["contype"]}" conaffinity="{g["conaffinity"]}" />'
            )
        lines.append("    </body>")
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets/robots/serialleg/mjcf/serialleg_stairs_1to9.xml"),
    )
    parser.add_argument(
        "--robot-mjcf",
        default="serialleg_closed_chain_v3_train_obb_trim.xml",
        help="被 include 的机器人 MJCF 文件名（必须与输出同目录）",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()

    xml = build(tuple(args.levels), args.robot_mjcf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(args.out))
    base = mujoco.MjModel.from_xml_path(str(args.out.parent / args.robot_mjcf))
    print(f"写出 {args.out}")
    print(
        f"  台阶 mocap body {len(args.levels)} 个（nmocap={model.nmocap}），"
        f"geom +{model.ngeom - base.ngeom}"
    )
    print(f"  nq {base.nq} -> {model.nq}，nv {base.nv} -> {model.nv}（必须不变）")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for level in args.levels:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"se3_stair_level_{level}")
        assert model.body_mocapid[bid] >= 0, f"第 {level} 级不是 mocap body"
        zs = [
            float(data.geom_xpos[g][2] + model.geom_size[g][2])
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid
        ]
        print(f"  第 {level} 级：盒顶 z 最低 {min(zs):.4f} 最高 {max(zs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
