"""评测场景：每次评测按当前 checkout 的 Rough 训练地形现生成台阶网格，保证与 checkpoint 训练时一致。

复用 `scripts/make_terrain_grid_scene.py` 的 `build()`（mjlab TerrainGenerator 编译 rough_terrains_cfg 后逐块搬出），
只取规则台阶列。场景文件必须与机器人 MJCF 同目录（MJCF 的 meshdir 按主模型目录解析），
所以写成 `assets/robots/serialleg/mjcf/_se3_bench_<hash>.xml`（.gitignore 忽略），按内容哈希复用。
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STAIR_COLUMNS = ("stairs_up", "stairs_two_step_up", "stairs_down", "stairs_two_step_down")


@dataclass(frozen=True)
class GridScene:
    path: str
    spawns: dict[tuple[str, int], tuple[float, float, float]]
    """(列名, 行) → 出生点世界坐标（含地面 z）。"""


def _load_grid_builder():
    path = REPO_ROOT / "scripts" / "make_terrain_grid_scene.py"
    spec = importlib.util.spec_from_file_location("_se3_bench_grid_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"找不到场景生成脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_stair_grid(robot_mjcf: Path, rows: tuple[int, ...]) -> GridScene:
    """生成台阶类四列 × 指定行的网格场景（include 被测契约指向的机器人 MJCF）。"""
    builder = _load_grid_builder()
    xml, sites, _ = builder.build(
        random_terrains=True,
        rows=tuple(sorted(set(rows))),
        columns=STAIR_COLUMNS,
        robot_mjcf=robot_mjcf.name,
    )
    digest = hashlib.sha1(xml.encode("utf-8")).hexdigest()[:12]
    out = robot_mjcf.parent / f"_se3_bench_{digest}.xml"
    if not out.exists():
        tmp = out.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        tmp.write_text(xml, encoding="utf-8")
        tmp.replace(out)
    spawns: dict[tuple[str, int], tuple[float, float, float]] = {}
    for name, pos in sites:
        # site 名：se3_env_<index>_<列>_r<行>
        body = name.split("_", 3)[3]
        column, row = body.rsplit("_r", 1)
        spawns[(column, int(row))] = tuple(float(v) for v in np.asarray(pos).ravel()[:3])
    return GridScene(path=str(out), spawns=spawns)


def robot_mjcf_for(onnx: str) -> Path:
    """按 ONNX 契约解析被测策略对应的机器人 MJCF（与 adapter 同一规则）。"""
    from se3_runtime import PolicyBundle
    from se3_runtime_mujoco.adapter import resolve_mjcf_path

    contract = PolicyBundle.load(onnx).contract
    return Path(resolve_mjcf_path(contract, override=None, artifact_path=onnx)).resolve()


__all__ = ["STAIR_COLUMNS", "GridScene", "build_stair_grid", "robot_mjcf_for"]
