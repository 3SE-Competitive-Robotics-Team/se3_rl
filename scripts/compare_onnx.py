"""比较两个 ONNX 文件是否同构：图结构、全部权重逐位相同、元数据键值差异。

用法：uv run python scripts/compare_onnx.py a.onnx b.onnx
退出码 0 表示图与权重完全一致（元数据差异只打印不判失败）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> int:
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = onnx.load(str(a_path)), onnx.load(str(b_path))
    same_bytes = (
        hashlib.sha256(a_path.read_bytes()).digest() == hashlib.sha256(b_path.read_bytes()).digest()
    )
    print(f"文件逐字节相同: {same_bytes}")
    ops_a = [(n.op_type, tuple(n.input), tuple(n.output)) for n in a.graph.node]
    ops_b = [(n.op_type, tuple(n.input), tuple(n.output)) for n in b.graph.node]
    print(f"图节点: {len(ops_a)} vs {len(ops_b)}，序列相同: {ops_a == ops_b}")
    ia = {t.name: numpy_helper.to_array(t) for t in a.graph.initializer}
    ib = {t.name: numpy_helper.to_array(t) for t in b.graph.initializer}
    weights_ok = set(ia) == set(ib) and all(np.array_equal(ia[k], ib[k]) for k in ia)
    max_diff = max(
        (
            float(np.max(np.abs(ia[k].astype(np.float64) - ib[k].astype(np.float64))))
            for k in ia
            if k in ib and ia[k].shape == ib[k].shape
        ),
        default=float("nan"),
    )
    print(f"权重: {len(ia)} vs {len(ib)} 个，逐位相同: {weights_ok}，最大差 {max_diff:.3e}")
    io_ok = [(i.name, str(i.type)) for i in a.graph.input] == [
        (i.name, str(i.type)) for i in b.graph.input
    ] and [(o.name, str(o.type)) for o in a.graph.output] == [
        (o.name, str(o.type)) for o in b.graph.output
    ]
    print(f"输入输出签名相同: {io_ok}")
    ma = {p.key: p.value for p in a.metadata_props}
    mb = {p.key: p.value for p in b.metadata_props}
    print(f"元数据键: {sorted(ma)} vs {sorted(mb)}")
    for k in sorted(set(ma) | set(mb)):
        va, vb = ma.get(k), mb.get(k)
        if va == vb:
            print(f"  {k}: 相同 ({len(va or '')} 字符)")
            continue
        try:
            ja, jb = json.loads(va or "null"), json.loads(vb or "null")
        except json.JSONDecodeError:
            print(f"  {k}: 不同（非 JSON）")
            continue

        def flat(d, p=""):
            out = {}
            if isinstance(d, dict):
                for kk, vv in d.items():
                    out.update(flat(vv, f"{p}{kk}."))
            elif isinstance(d, list):
                for i, vv in enumerate(d):
                    out.update(flat(vv, f"{p}{i}."))
            else:
                out[p[:-1]] = d
            return out

        fa, fb = flat(ja), flat(jb)
        diff = [kk for kk in sorted(set(fa) | set(fb)) if fa.get(kk) != fb.get(kk)]
        print(
            f"  {k}: JSON 差异 {len(diff)} 项: "
            + ", ".join(f"{kk}={fa.get(kk)!r}->{fb.get(kk)!r}" for kk in diff[:12])
        )
    return 0 if (ops_a == ops_b and weights_ok and io_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
