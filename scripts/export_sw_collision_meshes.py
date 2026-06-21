"""从 SolidWorks 装配体导出用于碰撞体重建的原始 STL 组件。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExportTarget:
    name: str
    rel_path: str
    doc_type: int


# SolidWorks API: 1=part, 2=assembly。
TARGETS: tuple[ExportTarget, ...] = (
    ExportTarget("base_link", "base/车架5.2.SLDASM", 2),
    ExportTarget("lf0_Link", "Left_thigh/Left_thigh.SLDASM", 2),
    ExportTarget("rf0_Link", "Right_thigh/Right_thigh.SLDASM", 2),
    ExportTarget("lf1_Link", "Left_calf/Left_calf.SLDASM", 2),
    ExportTarget("rf1_Link", "Right_calf/Right_calf.SLDASM", 2),
    ExportTarget("l_wheel_Link", "Left_wheel/减速箱.SLDASM", 2),
    ExportTarget("r_wheel_Link", "Right_wheel/Mirror减速箱.SLDASM", 2),
    ExportTarget("l_drive_bar_Link", "腿组6.2/腿组6.2/加工件/【加工件】小腿传动杆.SLDPRT", 1),
    ExportTarget("r_drive_bar_Link", "腿组6.2/腿组6.2/加工件/Mirror【加工件】小腿传动杆.SLDPRT", 1),
    ExportTarget("l_coupler_Link", "腿组6.2/腿组6.2/加工件/【加工件】异形连杆.SLDPRT", 1),
    ExportTarget("r_coupler_Link", "腿组6.2/腿组6.2/加工件/Mirror【加工件】异形连杆.SLDPRT", 1),
)


def _connect_solidworks() -> Any:
    try:
        import pythoncom  # noqa: F401
        import win32com.client
    except ImportError as exc:
        raise SystemExit(
            "缺少 pywin32；请用 `uv run --with pywin32 python scripts/export_sw_collision_meshes.py ...` 运行。"
        ) from exc

    try:
        return win32com.client.GetActiveObject("SldWorks.Application.32")
    except Exception:
        return win32com.client.Dispatch("SldWorks.Application.32")


def _byref_i32(value: int = 0) -> Any:
    import pythoncom
    from win32com.client import VARIANT

    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, value)


def _null_dispatch() -> Any:
    import pythoncom
    from win32com.client import VARIANT

    return VARIANT(pythoncom.VT_DISPATCH, None)


def _safe_clean_dir(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise RuntimeError(f"拒绝清理输出根目录之外的路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _export_target(
    sw: Any, source_root: Path, output_root: Path, target: ExportTarget
) -> dict[str, Any]:
    source_path = source_root / Path(target.rel_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    target_dir = output_root / target.name
    _safe_clean_dir(target_dir, output_root)

    errors = _byref_i32()
    warnings = _byref_i32()
    doc = sw.OpenDoc6(str(source_path), target.doc_type, 3, "", errors, warnings)
    if doc is None:
        raise RuntimeError(
            f"SolidWorks 打开失败: {source_path} errors={errors.value} warnings={warnings.value}"
        )

    save_errors = _byref_i32()
    save_warnings = _byref_i32()
    output_path = target_dir / f"{target.name}.STL"
    try:
        ok = doc.Extension.SaveAs(
            str(output_path), 0, 1, _null_dispatch(), save_errors, save_warnings
        )
        if not ok:
            raise RuntimeError(
                f"SolidWorks 导出失败: {source_path} errors={save_errors.value} warnings={save_warnings.value}"
            )
        exported = sorted(path.name for path in target_dir.glob("*.STL"))
        if not exported:
            exported = sorted(path.name for path in target_dir.glob("*.stl"))
        if not exported:
            raise RuntimeError(f"SolidWorks 未写出 STL: {source_path}")
        return {
            "name": target.name,
            "source": str(source_path),
            "open_errors": int(errors.value),
            "open_warnings": int(warnings.value),
            "save_errors": int(save_errors.value),
            "save_warnings": int(save_warnings.value),
            "files": exported,
        }
    finally:
        sw.CloseDoc(doc.GetTitle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="SolidWorks 总装目录，例如 D:/robomaster/图/底盘总装/底盘总装",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".tmp/sw_collision_raw"),
        help="原始导出 STL 目录；每个目标会单独清空重建。",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=[target.name for target in TARGETS],
        help="只导出指定目标；可重复传入。不传则导出全部。",
    )
    parser.add_argument(
        "--visible", action="store_true", help="显示 SolidWorks 主窗口，便于人工观察。"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.target or [])
    targets = [target for target in TARGETS if not selected or target.name in selected]

    sw = _connect_solidworks()
    sw.Visible = bool(args.visible)

    manifest = [_export_target(sw, source_root, output_root, target) for target in targets]
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"导出完成: {manifest_path}")
    for item in manifest:
        print(f"  {item['name']}: {len(item['files'])} files")


if __name__ == "__main__":
    main()
