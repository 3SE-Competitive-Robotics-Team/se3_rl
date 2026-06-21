"""把 SolidWorks 导出的 base_link.STL 按面片数切成 MuJoCo 可加载的小块。

该脚本只按二进制 STL 的 50 字节三角面记录切片，不做抽稀、不重算法线、
不合并顶点，几何面片保持原始导出结果。
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

STL_HEADER_SIZE = 80
STL_COUNT_SIZE = 4
STL_TRIANGLE_RECORD_SIZE = 50
DEFAULT_MAX_FACES = 180_000


def _read_binary_stl(path: Path) -> tuple[bytes, int]:
    data = path.read_bytes()
    if len(data) < STL_HEADER_SIZE + STL_COUNT_SIZE:
        raise SystemExit(f"{path} 不是有效的二进制 STL：文件太小")
    triangle_count = struct.unpack_from("<I", data, STL_HEADER_SIZE)[0]
    expected_size = STL_HEADER_SIZE + STL_COUNT_SIZE + triangle_count * STL_TRIANGLE_RECORD_SIZE
    if len(data) != expected_size:
        raise SystemExit(
            f"{path} 不是标准二进制 STL 或文件不完整："
            f"声明 {triangle_count} 面，期望 {expected_size} 字节，实际 {len(data)} 字节"
        )
    return data, triangle_count


def _write_chunk(path: Path, source_header: bytes, records: bytes, triangle_count: int) -> None:
    header = source_header[:STL_HEADER_SIZE]
    label = f"split no simplify faces={triangle_count}".encode("ascii")
    header = (label + b" " * STL_HEADER_SIZE)[:STL_HEADER_SIZE]
    path.write_bytes(header + struct.pack("<I", triangle_count) + records)


def split_stl(source: Path, output_dir: Path, max_faces: int) -> list[dict[str, object]]:
    data, triangle_count = _read_binary_stl(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("base_link_chunk_*.stl"):
        old_file.unlink()

    records_offset = STL_HEADER_SIZE + STL_COUNT_SIZE
    chunks: list[dict[str, object]] = []
    for chunk_index, start_face in enumerate(range(0, triangle_count, max_faces)):
        end_face = min(start_face + max_faces, triangle_count)
        byte_start = records_offset + start_face * STL_TRIANGLE_RECORD_SIZE
        byte_end = records_offset + end_face * STL_TRIANGLE_RECORD_SIZE
        chunk_name = f"base_link_chunk_{chunk_index:02d}.stl"
        chunk_path = output_dir / chunk_name
        _write_chunk(
            chunk_path,
            data[:STL_HEADER_SIZE],
            data[byte_start:byte_end],
            end_face - start_face,
        )
        chunks.append(
            {
                "file": chunk_name,
                "start_face": start_face,
                "end_face": end_face,
                "faces": end_face - start_face,
            }
        )

    manifest = {
        "source": str(source),
        "source_faces": triangle_count,
        "max_faces_per_chunk": max_faces,
        "chunks": chunks,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\robomaster\sw_urdf_work\serialleg\meshes\base_link.STL"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/robots/serialleg/meshes/sw_original_base"),
    )
    parser.add_argument("--max-faces", type=int, default=DEFAULT_MAX_FACES)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    chunks = split_stl(args.source.resolve(), args.output_dir.resolve(), int(args.max_faces))
    total_faces = sum(int(chunk["faces"]) for chunk in chunks)
    print(f"输出目录: {args.output_dir}")
    print(f"切块数量: {len(chunks)}")
    print(f"总面数: {total_faces}")
    for chunk in chunks:
        print(f"  {chunk['file']}: {chunk['faces']} faces")


if __name__ == "__main__":
    main()
