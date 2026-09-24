"""AMP 示范数据加载器，照 kyber_rl_lab 的 `utils/motion_loader.py` 结构移植。

kyber 的 MotionLoader 读的是 kyber_retarget 产出的关节/link 级 pkl（meta.available_fields、joint_order、
link_order、root_pos/root_rot/joint_pos……），按字段加载、重采样、镜像、按名称重排，最后给出统一的
`format` / `metadata` 与 `get_dataset_dict()`。本仓库的示范是 `se3.amp.pkl.v1`：payload 直接是 19 维运动帧
（契约 se3.amp.motion.v1，见 docs/amp_input.md），没有关节/link 层，所以这里保留同样的职责划分与接口，
把"字段"换成 19 维里的特征名子集，镜像规则按契约写死。

payload 键（scripts/export_fudan_amp_pkl.py）：
    format="se3.amp.pkl.v1", motion_contract="se3.amp.motion.v1", fps, dt, frame_dim=19, transition_dim=38,
    feature_names[19], sequences: list[np.ndarray[T, 19]], transitions: list[[T-1, 38]], lengths: list[int],
    time_s / source_time_s: list[[T]], annotations: list[dict], retargeted_to_serialleg: bool
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from se3_shared.amp import AMP_CONTROL_DT_S, AMP_FEATURE_NAMES, AMP_FRAME_DIM

DATASET_FORMAT = "se3.amp.pkl.v1"
MOTION_CONTRACT = "se3.amp.motion.v1"

# 关于机身 xz 平面的左右镜像：gravity_y / omega_x / omega_z / velocity_y 取反，左右轮各量互换。
_MIRROR_PERM = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 9, 10, 15, 16, 13, 14, 18, 17], dtype=np.int64)
_MIRROR_SIGN = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0] + [1.0] * 10, dtype=np.float32)


class MotionLoader:
    """加载 se3.amp.pkl.v1 示范数据集并给出 fork/kyber 约定的序列字典。"""

    CORE_FIELDS: tuple[str, ...] = tuple(AMP_FEATURE_NAMES)

    @staticmethod
    def supported_fields() -> list[str]:
        return list(MotionLoader.CORE_FIELDS)

    @staticmethod
    def _first_mismatch(lhs: list[str], rhs: list[str]) -> tuple[int, str, str] | None:
        if len(lhs) != len(rhs):
            return -1, f"len={len(lhs)}", f"len={len(rhs)}"
        for idx, (left_name, right_name) in enumerate(zip(lhs, rhs, strict=True)):
            if left_name != right_name:
                return idx, left_name, right_name
        return None

    @classmethod
    def _resolve_requested_fields(cls, fields: list[str] | tuple[str, ...] | None) -> list[str]:
        if fields is None:
            return list(cls.CORE_FIELDS)
        resolved = [str(field).strip() for field in fields]
        if not resolved:
            raise ValueError("AMP fields must not be empty.")
        supported = set(cls.CORE_FIELDS)
        for field in resolved:
            if field not in supported:
                raise ValueError(f"Unsupported motion field '{field}'. Supported fields: {cls.supported_fields()}")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"AMP fields contain duplicates: {resolved}")
        return resolved

    @staticmethod
    def _build_resample_times(
        num_frames: int, source_fps: float, simulation_dt: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if source_fps <= 0:
            raise ValueError(f"source_fps must be positive, got {source_fps}")
        src_times = np.arange(num_frames, dtype=np.float64) / float(source_fps)
        target_fps = 1.0 / float(simulation_dt)
        if np.isclose(source_fps, target_fps):
            return src_times, None
        duration = src_times[-1]
        dst_times = np.arange(0.0, duration + 1e-9, simulation_dt, dtype=np.float64)
        if dst_times[-1] < duration:
            dst_times = np.append(dst_times, duration)
        return src_times, dst_times

    @staticmethod
    def _resample_linear_if_needed(data: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray | None) -> np.ndarray:
        if dst_times is None:
            return data.copy()
        return np.stack([np.interp(dst_times, src_times, data[:, i]) for i in range(data.shape[1])], axis=1)

    @staticmethod
    def _mirror_frames(frames: np.ndarray) -> np.ndarray:
        return frames[:, _MIRROR_PERM] * _MIRROR_SIGN

    def __init__(
        self,
        dataset_path_root: str | Path,
        simulation_dt: float,
        dataset_glob: str = "*.pkl",
        device: str = "cpu",
        fields: list[str] | tuple[str, ...] | None = None,
        mirror_augmentation: bool = True,
    ) -> None:
        self.root = Path(dataset_path_root)
        self.simulation_dt = float(simulation_dt)
        self.dataset_glob = str(dataset_glob)
        self.device = torch.device(device)
        self.fields = self._resolve_requested_fields(fields)
        self.mirror_augmentation = bool(mirror_augmentation)

        if self.root.is_file():
            dataset_paths = [self.root]
        else:
            dataset_paths = sorted(self.root.glob(self.dataset_glob))
        if not dataset_paths:
            raise ValueError(f"No dataset file found in '{self.root}' with pattern '{self.dataset_glob}'.")

        self._build(dataset_paths)

    # ---- 单文件 ----
    def _load_dataset_file(self, path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Dataset file must contain a dict payload: {path}")
        return data

    def _extract_schema_metadata(self, path: Path, data: dict[str, Any]) -> tuple[float, list[str]]:
        fmt = str(data.get("format", ""))
        if fmt != DATASET_FORMAT:
            raise ValueError(f"Unsupported dataset format '{fmt}' in {path}; expected '{DATASET_FORMAT}'")
        contract = str(data.get("motion_contract", ""))
        if contract != MOTION_CONTRACT:
            raise ValueError(f"Unsupported motion contract '{contract}' in {path}; expected '{MOTION_CONTRACT}'")
        fps = float(data.get("fps", 0.0))
        if fps <= 0:
            raise ValueError(f"Invalid fps in dataset '{path}': {fps}")
        frame_dim = int(data.get("frame_dim", AMP_FRAME_DIM))
        if frame_dim != AMP_FRAME_DIM:
            raise ValueError(f"frame_dim must be {AMP_FRAME_DIM} in {path}, got {frame_dim}")
        feature_names = [str(name) for name in data.get("feature_names", [])]
        mismatch = self._first_mismatch(feature_names, list(self.CORE_FIELDS))
        if mismatch is not None:
            idx, lhs, rhs = mismatch
            raise ValueError(
                f"feature_names mismatch in {path} at index={idx}: dataset='{lhs}', contract='{rhs}'"
            )
        return fps, feature_names

    @staticmethod
    def _validate_sequences_present(path: Path, data: dict[str, Any]) -> list[np.ndarray]:
        sequences = data.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"Dataset '{path}' has no sequences")
        lengths = data.get("lengths")
        out: list[np.ndarray] = []
        for idx, seq in enumerate(sequences):
            arr = np.asarray(seq, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != AMP_FRAME_DIM:
                raise ValueError(f"sequence {idx} in {path} must be [T, {AMP_FRAME_DIM}], got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"sequence {idx} in {path} contains non-finite values")
            if isinstance(lengths, list) and idx < len(lengths) and int(lengths[idx]) != arr.shape[0]:
                raise ValueError(f"sequence {idx} in {path}: lengths[{idx}]={lengths[idx]} != {arr.shape[0]}")
            out.append(arr)
        return out

    def _field_indices(self) -> np.ndarray:
        index_by_name = {name: idx for idx, name in enumerate(self.CORE_FIELDS)}
        return np.asarray([index_by_name[name] for name in self.fields], dtype=np.int64)

    # ---- 全部文件 ----
    def _build(self, dataset_paths: list[Path]) -> None:
        sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        fps_values: list[float] = []
        file_names: list[str] = []
        annotations: list[dict[str, Any]] = []
        retargeted: list[bool] = []
        field_idx = self._field_indices()

        for path in dataset_paths:
            data = self._load_dataset_file(path)
            fps, _ = self._extract_schema_metadata(path, data)
            raw_sequences = self._validate_sequences_present(path, data)
            file_annotations = data.get("annotations") or [{}] * len(raw_sequences)
            retargeted.append(bool(data.get("retargeted_to_serialleg", False)))
            for seq_idx, frames in enumerate(raw_sequences):
                src_times, dst_times = self._build_resample_times(frames.shape[0], fps, self.simulation_dt)
                resampled = self._resample_linear_if_needed(frames, src_times, dst_times)
                if resampled.shape[0] < 2:
                    continue
                selected = resampled[:, field_idx]
                sequences.append(torch.tensor(selected, dtype=torch.float32, device=self.device))
                lengths.append(int(selected.shape[0]))
                fps_values.append(float(fps))
                file_names.append(f"{path.name}#{seq_idx}")
                annotation = file_annotations[seq_idx] if seq_idx < len(file_annotations) else {}
                annotations.append({k: v for k, v in dict(annotation).items() if not isinstance(v, np.ndarray)})
                if self.mirror_augmentation:
                    mirrored = self._mirror_frames(resampled.astype(np.float32))[:, field_idx]
                    sequences.append(torch.tensor(mirrored, dtype=torch.float32, device=self.device))
                    lengths.append(int(mirrored.shape[0]))
                    fps_values.append(float(fps))
                    file_names.append(f"{path.name}#{seq_idx}::mirror")
                    annotations.append(annotations[-1])

        if not sequences:
            raise ValueError("No valid motion sequence was loaded.")

        self.sequences = sequences
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.fps = torch.tensor(fps_values, dtype=torch.float32, device=self.device)

        feature_slices: dict[str, list[int]] = {}
        dims: list[str] = []
        for cursor, field in enumerate(self.fields):
            feature_slices[field] = [cursor, cursor + 1]
            dims.append(field)
        self.format = {"feature_slices": feature_slices, "dims": dims, "fields": list(self.fields)}
        self.metadata = {
            "files": file_names,
            "fields": list(self.fields),
            "contract": MOTION_CONTRACT,
            "dataset_format": DATASET_FORMAT,
            "control_dt_s": AMP_CONTROL_DT_S,
            "simulation_dt": self.simulation_dt,
            "mirror_augmentation": self.mirror_augmentation,
            "retargeted_to_serialleg": all(retargeted),
            "annotations": annotations,
        }

    @property
    def obs_dim(self) -> int:
        return int(self.sequences[0].shape[-1])

    def get_dataset_dict(self) -> dict[str, Any]:
        return {
            "sequences": self.sequences,
            "lengths": self.lengths,
            "fps": self.fps,
            "format": self.format,
            "metadata": self.metadata,
        }


__all__ = ["DATASET_FORMAT", "MOTION_CONTRACT", "MotionLoader"]
