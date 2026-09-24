"""AMP 示范数据集工厂，照 kyber_rl_lab 的 `utils/amp_dataset_factory.py` 移植。

流程与 kyber 一致：解析/校验字段 → MotionLoader 加载 → 校验加载后的字段顺序 → 校验 env 的 amp 观测组维数。
kyber 在 env 侧校验的是 observation term 的 joint/link 顺序；本仓库的 amp 观测组只有一项 19 维运动帧，
校验退化为"观测组维数 == 数据集维数"。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from se3_train.motion_loader import MotionLoader


def _resolve_and_validate_fields(fields: Sequence[str] | None) -> list[str]:
    resolved = MotionLoader.supported_fields() if fields is None else [str(field).strip() for field in fields]
    if not resolved:
        raise ValueError("AMP fields must not be empty.")
    seen: set[str] = set()
    duplicates = sorted({name for name in resolved if name in seen or seen.add(name)})  # type: ignore[func-returns-value]
    if duplicates:
        raise ValueError(f"AMP fields contain duplicates: {duplicates}")
    supported = set(MotionLoader.supported_fields())
    invalid = sorted(name for name in resolved if name not in supported)
    if invalid:
        raise ValueError(f"Unsupported AMP fields: {invalid}. Supported: {MotionLoader.supported_fields()}")
    return resolved


def _validate_env_dataset_layout(env: Any, obs_group: str, obs_dim: int) -> None:
    """env 的 amp 观测组维数必须等于数据集单帧维数。"""
    env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
    observation_manager = getattr(env_obj, "observation_manager", None)
    if observation_manager is None:
        return
    try:
        group_dim = int(observation_manager.compute()[obs_group].shape[-1])
    except (KeyError, AttributeError, TypeError):
        return
    if group_dim != obs_dim:
        raise ValueError(f"AMP obs dim mismatch: env '{obs_group}' is {group_dim}, dataset is {obs_dim}.")


def build_amp_dataset(
    env: Any | None = None,
    device: str = "cpu",
    *,
    dataset_root: str,
    dataset_glob: str = "*.pkl",
    mirror_augmentation: bool = True,
    obs_group: str = "amp",
    fields: Sequence[str] | None = None,
    simulation_dt: float | None = None,
) -> dict[str, Any]:
    """Build AMP dataset from se3.amp.pkl.v1 motion files.

    Args:
        env: Vectorized environment used by the runner（可为 None，此时必须给 simulation_dt）.
        device: Torch device string.
        dataset_root: Dataset root path（目录按 dataset_glob 匹配，或直接给单个 .pkl）.
        dataset_glob: Dataset file discovery glob.
        mirror_augmentation: Enable mirrored augmentation.
        obs_group: Observation group name used for AMP.
        fields: Ordered AMP fields to load from dataset（None = 契约全部 19 维）.
        simulation_dt: 训练控制周期；None 时从 env.step_dt 取.
    """
    if env is None and simulation_dt is None:
        raise ValueError("simulation_dt must be provided when building a dataset without an env object.")
    resolved_fields = _resolve_and_validate_fields(fields)
    if not dataset_root:
        raise ValueError("AMP dataset_root is not configured.")
    if simulation_dt is None:
        env_obj = env.unwrapped if hasattr(env, "unwrapped") else env
        simulation_dt = float(env_obj.step_dt)

    loader = MotionLoader(
        dataset_path_root=Path(dataset_root),
        simulation_dt=float(simulation_dt),
        dataset_glob=dataset_glob,
        device=device,
        fields=list(resolved_fields),
        mirror_augmentation=mirror_augmentation,
    )
    dataset = loader.get_dataset_dict()
    ds_fields = list(dataset.get("format", {}).get("fields", []))
    if ds_fields != list(resolved_fields):
        raise ValueError(
            "AMP dataset field order mismatch after loading.\n"
            f"requested fields: {list(resolved_fields)}\n"
            f"dataset fields: {ds_fields}"
        )
    if env is not None:
        _validate_env_dataset_layout(env=env, obs_group=str(obs_group), obs_dim=loader.obs_dim)
    print("[AMP] Dataset loaded")
    print(f"[AMP] files: {len(dataset['sequences'])}, obs_dim: {loader.obs_dim}")
    print(f"[AMP] format: {dataset['format']}")
    if not dataset["metadata"].get("retargeted_to_serialleg", False):
        print("[AMP] 提示：数据集标记 retargeted_to_serialleg=false，仍是源机器人运动")
    return dataset


__all__ = ["build_amp_dataset"]
