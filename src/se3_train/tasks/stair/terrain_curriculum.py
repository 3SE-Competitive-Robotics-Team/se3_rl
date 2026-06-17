"""台阶地形 level 的分桶采样工具。"""

from __future__ import annotations

import torch

DEFAULT_LEVEL_MAX_STAGES: tuple[tuple[int, int], ...] = (
    (0, 2),
    (250, 4),
    (500, 6),
    (780, 9),
)
DEFAULT_LEVEL_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (3, 6),
    (7, 9),
)
DEFAULT_BUCKET_WEIGHT_STAGES: tuple[tuple[int, tuple[float, float, float]], ...] = (
    (0, (0.75, 0.25, 0.00)),
    (250, (0.50, 0.40, 0.10)),
    (500, (0.30, 0.45, 0.25)),
    (750, (0.18, 0.32, 0.50)),
    (800, (0.15, 0.30, 0.55)),
)


def max_level_for_iteration(
    terrain,
    iteration: int,
    max_level_stages: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_MAX_STAGES,
) -> int:
    """根据训练迭代数查询当前可采样的最高地形 row。"""
    terrain_rows = int(terrain.terrain_origins.shape[0])
    target = 0
    for stage_iteration, stage_level in sorted(max_level_stages, key=lambda item: int(item[0])):
        if iteration >= int(stage_iteration):
            target = int(stage_level)
        else:
            break
    return max(0, min(terrain_rows - 1, target))


def bucket_weights_for_iteration(
    iteration: int,
    bucket_weight_stages: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = DEFAULT_BUCKET_WEIGHT_STAGES,
) -> tuple[float, ...]:
    """根据训练迭代数查询低/中/高 bucket 权重。"""
    weights: tuple[float, ...] = (1.0,)
    for stage_iteration, stage_weights in sorted(
        bucket_weight_stages,
        key=lambda item: int(item[0]),
    ):
        if iteration >= int(stage_iteration):
            weights = tuple(float(value) for value in stage_weights)
        else:
            break
    return weights


def sample_levels(
    env,
    terrain,
    count: int,
    iteration: int,
    max_level_stages: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_MAX_STAGES,
    level_buckets: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_BUCKETS,
    bucket_weight_stages: tuple[
        tuple[int, tuple[float, ...]],
        ...,
    ] = DEFAULT_BUCKET_WEIGHT_STAGES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """按 bucket 分布采样 terrain level，并返回诊断统计。"""
    count = max(0, int(count))
    device = env.device
    if count == 0:
        empty = torch.zeros(0, device=device, dtype=torch.long)
        return empty, _logs(device, 0, 0, 0.0, 0.0, 0.0)

    max_level = max_level_for_iteration(terrain, iteration, max_level_stages)
    bucket_weights = torch.tensor(
        bucket_weights_for_iteration(iteration, bucket_weight_stages),
        device=device,
        dtype=torch.float32,
    )
    if bucket_weights.numel() != len(level_buckets):
        raise ValueError("bucket 权重数量必须和 level bucket 数量一致")

    valid_ranges: list[tuple[int, int] | None] = []
    valid_weights = bucket_weights.clone()
    for index, (low_raw, high_raw) in enumerate(level_buckets):
        low = max(0, int(low_raw))
        high = min(int(high_raw), max_level)
        if low > high:
            valid_ranges.append(None)
            valid_weights[index] = 0.0
        else:
            valid_ranges.append((low, high))
    if float(valid_weights.sum().item()) <= 0.0:
        valid_ranges = [(0, max_level)]
        valid_weights = torch.ones(1, device=device, dtype=torch.float32)

    probs = valid_weights / torch.clamp(valid_weights.sum(), min=1.0e-6)
    bucket_ids = torch.multinomial(probs, count, replacement=True)
    levels = torch.zeros(count, device=device, dtype=torch.long)
    for bucket_id, range_value in enumerate(valid_ranges):
        if range_value is None:
            continue
        selected = bucket_ids == bucket_id
        if not torch.any(selected):
            continue
        low, high = range_value
        levels[selected] = torch.randint(
            low,
            high + 1,
            (int(selected.sum().item()),),
            device=device,
            dtype=torch.long,
        )

    low_rate = _bucket_rate(levels, level_buckets[0]) if len(level_buckets) > 0 else 0.0
    mid_rate = _bucket_rate(levels, level_buckets[1]) if len(level_buckets) > 1 else 0.0
    high_rate = _bucket_rate(levels, level_buckets[2]) if len(level_buckets) > 2 else 0.0
    return levels, _logs(device, max_level, int(levels.max().item()), low_rate, mid_rate, high_rate)


def _bucket_rate(levels: torch.Tensor, level_range: tuple[int, int]) -> float:
    low, high = int(level_range[0]), int(level_range[1])
    return float(((levels >= low) & (levels <= high)).float().mean().item())


def _logs(
    device: torch.device | str,
    max_allowed_level: int,
    sampled_max_level: int,
    low_rate: float,
    mid_rate: float,
    high_rate: float,
) -> dict[str, torch.Tensor]:
    return {
        "max_allowed_level": torch.tensor(float(max_allowed_level), device=device),
        "sampled_max_level": torch.tensor(float(sampled_max_level), device=device),
        "bucket_low_rate": torch.tensor(float(low_rate), device=device),
        "bucket_mid_rate": torch.tensor(float(mid_rate), device=device),
        "bucket_high_rate": torch.tensor(float(high_rate), device=device),
    }
