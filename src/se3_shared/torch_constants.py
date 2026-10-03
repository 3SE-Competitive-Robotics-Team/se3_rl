"""设备常量与索引张量缓存。

热路径里每次调用都写 `torch.as_tensor(常数, device=cuda)` 或用 Python 元组 / 列表做高级索引，
都会触发一次主机到设备的同步拷贝：CPU 必须等 GPU 把之前排队的物理与奖励 kernel 跑完才能继续，
rough 每个 env.step 因此有两百多次同步，GPU 三分之二时间空等（2026-10-03 whtws 实测，
见 docs/plan/rough_iteration_time_20261003.md）。这里按 (值, 设备, dtype) 只上传一次，之后复用同一个张量，数值逐位不变。

返回的张量是共享只读常量，调用方不得原地修改。
"""

from __future__ import annotations

from collections.abc import Hashable

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

_CACHE: dict[tuple[Hashable, str, object], torch.Tensor] = {}


def device_constant(
    value: Hashable, *, device: torch.device | str, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """返回 `torch.as_tensor(value, device=device, dtype=dtype)` 的缓存版本。"""
    key = (value, str(device), dtype)
    cached = _CACHE.get(key)
    if cached is None:
        # 在 inference_mode 里首次创建会得到 inference tensor，之后在普通模式下复用会受限，所以显式关掉。
        with torch.inference_mode(False):
            cached = torch.as_tensor(value, device=device, dtype=dtype)
        _CACHE[key] = cached
    return cached


def device_index(ids: tuple[int, ...] | list[int], *, device: torch.device | str) -> torch.Tensor:
    """返回整数索引序列在 device 上的 long 张量缓存，用于替代 `x[:, (0, 2)]` 这类元组 / 列表索引。"""
    return device_constant(tuple(int(i) for i in ids), device=device, dtype=torch.long)


__all__ = ["device_constant", "device_index"]
