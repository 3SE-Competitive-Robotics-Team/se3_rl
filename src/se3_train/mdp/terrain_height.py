"""机体相对地面高度：把"逐射线净空"换成"frame 高度 − 地面高度估计"。

## 为什么需要这个模块

MJLab 的 `TerrainHeightSensor.heights` 是逐射线的净空 `frame_z − hit_z`，并且对两种
异常情况返回**看起来合法的哨兵值**（见 `mjlab/sensor/terrain_height_sensor.py`）：

- 射线起点落在几何体内部（backface，机身陷进地形）→ 净空被钳成 **0.0**
- 射线在 `max_distance` 内什么都没打到 → 返回 **max_distance**（本仓库是 2.0）

再叠加 `reduction="min"`，一条 0 读数就能吃掉整圈射线。这两个值都不是 NaN 也不是 inf，
`torch.nan_to_num` 完全拦不住，下游没法从数值本身判断这一帧是不是废的。

2026-09-06 的 R1 崎岖地形训练就是这么崩的：机身陷进 heightfield → 高度读数 0.0 →
`flat_base_height` 的无界二次罚 `(0 − 0.30)²/0.05² × 4 = 144/s`（正常总奖励约 −10/s）→
critic 回归目标进万级（`Loss/value` 3 万）→ 优势函数失真 → 策略在 1000 轮内被摧毁。
平地上这两种退化都不可能发生，所以 Flat 基线一直没暴露它。

## 做法（对齐 scutrobotlab/wheeled-legged_RL）

参考仓库 `wheelbipe25_v3/env.py:5669` 的 `_update_ground_height_estimate()`：

1. 按 `isfinite` 掩掉无效射线，只用有效命中求**均值**（而不是 min，min 会被单条坏读数支配）；
2. 一个 env 的射线全无效时，退回 `terrain.env_origins[:, 2]`——该地形块的原点高度，
   一个已知正确的常数，而不是哨兵值；
3. 机身高度取 `root_z − ground_z_est`，机身自身的位姿永远是有限实数，
   "陷进地形"不再能污染高度本身。

本模块把这三条落到 MJLab 的传感器数据上。MJLab 没有 inf 可筛，有效性判据改为
`distances >= 0`（打中了）且 `normal_z > 0`（打中的是上表面而不是背面）。

平地上本模块与旧口径**逐位等价**：整圈射线都命中 z=0 的地面，均值等于最小值，
`frame_z − 0` 就是旧的净空。因此冻结的 Flat 基线不受影响，
由 `tests/test_terrain_height.py` 守护。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["frame_height_above_terrain", "ground_height_estimate"]


def _ray_view(sensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把传感器的原始射线数据摊平成 [B, F*N] 的 (hit_z, distances, normal_z)。"""
    data = sensor.data
    distances = data.distances
    batch = distances.shape[0]
    return (
        data.hit_pos_w[..., 2].reshape(batch, -1),
        distances.reshape(batch, -1),
        data.normals_w[..., 2].reshape(batch, -1),
    )


def ground_height_estimate(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """由地形高度传感器估计每个 env 脚下的地面世界高度，形状 [B]。

    只取"打中且命中上表面"的射线求均值；全部无效时退回该 env 的地形原点高度。
    """
    sensor = env.scene[sensor_name]
    hit_z, distances, normal_z = _ray_view(sensor)

    valid = (distances >= 0.0) & (normal_z > 0.0) & torch.isfinite(hit_z)
    count = valid.sum(dim=1)
    total = torch.where(valid, hit_z, torch.zeros_like(hit_z)).sum(dim=1)
    mean_z = total / count.clamp(min=1).to(hit_z.dtype)

    fallback = env.scene.env_origins[:, 2]
    return torch.where(count > 0, mean_z, fallback)


def frame_height_above_terrain(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    frame_index: int = 0,
) -> torch.Tensor:
    """传感器所在 frame 相对地面的高度，形状 [B]。

    等价于旧的 `sensor.data.heights[:, frame_index]`，但地面高度来自
    `ground_height_estimate()` 而不是逐射线净空，因此不会被 backface 的 0
    或射线打空的 max_distance 污染。base_link 上的传感器（`base_height_sensor` /
    `critic_height_sensor`）取到的就是参考仓库口径的 `root_z − ground_z_est`。
    """
    sensor = env.scene[sensor_name]
    frame_z = sensor.data.frame_pos_w[:, frame_index, 2]
    return frame_z - ground_height_estimate(env, sensor_name)
