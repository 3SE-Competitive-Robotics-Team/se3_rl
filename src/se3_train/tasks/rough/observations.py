"""本任务使用的 actor/critic 观测函数：平地那套 + critic 专用的地形高度扫描。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.observations import height_scan as _mjlab_height_scan

from se3_train.mdp.observations import (
    _finite_clamp,
    base_ang_vel_obs,
    base_height_obs,
    base_lin_vel_obs,
    commands_obs,
    jump_commands_obs,
    last_actions_obs,
    leg_joint_pos_obs,
    leg_joint_vel_obs,
    projected_gravity_obs,
    wheel_contact_force_obs,
    wheel_pos_obs,
    wheel_vel_obs,
)
from se3_train.mdp.terrain_height import frame_height_above_terrain

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_SELF_HIT_FOOTPRINT_M = (0.30, 0.25)
_SELF_HIT_MIN_RISE_M = 0.03


def _scan_footprint_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """网格中落在机器人自身足迹（机身系 |x|≤0.30、|y|≤0.25）内的射线掩码 [N]，按 env 缓存。"""
    cache_attr = f"_se3_scan_footprint_mask_{sensor_name}"
    mask = getattr(env, cache_attr, None)
    if mask is None:
        sensor = env.scene[sensor_name]
        offsets, _ = sensor.cfg.pattern.generate_rays(None, device=env.device)
        fx, fy = _SELF_HIT_FOOTPRINT_M
        mask = (offsets[:, 0].abs() <= fx) & (offsets[:, 1].abs() <= fy)
        setattr(env, cache_attr, mask)
    return mask


def height_scan_obs(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    reference_sensor_name: str = "critic_height_sensor",
    clip: float = 1.0,
) -> torch.Tensor:
    """critic 特权观测：机身周围网格各点的地面相对脚下地面的抬升(m)，形状 [B, N]。

    做法照 yly-true/fudan_rl_wheel_leg（legged_gym 配方）：机身系、yaw 对齐的网格向下打射线，
    只进 critic。那边的值是 `base_z − 0.5 − 地面高`（相对固定名义高度的净空）；这里改成
    `脚下地面高 − 各点地面高` 的相反数，即各点相对脚下的抬升：正值是前方台阶/上坡，负值是坑/下坡，
    与机身当前高度无关（本机器人高度指令 0.20–0.38 m 会变，净空口径会把高度指令混进地形信息）。

    脚下地面高取 `reference_sensor_name` 的稳健口径（有效射线均值，见 mdp/terrain_height.py）。
    两类读数要清洗：
    - 打空（distance < 0）：本地形上只在射线正好擦着台阶立面时出现，按"未知"记 0，
      不能按悬崖记 −clip。
    - 打到机器人自己（腿、轮、连杆与地形同为 geom group 0，射线只排除 base_link 本体）：
      落在自身足迹内且抬升超过 3 cm 的读数视为自击，记 0。代价是机身正下方真有台阶时看不到，
      但那时 critic 另有 base_height 与轮接触力。
    """
    sensor = env.scene[sensor_name]
    heights = _mjlab_height_scan(env, sensor_name)  # [B, N]：射线起点 z − 命中点 z
    under = frame_height_above_terrain(env, reference_sensor_name).unsqueeze(-1)  # [B, 1]
    rise = under - heights
    miss = sensor.data.distances.view(rise.shape) < 0
    self_hit = _scan_footprint_mask(env, sensor_name).unsqueeze(0) & (rise > _SELF_HIT_MIN_RISE_M)
    rise = torch.where(miss | self_hit, torch.zeros_like(rise), rise)
    return _finite_clamp(torch.clamp(rise, min=-float(clip), max=float(clip)))


__all__ = [
    "base_ang_vel_obs",
    "base_height_obs",
    "base_lin_vel_obs",
    "commands_obs",
    "height_scan_obs",
    "jump_commands_obs",
    "last_actions_obs",
    "leg_joint_pos_obs",
    "leg_joint_vel_obs",
    "projected_gravity_obs",
    "wheel_contact_force_obs",
    "wheel_pos_obs",
    "wheel_vel_obs",
]
