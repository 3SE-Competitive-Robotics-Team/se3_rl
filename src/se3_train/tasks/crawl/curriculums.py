"""趴地移动任务的课程：导轮承重达标前不推进速度课程；斜坡版按沿 +x 的前进距离升降地形等级。

v2（W&B nnbqq8h0）的病理：第 100 轮时导轮承重还接近 0、承重罚顶满，速度课程已按跟踪 EMA 0.52 推到 vx 1.4；
姿态在 300–700 轮才形成，此时跟踪 EMA 已掉到 0.16–0.18，但退级关闭、课程锁在 1.5，策略在宽指令下学成
“姿态对、速度不管”，sim2x 里任何指令都以 −1.7 m/s 后退。跟踪 EMA 在 vx=0 阶段由 sigma_stand 打分，
姿态没学会时也能过 0.5，所以必须另加姿态判据。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp.curriculums import commands_vel_adaptive

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")

_GUIDE_SHARE_EMA_ATTR = "_crawl_guide_share_ema"


def commands_vel_after_crawl_posture(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    *,
    guide_share_threshold: float = 0.5,
    guide_ema_alpha: float = 0.05,
    guide_log_key: str = "Crawl/guide_share_total",
    **vel_kwargs,
) -> dict[str, torch.Tensor]:
    """导轮总承重占比的 EMA 达到 guide_share_threshold 之后，才允许 commands_vel_adaptive 推进。

    导轮承重由 rewards.guide_wheel_support 写进日志（与跟踪分同一节流），缺失的步跳过 EMA 更新，
    口径与 commands_vel_adaptive 处理跟踪分一致。门控关闭时把推进阈值临时设为 +inf：
    速度上限不推进、跟踪 EMA 照常更新；姿态退化时门控重新关上，只停推进、不回退。
    阈值 0.5 = guide_wheel_support 每侧 25% 的目标之和。
    """
    log = getattr(env, "extras", {}).get("log", {})
    share = log.get(guide_log_key, None)
    ema = float(getattr(env, _GUIDE_SHARE_EMA_ATTR, 0.0))
    if share is not None:
        ema = (1.0 - float(guide_ema_alpha)) * ema + float(guide_ema_alpha) * float(share)
        setattr(env, _GUIDE_SHARE_EMA_ATTR, ema)
    ready = ema >= float(guide_share_threshold)
    if not ready:
        vel_kwargs = {**vel_kwargs, "advance_threshold": float("inf")}
    out = commands_vel_adaptive(env, env_ids, command_name, **vel_kwargs)
    out["guide_share_ema"] = torch.tensor(ema)
    out["posture_ready"] = torch.tensor(float(ready))
    return out


def terrain_levels_forward(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    *,
    promote_distance: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
    """「平地-上坡-平台」地形的升降级：只看沿世界系 +x 的前进距离（坡沿 +x 上，出生朝向 yaw≈0）。

    官方 terrain_levels_vel 按位移模长升级，倒车开远也会升级；这里改成：
    - 升级：前进距离 > promote_distance（机身整个上到坡顶平台）；
    - 降级：没升级、前向指令 vx > 0、前进距离不到指令应走距离的一半（与官方同口径，只取前向分量）。
    后退或零速指令的 episode 不升不降。首次 reset（common_step_counter == 0）冻结等级，同官方。
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    command = env.command_manager.get_command(command_name)
    assert command is not None

    forward = asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    cmd_vx = command[env_ids, 0]
    move_up = forward > float(promote_distance)
    move_down = (cmd_vx > 0.0) & (forward < cmd_vx * env.max_episode_length_s * 0.5) & ~move_up
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    return {"mean": torch.mean(levels), "max": torch.max(levels)}


__all__ = ["commands_vel_after_crawl_posture", "terrain_levels_forward"]
