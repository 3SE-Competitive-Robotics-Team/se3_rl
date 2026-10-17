"""趴地移动任务的课程：导轮承重达标前不推进速度课程。

v2（W&B nnbqq8h0）的病理：第 100 轮时导轮承重还接近 0、承重罚顶满，速度课程已按跟踪 EMA 0.52 推到 vx 1.4；
姿态在 300–700 轮才形成，此时跟踪 EMA 已掉到 0.16–0.18，但退级关闭、课程锁在 1.5，策略在宽指令下学成
“姿态对、速度不管”，sim2x 里任何指令都以 −1.7 m/s 后退。跟踪 EMA 在 vx=0 阶段由 sigma_stand 打分，
姿态没学会时也能过 0.5，所以必须另加姿态判据。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.curriculums import commands_vel_adaptive

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

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


__all__ = ["commands_vel_after_crawl_posture"]
