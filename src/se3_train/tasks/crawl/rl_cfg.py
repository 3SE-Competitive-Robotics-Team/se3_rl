"""趴地移动任务的 PPO 配置：Flat-MLP 基线（单帧 MLP、rollout 24 步、3500 轮），actor 换成 Rough 的小网络。

2026-10-17（v5，用户定）：actor 隐藏层 512/256/128 → 128/64/32（与 Rough 的 ROUGH_ACTOR_HIDDEN_DIMS 相同），
critic 不变。v4 的 actor 共 18.3 万参数（FP32 715 KB），超出 STM32H723 片内 Flash 留给权重的约 600 KB，
放外部 Flash 时 18 万 MACC 实测要约 42 ms、超过 20 ms 控制周期；小 actor 约 1.5 万参数（约 60 KB）。
"""

from __future__ import annotations

from dataclasses import replace

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg as flat_mlp_rl_cfg
from se3_train.tasks.rough.rl_cfg import ROUGH_ACTOR_HIDDEN_DIMS

CRAWL_ACTOR_HIDDEN_DIMS = ROUGH_ACTOR_HIDDEN_DIMS


def mlp_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Flat-MLP 基线的 PPO 配置，只把 actor 隐藏层换成 CRAWL_ACTOR_HIDDEN_DIMS。"""
    cfg = flat_mlp_rl_cfg(smoke=smoke)
    cfg.actor = replace(cfg.actor, hidden_dims=CRAWL_ACTOR_HIDDEN_DIMS)
    return cfg


__all__ = ["CRAWL_ACTOR_HIDDEN_DIMS", "mlp_rl_cfg"]
