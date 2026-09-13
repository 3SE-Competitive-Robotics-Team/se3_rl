"""崎岖地形任务的 PPO 配置：直接复用冻结的 Flat 基线，只改训练轮数。

2026-09-06 之前这里是 Flat 基线定型前拷贝出去的一份旧值（clip 0.167 / entropy 0.00516 /
epochs 7 / lr 6.5e-4 / 32 步），与平地已经不是同一套 PPO。移植 rough 时改为直接调用
`flat.mlp_rl_cfg()`，让 rough 与 Flat 基线共用一套超参数：这样 rough 相对 flat 的唯一变量
就是地形、地形课程和台阶状态机，不掺 PPO 差异。参考仓库同样是 rough 继承 flat 的 PPO 配置。
"""

from __future__ import annotations

import os

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

# 地形课程要爬 10 级难度，比平地的 3500 轮长。沿用本仓库非 Flat 线的 5000 轮惯例。
ROUGH_MAX_ITERATIONS = 5000


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置（超参数与 Flat 基线逐项相同）。"""
    cfg = mlp_rl_cfg(smoke=smoke)
    if not (smoke or os.environ.get("SE3_SMOKE", "0") == "1"):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


__all__ = ["ROUGH_MAX_ITERATIONS", "rl_cfg"]
