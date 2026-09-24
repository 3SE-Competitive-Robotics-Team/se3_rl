"""崎岖地形任务的 PPO 配置：直接复用冻结的 Flat 基线，只改训练轮数。

2026-09-06 之前这里是 Flat 基线定型前拷贝出去的一份旧值（clip 0.167 / entropy 0.00516 /
epochs 7 / lr 6.5e-4 / 32 步），与平地已经不是同一套 PPO。移植 rough 时改为直接调用
`flat.mlp_rl_cfg()`，让 rough 与 Flat 基线共用一套超参数：这样 rough 相对 flat 的唯一变量
就是地形、地形课程和台阶状态机，不掺 PPO 差异。参考仓库同样是 rough 继承 flat 的 PPO 配置。

M16（2026-09-20 用户定）：增加 GRU 入口 `gru_rl_cfg`。网络换成 Flat 的 GRU 配置（单层 GRU 512 +
MLP 512/256/128，actor/critic 同构，见 flat.rl_cfg._model_cfg），PPO 超参数、轮数、保存间隔与
MLP 入口逐项相同。rollout 特意**不用** Flat GRU 的 64 步而保持 24 步：64 步会同时改每轮采样量
（2.7 倍）、按轮计数的平地热身/ramp（env_cfg.ROUGH_STEPS_PER_POLICY_ITER=24）和推力课程的时间轴，
就不再是单变量对照；rsl_rl 采集期 hidden state 跨 rollout 持续、只在 done 时清零，24 步只截断
BPTT 梯度、不截断推理时的记忆长度。假设与验收见 docs/plan/m16_gru24_20260920.md。
"""

from __future__ import annotations

import os

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg
from se3_train.tasks.flat.rl_cfg import FLAT_NUM_STEPS_PER_ENV, mlp_rl_cfg
from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_gru_rl_cfg

# 地形课程要爬 10 级难度，比平地的 3500 轮长。沿用本仓库非 Flat 线的 5000 轮惯例。
ROUGH_MAX_ITERATIONS = 5000


def _is_smoke(smoke: bool) -> bool:
    return smoke or os.environ.get("SE3_SMOKE", "0") == "1"


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置（超参数与 Flat 基线逐项相同）。"""
    cfg = mlp_rl_cfg(smoke=smoke)
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


def gru_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置：只换网络，rollout 仍 24 步，其余与 `rl_cfg` 逐项相同（M16）。"""
    cfg = flat_gru_rl_cfg(smoke=smoke)
    cfg.num_steps_per_env = FLAT_NUM_STEPS_PER_ENV
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


__all__ = ["ROUGH_MAX_ITERATIONS", "gru_rl_cfg", "rl_cfg"]
