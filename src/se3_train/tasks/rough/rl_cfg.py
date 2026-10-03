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
from dataclasses import asdict, replace

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg, Se3PpoAlgorithmCfg
from se3_train.tasks.flat.rl_cfg import FLAT_NUM_STEPS_PER_ENV, mlp_rl_cfg
from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_gru_rl_cfg

# 地形课程要爬 10 级难度，比平地的 3500 轮长。沿用本仓库非 Flat 线的 5000 轮惯例。
ROUGH_MAX_ITERATIONS = 5000
# 2026-10-03 用户定的对照：actor 隐藏层 512/256/128 → 128/64/32，critic 不变（部署跑的是 actor，单变量看小网络够不够）。
ROUGH_SMALL_ACTOR_HIDDEN_DIMS = (128, 64, 32)


def _is_smoke(smoke: bool) -> bool:
    return smoke or os.environ.get("SE3_SMOKE", "0") == "1"


def _use_se3_ppo(cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """算法类换成 se3_train.ppo.Se3PPO（不设 critic 固定 LR，更新与 rsl_rl.PPO 逐位相同），多卡时记录梯度噪声尺度。"""
    fields = {k: v for k, v in asdict(cfg.algorithm).items() if k != "class_name"}
    cfg.algorithm = Se3PpoAlgorithmCfg(**fields, critic_learning_rate=None)
    return cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置（超参数与 Flat 基线逐项相同）。"""
    cfg = _use_se3_ppo(mlp_rl_cfg(smoke=smoke))
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


def small_actor_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """`rl_cfg` 的 actor 隐藏层换成 ROUGH_SMALL_ACTOR_HIDDEN_DIMS，其余（含 critic）逐项相同。临时对照入口用。"""
    cfg = rl_cfg(smoke=smoke)
    cfg.actor = replace(cfg.actor, hidden_dims=ROUGH_SMALL_ACTOR_HIDDEN_DIMS)
    return cfg


ROUGH_LONG_ROLLOUT_STEPS = 48
"""2026-10-03 推理 100 Hz 对照：每轮 48 步，使每轮仍覆盖 0.48 s（同复旦 wheel_legged_gym 100 Hz 配 48 步）。"""


def long_rollout_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """`rl_cfg` 只把每轮步数换成 ROUGH_LONG_ROLLOUT_STEPS，其余逐项相同。临时对照入口用。"""
    cfg = rl_cfg(smoke=smoke)
    cfg.num_steps_per_env = ROUGH_LONG_ROLLOUT_STEPS
    return cfg


def gru_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置：只换网络，rollout 仍 24 步，其余与 `rl_cfg` 逐项相同（M16）。"""
    cfg = _use_se3_ppo(flat_gru_rl_cfg(smoke=smoke))
    cfg.num_steps_per_env = FLAT_NUM_STEPS_PER_ENV
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


__all__ = [
    "ROUGH_LONG_ROLLOUT_STEPS",
    "ROUGH_MAX_ITERATIONS",
    "ROUGH_SMALL_ACTOR_HIDDEN_DIMS",
    "gru_rl_cfg",
    "long_rollout_rl_cfg",
    "rl_cfg",
    "small_actor_rl_cfg",
]
