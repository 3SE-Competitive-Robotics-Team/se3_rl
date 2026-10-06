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

2026-10-04（用户定）：MLP 入口 actor 隐藏层默认 512/256/128 → 128/64/32（ROUGH_ACTOR_HIDDEN_DIMS），
critic 不变。依据是 Actor128 对照（nulltask1 `4am48uzq` 五卡 × 8192、`qtpyzxtf` 六卡 × 1365）。
GRU 入口仍用 Flat 的 GRU 配置。同日删掉推理 100 Hz 对照（Exp-Dec2 / Exp-Dec2-Steps48），维持 50 Hz，
结论见 docs/plan/rough_dec2_100hz_20261003.md。

2026-10-05（用户定）：显式 vx 观测器入口 `vx_observer_rl_cfg`（见 se3_train.vx_observer）。actor 换成
VxObserverMLPModel（policy MLP 仍为 ROUGH_ACTOR_HIDDEN_DIMS，估计器 ROUGH_VX_ESTIMATOR_HIDDEN_DIMS），估计器用独立 Adam
（ROUGH_VX_ESTIMATOR_LEARNING_RATE）做 vx 的 MSE 回归；PPO 超参数、critic 与 `rl_cfg` 逐项相同。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace

from mjlab.rl import RslRlModelCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg, Se3PpoAlgorithmCfg
from se3_train.tasks.flat.rl_cfg import FLAT_NUM_STEPS_PER_ENV, mlp_rl_cfg
from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_gru_rl_cfg

# 地形课程要爬 10 级难度，比平地的 3500 轮长。沿用本仓库非 Flat 线的 5000 轮惯例。
ROUGH_MAX_ITERATIONS = 5000
# 2026-10-04 用户定：actor 隐藏层默认 128/64/32（部署跑的是 actor），critic 沿用 Flat 的 512/256/128。
ROUGH_ACTOR_HIDDEN_DIMS = (128, 64, 32)
# 显式 vx 观测器（2026-10-05 用户定）：估计器隐藏层与独立 Adam 学习率。lr 1e-3 是 HIMLoco / Ji et al. 估计器的常用值；
# 估计器是监督回归，不受 PPO 的 KL 自适应学习率约束。
ROUGH_VX_ESTIMATOR_HIDDEN_DIMS = (128, 64)
ROUGH_VX_ESTIMATOR_LEARNING_RATE = 1.0e-3


@dataclass
class VxObserverModelCfg(RslRlModelCfg):
    """VxObserverMLPModel 的网络配置：在 RslRlModelCfg 之上加历史帧数、单帧各项宽度与估计器隐藏层。"""

    history_length: int = 0
    frame_term_dims: tuple[int, ...] = ()
    estimator_hidden_dims: tuple[int, ...] = ROUGH_VX_ESTIMATOR_HIDDEN_DIMS
    latent_dim: int = 0
    class_name: str = "se3_train.vx_observer:VxObserverMLPModel"


def _is_smoke(smoke: bool) -> bool:
    return smoke or os.environ.get("SE3_SMOKE", "0") == "1"


def _use_se3_ppo(cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """算法类换成 se3_train.ppo.Se3PPO（不设 critic 固定 LR，更新与 rsl_rl.PPO 逐位相同），多卡时记录梯度噪声尺度。"""
    fields = {k: v for k, v in asdict(cfg.algorithm).items() if k != "class_name"}
    cfg.algorithm = Se3PpoAlgorithmCfg(**fields, critic_learning_rate=None)
    return cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置：PPO 超参数与 Flat 基线逐项相同，actor 隐藏层为 ROUGH_ACTOR_HIDDEN_DIMS。"""
    cfg = _use_se3_ppo(mlp_rl_cfg(smoke=smoke))
    cfg.actor = replace(cfg.actor, hidden_dims=ROUGH_ACTOR_HIDDEN_DIMS)
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


def vx_observer_rl_cfg(
    *,
    history_length: int,
    frame_term_dims: tuple[int, ...],
    target_group: str,
    latent_dim: int = 0,
    smoke: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """生成显式 vx 观测器 PPO 配置：只换 actor 类并打开估计器监督，其余与 `rl_cfg` 逐项相同。

    frame_term_dims 必须按 actor 观测组的 term 顺序给出单帧宽度（由入口从 env_cfg 推出，见 rough.__init__）。
    latent_dim：估计器额外输出的隐向量维数（PPO 端到端训练，见 se3_train.vx_observer），0 = 只估 vx。
    """
    cfg = rl_cfg(smoke=smoke)
    actor_fields = {k: v for k, v in asdict(cfg.actor).items() if k != "class_name"}
    cfg.actor = VxObserverModelCfg(
        **actor_fields,
        history_length=int(history_length),
        frame_term_dims=tuple(int(d) for d in frame_term_dims),
        latent_dim=int(latent_dim),
    )
    cfg.algorithm = replace(
        cfg.algorithm,
        estimator_learning_rate=ROUGH_VX_ESTIMATOR_LEARNING_RATE,
        estimator_target_group=target_group,
    )
    return cfg


def gru_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置：只换网络，rollout 仍 24 步，其余与 `rl_cfg` 逐项相同（M16）。"""
    cfg = _use_se3_ppo(flat_gru_rl_cfg(smoke=smoke))
    cfg.num_steps_per_env = FLAT_NUM_STEPS_PER_ENV
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


__all__ = [
    "ROUGH_ACTOR_HIDDEN_DIMS",
    "ROUGH_MAX_ITERATIONS",
    "ROUGH_VX_ESTIMATOR_HIDDEN_DIMS",
    "ROUGH_VX_ESTIMATOR_LEARNING_RATE",
    "VxObserverModelCfg",
    "gru_rl_cfg",
    "rl_cfg",
    "vx_observer_rl_cfg",
]
