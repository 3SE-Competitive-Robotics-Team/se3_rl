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
BPTT 梯度、不截断推理时的记忆长度。假设与验收见 m16_gru24_20260920。

2026-10-04（用户定）：MLP 入口 actor 隐藏层默认 512/256/128 → 128/64/32（ROUGH_ACTOR_HIDDEN_DIMS），
critic 不变。依据是 Actor128 对照（nulltask1 `4am48uzq` 五卡 × 8192、`qtpyzxtf` 六卡 × 1365）。
GRU 入口仍用 Flat 的 GRU 配置。同日删掉推理 100 Hz 对照（Exp-Dec2 / Exp-Dec2-Steps48），维持 50 Hz，
结论见 rough_dec2_100hz_20261003。

2026-10-07 起 CTS 入口 `cts_rl_cfg`（见 se3_train.cts_observer）；单独的显式 vx 观测器入口（VxObserver）2026-10-19 删除，
复现用 846c4fcd。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from mjlab.rl import RslRlModelCfg

from se3_train.rl_cfg import (
    ImitationLossCfg,
    PolicyDistillCfg,
    RslRlOnPolicyRunnerCfg,
    Se3PpoAlgorithmCfg,
)
from se3_train.tasks.flat.rl_cfg import FLAT_NUM_STEPS_PER_ENV, mlp_rl_cfg
from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_gru_rl_cfg

# 地形课程要爬 10 级难度，比平地的 3500 轮长。沿用本仓库非 Flat 线的 5000 轮惯例。
ROUGH_MAX_ITERATIONS = 5000
# 2026-10-19 用户定：Rough 系入口训练中每 1000 轮与训练结束时旁路跑 full benchmark（se3_tools.bench）。
ROUGH_BENCH_INTERVAL = 1000
# 2026-10-04 用户定：actor 隐藏层默认 128/64/32（部署跑的是 actor），critic 沿用 Flat 的 512/256/128。
ROUGH_ACTOR_HIDDEN_DIMS = (128, 64, 32)
# 学生编码器（沿用 estimator 名）独立 Adam 学习率。lr 1e-3 是 HIMLoco / Ji et al. 估计器的常用值；
# 估计器是监督回归，不受 PPO 的 KL 自适应学习率约束。
ROUGH_VX_ESTIMATOR_LEARNING_RATE = 1.0e-3


# CTS（2026-10-07 用户定）：隐向量维数与编码器隐藏层。教师编码器照 CTS 论文 512/256；学生编码器 256/128
# （480 → 256 → 128 → 33 约 16 万次乘加）。
ROUGH_CTS_LATENT_DIM = 32
ROUGH_CTS_TEACHER_HIDDEN_DIMS = (512, 256)
ROUGH_CTS_STUDENT_HIDDEN_DIMS = (256, 128)
# 单片机部署版学生编码器（2026-10-09 用户定）：256/128 的部署网络约 18 万参数、FP32 约 720 KB，
# 加固件代码约 146 KB 超出 STM32H723 片内 768 KiB 权重区；128/128 降到约 10.2 万参数、约 408 KB，片内放得下。
ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU = (128, 128)
# 2026-10-19 用户定：Student128 入口删掉「换编码器续训」用的 200 轮学生预热默认值，改为与 LowGains 一样从零训；
# 预热机制本身仍在 cts_rl_cfg(estimator_warmup_iterations=...) / se3_train.ppo，续训时按需显式传。
# 倒地教师的 PPO 模仿损失（2026-10-23 用户定，见 se3_train.ppo）：系数沿用 6 月教师-学生实验的 teacher_coef 0.2，
# 有效权重再乘 α（随教师一起退火到 0）；损失用 Huber（δ = 1 个动作单位）而不是 6 月的 MSE——kf3o74t7 前 200 轮
# 策略与教师平均差约 5 个动作单位，MSE 会让模仿项压过 PPO 损失。
ROUGH_FALL_TEACHER_IMITATION_LOSS_COEF = 0.2
ROUGH_FALL_TEACHER_IMITATION_LOSS_HUBER_DELTA = 1.0
# recover 组 AMP（2026-10-24 用户定，见 env_cfg.ROUGH_RECOVER_AMP_* 与 se3_train.amp）。参考数据 = gyi9exiz model_2200 在 recover 组
# 条件下的成功起身片段（出生 → 站稳满 1 s，scripts/record_recover_amp_dataset.py），判别器看全部 19 维（轮自转也算：参考来自同一仿真，
# 读数可靠，正好约束起身时乱甩轮子）、相邻 2 帧，左右镜像增广。
# 风格权重按 recover 组账本定：r ∈ [0, 1] 每秒最多 w 分；起身约 1–1.5 s + 站稳 1 s，满分风格约 2.5w = 10；成功 +20、失败 −20、
# 暴起 vs 温和的超限罚差约 10 分。w = 4 时风格与超限罚同量级、远小于成败差 40 分：8 s 躺着即便风格 0.3 分也只有约 10 分，
# 抵不过失败 −20，"摆烂"不会被风格奖励收买。
# 判别器每卡每轮 4 步、batch 4096（recover 组每卡约 1.4k env × 24 步 ≈ 3.4 万个窗口），学习率 / 梯度惩罚 / 热身沿用 fork 默认。
ROUGH_RECOVER_AMP_DATASET_ROOT = (
    Path(__file__).resolve().parents[4] / "assets" / "amp" / "recover_gyi9exiz_model2200"
)
ROUGH_RECOVER_AMP_REWARD_WEIGHT = 4.0
ROUGH_RECOVER_AMP_TRANSITION_FRAMES = 2
ROUGH_RECOVER_AMP_DISCRIMINATOR_BATCH = 4096
# recover 组网络老师在线蒸馏（2026-10-25 用户定，见 se3_train.ppo 的 policy_distill）：老师 = gyi9exiz model_2200（与 Recover-AMP
# 同结构同观测，部署路径前向），只在 recover 组样本上算，系数沿用倒地教师模仿损失的 0.2 / Huber δ 1，全程不退火。
# 老师权重与 AMP 数据集放在一起（assets/amp/ 不进 git，换机器要拷）。
ROUGH_RECOVER_DISTILL_TEACHER_CKPT = ROUGH_RECOVER_AMP_DATASET_ROOT / "gyi9exiz_model_2200.pt"


@dataclass
class CTSModelCfg(RslRlModelCfg):
    """CTSVxObserverModel 的网络配置（见 se3_train.cts_observer）。"""

    history_length: int = 0
    frame_term_dims: tuple[int, ...] = ()
    estimator_hidden_dims: tuple[int, ...] = ROUGH_CTS_STUDENT_HIDDEN_DIMS
    teacher_hidden_dims: tuple[int, ...] = ROUGH_CTS_TEACHER_HIDDEN_DIMS
    latent_dim: int = ROUGH_CTS_LATENT_DIM
    teacher_obs_group: str = "critic"
    vx_target_group: str = "estimator_target"
    role_group: str = "cts_role"
    student_latent_grad: str = "mse"
    class_name: str = "se3_train.cts_observer:CTSVxObserverModel"


def _is_smoke(smoke: bool) -> bool:
    return smoke or os.environ.get("SE3_SMOKE", "0") == "1"


def _use_se3_ppo(cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """算法类换成 se3_train.ppo.Se3PPO（不设 critic 固定 LR，更新与 rsl_rl.PPO 逐位相同），多卡时记录梯度噪声尺度。"""
    fields = {k: v for k, v in asdict(cfg.algorithm).items() if k != "class_name"}
    cfg.algorithm = Se3PpoAlgorithmCfg(**fields, critic_learning_rate=None)
    return cfg


def rl_cfg(
    smoke: bool = False, *, imitation_loss: ImitationLossCfg | None = None
) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置：PPO 超参数与 Flat 基线逐项相同，actor 隐藏层为 ROUGH_ACTOR_HIDDEN_DIMS。

    imitation_loss：PPO 总损失里的模仿项（倒地教师用，见 se3_train.ppo），None 不加。
    """
    cfg = _use_se3_ppo(mlp_rl_cfg(smoke=smoke))
    cfg.actor = replace(cfg.actor, hidden_dims=ROUGH_ACTOR_HIDDEN_DIMS)
    cfg.algorithm = replace(cfg.algorithm, imitation_loss=imitation_loss)
    cfg.bench_interval = ROUGH_BENCH_INTERVAL
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


def cts_rl_cfg(
    *,
    history_length: int,
    frame_term_dims: tuple[int, ...],
    target_group: str,
    role_group: str,
    teacher_obs_group: str = "critic",
    student_hidden_dims: tuple[int, ...] = ROUGH_CTS_STUDENT_HIDDEN_DIMS,
    estimator_warmup_iterations: int = 0,
    student_latent_grad: str = "mse",
    imitation_loss: ImitationLossCfg | None = None,
    policy_distill: PolicyDistillCfg | None = None,
    amp_cfg: dict | None = None,
    smoke: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """生成 CTS PPO 配置：actor 换 CTSVxObserverModel，学生编码器用独立 Adam 做隐向量重建；PPO 超参数与 `rl_cfg` 相同。

    student_hidden_dims：学生编码器隐藏层；estimator_warmup_iterations：本进程前若干轮只训练学生编码器（换编码器续训用）；
    student_latent_grad：学生隐向量的训练信号，"mse" 回归教师隐向量（默认）、"ppo" 由 PPO 端到端训练（见 se3_train.cts_observer）；
    imitation_loss：PPO 总损失里的模仿项（倒地教师用，见 se3_train.ppo），None 不加；
    policy_distill：网络老师在线蒸馏（recover 组起身用，见 se3_train.ppo），None 不加，与 imitation_loss 互斥；
    amp_cfg：AMP 风格奖励配置（recover 组起身风格用，见 se3_train.amp），None 不加。
    """
    cfg = rl_cfg(smoke=smoke)
    actor_fields = {
        k: v
        for k, v in asdict(cfg.actor).items()
        if k in {f.name for f in fields(RslRlModelCfg)} and k != "class_name"
    }
    cfg.actor = CTSModelCfg(
        **actor_fields,
        history_length=int(history_length),
        frame_term_dims=tuple(int(d) for d in frame_term_dims),
        vx_target_group=target_group,
        role_group=role_group,
        teacher_obs_group=teacher_obs_group,
        estimator_hidden_dims=tuple(int(d) for d in student_hidden_dims),
        student_latent_grad=str(student_latent_grad),
    )
    cfg.algorithm = replace(
        cfg.algorithm,
        estimator_learning_rate=ROUGH_VX_ESTIMATOR_LEARNING_RATE,
        estimator_target_group=target_group,
        estimator_warmup_iterations=int(estimator_warmup_iterations),
        imitation_loss=imitation_loss,
        policy_distill=policy_distill,
        amp_cfg=None if amp_cfg is None else dict(amp_cfg),
    )
    return cfg


def recover_amp_cfg(*, obs_group: str, mask_obs_group: str) -> dict:
    """recover 组 AMP 的 amp_cfg（键见 se3_train.amp.AMP）。"""
    return {
        "obs_group": obs_group,
        "mask_obs_group": mask_obs_group,
        "transition_frames": ROUGH_RECOVER_AMP_TRANSITION_FRAMES,
        "reward_weight": ROUGH_RECOVER_AMP_REWARD_WEIGHT,
        "reward_warmup_updates": 50,
        "discriminator_updates": 4,
        "discriminator_batch_size": ROUGH_RECOVER_AMP_DISCRIMINATOR_BATCH,
        "discriminator_grad_penalty_weight": 5.0,
        "learning_rate": 5.0e-4,
        "model_cfg": {"hidden_dims": [512, 256], "activation": "elu", "state_normalization": True},
        "dataset_kwargs": {
            "dataset_root": str(ROUGH_RECOVER_AMP_DATASET_ROOT),
            "mirror_augmentation": True,
        },
    }


def gru_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置：只换网络，rollout 仍 24 步，其余与 `rl_cfg` 逐项相同（M16）。"""
    cfg = _use_se3_ppo(flat_gru_rl_cfg(smoke=smoke))
    cfg.num_steps_per_env = FLAT_NUM_STEPS_PER_ENV
    cfg.bench_interval = ROUGH_BENCH_INTERVAL
    if not _is_smoke(smoke):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


__all__ = [
    "ROUGH_ACTOR_HIDDEN_DIMS",
    "ROUGH_BENCH_INTERVAL",
    "ROUGH_CTS_LATENT_DIM",
    "ROUGH_CTS_STUDENT_HIDDEN_DIMS",
    "ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU",
    "ROUGH_CTS_TEACHER_HIDDEN_DIMS",
    "ROUGH_FALL_TEACHER_IMITATION_LOSS_COEF",
    "ROUGH_FALL_TEACHER_IMITATION_LOSS_HUBER_DELTA",
    "ROUGH_MAX_ITERATIONS",
    "ROUGH_RECOVER_AMP_DATASET_ROOT",
    "ROUGH_RECOVER_DISTILL_TEACHER_CKPT",
    "ROUGH_VX_ESTIMATOR_LEARNING_RATE",
    "CTSModelCfg",
    "cts_rl_cfg",
    "gru_rl_cfg",
    "recover_amp_cfg",
    "rl_cfg",
]
