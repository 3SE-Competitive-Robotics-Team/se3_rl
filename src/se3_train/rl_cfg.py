"""SE3 训练任务共享的 RSL-RL runner 默认配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg as MjlabRslRlOnPolicyRunnerCfg
from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class ImitationLossCfg:
    """PPO 总损失里的模仿项（见 se3_train.ppo 的 imitation_loss）：coef·α·Huber(μ − T)，只在教师接管的样本上求均值。

    group：监督目标观测组 [接管, α, 教师动作 6 维]；front_period / back_period：前杆 / 后杆整圈对应的动作量（2π / 缩放），
    T 先按前杆整圈平移到 μ 那一圈，shift_back=True（joint 语义）时同侧后杆同步平移、夹角不变。
    """

    group: str = "fall_teacher_target"
    coef: float = 0.2
    huber_delta: float = 1.0
    front_period: float = 0.0
    back_period: float = 0.0
    shift_back: bool = True


@dataclass
class Se3PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO 算法配置：可选固定 critic 学习率、可选显式 vx 估计器监督、可选模仿损失（见 se3_train.ppo）。"""

    critic_learning_rate: float | None = None
    estimator_learning_rate: float | None = None
    estimator_target_group: str | None = None
    estimator_warmup_iterations: int = 0
    imitation_loss: ImitationLossCfg | None = None
    class_name: str = "se3_train.ppo:Se3PPO"


@dataclass
class RslRlOnPolicyRunnerCfg(MjlabRslRlOnPolicyRunnerCfg):
    """提供在线 W&B 默认配置，并上传 PyTorch checkpoint。"""

    logger: Literal["wandb", "tensorboard"] = "wandb"
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "se3-wheel-leg")
    )
    upload_model: bool = True
    bench_interval: int = 0
    """每 N 轮在保存后与训练结束时旁路跑一次 full benchmark（见 se3_train.bench_hook）；0 = 关闭。"""
    bench_workers: int = 0
    """benchmark 进程数；0 = CPU 核数的 1/4（上限 16）。"""


def bind_task_name(
    cfg: RslRlOnPolicyRunnerCfg,
    task_name: str,
) -> RslRlOnPolicyRunnerCfg:
    """将注册任务名绑定为本地 experiment 和 W&B Project。"""
    cfg.experiment_name = task_name
    cfg.wandb_project = os.environ.get("WANDB_PROJECT", task_name)
    return cfg
