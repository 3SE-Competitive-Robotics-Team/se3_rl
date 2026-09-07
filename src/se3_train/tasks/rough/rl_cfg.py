"""崎岖地形任务的 PPO 配置：直接复用冻结的 Flat 基线，只改训练轮数。

2026-09-06 之前这里是 Flat 基线定型前拷贝出去的一份旧值（clip 0.167 / entropy 0.00516 /
epochs 7 / lr 6.5e-4 / 32 步），与平地已经不是同一套 PPO。移植 rough 时改为直接调用
`flat.mlp_rl_cfg()`，让 rough 与 Flat 基线共用一套超参数：这样 rough 相对 flat 的唯一变量
就是地形、地形课程和台阶状态机，不掺 PPO 差异。参考仓库同样是 rough 继承 flat 的 PPO 配置。
"""

from __future__ import annotations

import os
from dataclasses import asdict

from se3_train.amp import default_dataset_root
from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg, Se3PpoAlgorithmCfg
from se3_train.tasks.flat.rl_cfg import mlp_rl_cfg

# 地形课程要爬 10 级难度，比平地的 3500 轮长。沿用本仓库非 Flat 线的 5000 轮惯例。
ROUGH_MAX_ITERATIONS = 5000

# AMP 默认参数照 kyber_rl_lab 的 g1 velocity AMP：reward_weight 3.0、热身 100 次更新、
# 每轮 2 步判别器更新、batch 4096、lr 1e-4、R1 惩罚 10、判别器 [512,256]+输入归一化。
ROUGH_AMP_DATASET_ROOT = "assets/amp/fudan_stairs20_20260907"


def amp_cfg_dict(*, dataset_root: str) -> dict:
    """Se3PpoAlgorithmCfg.amp_cfg 的内容（键与 kyber RslRlAmpCfg 一致，数据集走 dataset_kwargs）。"""
    return {
        "obs_group": "amp",
        "transition_frames": 2,
        "reward_weight": 3.0,
        "reward_warmup_updates": 100,
        "discriminator_updates": 2,
        "discriminator_batch_size": 4096,
        "discriminator_grad_penalty_weight": 10.0,
        "learning_rate": 1.0e-4,
        "max_grad_norm": None,
        "model_cfg": {"hidden_dims": [512, 256], "activation": "elu", "state_normalization": True},
        "dataset_kwargs": {
            "dataset_root": dataset_root,
            "mirror_augmentation": True,
        },
    }


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置（超参数与 Flat 基线逐项相同）。"""
    cfg = mlp_rl_cfg(smoke=smoke)
    if not (smoke or os.environ.get("SE3_SMOKE", "0") == "1"):
        cfg.max_iterations = ROUGH_MAX_ITERATIONS
    return cfg


def amp_rl_cfg(smoke: bool = False, *, dataset_root: str | None = None) -> RslRlOnPolicyRunnerCfg:
    """带 AMP 的 PPO 配置：其余与 rl_cfg 逐项相同，算法换成 Se3PPO + amp_cfg。

    数据集目录默认 assets/amp/fudan_stairs20_20260907（package_fudan_amp_dataset.py 产物），
    可用环境变量 SE3_AMP_DATASET_ROOT 覆盖；注册时不要求目录存在，真正加载在训练启动、构造算法时。
    """
    cfg = rl_cfg(smoke=smoke)
    root = default_dataset_root(ROUGH_AMP_DATASET_ROOT) if dataset_root is None else dataset_root
    algorithm = asdict(cfg.algorithm)
    algorithm.pop("class_name", None)
    algorithm.pop("critic_learning_rate", None)
    algorithm.pop("amp_cfg", None)
    cfg.algorithm = Se3PpoAlgorithmCfg(**algorithm, amp_cfg=amp_cfg_dict(dataset_root=root))
    return cfg


__all__ = ["ROUGH_AMP_DATASET_ROOT", "ROUGH_MAX_ITERATIONS", "amp_cfg_dict", "amp_rl_cfg", "rl_cfg"]
