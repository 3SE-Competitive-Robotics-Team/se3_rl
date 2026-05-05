from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def se3_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 PPO 训练配置。

    Args:
        smoke: 是否使用 smoke 模式（5 轮训练，不上传 SwanLab）

    Returns:
        RslRlOnPolicyRunnerCfg 实例
    """
    # smoke 模式：5 轮，tensorboard logger
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
    else:
        max_iterations = 2000
        logger = "swanlab"

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.167,
            entropy_coef=0.00516,
            num_learning_epochs=7,
            num_mini_batches=4,
            learning_rate=6.5e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="se3_wheel_leg",
        save_interval=100,
        num_steps_per_env=32,
        max_iterations=max_iterations,
        logger=logger,
        wandb_project="se3_wheel_leg",
    )
