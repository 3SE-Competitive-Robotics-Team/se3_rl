"""WheelDog 平地任务 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置。"""
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
    else:
        max_iterations = 5000
        logger = os.environ.get("SE3_LOGGER", "wandb")

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name="RNNModel",
            rnn_type="gru",
            rnn_hidden_dim=512,
            rnn_num_layers=1,
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
            class_name="RNNModel",
            rnn_type="gru",
            rnn_hidden_dim=512,
            rnn_num_layers=1,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="se3_wheel_dog_flat",
        save_interval=100,
        num_steps_per_env=64,
        max_iterations=max_iterations,
        logger=logger,
        wandb_project="se3_wheel_leg",
    )


__all__ = ["rl_cfg"]
