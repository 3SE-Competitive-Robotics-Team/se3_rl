"""倒金字塔台阶任务 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

_BASE_MODEL_DIR = "base_model"


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """台阶 GRU PPO 配置，从当前 32 维 GRU 基模 warm-start。"""
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
        resume = False
    else:
        max_iterations = 2200
        logger = os.environ.get("SE3_LOGGER", "wandb")
        resume = True

    learning_rate = float(os.environ.get("SE3_STAIR_LEARNING_RATE", "6.5e-4"))
    init_std = float(os.environ.get("SE3_STAIR_INIT_STD", "0.5"))
    entropy_coef = float(os.environ.get("SE3_STAIR_ENTROPY_COEF", "0.00516"))

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
                "init_std": init_std,
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
            clip_param=0.167,
            entropy_coef=entropy_coef,
            num_learning_epochs=7,
            num_mini_batches=4,
            learning_rate=learning_rate,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="se3_wheel_leg",
        save_interval=100,
        num_steps_per_env=64,
        max_iterations=max_iterations,
        logger=logger,
        wandb_project="se3_wheel_leg",
        resume=resume,
        load_run=_BASE_MODEL_DIR,
        load_checkpoint="model_[0-9]+_gru\\.pt",
    )
