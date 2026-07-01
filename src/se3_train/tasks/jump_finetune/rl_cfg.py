from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """跳跃精细训练 GRU PPO 配置（从 PreTrain checkpoint 加载）。

    使用方式：
        训练前确认 logs/rsl_rl/se3_wheel_leg/ 下有 PreTrain 的 run 目录。
        load_run 默认匹配最新 run（".*"），如需指定特定 run 可通过 CLI 覆盖：
            uv run --env-file .env se3-train SE3-WheelLegged-Jump-FineTune-GRU \\
                --agent.load-run 2025-01-01_12-00-00_jump_pretrain

    max_iterations = 5000（精细训练阶段）。
    """
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
        resume = False
    else:
        max_iterations = 5000
        logger = os.environ.get("SE3_LOGGER", "tensorboard")
        resume = True

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
        num_steps_per_env=64,
        max_iterations=max_iterations,
        logger=logger,
        # 从 PreTrain checkpoint 加载，指定 run 目录避免匹配到其他 run
        resume=resume,
        load_run="2026-05-18_15-14-03",
        load_checkpoint="model_.*.pt",
    )
