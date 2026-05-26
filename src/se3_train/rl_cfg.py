from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# base model checkpoint 相对于项目根目录的默认路径（用于 load_run）
_BASE_MODEL_DIR = "base_model"


def se3_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 MLP PPO 训练配置。"""
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
    else:
        max_iterations = 5000
        logger = os.environ.get("SE3_LOGGER", "wandb")

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


def se3_gru_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置（RNNModel, hidden=512, 1 layer, 64 步 BPTT）。"""
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
        wandb_project="se3_wheel_leg",
    )


def se3_jump_pretrain_gru_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """跳跃预训练 GRU PPO 配置（从 assets/base_model checkpoint 加载）。

    使用方式：
        1. 将行走 checkpoint 放入 assets/base_model/model_xxx.pt
        2. 在训练机器上建立软链接：
               ln -s $(pwd)/assets/base_model logs/rsl_rl/se3_wheel_leg/base_model
        3. 然后启动训练，mjlab 会自动从 base_model/ 目录加载最新 checkpoint。

    max_iterations = 1500（预训练阶段）。
    """
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
        resume = False
    else:
        max_iterations = 1500
        logger = os.environ.get("SE3_LOGGER", "wandb")
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
        wandb_project="se3_wheel_leg",
        # 从 GRU 行走基模加载，仅匹配数字迭代号，避免误加载跳跃阶段 checkpoint。
        resume=resume,
        load_run=_BASE_MODEL_DIR,
        load_checkpoint="model_[0-9]+_gru\\.pt",
    )


def se3_jump_gru_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """跳跃精细训练 GRU PPO 配置（从 PreTrain checkpoint 加载）。

    使用方式：
        训练前确认 logs/rsl_rl/se3_wheel_leg/ 下有 PreTrain 的 run 目录。
        load_run 默认匹配最新 run（".*"），如需指定特定 run 可通过 CLI 覆盖：
            uv run --env-file .env se3-train SE3-WheelLegged-Jump-GRU \\
                --agent.load-run 2025-01-01_12-00-00_jump_pretrain

    max_iterations = 5000（精细训练阶段）。
    """
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
        resume = False
    else:
        max_iterations = 5000
        logger = os.environ.get("SE3_LOGGER", "wandb")
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
        wandb_project="se3_wheel_leg",
        # 从 PreTrain checkpoint 加载，指定 run 目录避免匹配到其他 run
        resume=resume,
        load_run="2026-05-18_15-14-03",
        load_checkpoint="model_.*.pt",
    )
