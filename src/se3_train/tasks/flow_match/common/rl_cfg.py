"""FlowMatch task 组共享 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _gru_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
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


def single_label_gru_ppo_runner_cfg(
    label: str,
    smoke: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """FlowMatch 单标签 GRU PPO 配置。"""
    cfg = _gru_ppo_runner_cfg(smoke=smoke)
    cfg.experiment_name = f"se3_wheel_leg_flow_match_{label}"
    cfg.resume = False
    return cfg


def gait_pretrain_gru_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """纯 GAIT 预训练 GRU PPO 配置。"""
    cfg = _gru_ppo_runner_cfg(smoke=smoke)
    cfg.experiment_name = "se3_wheel_leg_flow_match_gait_pretrain"
    cfg.resume = False
    if not (smoke or os.environ.get("SE3_SMOKE", "0") == "1"):
        cfg.max_iterations = 5000
    return cfg


def gait_finetune_gru_ppo_runner_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """纯 GAIT fine-tune GRU PPO 配置。"""
    cfg = _gru_ppo_runner_cfg(smoke=smoke)
    cfg.experiment_name = "se3_wheel_leg_flow_match_gait_finetune"
    cfg.resume = False
    cfg.algorithm.learning_rate = 2.0e-4
    cfg.algorithm.desired_kl = 0.006
    cfg.algorithm.entropy_coef = 0.003
    if not (smoke or os.environ.get("SE3_SMOKE", "0") == "1"):
        cfg.max_iterations = 2000
    return cfg


__all__ = [
    "gait_finetune_gru_ppo_runner_cfg",
    "gait_pretrain_gru_ppo_runner_cfg",
    "single_label_gru_ppo_runner_cfg",
]
