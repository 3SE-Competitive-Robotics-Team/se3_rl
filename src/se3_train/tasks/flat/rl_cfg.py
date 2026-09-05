from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg, Se3PpoAlgorithmCfg

# actor 的初始学习率（KL 自适应从这里出发）；critic 解耦实验用同一个值作为固定 LR，不引入新数字。
FLAT_LEARNING_RATE = 6.5e-4


def _model_cfg(
    *,
    recurrent: bool,
    distribution_cfg: dict[str, object] | None = None,
) -> RslRlModelCfg:
    """生成共享网络配置，仅按任务版本切换 GRU 或 MLP。"""
    if recurrent:
        return RslRlModelCfg(
            class_name="RNNModel",
            rnn_type="gru",
            rnn_hidden_dim=512,
            rnn_num_layers=1,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=distribution_cfg,
        )
    return RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=distribution_cfg,
    )


def _rl_cfg(
    *, smoke: bool, recurrent: bool, critic_learning_rate: float | None = None
) -> RslRlOnPolicyRunnerCfg:
    """生成网络类型以外完全一致的 PPO 训练配置。

    critic_learning_rate：None 沿用 rsl_rl.PPO（actor/critic 共用 KL 自适应 LR）；给定时切到
    se3_train.ppo.Se3PPO，critic 用该固定 LR，actor 仍走 KL 自适应，见 se3_train/ppo.py。
    """
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
    else:
        max_iterations = 5000
        logger = os.environ.get("SE3_LOGGER", "wandb")

    algorithm_kwargs = dict(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.167,
        entropy_coef=0.00516,
        num_learning_epochs=7,
        num_mini_batches=4,
        learning_rate=FLAT_LEARNING_RATE,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    return RslRlOnPolicyRunnerCfg(
        actor=_model_cfg(
            recurrent=recurrent,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=_model_cfg(recurrent=recurrent),
        algorithm=(
            Se3PpoAlgorithmCfg(**algorithm_kwargs, critic_learning_rate=critic_learning_rate)
            if critic_learning_rate is not None
            else RslRlPpoAlgorithmCfg(**algorithm_kwargs)
        ),
        experiment_name="se3_wheel_leg",
        save_interval=100,
        num_steps_per_env=64,
        max_iterations=max_iterations,
        logger=logger,
    )


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 GRU PPO 训练配置（hidden=512，单层 GRU，64 步 BPTT）。"""
    return _rl_cfg(smoke=smoke, recurrent=True)


def mlp_rl_cfg(
    smoke: bool = False, *, critic_learning_rate: float | None = None
) -> RslRlOnPolicyRunnerCfg:
    """生成除网络类型外与 GRU 版本一致的 MLP PPO 配置。"""
    return _rl_cfg(smoke=smoke, recurrent=False, critic_learning_rate=critic_learning_rate)
