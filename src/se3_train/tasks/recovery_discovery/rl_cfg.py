"""倒地自启 Discovery 阶段 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


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


def _rl_cfg(*, smoke: bool, recurrent: bool) -> RslRlOnPolicyRunnerCfg:
    """生成网络类型以外完全一致的 Discovery PPO 配置。"""
    smoke_enabled = smoke or os.environ.get("SE3_SMOKE", "0") == "1"
    logger = "tensorboard" if smoke_enabled else os.environ.get("SE3_LOGGER", "tensorboard")

    learning_rate = float(os.environ.get("SE3_RECOVERY_LEARNING_RATE", "3.0e-4"))
    init_std = float(os.environ.get("SE3_RECOVERY_INIT_STD", "0.5"))
    entropy_coef = float(os.environ.get("SE3_RECOVERY_ENTROPY_COEF", "0.00516"))
    max_iterations = (
        5 if smoke_enabled else int(os.environ.get("SE3_RECOVERY_DISCOVERY_MAX_ITERATIONS", "5000"))
    )

    return RslRlOnPolicyRunnerCfg(
        actor=_model_cfg(
            recurrent=recurrent,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": "scalar",
            },
        ),
        critic=_model_cfg(recurrent=recurrent),
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
            desired_kl=0.008,
            max_grad_norm=1.0,
        ),
        experiment_name="se3_wheel_leg",
        save_interval=100,
        num_steps_per_env=64,
        max_iterations=max_iterations,
        logger=logger,
        resume=False,
    )


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成 Discovery 阶段从零训练的 GRU PPO 配置。"""
    return _rl_cfg(smoke=smoke, recurrent=True)


def mlp_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成除网络类型外与 GRU 版本一致的 MLP PPO 配置。"""
    return _rl_cfg(smoke=smoke, recurrent=False)
