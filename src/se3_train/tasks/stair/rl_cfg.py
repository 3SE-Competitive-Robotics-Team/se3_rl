"""倒金字塔台阶任务 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

_DEFAULT_LOAD_RUN = "base_model"
_DEFAULT_GRU_LOAD_CHECKPOINT = "rough_base\\.pt"
_DEFAULT_MLP_LOAD_CHECKPOINT = "mlp\\.pt"
_NUM_STEPS_PER_ENV = 64


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
    """生成网络类型和对应基模以外完全一致的台阶 PPO 配置。"""
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        max_iterations = 5
        logger = "tensorboard"
        resume = False
    else:
        # 直接从台阶 level 0 开始训练；保持源仓库 stair GRU 的 64-step rollout。
        # CTBC 第 0-200 轮满幅启用，200->500 轮退火，之后进入无辅助台阶课程。
        max_iterations = int(os.environ.get("SE3_STAIR_MAX_ITERATIONS", "3200"))
        logger = os.environ.get("SE3_LOGGER", "tensorboard")
        resume = True

    # 从低 std 的 34 维 GRU 基模 warm-start 时，3e-4 的首个 Adam step 会让 KL 爆炸。
    learning_rate = float(os.environ.get("SE3_STAIR_LEARNING_RATE", "1.0e-5"))
    init_std = float(os.environ.get("SE3_STAIR_INIT_STD", "0.5"))
    entropy_coef = float(os.environ.get("SE3_STAIR_ENTROPY_COEF", "0.00516"))
    if recurrent:
        load_checkpoint = os.environ.get(
            "SE3_STAIR_LOAD_CHECKPOINT",
            _DEFAULT_GRU_LOAD_CHECKPOINT,
        )
    else:
        load_checkpoint = os.environ.get(
            "SE3_STAIR_MLP_LOAD_CHECKPOINT",
            _DEFAULT_MLP_LOAD_CHECKPOINT,
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
        num_steps_per_env=_NUM_STEPS_PER_ENV,
        max_iterations=max_iterations,
        logger=logger,
        resume=resume,
        load_run=os.environ.get("SE3_STAIR_LOAD_RUN", _DEFAULT_LOAD_RUN),
        load_checkpoint=load_checkpoint,
    )


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成从 34 维 GRU 基模 warm-start 的台阶 PPO 配置。"""
    return _rl_cfg(smoke=smoke, recurrent=True)


def mlp_rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """生成从 34 维 MLP 基模 warm-start 的台阶 PPO 配置。"""
    return _rl_cfg(smoke=smoke, recurrent=False)
