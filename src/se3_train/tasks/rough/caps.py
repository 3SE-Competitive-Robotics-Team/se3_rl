"""CTS 学生历史编码器的 CAPS 空间一致性微调配置。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.noise import UniformNoiseCfg

from se3_train.onnx_metadata import observation_term_width
from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg


def configure_student_caps(
    train_env: ManagerBasedRlEnvCfg,
    play_env: ManagerBasedRlEnvCfg,
    agent: RslRlOnPolicyRunnerCfg,
) -> None:
    """在独立实验配置上加干净历史，只微调原有学生；其他训练任务不变。

    初始实验取空间损失权重 1、Adam 学习率 1e-4、1000 轮；均为待验证超参数。
    原有隐向量/vx MSE、3:1 师生采样、PD、动作缩放和域随机化保持一致。
    干净历史只用于生成训练噪声视图，不进入部署 actor 或 ONNX 输入。
    """
    actor = train_env.observations["actor"]
    if not actor.enable_corruption or not actor.history_length:
        raise ValueError("学生 CAPS 需要开启观测噪声及历史的 actor")
    amplitudes: list[float] = []
    for name, term in actor.terms.items():
        # 本实验的噪声在观测函数完成物理量缩放后相加。拒绝改变处理顺序的配置。
        if term.clip is not None or term.scale is not None or term.delay_max_lag:
            raise ValueError(f"CAPS 未支持 {name} 的 clip/scale/观测延迟")
        amplitude = 0.0
        if term.noise is not None:
            noise = term.noise
            if (
                not isinstance(noise, UniformNoiseCfg)
                or noise.operation != "add"
                or not isinstance(noise.n_max, float)
                or noise.n_min != -noise.n_max
            ):
                raise ValueError(f"CAPS 需要 {name} 为对称的加性均匀噪声")
            amplitude = noise.n_max
        amplitudes.extend([amplitude] * (observation_term_width(name) * actor.history_length))
    for cfg in (train_env, play_env):
        clean = deepcopy(cfg.observations["actor"])
        clean.enable_corruption = False
        cfg.observations["caps_clean_history"] = clean
    agent.actor = replace(
        agent.actor,
        caps_spatial_weight=1.0,
        caps_noise_amplitudes=tuple(amplitudes),
    )
    agent.algorithm = replace(
        agent.algorithm,
        estimator_only=True,
        estimator_warmup_iterations=0,
        estimator_learning_rate=1.0e-4,
    )
    # smoke 仍遵循既有 5 轮限制，正式入口默认只做有界的微调实验。
    if agent.max_iterations > 5:
        agent.max_iterations = 1000
    agent.save_interval = 100
