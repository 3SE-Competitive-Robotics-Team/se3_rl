"""从 A22 同一检查点比较关闭台阶高度罚与四分之一强度高度罚。"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a22_stair_command import a22_env_cfg, a22_rl_cfg


def a24_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """对照组保持 A22 的全部环境配置。"""
    return a22_env_cfg(play=play)


def a25_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """仅恢复台阶高度罚至非台阶列的四分之一，有效权重为 -1。"""
    cfg = a24_env_cfg(play=play)
    term = cfg.rewards["flat_base_height"]
    cfg.rewards["flat_base_height"] = replace(
        term, params={**term.params, "terrain_penalty_scale": 0.25}
    )
    return cfg


def a24_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """两组均从 A22 model_400 网络参数开始，重置优化器和训练轮数。"""
    cfg = a22_rl_cfg(smoke=smoke)
    cfg.load_run = "2026-09-12_19-18-15_rough-A22-stopgate-seed42-3x8192-1k"
    cfg.load_checkpoint = "model_400.pt"
    return cfg
