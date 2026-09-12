"""A22 的零速爬升奖励不门控对照。"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a22_stair_command import a22_env_cfg, a22_rl_cfg


def a23_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """保留 A22 采样，仅取消两项爬升奖励的零速门控。"""
    cfg = a22_env_cfg(play=play)
    for name in ("stair_climb_progress", "stair_support_height"):
        term = cfg.rewards[name]
        cfg.rewards[name] = replace(
            term,
            params={
                key: value for key, value in term.params.items() if key != "movement_command_name"
            },
        )
    return cfg


def a23_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """与 A22 使用同一 A21 初始模型和训练预算。"""
    cfg = a22_rl_cfg(smoke=smoke)
    cfg.load_run = os.environ.get("SE3_A23_LOAD_RUN", cfg.load_run)
    cfg.load_checkpoint = os.environ.get("SE3_A23_LOAD_CHECKPOINT", cfg.load_checkpoint)
    return cfg
