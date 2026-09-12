"""A21 基础上的台阶零速到前进指令采样对照。"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a21_full_height import a21_env_cfg, a21_rl_cfg


def a22_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """扩大台阶速度范围，并在零速指令下停发新增爬升奖励。"""
    cfg = a21_env_cfg(play=play)
    cfg.commands["velocity_height"] = replace(
        cfg.commands["velocity_height"], stair_lin_vel_x_range=(0.0, 2.4)
    )
    for name in ("stair_climb_progress", "stair_support_height"):
        term = cfg.rewards[name]
        cfg.rewards[name] = replace(
            term, params={**term.params, "movement_command_name": "velocity_height"}
        )
    return cfg


def a22_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """从 A21 最终网络参数开始，优化器与 iteration 重新初始化。"""
    cfg = a21_rl_cfg(smoke=smoke)
    cfg.load_run = os.environ.get(
        "SE3_A22_LOAD_RUN",
        "2026-09-12_15-04-09_rough-A21-fullheight-transition-seed42-6x8192-1k",
    )
    cfg.load_checkpoint = os.environ.get("SE3_A22_LOAD_CHECKPOINT", "model_999.pt")
    return cfg
