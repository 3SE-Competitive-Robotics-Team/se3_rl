"""A15 warm-start 的高姿态静站到前进指令转移实验。"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a15_pricing import a15_rl_cfg
from .a18_height_gate import a18_height_gate_env_cfg

A20_TRANSITION_PROB = 0.5
A20_HIGH_STAND_HEIGHT_RANGE = (0.36, 0.38)
A20_HIGH_STAND_DURATION_RANGE_S = (1.5, 2.5)
A20_MOVE_VX_RANGE = (0.8, 2.4)


def a20_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """保留 A15 奖励与高度门控，并在平地注入高姿态启动序列。"""
    cfg = a18_height_gate_env_cfg(play=play)
    term = cfg.commands["velocity_height"]
    cfg.commands["velocity_height"] = replace(
        term,
        high_stand_transition_prob=A20_TRANSITION_PROB,
        high_stand_height_range=A20_HIGH_STAND_HEIGHT_RANGE,
        high_stand_duration_range_s=A20_HIGH_STAND_DURATION_RANGE_S,
        high_stand_move_vx_range=A20_MOVE_VX_RANGE,
    )
    return cfg


def a20_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """从 A15 model_4999 加载网络参数，优化器和 iteration 重新开始。"""
    smoke = bool(smoke or os.environ.get("SE3_SMOKE", "0") == "1")
    cfg = a15_rl_cfg()
    cfg.max_iterations = 5 if smoke else 1500
    if smoke:
        cfg.logger = "tensorboard"
    cfg.resume = not smoke
    cfg.load_run = os.environ.get(
        "SE3_A20_LOAD_RUN",
        "2026-09-11_13-04-02_rough-A15-pricing-seed42-6x8192-5k",
    )
    cfg.load_checkpoint = os.environ.get("SE3_A20_LOAD_CHECKPOINT", "model_4999.pt")
    return cfg


__all__ = ["a20_env_cfg", "a20_rl_cfg"]
