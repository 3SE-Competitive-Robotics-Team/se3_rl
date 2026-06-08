"""CTBC 台阶任务 PPO 配置。"""

from __future__ import annotations

import os

from se3_train.tasks.recovery.rl_cfg import rl_cfg as recovery_rl_cfg


def rl_cfg(smoke: bool = False):
    """复用 recovery 的 GRU 结构，保持 checkpoint 兼容。"""
    cfg = recovery_rl_cfg(smoke=smoke)
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        cfg.max_iterations = 5
    else:
        cfg.max_iterations = int(os.environ.get("SE3_STAIR_MAX_ITERATIONS", "2000"))
    cfg.experiment_name = "se3_wheel_leg_stair_ctbc"
    cfg.save_interval = int(os.environ.get("SE3_STAIR_SAVE_INTERVAL", "100"))
    return cfg
