"""CTBC 台阶任务 PPO 配置。"""

from __future__ import annotations

import os

from se3_train.tasks.recovery.rl_cfg import rl_cfg as recovery_rl_cfg

_BASE_MODEL_DIR = "base_model"
_RECOVERY_BASE_CHECKPOINT = "model_4999_gru.pt"


def rl_cfg(smoke: bool = False):
    """复用 recovery 的 GRU 结构，并从最新 recovery 基模 warm-start。"""
    cfg = recovery_rl_cfg(smoke=smoke)
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        cfg.max_iterations = 5
        cfg.resume = False
    else:
        cfg.max_iterations = int(os.environ.get("SE3_STAIR_MAX_ITERATIONS", "2000"))
        cfg.resume = True
    cfg.experiment_name = "se3_wheel_leg_stair_ctbc"
    cfg.save_interval = int(os.environ.get("SE3_STAIR_SAVE_INTERVAL", "100"))
    cfg.load_run = _BASE_MODEL_DIR
    cfg.load_checkpoint = _RECOVERY_BASE_CHECKPOINT
    return cfg
