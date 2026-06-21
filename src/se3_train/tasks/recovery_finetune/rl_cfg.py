"""倒地自启 Stage II FineTune PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.recovery.rl_cfg import rl_cfg as recovery_rl_cfg

_DEFAULT_LOAD_RUN = "recovery_discovery"
_DEFAULT_LOAD_CHECKPOINT = "model_1500\\.pt"


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Stage II 从 Discovery checkpoint warm-start。"""
    cfg = recovery_rl_cfg(smoke=smoke)
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        cfg.resume = False
        cfg.max_iterations = 5
        cfg.logger = "tensorboard"
    else:
        cfg.resume = True
        cfg.max_iterations = int(os.environ.get("SE3_RECOVERY_FINETUNE_MAX_ITERATIONS", "5000"))
        cfg.logger = os.environ.get("SE3_LOGGER", "wandb")
        cfg.load_run = os.environ.get("SE3_RECOVERY_FINETUNE_LOAD_RUN", _DEFAULT_LOAD_RUN)
        cfg.load_checkpoint = os.environ.get(
            "SE3_RECOVERY_FINETUNE_LOAD_CHECKPOINT",
            _DEFAULT_LOAD_CHECKPOINT,
        )
    return cfg
