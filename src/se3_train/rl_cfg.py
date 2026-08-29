"""SE3 训练任务共享的 RSL-RL runner 默认配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg as MjlabRslRlOnPolicyRunnerCfg


@dataclass
class RslRlOnPolicyRunnerCfg(MjlabRslRlOnPolicyRunnerCfg):
    """提供在线 W&B 默认配置，并上传 PyTorch checkpoint。"""

    logger: Literal["wandb", "tensorboard"] = "wandb"
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "se3-wheel-leg")
    )
    upload_model: bool = True


def bind_task_name(
    cfg: RslRlOnPolicyRunnerCfg,
    task_name: str,
) -> RslRlOnPolicyRunnerCfg:
    """将注册任务名绑定为本地 experiment 和 W&B Project。"""
    cfg.experiment_name = task_name
    cfg.wandb_project = os.environ.get("WANDB_PROJECT", task_name)
    return cfg
