"""SE3 训练任务共享的 RSL-RL runner 默认配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg as MjlabRslRlOnPolicyRunnerCfg


@dataclass
class RslRlOnPolicyRunnerCfg(MjlabRslRlOnPolicyRunnerCfg):
    """统一在线 W&B 项目，并默认禁止上传 checkpoint。"""

    logger: Literal["wandb", "tensorboard"] = "wandb"
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "se3-wheel-leg")
    )
    upload_model: bool = False
