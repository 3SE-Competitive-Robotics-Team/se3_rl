"""SE3 训练 runner 扩展。"""

from __future__ import annotations

import torch
from mjlab.rl import MjlabOnPolicyRunner


class Se3WarmStartRunner(MjlabOnPolicyRunner):
    """阶段切换专用 warm-start runner。

    上一阶段 checkpoint 只提供 actor/critic 初始权重。新阶段必须从新的 runner
    迭代、optimizer 和环境计数开始，否则课程和日志会继承旧 checkpoint 的训练进度。
    """

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """只加载策略和价值网络权重，不恢复 optimizer、iter 和 env_state。"""
        warm_start_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }

        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        actor_state_dict = loaded_dict.get("actor_state_dict", {})
        if "std" in actor_state_dict:
            actor_state_dict["distribution.std_param"] = actor_state_dict.pop("std")
        if "log_std" in actor_state_dict:
            actor_state_dict["distribution.log_std_param"] = actor_state_dict.pop("log_std")

        self.alg.load(loaded_dict, warm_start_cfg if load_cfg is None else load_cfg, strict)
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        return loaded_dict.get("infos", {})


Se3PretrainWarmStartRunner = Se3WarmStartRunner
