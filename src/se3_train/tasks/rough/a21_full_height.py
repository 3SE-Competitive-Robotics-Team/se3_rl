"""A20 高姿态启动序列配合完整高度惩罚的单变量对照。"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a15_pricing import a15_rl_cfg
from .a20_high_stand_transition import a20_env_cfg

_HEIGHT_GATE_PARAMS = {
    "command_speed_gate_range",
    "min_clearance",
    "min_clearance_sigma",
}


def a21_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """保留 A20 指令转移，只把高度项恢复为 A15 的完整目标高度惩罚。"""
    cfg = a20_env_cfg(play=play)
    term = cfg.rewards["flat_base_height"]
    cfg.rewards["flat_base_height"] = replace(
        term,
        params={key: value for key, value in term.params.items() if key not in _HEIGHT_GATE_PARAMS},
    )
    return cfg


def a21_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """从 A15 网络参数开始新实验，不恢复优化器与 iteration。"""
    smoke = bool(smoke or os.environ.get("SE3_SMOKE", "0") == "1")
    cfg = a15_rl_cfg()
    cfg.max_iterations = 5 if smoke else 1000
    if smoke:
        cfg.logger = "tensorboard"
    cfg.resume = not smoke
    cfg.load_run = os.environ.get(
        "SE3_A21_LOAD_RUN",
        "2026-09-11_13-04-02_rough-A15-pricing-seed42-6x8192-5k",
    )
    cfg.load_checkpoint = os.environ.get("SE3_A21_LOAD_CHECKPOINT", "model_4999.pt")
    return cfg


__all__ = ["a21_env_cfg", "a21_rl_cfg"]
