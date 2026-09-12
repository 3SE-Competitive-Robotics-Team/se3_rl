"""A15 warm-start 的无平地热身对照与指令速度门控高度实验。"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a13_tuned import A13_ENV_KWARGS
from .a15_pricing import (
    A15_BASE_HEIGHT_SIGMA,
    A15_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES,
    A15_FLAT_VZ_WEIGHT,
    A15_OFF_STAIR_TRACKING_SIGMA_MOVE,
    a15_rl_cfg,
)
from .env_cfg import ROUGH_BODY_COLLISION_BOTTOM_OFFSET, ROUGH_TERRAIN_HEIGHT_CLEARANCE, env_cfg

A18_HEIGHT_GATE_RANGE = (0.10, 0.40)
"""前向指令从 0.10 到 0.40 m/s 时，目标高度惩罚由全幅平滑退到零。"""

A18_MIN_CLEARANCE = -ROUGH_BODY_COLLISION_BOTTOM_OFFSET + ROUGH_TERRAIN_HEIGHT_CLEARANCE
"""机身碰撞底面与局部地面的最低安全净空对应的 base 高度。"""


def a18_control_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A15 奖励原样保留，只取消 warm-start 后重复的平地热身。"""
    return env_cfg(
        play=play,
        flat_warmup_iterations=0,
        flat_warmup_ramp_iterations=0,
        **_a15_env_kwargs(),
    )


def a18_height_gate_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """对照组基础上，仅给原 flat_base_height 项增加指令速度门控。"""
    cfg = a18_control_env_cfg(play=play)
    term = cfg.rewards["flat_base_height"]
    cfg.rewards["flat_base_height"] = replace(
        term,
        params={
            **term.params,
            "command_speed_gate_range": A18_HEIGHT_GATE_RANGE,
            "min_clearance": A18_MIN_CLEARANCE,
            "min_clearance_sigma": 0.05,
        },
    )
    return cfg


def a18_warmstart_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """从 A15 model_4999 只加载 actor/critic，优化器和 iteration 重新开始。"""
    smoke = bool(smoke or os.environ.get("SE3_SMOKE", "0") == "1")
    cfg = a15_rl_cfg()
    if smoke:
        cfg.max_iterations = 5
        cfg.logger = "tensorboard"
    cfg.resume = not smoke
    cfg.load_run = os.environ.get(
        "SE3_A18_LOAD_RUN",
        "2026-09-11_13-04-02_rough-A15-pricing-seed42-6x8192-5k",
    )
    cfg.load_checkpoint = os.environ.get("SE3_A18_LOAD_CHECKPOINT", "model_4999.pt")
    return cfg


def _a15_env_kwargs() -> dict[str, object]:
    """复用 A15 已冻结参数，同时允许覆盖 warmup。"""
    return {
        **A13_ENV_KWARGS,
        "base_height_sigma": A15_BASE_HEIGHT_SIGMA,
        "off_stair_tracking_sigma_move": A15_OFF_STAIR_TRACKING_SIGMA_MOVE,
        "flat_vz_weight": A15_FLAT_VZ_WEIGHT,
        "command_velocity_error_terrain_names": A15_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES,
    }


__all__ = [
    "A18_HEIGHT_GATE_RANGE",
    "A18_MIN_CLEARANCE",
    "a18_control_env_cfg",
    "a18_height_gate_env_cfg",
    "a18_warmstart_rl_cfg",
]
