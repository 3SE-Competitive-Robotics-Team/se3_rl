"""A15 warm-start 对照：stairs_up 奖励列与 flat 列逐项同价。"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a13_tuned import A13_ENV_KWARGS
from .a15_pricing import (
    A15_BASE_HEIGHT_SIGMA,
    A15_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES,
    A15_FLAT_VZ_WEIGHT,
    A15_OFF_STAIR_TRACKING_SIGMA_MOVE,
)
from .env_cfg import env_cfg
from .rl_cfg import rl_cfg as rough_rl_cfg


def flat_stairs_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 A15 环境，并移除 stairs_up 的专属奖励列门控。

    A15 的四个定价修正仍保留；stairs_up 不再置零高度/角速度奖励，
    也不再使用台阶专属速度核。两个台阶专属奖励项权重设为 0，
    因而 stairs_up 与 flat 使用同一套 RewardManager 奖励数学。
    """
    kwargs = dict(A13_ENV_KWARGS)
    kwargs.update(
        reward_terrain_type_names=(),
        zero_base_height_on_terrain=False,
        zero_tracking_ang_vel_on_terrain=False,
        base_height_sigma=A15_BASE_HEIGHT_SIGMA,
        off_stair_tracking_sigma_move=A15_OFF_STAIR_TRACKING_SIGMA_MOVE,
        flat_vz_weight=A15_FLAT_VZ_WEIGHT,
        command_velocity_error_terrain_names=A15_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES,
    )
    cfg = env_cfg(play=play, **kwargs)
    cfg.rewards["stair_climb_progress"].weight = 0.0
    cfg.rewards["stair_support_height"].weight = 0.0
    return cfg


def flat_stairs_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """A15 PPO 配置，正式运行时仅 warm-start actor/critic。"""
    smoke = bool(smoke or os.environ.get("SE3_SMOKE", "0") == "1")
    cfg = rough_rl_cfg(smoke=smoke)
    cfg.resume = not smoke
    cfg.load_run = os.environ.get(
        "SE3_A15_FLAT_STAIRS_LOAD_RUN",
        "2026-09-11_13-04-02_rough-A15-pricing-seed42-6x8192-5k",
    )
    cfg.load_checkpoint = os.environ.get("SE3_A15_FLAT_STAIRS_LOAD_CHECKPOINT", "model_4999.pt")
    return cfg


__all__ = ["flat_stairs_env_cfg", "flat_stairs_rl_cfg"]
