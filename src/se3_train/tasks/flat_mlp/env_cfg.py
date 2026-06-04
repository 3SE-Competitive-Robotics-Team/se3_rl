"""平地 MLP 行走任务环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

_MLP_STEPS_PER_POLICY_ITER = 32
_INACTIVE_JUMP_HEIGHT = 0.0


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成面向真机平地基模的 MLP 环境配置。"""
    cfg = flat_env_cfg(play=play)
    _disable_jump_contract(cfg)
    if not play:
        _align_curriculum_progress(cfg)
    return cfg


def _disable_jump_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    """保留 32 维观测接口，但让跳跃扩展维度固定为未激活状态。"""
    command_cfg = cfg.commands["velocity_height"]
    command_cfg.jump_prob = 0.0
    command_cfg.jump_height_range = (_INACTIVE_JUMP_HEIGHT, _INACTIVE_JUMP_HEIGHT)
    command_cfg.rsi_takeoff_prob = 0.0
    command_cfg.rsi_random_frame = False


def _align_curriculum_progress(cfg: ManagerBasedRlEnvCfg) -> None:
    """让课程进度使用 MLP rollout 的真实 PPO iteration 长度。"""
    if cfg.curriculum is None:
        return

    for term_name in ("command_vel", "push_disturbance"):
        term = cfg.curriculum.get(term_name)
        if term is None:
            continue
        term.params["steps_per_policy_iter"] = _MLP_STEPS_PER_POLICY_ITER


__all__ = ["env_cfg"]
