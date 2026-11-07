"""Rough-SuperMimic 环境（2026-11-06 用户定，第 1 阶段）：Rough 默认配置换纯平地地形，再去掉 recover 组与推扰。

- 地形：`rough.terrains.flat_only_terrains_cfg()`。全地形要把局部高度图接进世界模型，留到第 2 阶段
  （docs/plan/rough_supermimic_20261106.md）。纯平地里没有跳跃列，`jump_commands` 恒 0、`mimic_*` 奖励恒 0。
- 去掉 recover 组整套（倒地出生、零速指令、recover_* 奖励 / 终止、AMP 与 recover_mask 观测组、critic 的
  recover_* 项）：用户定的是速度跟踪，倒地起身是另一个 MDP。
- 去掉推扰（`push_robots` 事件与 `push_disturbance` 课程）：与论文一致，状态突变无法被确定性世界模型拟合。
- 其余（DR1、动作延迟 0–10 ms、轮力矩滞后 5–15 ms、30 维 actor 观测、ONNX 契约）与 Rough 默认相同。

SuperMimic 入口额外挂一个 recorder 把 transition 写进回放缓存；PPO 对照入口不挂（recorder 不改 MDP）。
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.supermimic.recorder import RECORDER_NAME, SuperMimicRecorderCfg
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.terrains import flat_only_terrains_cfg

ROUGH_SUPERMIMIC_DROPPED_EVENTS = (
    "reset_from_fall_states",
    "reset_from_recover_trajectories",
    "assign_recover_group",
    "force_recover_commands",
    "snapshot_recover_potential",
    "log_fall_recovery",
    "push_robots",
)
ROUGH_SUPERMIMIC_DROPPED_REWARD_PREFIXES = ("recover_", "stair_", "mimic_")
"""recover 组奖励；台阶类与跳跃 mimic 奖励按设计只在台阶列 / 跳跃列生效，纯平地上应恒 0。

注意：`stair_heading_hold` 在地形里找不到台阶列时列掩码为 None，会退化成全体 env 都罚（−6 × 航向误差²，
2026-11-06 GPU 核对实测平地上均值 2.9/步），所以这里必须显式去掉，不能靠"平地上自然为 0"。"""
ROUGH_SUPERMIMIC_DROPPED_CURRICULA = ("push_disturbance",)
ROUGH_SUPERMIMIC_DROPPED_OBS_GROUPS = ("amp", "recover_mask")
ROUGH_SUPERMIMIC_REPLAY_CAPACITY = 1000
"""回放缓存每 env 保留的步数（论文 Table A.5）。"""


def env_cfg(play: bool = False, *, replay: bool = True) -> ManagerBasedRlEnvCfg:
    """纯平地 Rough + 去 recover 组与推扰；replay=True 时挂 SuperMimic 回放 recorder。"""
    cfg = rough_env_cfg(play=play, terrain_generator=flat_only_terrains_cfg())
    cfg.events = {k: v for k, v in cfg.events.items() if k not in ROUGH_SUPERMIMIC_DROPPED_EVENTS}
    cfg.rewards = {
        k: v
        for k, v in cfg.rewards.items()
        if not k.startswith(ROUGH_SUPERMIMIC_DROPPED_REWARD_PREFIXES)
    }
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if not k.startswith("recover_")}
    if cfg.curriculum:
        cfg.curriculum = {
            k: v for k, v in cfg.curriculum.items() if k not in ROUGH_SUPERMIMIC_DROPPED_CURRICULA
        }
    observations = {
        k: v for k, v in cfg.observations.items() if k not in ROUGH_SUPERMIMIC_DROPPED_OBS_GROUPS
    }
    critic = observations["critic"]
    observations["critic"] = replace(
        critic, terms={k: v for k, v in critic.terms.items() if not k.startswith("recover_")}
    )
    cfg.observations = observations
    if replay and not play:
        cfg.recorders = {
            **dict(cfg.recorders or {}),
            RECORDER_NAME: SuperMimicRecorderCfg(
                params={"capacity": ROUGH_SUPERMIMIC_REPLAY_CAPACITY}
            ),
        }
    return cfg


__all__ = [
    "ROUGH_SUPERMIMIC_DROPPED_CURRICULA",
    "ROUGH_SUPERMIMIC_DROPPED_EVENTS",
    "ROUGH_SUPERMIMIC_DROPPED_OBS_GROUPS",
    "ROUGH_SUPERMIMIC_REPLAY_CAPACITY",
    "env_cfg",
]
