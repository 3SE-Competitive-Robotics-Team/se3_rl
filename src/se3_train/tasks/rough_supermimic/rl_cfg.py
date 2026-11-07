"""Rough-SuperMimic 的 runner 配置：Rough MLP 的 runner 配置（actor 128/64/32、轮数、保存间隔、bench）+ SuperMimic 超参。

PPO 超参仍在（runner 用 PPO 构造 actor 与 checkpoint 格式），但 SuperMimic 入口不用它们更新；
PPO 对照入口直接用 `rough.rl_cfg.rl_cfg()`。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg
from se3_train.supermimic.config import SuperMimicCfg
from se3_train.tasks.rough.rl_cfg import rl_cfg as rough_rl_cfg


@dataclass
class SuperMimicRunnerCfg(RslRlOnPolicyRunnerCfg):
    """在 SE3 runner 配置上加一个 `supermimic` 字段。"""

    supermimic: SuperMimicCfg = field(default_factory=SuperMimicCfg)


def supermimic_rl_cfg(smoke: bool = False) -> SuperMimicRunnerCfg:
    """Rough MLP runner 配置原样搬过来，每轮 env step 数与 SuperMimic 采集节奏对齐。"""
    base = rough_rl_cfg(smoke=smoke)
    cfg = SuperMimicRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
    cfg.supermimic.schedule.steps_per_iteration = int(cfg.num_steps_per_env)
    return cfg


__all__ = ["SuperMimicRunnerCfg", "supermimic_rl_cfg"]
