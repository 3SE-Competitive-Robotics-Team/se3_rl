"""SuperMimic 训练超参（挂在 runner 配置的 `supermimic` 字段上，tyro 可从命令行逐项覆盖）。

默认值取论文 Table A.5（arXiv 2610.28480），出处与未指明事项的取舍见
docs/plan/rough_supermimic_20261106.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldModelCfg:
    """世界模型网络与 k 步窗口拟合。"""

    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    termination_hidden_dims: tuple[int, ...] = (256, 256)
    reward_hidden_dims: tuple[int, ...] = (512, 512)
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    learning_rate: float = 2.0e-4
    grad_clip_norm: float = 1.0
    termination_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0


@dataclass
class ImaginationCfg:
    """想象 rollout、actor BPTT 与 critic 拟合。"""

    horizon: int = 32
    batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_learning_rate: float = 2.0e-4
    actor_betas: tuple[float, float] = (0.9, 0.95)
    actor_grad_clip_norm: float = 1.0
    actor_grad_clip_value: float = 0.1
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_learning_rate: float = 2.0e-4
    critic_grad_clip_norm: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    observation_noise: bool = True


@dataclass
class ScheduleCfg:
    """采集与更新的节奏。一次 learning iteration = `steps_per_iteration` 个 env step。"""

    steps_per_iteration: int = 24
    """与 Rough PPO 每轮 24 步相同，使 max_iterations、save_interval、W&B 横轴与 PPO 对照一一对应。"""
    updates_per_env_step: float = 0.4
    """UTD：每个 env step 平均做几次（世界模型 + 策略）更新。"""
    warmup_env_steps: int = 40
    """首次更新前先采这么多步，保证缓存里有完整的 k 步窗口。"""
    action_noise_std: float = 0.1
    """采集时执行动作 = 策略动作 + N(0, std²)，只为拓宽世界模型拟合的动作分布。"""
    replay_capacity: int = 1000
    """缓存每 env 保留的步数。"""
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000


@dataclass
class SuperMimicCfg:
    """SuperMimic 全部超参。"""

    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    imagination: ImaginationCfg = field(default_factory=ImaginationCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)


def supermimic_cfg_from_dict(raw: dict | SuperMimicCfg | None) -> SuperMimicCfg:
    """runner 拿到的是 asdict 后的字典，这里还原成数据类（缺省项取默认值）。"""
    if raw is None:
        return SuperMimicCfg()
    if isinstance(raw, SuperMimicCfg):
        return raw

    def build(cls, values: dict | None):
        values = dict(values or {})
        kwargs = {}
        for name, value in values.items():
            if name not in cls.__dataclass_fields__:
                raise KeyError(f"{cls.__name__} 没有字段 {name}")
            kwargs[name] = tuple(value) if isinstance(value, list) else value
        return cls(**kwargs)

    return SuperMimicCfg(
        world_model=build(WorldModelCfg, raw.get("world_model")),
        imagination=build(ImaginationCfg, raw.get("imagination")),
        schedule=build(ScheduleCfg, raw.get("schedule")),
    )


__all__ = [
    "ImaginationCfg",
    "ScheduleCfg",
    "SuperMimicCfg",
    "WorldModelCfg",
    "supermimic_cfg_from_dict",
]
