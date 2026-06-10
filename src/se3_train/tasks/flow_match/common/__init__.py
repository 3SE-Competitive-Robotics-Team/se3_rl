"""FlowMatch task 组共享配置。"""

from __future__ import annotations

from .env import (
    apply_loco_task_mode_rewards,
    apply_task_mode_command,
    apply_task_mode_observations,
    apply_task_mode_rewards,
    loco_env_cfg,
    loco_light_terrain_cfg,
    loco_light_terrain_env_cfg,
    loco_script_play_env_cfg,
    single_label_env_cfg,
    task_mode_env_cfg,
    wheel_env_cfg,
)
from .rl_cfg import single_label_gru_ppo_runner_cfg

__all__ = [
    "apply_loco_task_mode_rewards",
    "apply_task_mode_command",
    "apply_task_mode_observations",
    "apply_task_mode_rewards",
    "loco_env_cfg",
    "loco_light_terrain_cfg",
    "loco_light_terrain_env_cfg",
    "loco_script_play_env_cfg",
    "single_label_env_cfg",
    "single_label_gru_ppo_runner_cfg",
    "task_mode_env_cfg",
    "wheel_env_cfg",
]
