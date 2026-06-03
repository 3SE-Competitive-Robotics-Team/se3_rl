"""42 维观测跳跃任务环境配置。"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from se3_train.tasks.jump.env_cfg import env_cfg as jump_env_cfg

from . import observations


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造 42 维观测跳跃 fine-tune 环境。"""
    cfg = jump_env_cfg(play=play)

    actor_terms = dict(cfg.observations["actor"].terms)
    actor_terms.update(
        {
            "base_lin_vel_actor": ObservationTermCfg(func=observations.base_lin_vel_actor_obs),
            "base_height_actor": ObservationTermCfg(
                func=observations.base_height_actor_obs,
                params={"sensor_name": "base_height_sensor"},
            ),
            "wheel_contact_forces_actor": ObservationTermCfg(
                func=observations.wheel_contact_force_actor_obs,
                params={"sensor_name": "wheel_sensor"},
            ),
            "wheel_height_actor": ObservationTermCfg(
                func=observations.wheel_height_actor_obs,
                params={"sensor_name": "wheel_height_sensor"},
            ),
            "leg_contact_forces_actor": ObservationTermCfg(
                func=observations.leg_contact_force_actor_obs,
                params={"sensor_name": "leg_contact_sensor"},
            ),
        }
    )
    cfg.observations["actor"] = replace(cfg.observations["actor"], terms=actor_terms)

    critic_terms = dict(cfg.observations["critic"].terms)
    critic_terms.update(actor_terms)
    cfg.observations["critic"] = replace(cfg.observations["critic"], terms=critic_terms)
    return cfg
