"""SE3 轮腿机器人训练环境。"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    se3_flat_env_cfg,
    se3_jump_env_cfg,
    se3_jump_pretrain_env_cfg,
    se3_rough_env_cfg,
)
from .rl_cfg import (
    se3_gru_ppo_runner_cfg,
    se3_jump_gru_ppo_runner_cfg,
    se3_jump_pretrain_gru_ppo_runner_cfg,
    se3_ppo_runner_cfg,
)
from .runner import Se3WarmStartRunner

register_mjlab_task(
    task_id="SE3-WheelLegged-Rough",
    env_cfg=se3_rough_env_cfg(),
    play_env_cfg=se3_rough_env_cfg(play=True),
    rl_cfg=se3_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="SE3-WheelLegged-Flat-GRU",
    env_cfg=se3_flat_env_cfg(),
    play_env_cfg=se3_flat_env_cfg(play=True),
    rl_cfg=se3_gru_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="SE3-WheelLegged-Jump-PreTrain-GRU",
    env_cfg=se3_jump_pretrain_env_cfg(),
    play_env_cfg=se3_jump_pretrain_env_cfg(play=True),
    rl_cfg=se3_jump_pretrain_gru_ppo_runner_cfg(),
    runner_cls=Se3WarmStartRunner,
)

register_mjlab_task(
    task_id="SE3-WheelLegged-Jump-GRU",
    env_cfg=se3_jump_env_cfg(),
    play_env_cfg=se3_jump_env_cfg(play=True),
    rl_cfg=se3_jump_gru_ppo_runner_cfg(),
    runner_cls=Se3WarmStartRunner,
)
