"""SE3 轮腿机器人训练环境。"""

from mjlab.tasks.registry import register_mjlab_task
from swanlab_rsl_rl import patch_rsl_rl_logger

from .env_cfg import se3_flat_env_cfg, se3_rough_env_cfg
from .rl_cfg import se3_ppo_runner_cfg

# 应用 monkey patch 以支持 SwanLab
patch_rsl_rl_logger()

register_mjlab_task(
    task_id="SE3-WheelLegged-Flat",
    env_cfg=se3_flat_env_cfg(),
    play_env_cfg=se3_flat_env_cfg(play=True),
    rl_cfg=se3_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="SE3-WheelLegged-Rough",
    env_cfg=se3_rough_env_cfg(),
    play_env_cfg=se3_rough_env_cfg(play=True),
    rl_cfg=se3_ppo_runner_cfg(),
)
