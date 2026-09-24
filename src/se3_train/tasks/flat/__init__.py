"""平地行走任务：仅注册 D11 单帧 MLP 基线。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import FLAT_ACTION_SMOOTHNESS_SPRING, FLAT_WHEEL_ACTION_SCALE, env_cfg
from .rl_cfg import mlp_rl_cfg

TASK_ID = "SE3-WheelLegged-Flat-MLP"


def register() -> None:
    """注册保留的 Flat-MLP，沿用原有动作与奖励配置。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = ["TASK_ID", "register"]
