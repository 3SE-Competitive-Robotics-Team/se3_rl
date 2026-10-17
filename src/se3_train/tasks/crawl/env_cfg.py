"""趴地移动任务环境配置：以冻结的 Flat 基线为底座，只覆盖趴地相关的差异。

与 Flat 的差异（其余全部沿用，包括 34 维 actor 观测、动作契约、奖励权重与域随机化）：
- 机器人换成带 6 个被动底盘导轮的变体 MJCF；碰撞 geom 挪到 group 3，高度射线只看地形。
- 高度指令固定为趴地高度（导轮着地时 base_link 原点离地高度）。
- 出生即趴姿（reset_crawl_pose），不再用站立姿态 ±90° hip 随机化。
- 删除 stand_still：它把零指令时的腿往站立默认姿态拉。
- 速度课程终值 vx 1.5 m/s、yaw 3 rad/s：导轮横向不打滑，原地转向要靠主轮侧滑，先给保守包络。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.robot_cfg import get_serialleg_closedchain_cfg
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import events

# 导轮轴心 z = -0.1259、导轮半径 0.010（见 MJCF），机身水平且导轮着地时 base_link 原点离地 0.1359 m。
CRAWL_BASE_HEIGHT = 0.1359
CRAWL_MAX_LIN_VEL_X = 1.5
CRAWL_MAX_ANG_VEL_YAW = 3.0
# mjlab 资产库约定：碰撞 geom group 3，射线传感器 include_geom_groups=(0,) 只看地形。
# 趴地时 critic 高度环（半径 0.15 m）会扫到导轮、主轮和腿，必须挪开。
CRAWL_ROBOT_COLLISION_GEOM_GROUP = 3


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造趴地移动任务配置。"""
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        max_ang_vel_yaw=CRAWL_MAX_ANG_VEL_YAW,
    )

    cfg.scene.entities["robot"] = get_serialleg_closedchain_cfg(
        collision_geom_group=CRAWL_ROBOT_COLLISION_GEOM_GROUP,
        guide_wheels=True,
    )

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.height_range = (CRAWL_BASE_HEIGHT, CRAWL_BASE_HEIGHT)
    command_cfg.standing_height_range = (CRAWL_BASE_HEIGHT, CRAWL_BASE_HEIGHT)
    command_cfg.deployment_ranges = {
        **command_cfg.deployment_ranges,
        "lin_vel_x": (-CRAWL_MAX_LIN_VEL_X, CRAWL_MAX_LIN_VEL_X),
        "height": (CRAWL_BASE_HEIGHT, CRAWL_BASE_HEIGHT),
    }

    del cfg.rewards["stand_still"]

    if not play:
        cfg.curriculum["command_vel"].params["max_lin_vel_x"] = CRAWL_MAX_LIN_VEL_X

    cfg.events["reset_joints"] = EventTermCfg(
        func=events.reset_crawl_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    return cfg
