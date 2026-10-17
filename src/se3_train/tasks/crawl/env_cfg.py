"""趴地移动任务环境配置：以冻结的 Flat 基线为底座，只覆盖趴地相关的差异。

与 Flat 的差异（其余全部沿用，包括 34 维 actor 观测、动作契约、奖励权重与域随机化）：
- 机器人换成带 6 个被动底盘导轮的变体 MJCF；碰撞 geom 挪到 group 3，高度射线只看地形。
- 高度指令固定为趴地高度（导轮着地时 base_link 原点离地高度）。
- 出生即趴姿（reset_crawl_pose），不再用站立姿态 ±90° hip 随机化。
- 删除 stand_still：它把零指令时的腿往站立默认姿态拉。
- 速度课程终值 vx 1.5 m/s；不给转向指令（yaw 恒 0，2026-10-17 用户定：趴地技能不需要转向），
  tracking_ang_vel 照常计酬即奖励保持航向；tracking_lin_yaw_joint 只在 |yaw 指令| ≥ 0.5 时生效，恒为 0，删除。
- 新增导轮承重罚 guide_wheel_support 与左右主轮对称罚 wheel_symmetry_offset（见 rewards.py 与下方权重注释），
  critic 额外看左右导轮接触力。
- 速度课程加导轮承重门控：导轮总承重 EMA 达标前不推进 vx 上限（见 curriculums.py）。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from se3_train.robot_cfg import get_serialleg_closedchain_cfg
from se3_train.tasks.flat import observations
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, events, rewards

# 导轮轴心 z = -0.1259、导轮半径 0.010（见 MJCF），机身水平且导轮着地时 base_link 原点离地 0.1359 m。
CRAWL_BASE_HEIGHT = 0.1359
CRAWL_MAX_LIN_VEL_X = 1.5
# 不给转向指令：课程 yaw 上限与部署包络都为 0。
CRAWL_MAX_ANG_VEL_YAW = 0.0
# mjlab 资产库约定：碰撞 geom group 3，射线传感器 include_geom_groups=(0,) 只看地形。
# 趴地时 critic 高度环（半径 0.15 m）会扫到导轮、主轮和腿，必须挪开。
CRAWL_ROBOT_COLLISION_GEOM_GROUP = 3
# 导轮承重罚（2026-10-17）：每侧导轮承重占整机重力低于 25% 的缺口归一化后左右平均，∈ [0, 1]。
# 第一版 model_3499 每侧约 2%，罚 ≈0.92 → −4.6/s，与 tracking_lin_vel（+4 满分）同量级；名义趴姿 ≥40%/侧，不扣分。
CRAWL_GUIDE_WHEEL_SUPPORT_WEIGHT = -5.0
CRAWL_GUIDE_WHEEL_SIDE_SHARE_TARGET = 0.25
# 左右主轮对称罚（2026-10-17）：机身系 Δx² + Δz²，定价沿用 Rough M49 的 −50/m²。
# model_3499（Δx 0.18、Δz 0.056 m）−1.8/s；model_500（Δx 0.35、Δz 0.07 m）−6.4/s；对称趴姿为 0。
CRAWL_WHEEL_SYMMETRY_WEIGHT = -50.0
# 速度课程门控（2026-10-17）：导轮总承重 EMA ≥ 0.5（每侧目标 25% 之和）后才推进 vx 上限，见 curriculums.py。
CRAWL_VEL_CURRICULUM_GUIDE_SHARE_THRESHOLD = 0.5


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
    del cfg.rewards["tracking_lin_yaw_joint"]

    guide_sensors = tuple(
        ContactSensorCfg(
            name=f"{side}_guide_wheel_sensor",
            primary=ContactMatch(
                mode="body",
                pattern=rf"^({side}_guide_wheel_[0-2]_Link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        )
        for side in ("l", "r")
    )
    cfg.scene.sensors = (*cfg.scene.sensors, *guide_sensors)
    for side in ("l", "r"):
        cfg.observations["critic"].terms[f"{side}_guide_wheel_contact_forces"] = ObservationTermCfg(
            func=observations.contact_force_norm_obs,
            params={"sensor_name": f"{side}_guide_wheel_sensor"},
        )
    cfg.rewards["guide_wheel_support"] = RewardTermCfg(
        func=rewards.guide_wheel_support,
        weight=CRAWL_GUIDE_WHEEL_SUPPORT_WEIGHT,
        params={
            "left_sensor_name": "l_guide_wheel_sensor",
            "right_sensor_name": "r_guide_wheel_sensor",
            "side_share_target": CRAWL_GUIDE_WHEEL_SIDE_SHARE_TARGET,
        },
    )
    cfg.rewards["wheel_symmetry_offset"] = RewardTermCfg(
        func=rewards.wheel_symmetry_offset,
        weight=CRAWL_WHEEL_SYMMETRY_WEIGHT,
    )

    if not play:
        # 速度课程改由导轮承重门控（见 curriculums.py），其余参数沿用 Flat。
        command_vel = cfg.curriculum["command_vel"]
        command_vel.func = curriculums.commands_vel_after_crawl_posture
        command_vel.params["max_lin_vel_x"] = CRAWL_MAX_LIN_VEL_X
        command_vel.params["guide_share_threshold"] = CRAWL_VEL_CURRICULUM_GUIDE_SHARE_THRESHOLD

    cfg.events["reset_joints"] = EventTermCfg(
        func=events.reset_crawl_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    return cfg
