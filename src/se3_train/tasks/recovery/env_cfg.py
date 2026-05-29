"""倒地自启任务环境配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import events, rewards


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """统一全姿态随机的反倒自起训练环境。"""

    cfg = flat_env_cfg(play=play)

    cfg.events["reset_root_state"] = EventTermCfg(
        func=events.reset_root_state_full_angle_random,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "tilt_range": (0.0, 3.141592653589793),
            "tilt_axis_range": (-3.141592653589793, 3.141592653589793),
            "yaw_range": (-3.141592653589793, 3.141592653589793),
            "height_range": (0.26, 0.36),
            "clearance_range": (0.02, 0.05),
            "lin_vel_range": (-0.15, 0.15),
            "ang_vel_range": (-0.8, 0.8),
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.25,
            "joint_vel_range": (-0.8, 0.8),
        },
    )

    # 自起训练不再区分 normal/fallen/recovery episode，所有样本只受超时等硬错误终止约束。
    cfg.terminations.pop("bad_orientation", None)
    cfg.terminations.pop("leg_contact", None)
    cfg.terminations.pop("recovery_stagnation", None)
    cfg.events.pop("push_robots", None)
    if cfg.curriculum is not None:
        cfg.curriculum.pop("push_disturbance", None)

    # 去掉轴向姿态惩罚和旧 recovery 专属奖励；自起能力由 upward + 同一套 locomotion reward 学出来。
    for reward_name in (
        "tracking_orientation_l2",
        "bad_tilt",
        "flat_base_height",
        "flat_base_lin_vel_z",
        "flat_wheel_ground_slip",
        "flat_wheel_center_alignment",
        "is_alive",
    ):
        cfg.rewards.pop(reward_name, None)

    # 对齐 robot_lab 的自起训练比例:upward 是主目标,动作平滑只保留很轻的正则。
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=3.0)
    cfg.rewards["action_rate"] = RewardTermCfg(func=rewards.action_rate, weight=-0.01)
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=2.49,
        params={
            "command_name": "velocity_height",
            "sigma": 0.05,
            "height_sensor_name": "base_height_sensor",
            "use_upright_gate": True,
        },
    )

    return cfg
