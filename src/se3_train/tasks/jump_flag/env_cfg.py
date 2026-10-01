"""跳跃 flag 任务环境配置（J5，2026-10-01 用户定）。

单独一个跳跃策略：MLP、单帧 34 维本体（与 rough 部署契约同布局，不加参考帧、不给相位）、从头训、平地。
- 执行链与 J4 / M54 对齐：膝气弹簧 300 N + 电机侧前馈补偿，腿部 T-N 包络 ×0.8（平台 32 N·m）。
- 指令：vx ±2.4（保留 20% 零速）、yaw / pitch / roll 恒 0、站姿高度 0.20–0.24（低站姿留足蹬地行程，
  用户定）；站满 1 s 后每秒 0.5 概率触发，jump_flag 置 1 保持 0.5 s 再置 0，目标轮底离地间隙 0.20–0.50 连续。
- 奖励：Flat 原定价 + 按状态触发的跳跃项（上升速度、离地、收腿、触地时一次性最高点奖励、漏跳一次性罚、
  二次起跳罚）；跳跃事件期间（触发 → 落地稳住）屏蔽轮子离地罚、高度罚、静站罚、接触力罚与速度跟踪的 vz 项。
- reset：Flat 根状态 + 小幅关节扰动（不用 Flat 的 ±90° 髋随机化）；去掉速度课程（vx 包络固定），保留推扰课程。
"""

from __future__ import annotations

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp import events as mdp_events
from se3_train.mdp.commands import VelocityHeightCommandCfg
from se3_train.robot_cfg import get_serialleg_closedchain_cfg
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import mdp
from .commands import JumpFlagCommandCfg

JUMP_FLAG_LEG_TORQUE_ENVELOPE_SCALE = 0.8
JUMP_FLAG_EPISODE_LENGTH_S = 10.0
JUMP_FLAG_MAX_LIN_VEL_X = 2.4
JUMP_FLAG_STANDING_RATIO = 0.2
JUMP_FLAG_HEIGHT_RANGE = (0.20, 0.24)
"""站姿高度指令（用户定）：低站姿到起跳高度（约 0.374 m）留 0.13–0.17 m 蹬地行程，直接蹬即可跳到 0.40–0.50 m。"""
JUMP_FLAG_TARGET_CLEARANCE_RANGE = (0.20, 0.50)
JUMP_FLAG_WINDOW_S = 0.5
JUMP_FLAG_RESET_JOINT_OFFSET = 0.1
JUMP_FLAG_REWARD_WEIGHTS = {
    "jump_upward_velocity": 3.0,
    "jump_airborne": 2.0,
    "jump_tuck": 1.0,
    "jump_rehop": -4.0,
}
"""按秒计的跳跃奖励权重。"""
JUMP_FLAG_APEX_BONUS = 10.0
"""触地时一次性最高点奖励的满分（按 step_dt 反算权重，同 rough 的摔倒罚写法）。"""
JUMP_FLAG_MISSED_PENALTY = 10.0
"""窗口结束仍未离地的一次性罚。"""
JUMP_FLAG_EVENT_MASKED_REWARDS = (
    "flat_wheel_contact",
    "flat_base_height",
    "stand_still",
    "contact_forces",
)
"""跳跃事件期间置零的行走项。"""


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )

    # 执行链：与 J4 / M54 对齐
    cfg.scene.entities = {
        "robot": get_serialleg_closedchain_cfg(
            leg_torque_envelope_scale=JUMP_FLAG_LEG_TORQUE_ENVELOPE_SCALE
        )
    }
    action = cfg.actions["delayed_action"]
    if action.knee_gas_spring_force <= 0.0:
        raise ValueError("跳跃 flag 任务需要正的气弹簧前馈补偿力")
    action.knee_gas_spring_compensation_enabled = True

    # 指令
    base = cfg.commands["velocity_height"]
    kwargs = {f.name: getattr(base, f.name) for f in fields(VelocityHeightCommandCfg) if f.init}
    vx = (-JUMP_FLAG_MAX_LIN_VEL_X, JUMP_FLAG_MAX_LIN_VEL_X)
    kwargs.update(
        lin_vel_x_range=vx,
        ang_vel_yaw_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        roll_range=(0.0, 0.0),
        height_range=JUMP_FLAG_HEIGHT_RANGE,
        standing_height_range=JUMP_FLAG_HEIGHT_RANGE,
        standing_ratio=JUMP_FLAG_STANDING_RATIO,
        deployment_ranges={
            "lin_vel_x": vx,
            "ang_vel_yaw": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "height": JUMP_FLAG_HEIGHT_RANGE,
            "jump_flag": (0.0, 1.0),
            "jump_target_height": (0.0, JUMP_FLAG_TARGET_CLEARANCE_RANGE[1]),
            "jump_phase": (0.0, 0.0),
        },
    )
    cfg.commands = {
        "velocity_height": JumpFlagCommandCfg(
            window_s=JUMP_FLAG_WINDOW_S,
            target_clearance_range=JUMP_FLAG_TARGET_CLEARANCE_RANGE,
            **kwargs,
        )
    }

    # 观测：actor 维持 34 维；critic 加跳跃事件状态
    cfg.observations = dict(cfg.observations)
    critic = cfg.observations["critic"]
    critic_terms = dict(critic.terms)
    critic_terms["jump_event_state"] = ObservationTermCfg(func=mdp.jump_event_state_obs)
    cfg.observations["critic"] = replace(critic, terms=critic_terms)

    # 奖励
    cfg.rewards = dict(cfg.rewards)
    for name in JUMP_FLAG_EVENT_MASKED_REWARDS:
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=mdp.outside_jump_event,
            weight=float(term.weight),
            params={"inner": term.func, "params": dict(term.params or {})},
        )
    track = cfg.rewards["tracking_lin_vel"]
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=mdp.tracking_lin_vel_jump_event, weight=float(track.weight), params=dict(track.params)
    )
    for name, weight in JUMP_FLAG_REWARD_WEIGHTS.items():
        cfg.rewards[name] = RewardTermCfg(func=getattr(mdp, name), weight=float(weight))
    step_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    cfg.rewards["jump_apex_height"] = RewardTermCfg(
        func=mdp.jump_apex_height, weight=JUMP_FLAG_APEX_BONUS / step_dt
    )
    cfg.rewards["jump_missed"] = RewardTermCfg(
        func=mdp.jump_missed, weight=-JUMP_FLAG_MISSED_PENALTY / step_dt
    )

    # reset：Flat 根状态 + 小幅关节扰动（Flat 默认把髋随机到 ±90°）
    cfg.events = dict(cfg.events)
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "align_root_height_to_wheels": True,
            "wheel_clearance": 0.001,
            "joint_offset_range": JUMP_FLAG_RESET_JOINT_OFFSET,
        },
    )

    # 课程：去掉速度课程（vx 包络固定），保留推扰
    if cfg.curriculum:
        cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k != "command_vel"}
    if not play:
        cfg.episode_length_s = JUMP_FLAG_EPISODE_LENGTH_S
    return cfg


__all__ = [
    "JUMP_FLAG_APEX_BONUS",
    "JUMP_FLAG_EPISODE_LENGTH_S",
    "JUMP_FLAG_EVENT_MASKED_REWARDS",
    "JUMP_FLAG_HEIGHT_RANGE",
    "JUMP_FLAG_LEG_TORQUE_ENVELOPE_SCALE",
    "JUMP_FLAG_MAX_LIN_VEL_X",
    "JUMP_FLAG_MISSED_PENALTY",
    "JUMP_FLAG_RESET_JOINT_OFFSET",
    "JUMP_FLAG_REWARD_WEIGHTS",
    "JUMP_FLAG_STANDING_RATIO",
    "JUMP_FLAG_TARGET_CLEARANCE_RANGE",
    "JUMP_FLAG_WINDOW_S",
    "env_cfg",
]
