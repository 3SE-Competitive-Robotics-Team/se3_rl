"""跳跃 mimic 任务环境配置（2026-10-01 用户定）。

单独一个跳跃策略：MLP、单帧本体 34 维 + 参考帧 20 维、从头训、平地原地跳。以 Flat 基线为底：
- 执行链与 rough M54 对齐：膝气弹簧 300 N + 电机侧前馈补偿，腿部 T-N 包络按物理口径 ×0.8（平台 32 N·m）。
- 指令：vx / yaw / pitch / roll 恒 0、站姿高度恒为参考站姿（0.28 m）；跳跃触发与参考时钟见 commands.py，
  jump_phase 恒 0（不把相位输入网络）。
- 奖励：Flat 原定价 + 腿长 / 机身高度 / 竖直速度 / 轮接触四项模仿奖励；Flat 的速度跟踪、轮/腿离地罚、高度罚
  原本就按 jump_flag 屏蔽，另把静站罚与轮子大接触力罚在跳跃期间置零。
- reset：rsi_prob 的回合从参考随机时刻开始（RSI），其余从站姿开始；跳跃期间偏离参考过大提前终止。
- 去掉速度课程（会自动放开 vx），保留推扰课程。
"""

from __future__ import annotations

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_train.mdp.commands import VelocityHeightCommandCfg
from se3_train.robot_cfg import get_serialleg_closedchain_cfg
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import mdp
from .commands import JumpMimicCommandCfg
from .reference import DEFAULT_REFERENCE_PATHS, JumpReferenceLibrary

JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE = 0.8
JUMP_MIMIC_RSI_PROB = 0.5
JUMP_MIMIC_EPISODE_LENGTH_S = 10.0
JUMP_MIMIC_REWARD_WEIGHTS = {
    "mimic_leg_length": 3.0,
    "mimic_base_height": 3.0,
    "mimic_base_vz": 1.5,
    "mimic_contact": 1.0,
}


def _stand_height() -> float:
    return JumpReferenceLibrary(DEFAULT_REFERENCE_PATHS, "cpu").stand_height


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )
    stand = _stand_height()

    # 执行链：与 M54 对齐
    cfg.scene.entities = {
        "robot": get_serialleg_closedchain_cfg(
            leg_torque_envelope_scale=JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE
        )
    }
    action = cfg.actions["delayed_action"]
    if action.knee_gas_spring_force <= 0.0:
        raise ValueError("跳跃 mimic 需要正的气弹簧前馈补偿力")
    action.knee_gas_spring_compensation_enabled = True

    # 指令：原地、固定站姿高度 + 跳跃触发
    base = cfg.commands["velocity_height"]
    kwargs = {f.name: getattr(base, f.name) for f in fields(VelocityHeightCommandCfg) if f.init}
    kwargs.update(
        lin_vel_x_range=(0.0, 0.0),
        ang_vel_yaw_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        roll_range=(0.0, 0.0),
        height_range=(stand, stand),
        standing_height_range=(stand, stand),
        standing_ratio=1.0,
        deployment_ranges={
            "lin_vel_x": (0.0, 0.0),
            "ang_vel_yaw": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "height": (stand, stand),
            "jump_flag": (0.0, 1.0),
            "jump_target_height": (0.0, 0.40),
            "jump_phase": (0.0, 0.0),
        },
    )
    cfg.commands = {"velocity_height": JumpMimicCommandCfg(**kwargs)}

    # 观测：actor 34 维本体 + 参考帧；critic 再加参考时钟
    cfg.observations = dict(cfg.observations)
    actor = cfg.observations["actor"]
    actor_terms = dict(actor.terms)
    actor_terms["jump_reference"] = ObservationTermCfg(func=mdp.jump_reference_obs)
    cfg.observations["actor"] = replace(actor, terms=actor_terms)
    critic = cfg.observations["critic"]
    critic_terms = dict(critic.terms)
    critic_terms["jump_reference"] = ObservationTermCfg(func=mdp.jump_reference_obs)
    critic_terms["jump_reference_state"] = ObservationTermCfg(func=mdp.jump_reference_state_obs)
    cfg.observations["critic"] = replace(critic, terms=critic_terms)

    # 奖励
    cfg.rewards = dict(cfg.rewards)
    for name in ("stand_still", "contact_forces"):
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=mdp.not_jumping,
            weight=float(term.weight),
            params={"inner": term.func, "params": dict(term.params or {})},
        )
    for name, weight in JUMP_MIMIC_REWARD_WEIGHTS.items():
        cfg.rewards[name] = RewardTermCfg(func=getattr(mdp, name), weight=float(weight))

    # 终止：跳跃期间偏离参考过大
    cfg.terminations = dict(cfg.terminations)
    cfg.terminations["mimic_deviation"] = TerminationTermCfg(
        func=mdp.mimic_deviation, time_out=False
    )

    # reset：参考状态初始化替代 Flat 的根状态 / 关节随机 reset（后者把髋随机到 ±90°）
    cfg.events = dict(cfg.events)
    cfg.events.pop("reset_root_state", None)
    cfg.events.pop("reset_joints", None)
    cfg.events["reset_jump_mimic"] = EventTermCfg(
        func=mdp.reset_jump_mimic,
        mode="reset",
        params={"rsi_prob": 0.0 if play else JUMP_MIMIC_RSI_PROB},
    )

    # 课程：去掉速度课程，保留推扰
    if cfg.curriculum:
        cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k != "command_vel"}
    if not play:
        cfg.episode_length_s = JUMP_MIMIC_EPISODE_LENGTH_S
    return cfg


__all__ = [
    "JUMP_MIMIC_EPISODE_LENGTH_S",
    "JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE",
    "JUMP_MIMIC_REWARD_WEIGHTS",
    "JUMP_MIMIC_RSI_PROB",
    "env_cfg",
]
