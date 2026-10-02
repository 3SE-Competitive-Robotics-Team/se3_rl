"""跳跃 mimic 任务环境配置（2026-10-01 用户定）。

单独一个跳跃策略：MLP、单帧本体 34 维 + 参考帧 20 维、从头训、平地跳。以 Flat 基线为底：
- 执行链与 rough M54 对齐：膝气弹簧 300 N + 电机侧前馈补偿，腿部 T-N 包络按物理口径 ×0.8（平台 32 N·m）。
- 指令：yaw / pitch / roll 恒 0（vx 默认恒 0，J3 起放开做前进跳）、站姿高度恒为参考站姿（0.28 m）；跳跃触发与参考时钟见 commands.py，
  jump_phase 恒 0（不把相位输入网络）。
- 奖励：Flat 原定价 + 腿长 / 机身高度 / 竖直速度 / 轮接触四项模仿奖励；Flat 的轮/腿离地罚、高度罚
  原本就按 jump_flag 屏蔽，另把静站罚与轮子大接触力罚在跳跃期间置零。
- reset：rsi_prob 的回合从参考随机时刻开始（RSI），其余从站姿开始；跳跃期间偏离参考过大提前终止。
- 前进跳（max_lin_vel_x > 0，J3 起）：跳跃中速度 / 姿态指令冻结；速度跟踪核在跳跃期间去掉 vz 项只看 vx
  （Flat 核含 vz，腾空时整项归零）；RSI 回合按 vx 指令给机身水平速度与无滑轮速。
- 去掉速度课程（会自动放开 vx），保留推扰课程。
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

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
from .reference import (
    DEFAULT_REFERENCE_HEIGHTS,
    REFERENCE_DIR,
    JumpReferenceLibrary,
    reference_paths,
)

JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE = 0.8
JUMP_MIMIC_RSI_PROB = 0.5
JUMP_MIMIC_EPISODE_LENGTH_S = 10.0
JUMP_MIMIC_MAX_HEIGHT_ERROR = 0.25
"""J1 机身高度偏离终止阈值；0.25 m 大于 0.20/0.30 参考的最高点（不起跳也不触发），J2 收紧到 0.12 m。"""
JUMP_MIMIC_J2_MAX_HEIGHT_ERROR = 0.12
JUMP_MIMIC_J3_MAX_LIN_VEL_X = 1.5
"""J3 前进跳 vx 指令包络 ±1.5 m/s（Flat 部署上限 2.4 的约 60%，先在中速段学会跑着跳）。"""
JUMP_MIMIC_MOVING_STANDING_RATIO = 0.2
"""放开 vx 后仍保留 20% 零速回合，原地跳不丢。"""
JUMP_MIMIC_J4_REFERENCE_HEIGHTS = (0.20, 0.30, 0.40, 0.50)
"""J4 在 J1–J3 三条参考上加 0.50 m（起跳 2.62 m/s、蹬地 20.7 m/s²，准静态力矩利用率峰值 0.72）。"""
JUMP_MIMIC_REWARD_WEIGHTS = {
    "mimic_leg_length": 3.0,
    "mimic_base_height": 3.0,
    "mimic_base_vz": 1.5,
    "mimic_contact": 1.0,
}


def _stand_height(paths: tuple[str, ...]) -> float:
    return JumpReferenceLibrary(paths, "cpu").stand_height


def env_cfg(
    play: bool = False,
    max_height_error: float = JUMP_MIMIC_MAX_HEIGHT_ERROR,
    max_lin_vel_x: float = 0.0,
    reference_heights: tuple[float, ...] = DEFAULT_REFERENCE_HEIGHTS,
    reference_dir: Path = REFERENCE_DIR,
    reference_obs: bool = True,
    rsi_prob: float = JUMP_MIMIC_RSI_PROB,
) -> ManagerBasedRlEnvCfg:
    """跳跃 mimic 环境。

    max_height_error：跳跃期间机身高度偏离参考的提前终止阈值（m）。
    max_lin_vel_x：vx 指令包络（m/s）；0 = 原地跳（J1/J2）。大于 0 时为前进跳：vx 在 ±max 内均匀采样
    （保留 JUMP_MIMIC_MOVING_STANDING_RATIO 的零速回合），跳跃期间速度跟踪只看 vx，RSI 带指令速度。
    reference_heights：参考轨迹的目标离地间隙（m），触发时均匀选一条；部署包络 jump_target_height 上界取其最大值。
    reference_dir：参考轨迹目录（默认 jump_ref_v1）；站姿高度指令取参考的站姿，J7 换成无下蹲参考（站姿 0.22 m）。
    reference_obs：观测里是否有参考信息。False（J8，用户定）时 actor 与 critic 都不看 20 维参考帧，critic 也不看参考时钟 /
    参考编号，actor 只有 34 维本体（含 jump_flag / 目标高度），是 POMDP；模仿奖励、偏离终止、RSI 不变。
    rsi_prob：回合从参考随机时刻开始（RSI）的比例；J9 置 0（所有回合从站姿开始，跳跃只能由触发进入）。
    """
    moving = float(max_lin_vel_x) > 0.0
    vx_range = (-float(max_lin_vel_x), float(max_lin_vel_x)) if moving else (0.0, 0.0)
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )
    paths = reference_paths(tuple(reference_heights), reference_dir)
    stand = _stand_height(paths)

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
        lin_vel_x_range=vx_range,
        ang_vel_yaw_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        roll_range=(0.0, 0.0),
        height_range=(stand, stand),
        standing_height_range=(stand, stand),
        standing_ratio=JUMP_MIMIC_MOVING_STANDING_RATIO if moving else 1.0,
        deployment_ranges={
            "lin_vel_x": vx_range,
            "ang_vel_yaw": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "height": (stand, stand),
            "jump_flag": (0.0, 1.0),
            "jump_target_height": (0.0, float(max(reference_heights))),
            "jump_phase": (0.0, 0.0),
        },
    )
    cfg.commands = {"velocity_height": JumpMimicCommandCfg(reference_paths=paths, **kwargs)}

    # 观测：actor 34 维本体 + 参考帧；critic 再加参考时钟
    cfg.observations = dict(cfg.observations)
    actor = cfg.observations["actor"]
    actor_terms = dict(actor.terms)
    if reference_obs:
        actor_terms["jump_reference"] = ObservationTermCfg(func=mdp.jump_reference_obs)
    cfg.observations["actor"] = replace(actor, terms=actor_terms)
    critic = cfg.observations["critic"]
    critic_terms = dict(critic.terms)
    if reference_obs:
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
    if moving:
        term = cfg.rewards["tracking_lin_vel"]
        cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
            func=mdp.tracking_lin_vel_jump, weight=float(term.weight), params=dict(term.params)
        )
    for name, weight in JUMP_MIMIC_REWARD_WEIGHTS.items():
        cfg.rewards[name] = RewardTermCfg(func=getattr(mdp, name), weight=float(weight))

    # 终止：跳跃期间偏离参考过大
    cfg.terminations = dict(cfg.terminations)
    cfg.terminations["mimic_deviation"] = TerminationTermCfg(
        func=mdp.mimic_deviation,
        time_out=False,
        params={"max_height_error": float(max_height_error)},
    )

    # reset：参考状态初始化替代 Flat 的根状态 / 关节随机 reset（后者把髋随机到 ±90°）
    cfg.events = dict(cfg.events)
    cfg.events.pop("reset_root_state", None)
    cfg.events.pop("reset_joints", None)
    cfg.events["reset_jump_mimic"] = EventTermCfg(
        func=mdp.reset_jump_mimic,
        mode="reset",
        params={"rsi_prob": 0.0 if play else float(rsi_prob)},
    )

    # 课程：去掉速度课程，保留推扰
    if cfg.curriculum:
        cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k != "command_vel"}
    if not play:
        cfg.episode_length_s = JUMP_MIMIC_EPISODE_LENGTH_S
    return cfg


__all__ = [
    "JUMP_MIMIC_EPISODE_LENGTH_S",
    "JUMP_MIMIC_J2_MAX_HEIGHT_ERROR",
    "JUMP_MIMIC_J3_MAX_LIN_VEL_X",
    "JUMP_MIMIC_J4_REFERENCE_HEIGHTS",
    "JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE",
    "JUMP_MIMIC_MAX_HEIGHT_ERROR",
    "JUMP_MIMIC_MOVING_STANDING_RATIO",
    "JUMP_MIMIC_REWARD_WEIGHTS",
    "JUMP_MIMIC_RSI_PROB",
    "env_cfg",
]
