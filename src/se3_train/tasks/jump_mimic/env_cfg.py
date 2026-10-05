"""跳跃 mimic 任务环境配置（2026-10-01 用户定，J10 起为默认）。

单独一个跳跃策略：MLP、34 维观测（jump_phase 槽位给一维相位）、从头训、平地前进跳。以 Flat 基线为底：
- 执行链与 rough 对齐：膝气弹簧 300 N + 电机侧前馈补偿，腿部 T-N 包络按物理口径 ×0.8（平台 32 N·m）。
- 参考：无下蹲参考 jump_ref_v2_nocrouch_h022（站姿 0.22 m，flag 一到就蹬），目标离地间隙 0.20/0.30/0.40/0.50 m。
- 指令：vx ±1.5 m/s（保留 20% 零速回合）、yaw / pitch / roll 恒 0、站姿高度恒为参考站姿；跳跃触发与参考时钟见 commands.py，
  跳跃中 jump_phase = 参考时刻 / 1.5 s（J8 证明网络必须有时间信息，J10 证明一维相位可代替 20 维参考帧）。
- 奖励：Flat 原定价 + 腿长 / 机身高度 / 竖直速度 / 轮接触四项模仿奖励；Flat 的轮/腿离地罚、高度罚原本就按 jump_flag 屏蔽，
  另把静站罚与轮子大接触力罚在跳跃期间置零；跳跃期间速度跟踪核去掉 vz 项只看 vx（Flat 核含 vz，腾空时整项归零）。
- reset：全部从站姿开始（J9 证明不需要 RSI）；跳跃期间机身高度偏离参考超过 0.12 m 提前终止。
- 去掉速度课程（会自动放开 vx），保留推扰课程。
"""

from __future__ import annotations

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_train.mdp.commands import VelocityHeightCommandCfg
from se3_train.robot_cfg import get_serialleg_closedchain_cfg
from se3_train.tasks.common.no_attitude import apply_no_attitude_layout
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import mdp
from .commands import JumpMimicCommandCfg
from .reference import REFERENCE_DIR, JumpReferenceLibrary, reference_paths

JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE = 0.8
JUMP_MIMIC_PHASE_TIME_SCALE_S = 1.5
"""jump_phase = 触发后参考时刻 / 该常数；参考最长 1.46 s，相位约落在 0–0.97。用固定常数而不是各条参考时长归一化，
使同一相位值对应同一物理时刻（起蹬都在 0）。"""
JUMP_MIMIC_EPISODE_LENGTH_S = 10.0
JUMP_MIMIC_MAX_HEIGHT_ERROR = 0.12
"""跳跃期间机身高度偏离参考的提前终止阈值（J2 起；J1 的 0.25 m 大于低参考的最高点，不起跳也不触发）。"""
JUMP_MIMIC_MAX_LIN_VEL_X = 1.5
"""前进跳 vx 指令包络 ±1.5 m/s（Flat 部署上限 2.4 的约 60%，J3 起）。"""
JUMP_MIMIC_MOVING_STANDING_RATIO = 0.2
"""放开 vx 后仍保留 20% 零速回合，原地跳不丢。"""
JUMP_MIMIC_REFERENCE_HEIGHTS = (0.20, 0.30, 0.40, 0.50)
"""参考的目标轮底离地间隙（m），触发时均匀选一条；0.50 m 起跳 2.62 m/s、蹬地 20.7 m/s²，准静态力矩利用率峰值 0.72。"""
JUMP_MIMIC_REWARD_WEIGHTS = {
    "mimic_leg_length": 3.0,
    "mimic_base_height": 3.0,
    "mimic_base_vz": 1.5,
    "mimic_contact": 1.0,
}


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """跳跃 mimic 环境（各定价与依据见模块 docstring）。"""
    vx_range = (-JUMP_MIMIC_MAX_LIN_VEL_X, JUMP_MIMIC_MAX_LIN_VEL_X)
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )
    paths = reference_paths(JUMP_MIMIC_REFERENCE_HEIGHTS, REFERENCE_DIR)
    library = JumpReferenceLibrary(paths, "cpu")
    stand = library.stand_height

    # 执行链：与 rough 对齐
    cfg.scene.entities = {
        "robot": get_serialleg_closedchain_cfg(
            leg_torque_envelope_scale=JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE
        )
    }
    action = cfg.actions["delayed_action"]
    if action.knee_gas_spring_force <= 0.0:
        raise ValueError("跳跃 mimic 需要正的气弹簧前馈补偿力")
    action.knee_gas_spring_compensation_enabled = True

    # 指令：前进跳、固定站姿高度 + 跳跃触发
    base = cfg.commands["velocity_height"]
    kwargs = {f.name: getattr(base, f.name) for f in fields(VelocityHeightCommandCfg) if f.init}
    kwargs.update(
        lin_vel_x_range=vx_range,
        ang_vel_yaw_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        roll_range=(0.0, 0.0),
        height_range=(stand, stand),
        standing_height_range=(stand, stand),
        standing_ratio=JUMP_MIMIC_MOVING_STANDING_RATIO,
        deployment_ranges={
            "lin_vel_x": vx_range,
            "ang_vel_yaw": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "height": (stand, stand),
            "jump_flag": (0.0, 1.0),
            "jump_target_height": (0.0, float(library.target_clearance.max())),
            "jump_phase": (0.0, float(library.duration.max()) / JUMP_MIMIC_PHASE_TIME_SCALE_S),
        },
    )
    cfg.commands = {
        "velocity_height": JumpMimicCommandCfg(
            reference_paths=paths, phase_time_scale_s=JUMP_MIMIC_PHASE_TIME_SCALE_S, **kwargs
        )
    }
    # 观测 30 维、部署指令六维（去掉 pitch / roll 与 wheel_pos_zero，见 tasks.common.no_attitude）
    apply_no_attitude_layout(cfg)

    # 奖励
    cfg.rewards = dict(cfg.rewards)
    for name in ("stand_still", "contact_forces"):
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=mdp.not_jumping,
            weight=float(term.weight),
            params={"inner": term.func, "params": dict(term.params or {})},
        )
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
        params={"max_height_error": JUMP_MIMIC_MAX_HEIGHT_ERROR},
    )

    # reset：从参考站姿开始，替代 Flat 的根状态 / 关节随机 reset（后者把髋随机到 ±90°）
    cfg.events = dict(cfg.events)
    cfg.events.pop("reset_root_state", None)
    cfg.events.pop("reset_joints", None)
    cfg.events["reset_jump_mimic"] = EventTermCfg(func=mdp.reset_jump_mimic, mode="reset")

    # 课程：去掉速度课程，保留推扰
    if cfg.curriculum:
        cfg.curriculum = {k: v for k, v in cfg.curriculum.items() if k != "command_vel"}
    if not play:
        cfg.episode_length_s = JUMP_MIMIC_EPISODE_LENGTH_S
    return cfg


__all__ = [
    "JUMP_MIMIC_EPISODE_LENGTH_S",
    "JUMP_MIMIC_LEG_TORQUE_ENVELOPE_SCALE",
    "JUMP_MIMIC_MAX_HEIGHT_ERROR",
    "JUMP_MIMIC_MAX_LIN_VEL_X",
    "JUMP_MIMIC_MOVING_STANDING_RATIO",
    "JUMP_MIMIC_PHASE_TIME_SCALE_S",
    "JUMP_MIMIC_REFERENCE_HEIGHTS",
    "JUMP_MIMIC_REWARD_WEIGHTS",
    "env_cfg",
]
