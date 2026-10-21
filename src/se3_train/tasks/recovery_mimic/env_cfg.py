"""起身 mimic 任务环境配置（2026-10-21 用户定）：训练一个跟踪手调起身参考的教师，之后在 Rough 里倒地时监督主策略。

用奖励约束让主策略"柔和起身"会被绕过（策略选择付罚款换快速翻身：b3odjuk6 收到 s=1 后起身腿速仍 20 rad/s、
功率 1.3 kW），所以改成模仿明确的参考动作：参考由 se3_tools.recovery_reference 开环生成（两腿镜像同步转大腿），
R1（倒置·腿折叠）峰值腿速 4.4 rad/s、功率 233 W。

以 Flat 基线为底：
- 执行链与 Rough 完全一致（60/3/0.2、腿 T-N 包络 ×0.8、轮 rated_point 包络、轮力矩滞后、气弹簧前馈、碰撞组），
  动作空间相同，教师输出可直接监督主策略。
- 观测：Rough 的 30 维本体观测（tasks.common.no_attitude）+ 9 维参考信息（相位、参考重力投影、腿角误差、参考高度），
  后者是教师特权观测，蒸馏时在训练端算。指令恒为零速、高度 0.22（参考不看指令）。
- 奖励：以跟踪参考为主（姿态、腿角、机身高度、腿速、机身角速度），保留 Flat 的动作平滑与力矩 / 功率正则；
  去掉速度 / 姿态 / 高度 / 接触等与参考冲突的项。
- 终止：超时（参考时长 + 0.5 s）、数值爆炸（不查腿角偏离默认）、姿态偏离参考 > 60°（出生 0.3 s 后）；
  去掉 Flat 的姿态过差与腿着地终止。
- reset：从倒地状态库同家族状态出生（随机 yaw），参考时钟从 0 开始；去掉推扰与速度课程，保留 DR。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_train.robot_cfg import _RESOURCES, get_serialleg_closedchain_cfg
from se3_train.tasks.common.no_attitude import apply_no_attitude_layout
from se3_train.tasks.flat.env_cfg import FLAT_ACTION_SMOOTHNESS_SPRING, FLAT_WHEEL_ACTION_SCALE
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough.env_cfg import (
    ROUGH_LEG_TORQUE_ENVELOPE_SCALE,
    ROUGH_ROBOT_COLLISION_GEOM_GROUP,
    ROUGH_WHEEL_TORQUE_ENVELOPE,
    ROUGH_WHEEL_TORQUE_LAG_RANGE_S,
)

from . import mdp
from .reference import REFERENCE_DIR, RecoveryReferenceLibrary

RECOVERY_MIMIC_REFERENCES = (str(REFERENCE_DIR / "r1_inverted_folded.npz"),)
RECOVERY_MIMIC_FALL_STATE_CACHE = str(
    _RESOURCES / "recovery_states" / "fall_states_d1ed876b_40k.npz"
)
RECOVERY_MIMIC_START_FAMILY = (2, 1.0, True)
"""出生家族（重力投影主轴, 符号, 腿折叠）：(2, +1, 折叠) = R1 倒置·腿折叠，状态库里约 4900 条。"""
RECOVERY_MIMIC_STAND_HEIGHT = 0.22
RECOVERY_MIMIC_EXTRA_EPISODE_S = 0.5
RECOVERY_MIMIC_MAX_ORIENTATION_ERROR_DEG = 60.0
RECOVERY_MIMIC_REWARD_WEIGHTS = {
    "mimic_orientation": 3.0,
    "mimic_leg_pos": 2.0,
    "mimic_base_height": 1.0,
    "mimic_leg_vel": 0.5,
    "mimic_body_ang_vel": 0.5,
}
_KEPT_FLAT_REWARDS = (
    "action_rate",
    "action_smoothness",
    "leg_torques",
    "wheel_torques",
    "leg_power",
    "dof_pos_limits",
)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """起身 mimic 环境（各项依据见模块 docstring）。"""
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )
    refs = RECOVERY_MIMIC_REFERENCES
    library = RecoveryReferenceLibrary(refs, "cpu")

    lag = ROUGH_WHEEL_TORQUE_LAG_RANGE_S
    cfg.scene.entities = {
        "robot": get_serialleg_closedchain_cfg(
            collision_geom_group=ROUGH_ROBOT_COLLISION_GEOM_GROUP,
            leg_torque_envelope_scale=ROUGH_LEG_TORQUE_ENVELOPE_SCALE,
            wheel_torque_envelope=ROUGH_WHEEL_TORQUE_ENVELOPE,
            wheel_torque_lag_tau_range_s=(
                None if lag is None else ((0.5 * sum(lag),) * 2 if play else tuple(lag))
            ),
        )
    }
    action = cfg.actions["delayed_action"]
    if action.knee_gas_spring_force <= 0.0:
        raise ValueError("起身 mimic 需要正的气弹簧前馈补偿力（与 Rough 一致）")
    action.knee_gas_spring_compensation_enabled = True

    # 指令：沿用 Flat 的 8 维跳跃指令项（跳跃概率 0，观测里的 jump_commands 槽位需要它），改成零速、固定高度；
    # 参考不看指令，观测槽位保持与 Rough 同布局。
    command = cfg.commands["velocity_height"]
    h = RECOVERY_MIMIC_STAND_HEIGHT
    command.lin_vel_x_range = (0.0, 0.0)
    command.ang_vel_yaw_range = (0.0, 0.0)
    command.pitch_range = (0.0, 0.0)
    command.roll_range = (0.0, 0.0)
    command.height_range = (h, h)
    command.standing_height_range = (h, h)
    apply_no_attitude_layout(cfg)

    # 观测：actor / critic 末尾追加参考信息
    reference_obs = ObservationTermCfg(
        func=mdp.recovery_reference_obs, params={"reference_paths": refs}
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms = {
            **cfg.observations[group].terms,
            "recovery_reference": reference_obs,
        }

    # 奖励：跟踪参考 + Flat 的平滑与力矩正则
    kept = {
        name: cfg.rewards[name] for name in _KEPT_FLAT_REWARDS if cfg.rewards.get(name) is not None
    }
    cfg.rewards = {
        **{
            name: RewardTermCfg(
                func=getattr(mdp, name), weight=float(w), params={"reference_paths": refs}
            )
            for name, w in RECOVERY_MIMIC_REWARD_WEIGHTS.items()
        },
        **kept,
    }

    # 终止：超时、数值爆炸、姿态偏离参考
    cfg.terminations = {
        name: term
        for name, term in cfg.terminations.items()
        if name in ("time_out", "catastrophic_state")
    }
    # 参考里大腿转约 200°、腿角离默认站姿最远约 5 rad，Flat 的"腿角偏离默认 >3 rad 判发散"会在参考 0.95 s 处
    # 把正确跟踪的 env 杀掉（kirud7sl 因此学成倒着不动等姿态终止），这里只保留数值发散类检查。
    cfg.terminations["catastrophic_state"].params["max_leg_pos_error"] = None
    cfg.terminations["orientation_deviation"] = TerminationTermCfg(
        func=mdp.orientation_deviation,
        time_out=False,
        params={"reference_paths": refs, "max_error_deg": RECOVERY_MIMIC_MAX_ORIENTATION_ERROR_DEG},
    )

    # reset：从倒地状态库同家族出生；去掉 Flat 的站姿 reset 与推扰
    cfg.events = {
        name: ev
        for name, ev in cfg.events.items()
        if name not in ("reset_root_state", "reset_joints", "push_robots")
    }
    cfg.events["reset_recovery_mimic"] = EventTermCfg(
        func=mdp.reset_recovery_mimic,
        mode="reset",
        params={
            "reference_paths": refs,
            "cache_path": RECOVERY_MIMIC_FALL_STATE_CACHE,
            "family": RECOVERY_MIMIC_START_FAMILY,
        },
    )
    cfg.curriculum = {}
    cfg.episode_length_s = float(library.duration.max()) + RECOVERY_MIMIC_EXTRA_EPISODE_S
    return cfg


__all__ = [
    "RECOVERY_MIMIC_EXTRA_EPISODE_S",
    "RECOVERY_MIMIC_FALL_STATE_CACHE",
    "RECOVERY_MIMIC_MAX_ORIENTATION_ERROR_DEG",
    "RECOVERY_MIMIC_REFERENCES",
    "RECOVERY_MIMIC_REWARD_WEIGHTS",
    "RECOVERY_MIMIC_STAND_HEIGHT",
    "RECOVERY_MIMIC_START_FAMILY",
    "env_cfg",
]
