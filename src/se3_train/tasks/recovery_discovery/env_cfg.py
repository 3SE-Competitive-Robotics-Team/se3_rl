"""倒地自启 Discovery 阶段环境配置。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_shared import JointGroup
from se3_train.mdp import env_groups, terminations
from se3_train.mdp import events as mdp_events
from se3_train.mdp import observations as mdp_observations
from se3_train.mdp.recovery_teacher import RecoveryTeacherCfg
from se3_train.mdp.recovery_torque_assist import RecoveryTorqueAssistCfg
from se3_train.tasks.recovery import curriculums, rewards
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RECOVERY_STATE_CACHE_PATH = (
    _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_closedchain_stair_v3_40k.npz"
)
_DISCOVERY_MAX_LIN_VEL_X = 1.89
_DISCOVERY_MAX_ANG_VEL_YAW = 9.41
_DISCOVERY_FINAL_HEIGHT_RANGE = (0.195, 0.390)
_DISCOVERY_DEPLOYMENT_RANGES = {
    "lin_vel_x": (-_DISCOVERY_MAX_LIN_VEL_X, _DISCOVERY_MAX_LIN_VEL_X),
    "ang_vel_yaw": (-_DISCOVERY_MAX_ANG_VEL_YAW, _DISCOVERY_MAX_ANG_VEL_YAW),
    "pitch": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "height": _DISCOVERY_FINAL_HEIGHT_RANGE,
    "jump_flag": (0.0, 0.0),
    "jump_target_height": (0.0, 0.0),
    "jump_phase": (0.0, 0.0),
}
_STEPS_PER_POLICY_ITER = 24
_TRAIN_ENV_GROUPS = {"loco": 0.5, "recover": 0.5}
_PLAY_ENV_GROUPS = {"loco": 0.0, "recover": 1.0}
_TRAIN_NUM_ENVS_PER_RANK = 8192
RECOVERY_DISCOVERY_HISTORY_LENGTH = 5
_LOCO_TN_SAFE_RATIO = 0.80
_LOCO_TN_REWARD_WEIGHT = -0.50
# Grouped 的分组 TN 项已删（5000 轮实测恒零），这两个常量仅供 recovery_expert 引用。
_RECOVER_TN_SAFE_RATIO = 0.80
_RECOVER_TN_REWARD_WEIGHT = -0.50


class DiscoveryRewardProfile(StrEnum):
    """Recovery-Discovery 奖励记账配置，标准任务必须显式保持 baseline。"""

    BASELINE = "baseline"
    GENTLE = "gentle"
    REFORM_A = "reform-a"
    REFORM_AB = "reform-ab"


_DISCOVERY_REWARD_WEIGHTS = {
    "tracking_lin_vel": 3.0,
    "tracking_ang_vel": 1.5,
    "upward": 3.0,
    "tracking_height": -1500.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "upright_orientation_l2": -0.5,
    "upright_zero_velocity": -0.20,
    "stand_still": -2.0,
    "joint_pos_penalty": -1.0,
    "leg_action_rate": -0.001,
    # 轮 scale 45→15 补偿（2026-09-01）：同一物理轮速轨迹动作幅值 ×3、差分平方 ×9，
    # 权重 ÷9 保持物理定价不变。
    "wheel_action_rate": -1.1e-4,
    # -0.12 + cap 320 = 4gs3te0p 验证有效的组合（σ 退火至 0.24、reward 272），沿用为基线。
    "action_smoothness": -0.12,
    "leg_torques": -2.0e-4,
    "leg_dof_acc": -2.5e-7,
    "leg_power": -1.0e-4,
    "joint_mirror": -0.05,
    "dof_pos_limits": -5.0,
    "collision": -1.0,
    "contact_forces": -1.5e-4,
    "wheel_air_velocity": -1.0e-3,
    "leg_contact": -1.0,
    "wheel_contact_without_cmd": 0.1,
    "diagnostics": 1.0,
}

# GENTLE（起身温柔化，零新增项）：R1 翻滚速率 ×5、R2 落地冲击生效化、R3 躺地税降档
# （门参数见 _apply_discovery_reward_profile）。定标依据 = yzau6pg5@4999 回放实测
# （弹道 ω 7.4-11.3 rad/s、落地 p50 420-980N/p95 1475N）。
_DISCOVERY_GENTLE_REWARD_WEIGHTS = {
    **_DISCOVERY_REWARD_WEIGHTS,
    "ang_vel_xy": -0.25,
    "contact_forces": -15.0,
    # 静止定价修正 Z1/Z2（2026-09-01）。实测依据 = d9m1mi7p@3936：upward 11.7/s 占
    # episode 回报 88%，静止抖动总代价 ~1.4/s 仅为站立收入的 10%，recover_lin_score
    # 0.613 = exp(-0.1²/0.02) 反推静止漂移 RMS ≈ 0.1 m/s 且全程无改善。
    # Z1 upward 3.0 → 1.0：直立常驻收入 12/s → 4/s，跟踪与静止定价的相对权重 ×3。
    "upward": 1.0,
    # Z2 静止漂移罚 -0.20 → -0.5：cap 后上限 1.6/s → 4/s，与降档后的 upward 同量级。
    "upright_zero_velocity": -0.5,
    # 起身动作整形 S2-S4（2026-09-01）。实测依据 = job21@3000-4999 收敛期贡献：
    # upward +11.38/s 一家独大，而甩轮子 1:245、抽搐 1:27、不对称 1:10200。
    # S2 对称性：解开直立门后按 ×40 提价，目标量级 ~1% of upward。
    "joint_mirror": -2.0,
    # S3 轮子空转：改为只在起身段计价（recovery_active_only），基数缩小故 ×30 提价。
    "wheel_air_velocity": -3.0e-2,
    # S4 抖动：全局 ×50 + 起身段再 ×10（recovery_scale），起身段等效 ×500。
    "leg_action_rate": -0.05,
    # Z5 轮抖动提价（2026-09-02）：σ_eq ∝ 1/√w，q21vt9zi 实测 wheel σ 在 0.40 平台
    # 9000 iter 不动（噪声成本 ~0.2/s 与熵红利均衡）。×4 目标 σ 0.40 -> ~0.20；
    # 起身段由 recovery_scale 10 -> 2.5 抵消，绝对价位 -5.5e-2 不变，只动行走段。
    "wheel_action_rate": -2.2e-2,
    # Z7 重力门控水平罚（2026-09-02）：训练端实测站立倾斜 7.6°，而
    # upright_orientation_l2 的转弯横滚尺度（pitch_scale 0.20）只给它定价 -0.22/s。
    # pg_xy L2 × 近直立门（30°->15° 渐入，起身/倒地免罚），-30 使 7.6° 计 -0.53/s、
    # 常规行驶 5° 倾计 -0.23/s、2° 站姿 -0.04/s——只压歪斜站立，不打加减速倾斜。
    "flat_orientation": -30.0,
}

_DISCOVERY_REFORM_A_REWARD_WEIGHTS = {
    **{
        name: weight
        for name, weight in _DISCOVERY_REWARD_WEIGHTS.items()
        if name != "contact_forces"
    },
    "leg_dof_vel": -0.08,
    "impact_forces": -0.15,
    "ang_vel_xy_excess": -0.5,
}
_DISCOVERY_REFORM_AB_REWARD_WEIGHTS = {
    **{
        name: weight
        for name, weight in _DISCOVERY_REFORM_A_REWARD_WEIGHTS.items()
        if name != "upward"
    },
    "upward_arrival": 12.5,
}
_DISCOVERY_REWARD_WEIGHTS_BY_PROFILE = {
    DiscoveryRewardProfile.BASELINE: _DISCOVERY_REWARD_WEIGHTS,
    DiscoveryRewardProfile.GENTLE: _DISCOVERY_GENTLE_REWARD_WEIGHTS,
    DiscoveryRewardProfile.REFORM_A: _DISCOVERY_REFORM_A_REWARD_WEIGHTS,
    DiscoveryRewardProfile.REFORM_AB: _DISCOVERY_REFORM_AB_REWARD_WEIGHTS,
}


def _ungrouped_velocity_curriculum_cfg() -> CurriculumTermCfg:
    """保留不分组对照实验原有的固定速度课程。"""
    return CurriculumTermCfg(
        func=curriculums.commands_vel,
        params={
            "command_name": "velocity_height",
            "use_iterations": True,
            "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
            "velocity_stages": [
                {
                    "iteration": 0,
                    "lin_vel_x_range": (0.0, 0.0),
                    "ang_vel_yaw_range": (0.0, 0.0),
                },
                {
                    "iteration": 1500,
                    "lin_vel_x_range": (0.0, 0.0),
                    "ang_vel_yaw_range": (0.0, 0.0),
                },
                {
                    "iteration": 2000,
                    "lin_vel_x_range": (-0.5, 0.5),
                    "ang_vel_yaw_range": (-1.0, 1.0),
                },
                {
                    "iteration": 2600,
                    "lin_vel_x_range": (-1.0, 1.0),
                    "ang_vel_yaw_range": (-2.5, 2.5),
                },
                {
                    "iteration": 3400,
                    "lin_vel_x_range": (-1.6, 1.6),
                    "ang_vel_yaw_range": (-5.0, 5.0),
                },
                {
                    "iteration": 4200,
                    "lin_vel_x_range": (
                        -_DISCOVERY_MAX_LIN_VEL_X,
                        _DISCOVERY_MAX_LIN_VEL_X,
                    ),
                    "ang_vel_yaw_range": (
                        -_DISCOVERY_MAX_ANG_VEL_YAW,
                        _DISCOVERY_MAX_ANG_VEL_YAW,
                    ),
                },
            ],
        },
    )


def _configure_discovery_reward_contract(
    cfg: ManagerBasedRlEnvCfg,
    reward_profile: DiscoveryRewardProfile,
) -> None:
    """先装配冻结的 baseline，再按显式 profile 叠加实验性记账。"""

    cfg.rewards.clear()
    cfg.rewards["tracking_lin_vel"] = RewardTermCfg(
        func=rewards.tracking_lin_vel,
        weight=3.0,
        params={
            "command_name": "velocity_height",
            "sigma_move": 0.25,
            "sigma_stand": 0.02,
            "vz_weight": 0.0,
            "use_upright_gate": True,
            "tracking_upright_full_cos": math.cos(math.radians(15.0)),
        },
    )
    cfg.rewards["tracking_ang_vel"] = RewardTermCfg(
        func=rewards.tracking_ang_vel,
        weight=1.5,
        params={
            "command_name": "velocity_height",
            "sigma": 0.25,
            "use_upright_gate": True,
            "tracking_upright_full_cos": math.cos(math.radians(15.0)),
        },
    )
    cfg.rewards["upward"] = RewardTermCfg(func=rewards.upward, weight=3.0)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=rewards.lin_vel_z, weight=-2.0)
    cfg.rewards["ang_vel_xy"] = RewardTermCfg(
        func=rewards.ang_vel_xy,
        weight=-0.05,
        params={"use_upright_gate": False},
    )
    cfg.rewards["tracking_height"] = RewardTermCfg(
        func=rewards.tracking_height,
        weight=-1500.0,
        params={
            "command_name": "velocity_height",
            "sigma": 0.0025,
            "height_sensor_name": "base_height_sensor",
            "kernel": "l2",
            "use_upright_gate": False,
            "min_upright_gate": 0.0,
            "use_pose_end_gate": False,
            "use_inverted_free_upright_height_gate": True,
            "upright_gate_angle_deg": 30.0,
            "inverted_gate_angle_deg": 150.0,
        },
    )
    cfg.rewards["upright_orientation_l2"] = RewardTermCfg(
        func=rewards.recovery_upright_orientation_l2,
        weight=-0.5,
        params={
            "command_name": "velocity_height",
            "gate_start_deg": 60.0,
            "gate_full_deg": 20.0,
            "roll_scale_rad": 0.14,
            "pitch_scale_rad": 0.20,
            "roll_weight": 1.5,
            "pitch_weight": 1.0,
            "max_penalty": 6.0,
        },
    )
    cfg.rewards["upright_zero_velocity"] = RewardTermCfg(
        func=rewards.recovery_upright_zero_velocity_penalty,
        weight=-0.20,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "gate_start_deg": 45.0,
            "gate_full_deg": 15.0,
            "base_speed_scale": 0.10,
            "wheel_speed_scale": 0.08,
            "base_ang_vel_scale": 0.30,
            "max_penalty": 8.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=rewards.stand_still,
        weight=-2.0,
        params={
            "command_name": "velocity_height",
            "command_threshold": 0.1,
            "default_height": 0.26,
            "height_tolerance": 40.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["joint_pos_penalty"] = RewardTermCfg(
        func=rewards.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "velocity_height",
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["leg_action_rate"] = RewardTermCfg(
        func=rewards.leg_action_rate,
        weight=-0.001,
    )
    cfg.rewards["wheel_action_rate"] = RewardTermCfg(
        func=rewards.wheel_action_rate,
        # 轮 scale 45→15 的 ×9 补偿，见 _DISCOVERY_REWARD_WEIGHTS 注释。
        weight=-1.1e-4,
    )
    # S1 生效的 action_smoothness 权威定义（会覆盖 recovery 基类里 weight=0 的同名项）。
    cfg.rewards["action_smoothness"] = RewardTermCfg(
        func=rewards.action_smoothness,
        weight=-0.12,
        params={
            "command_name": "velocity_height",
            # 姿态门控已禁用（181/180 → gate 恒 1），平滑罚全姿态生效。
            "gate_start_deg": 181.0,
            "gate_full_deg": 180.0,
            # 弹簧 plant 上 σ≈2 的噪声二阶差分期望 ≈288，cap=80 使罚项饱和、对噪声
            # 零梯度（σ 膨胀根源之一）；提到 320 保证 σ≤4 全程有梯度。
            "max_penalty": 320.0,
            "leg_scale": 1.0,
            # 轮 scale 45→15 补偿（2026-09-01）：2.0/9≈0.22，保持蓄意轮运动的平滑税
            # 物理价位不变（噪声地板 6σ² 是动作单位量、与 scale 无关，cap 320 定标仍成立）。
            # 不补偿会让轮加速的平滑税 ×9、压垮平衡校正——与改 scale 的目的相反。
            "wheel_scale": 0.22,
        },
    )
    cfg.rewards["leg_torques"] = RewardTermCfg(
        func=rewards.leg_torques,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_dof_acc"] = RewardTermCfg(
        func=rewards.leg_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["leg_power"] = RewardTermCfg(
        func=rewards.leg_power,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["joint_mirror"] = RewardTermCfg(
        func=rewards.joint_mirror,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["dof_pos_limits"] = RewardTermCfg(
        func=rewards.dof_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["collision"] = RewardTermCfg(
        func=rewards.collision,
        weight=-1.0,
        params={
            "sensor_name": "collision_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
            "use_recovery_gate": False,
        },
    )
    cfg.rewards["contact_forces"] = RewardTermCfg(
        func=rewards.contact_forces,
        weight=-1.5e-4,
        params={
            "threshold": 20.0,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
            "use_recovery_gate": False,
        },
    )
    cfg.rewards["wheel_air_velocity"] = RewardTermCfg(
        func=rewards.wheel_air_velocity_penalty,
        weight=-1.0e-3,
        params={
            "sensor_name": "wheel_sensor",
            "force_threshold": 1.0,
            "max_penalty": 10000.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "log_prefix": "Recovery",
        },
    )
    cfg.rewards["leg_contact"] = RewardTermCfg(
        func=rewards.leg_contact_penalty,
        weight=-1.0,
        params={
            "sensor_name": "leg_contact_sensor",
            "force_threshold": 1.0,
        },
    )
    cfg.rewards["wheel_contact_without_cmd"] = RewardTermCfg(
        func=rewards.feet_contact_without_cmd,
        weight=0.1,
        params={
            "command_name": "velocity_height",
            "force_threshold": 1.0,
            "cmd_threshold": 0.1,
            "sensor_name": "wheel_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["diagnostics"] = RewardTermCfg(
        func=rewards.recovery_diagnostics,
        weight=1.0,
        params={
            "command_name": "velocity_height",
            "base_height_sensor_name": "base_height_sensor",
            "wheel_sensor_name": "wheel_sensor",
            "leg_contact_sensor_name": "leg_contact_sensor",
            "collision_sensor_name": "collision_sensor",
            "asset_cfg": SceneEntityCfg("robot"),
            "force_threshold": 1.0,
            "contact_force_threshold": 20.0,
            "active_rod_margin_warning": 0.05,
            "log_interval_steps": 256,
            "core_log_interval_steps": 64,
        },
    )
    _apply_discovery_reward_profile(cfg, reward_profile)
    _assert_discovery_reward_contract(cfg, reward_profile)


def _apply_discovery_reward_profile(
    cfg: ManagerBasedRlEnvCfg,
    reward_profile: DiscoveryRewardProfile,
) -> None:
    """只在显式改革 profile 上叠加 A1-A4 与可选的到达式记账。"""

    if reward_profile is DiscoveryRewardProfile.BASELINE:
        return

    if reward_profile is DiscoveryRewardProfile.GENTLE:
        # R1 翻滚速率定价：弹道翻滚 ω 实测 7.4-11.3 rad/s -> 14-32/s，温柔 2-3 -> 1-2/s。
        cfg.rewards["ang_vel_xy"] = replace(cfg.rewards["ang_vel_xy"], weight=-0.25)
        # R2 落地冲击生效化：阈 250N ≈ 2x 动载（常规行驶/接推免税），-15 配合函数内 /100
        # 等效每牛 0.15 -> 724N 一次 ~4、p95 1475N ~9（baseline 的 -1.5e-4 为装饰品）。
        contact_params = dict(cfg.rewards["contact_forces"].params or {})
        contact_params["threshold"] = 250.0
        cfg.rewards["contact_forces"] = replace(
            cfg.rewards["contact_forces"],
            weight=-15.0,
            params=contact_params,
        )
        # R3 躺地时间税降档：倒置 34/s -> ~7/s（_upright_factor 门 + 0.2 下限，无免税区，
        # 反躺平双保险 = 下限 + upward）；<45° 站立带 ≈1，
        # 45-90° cos/0.7 斜靠税略强于 baseline（反停车加固）。经济学：倒置 urgency 18/s，
        # 快 0.9s 赚 ~16 < R1+R2 弹道账单 7-24 -> 「越快越赚」不等式翻转。
        height_params = dict(cfg.rewards["tracking_height"].params or {})
        for name in ("upright_gate_angle_deg", "inverted_gate_angle_deg"):
            height_params.pop(name, None)
        height_params.update(
            {
                "use_inverted_free_upright_height_gate": False,
                "use_upright_gate": True,
                "min_upright_gate": 0.2,
            }
        )
        cfg.rewards["tracking_height"] = replace(
            cfg.rewards["tracking_height"],
            params=height_params,
        )
        # S1 起身梯度：upward = (1-pg_z)^2 在完全倒置（pg_z=+1）处导数恰为 0，而 60%
        # 的 recover reset 正是 pitch≈180°，策略在最难的起点上拿不到局部梯度，只能靠
        # 爆发式甩轮跳出零梯度区。recovery_upward = 2u+2u^4 在倒置处保留导数 2，
        # 直立/倒置端点值（4 / 0）不变。Z1 同时把权重降到 1.0（见权重表注释）。
        cfg.rewards["upward"] = replace(
            cfg.rewards["upward"],
            func=rewards.recovery_upward,
            weight=_DISCOVERY_GENTLE_REWARD_WEIGHTS["upward"],
        )
        # Z2 静止漂移罚提价（参数不变，仅权重，见权重表注释）。
        cfg.rewards["upright_zero_velocity"] = replace(
            cfg.rewards["upright_zero_velocity"],
            weight=_DISCOVERY_GENTLE_REWARD_WEIGHTS["upright_zero_velocity"],
        )
        # Z3 sigma_stand 0.02 → 0.08：0.02 在实测 0.1-0.35 m/s 的静止误差区间是零梯度
        # 悬崖（exp(-0.35²/0.02)≈0.002），0.08 使 0.1 m/s 处得分 0.88、0.3 m/s 处 0.32，
        # 整个工作区间恢复梯度。
        lin_params = dict(cfg.rewards["tracking_lin_vel"].params or {})
        lin_params["sigma_stand"] = 0.08
        cfg.rewards["tracking_lin_vel"] = replace(
            cfg.rewards["tracking_lin_vel"],
            params=lin_params,
        )
        # S2 对称性：_upright_factor 在倾角 >= 90° 恒为 0，把起身全程的 joint_mirror
        # 乘成零。解门与提价必须同时做，只做一个都等于没做。
        mirror_params = dict(cfg.rewards["joint_mirror"].params or {})
        mirror_params["use_upright_gate"] = False
        cfg.rewards["joint_mirror"] = replace(
            cfg.rewards["joint_mirror"],
            weight=_DISCOVERY_GENTLE_REWARD_WEIGHTS["joint_mirror"],
            params=mirror_params,
        )
        # S3 轮子空转只在起身段计价，行走段完全不受影响。
        air_params = dict(cfg.rewards["wheel_air_velocity"].params or {})
        air_params["recovery_active_only"] = True
        cfg.rewards["wheel_air_velocity"] = replace(
            cfg.rewards["wheel_air_velocity"],
            weight=_DISCOVERY_GENTLE_REWARD_WEIGHTS["wheel_air_velocity"],
            params=air_params,
        )
        # S4 抖动：action_smoothness 没有 recovery 钩子，只能用这两项的 recovery_scale
        # 在起身段额外加价；行走段仍按全局权重计，量级仍低于 leg_torques。
        # Z5：wheel 的 recovery_scale 降为 2.5 抵消全局 ×4，起身段绝对价位不变。
        for name, recovery_scale in (("leg_action_rate", 10.0), ("wheel_action_rate", 2.5)):
            rate_params = dict(cfg.rewards[name].params or {})
            rate_params["recovery_scale"] = recovery_scale
            cfg.rewards[name] = replace(
                cfg.rewards[name],
                weight=_DISCOVERY_GENTLE_REWARD_WEIGHTS[name],
                params=rate_params,
            )
        # Z7 重力门控水平罚（定标见权重表注释）。
        cfg.rewards["flat_orientation"] = RewardTermCfg(
            func=rewards.flat_orientation_l2,
            weight=_DISCOVERY_GENTLE_REWARD_WEIGHTS["flat_orientation"],
            params={
                "gate_start_deg": 30.0,
                "gate_full_deg": 15.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Z6 静止漂移罚去饱和（2026-09-02）：q21vt9zi 评测端实测零指令下 ~0.19 m/s
        # 匀速漂移，此处 base(0.19/0.1)²+wheel(0.19/0.08)²≈9.2 已越过 cap=8 的
        # 零梯度区。cap 8 -> 32 恢复 ≤0.35 m/s 区间的梯度；保留上界，防止零指令
        # 站立被 push 时的瞬态速度（>0.5 m/s）产生无界爆罚污染 return。
        zero_vel_params = dict(cfg.rewards["upright_zero_velocity"].params or {})
        zero_vel_params["max_penalty"] = 32.0
        cfg.rewards["upright_zero_velocity"] = replace(
            cfg.rewards["upright_zero_velocity"],
            params=zero_vel_params,
        )
        return

    height_params = dict(cfg.rewards["tracking_height"].params or {})
    for name in (
        "use_inverted_free_upright_height_gate",
        "upright_gate_angle_deg",
        "inverted_gate_angle_deg",
    ):
        height_params.pop(name, None)
    height_params.update(
        {
            "use_inverted_free_upright_height_gate": False,
            "use_near_upright_gate": True,
            "near_upright_gate_start_deg": 30.0,
            "near_upright_gate_full_deg": 15.0,
        }
    )
    cfg.rewards["tracking_height"] = replace(
        cfg.rewards["tracking_height"],
        params=height_params,
    )

    cfg.rewards.pop("contact_forces")
    cfg.rewards["leg_dof_vel"] = RewardTermCfg(
        func=rewards.leg_dof_vel,
        weight=-0.08,
        params={"max_vel": 6.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["impact_forces"] = RewardTermCfg(
        func=rewards.impact_forces,
        weight=-0.15,
        params={
            "sensor_names": ("wheel_sensor", "leg_contact_sensor", "collision_sensor"),
            "threshold": 200.0,
            "max_excess": 3000.0,
        },
    )
    cfg.rewards["ang_vel_xy_excess"] = RewardTermCfg(
        func=rewards.base_ang_vel_xy_excess,
        weight=-0.5,
        params={"threshold": 4.0},
    )

    if reward_profile is DiscoveryRewardProfile.REFORM_AB:
        cfg.rewards.pop("upward")
        cfg.rewards["upward_arrival"] = RewardTermCfg(
            func=rewards.upward_arrival,
            weight=12.5,
            params={"rate_cap": 0.1},
        )


def _critical_reward_params(
    reward_profile: DiscoveryRewardProfile,
) -> dict[str, dict[str, object]]:
    """返回足以区分三种记账语义的关键参数。"""

    if reward_profile is DiscoveryRewardProfile.BASELINE:
        return {
            "tracking_height": {
                "use_inverted_free_upright_height_gate": True,
                "upright_gate_angle_deg": 30.0,
                "inverted_gate_angle_deg": 150.0,
            },
            "contact_forces": {
                "threshold": 20.0,
                "sensor_name": "wheel_sensor",
                "use_recovery_gate": False,
            },
        }

    if reward_profile is DiscoveryRewardProfile.GENTLE:
        return {
            "tracking_height": {
                "use_inverted_free_upright_height_gate": False,
                "use_upright_gate": True,
                "min_upright_gate": 0.2,
            },
            "contact_forces": {
                "threshold": 250.0,
                "sensor_name": "wheel_sensor",
                "use_recovery_gate": False,
            },
            # 起身动作整形 S2-S4：这三处一旦被改回默认值，对应的定价会静默失效。
            "joint_mirror": {"use_upright_gate": False},
            "wheel_air_velocity": {"recovery_active_only": True},
            "leg_action_rate": {"recovery_scale": 10.0},
            # Z5：全局 ×4 由 recovery_scale 2.5 抵消，改回 10.0 会让起身段隐式 ×4。
            "wheel_action_rate": {"recovery_scale": 2.5},
            # 静止定价修正 Z3：sigma_stand 改回 0.02 会让静止精度回到零梯度悬崖。
            "tracking_lin_vel": {"sigma_stand": 0.08},
            # Z6：cap 改回 8 会在 >=0.17 m/s 漂移区间重新制造零梯度饱和。
            "upright_zero_velocity": {"max_penalty": 32.0},
            # Z7：门若放宽到起身带（>30°），水平罚会重新惩罚起身翻滚。
            "flat_orientation": {"gate_start_deg": 30.0, "gate_full_deg": 15.0},
        }

    result: dict[str, dict[str, object]] = {
        "tracking_height": {
            "use_inverted_free_upright_height_gate": False,
            "use_near_upright_gate": True,
            "near_upright_gate_start_deg": 30.0,
            "near_upright_gate_full_deg": 15.0,
        },
        "leg_dof_vel": {"max_vel": 6.0},
        "impact_forces": {
            "sensor_names": ("wheel_sensor", "leg_contact_sensor", "collision_sensor"),
            "threshold": 200.0,
            "max_excess": 3000.0,
        },
        "ang_vel_xy_excess": {"threshold": 4.0},
    }
    if reward_profile is DiscoveryRewardProfile.REFORM_AB:
        result["upward_arrival"] = {"rate_cap": 0.1}
    return result


def _critical_reward_functions(
    reward_profile: DiscoveryRewardProfile,
) -> dict[str, object]:
    """返回 profile 关键奖励项应绑定的实现。"""

    result: dict[str, object] = {"tracking_height": rewards.tracking_height}
    if reward_profile is DiscoveryRewardProfile.BASELINE:
        result["contact_forces"] = rewards.contact_forces
        return result
    if reward_profile is DiscoveryRewardProfile.GENTLE:
        result["contact_forces"] = rewards.contact_forces
        # S1：GENTLE 的 upward 必须绑定倒置处保留梯度的实现，绑回 rewards.upward
        # 会静默恢复 (1-pg_z)^2 的零梯度起点。
        result["upward"] = rewards.recovery_upward
        # Z7：必须是重力门控实现；换成无门控 flat_orientation 会罚穿起身过程。
        result["flat_orientation"] = rewards.flat_orientation_l2
        return result
    result.update(
        {
            "leg_dof_vel": rewards.leg_dof_vel,
            "impact_forces": rewards.impact_forces,
            "ang_vel_xy_excess": rewards.base_ang_vel_xy_excess,
        }
    )
    if reward_profile is DiscoveryRewardProfile.REFORM_AB:
        result["upward_arrival"] = rewards.upward_arrival
    return result


def _forbidden_reward_params(
    reward_profile: DiscoveryRewardProfile,
) -> dict[str, tuple[str, ...]]:
    """返回会把当前 profile 偷偷改成另一种门控语义的禁用参数。"""

    if reward_profile is DiscoveryRewardProfile.BASELINE:
        return {
            "tracking_height": (
                "use_near_upright_gate",
                "near_upright_gate_start_deg",
                "near_upright_gate_full_deg",
            )
        }
    if reward_profile is DiscoveryRewardProfile.GENTLE:
        return {
            "tracking_height": (
                "use_near_upright_gate",
                "near_upright_gate_start_deg",
                "near_upright_gate_full_deg",
                "upright_gate_angle_deg",
                "inverted_gate_angle_deg",
            )
        }
    return {
        "tracking_height": (
            "upright_gate_angle_deg",
            "inverted_gate_angle_deg",
        )
    }


def _expected_discovery_reward_weights(
    reward_profile: DiscoveryRewardProfile,
    *,
    ungrouped: bool,
) -> dict[str, float]:
    """返回指定 profile 与环境布局的精确奖励权重表。"""

    expected = dict(_DISCOVERY_REWARD_WEIGHTS_BY_PROFILE[reward_profile])
    if ungrouped:
        expected["tn_envelope_violation"] = _LOCO_TN_REWARD_WEIGHT
    return expected


def _assert_discovery_reward_contract(
    cfg: ManagerBasedRlEnvCfg,
    reward_profile: DiscoveryRewardProfile,
    *,
    ungrouped: bool = False,
) -> None:
    """校验奖励名称、权重、关键参数与关键实现，阻止 profile 静默漂移。"""

    expected_weights = _expected_discovery_reward_weights(
        reward_profile,
        ungrouped=ungrouped,
    )
    actual = set(cfg.rewards)
    expected = set(expected_weights)
    if actual != expected:
        raise RuntimeError(
            f"Recovery-Discovery {reward_profile.value} reward contract drifted: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    bad_weights = {
        name: float(cfg.rewards[name].weight)
        for name, expected_weight in expected_weights.items()
        if abs(float(cfg.rewards[name].weight) - float(expected_weight)) > 1.0e-12
    }
    if bad_weights:
        raise RuntimeError(
            f"Recovery-Discovery {reward_profile.value} reward weight drifted: {bad_weights}"
        )

    bad_params: dict[str, object] = {}
    critical_params = _critical_reward_params(reward_profile)
    for reward_name, expected_params in critical_params.items():
        actual_params = cfg.rewards[reward_name].params or {}
        for param_name, expected_value in expected_params.items():
            actual_value = actual_params.get(param_name)
            if actual_value != expected_value:
                bad_params[f"{reward_name}.{param_name}"] = actual_value
    if bad_params:
        raise RuntimeError(
            f"Recovery-Discovery {reward_profile.value} reward params drifted: {bad_params}"
        )

    forbidden_params = {
        f"{reward_name}.{param_name}": (cfg.rewards[reward_name].params or {})[param_name]
        for reward_name, param_names in _forbidden_reward_params(reward_profile).items()
        for param_name in param_names
        if param_name in (cfg.rewards[reward_name].params or {})
    }
    if forbidden_params:
        raise RuntimeError(
            f"Recovery-Discovery {reward_profile.value} reward params conflict: {forbidden_params}"
        )

    bad_functions = {
        name: getattr(cfg.rewards[name].func, "__qualname__", repr(cfg.rewards[name].func))
        for name, expected_func in _critical_reward_functions(reward_profile).items()
        if cfg.rewards[name].func is not expected_func
    }
    if bad_functions:
        raise RuntimeError(
            f"Recovery-Discovery {reward_profile.value} reward functions drifted: {bad_functions}"
        )


def _reward_contract_hash(
    cfg: ManagerBasedRlEnvCfg,
    reward_profile: DiscoveryRewardProfile,
) -> str:
    """根据运行时最终奖励表生成稳定指纹。"""

    critical_params = _critical_reward_params(reward_profile)
    critical_functions = _critical_reward_functions(reward_profile)
    payload = {
        "profile": reward_profile.value,
        "terms": {name: float(term.weight) for name, term in sorted(cfg.rewards.items())},
        "critical_params": critical_params,
        "critical_functions": {
            name: f"{func.__module__}.{func.__qualname__}"
            for name, func in sorted(critical_functions.items())
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def detect_discovery_reward_contract(
    cfg: ManagerBasedRlEnvCfg,
) -> tuple[str, str] | None:
    """从运行时最终奖励表识别 profile，并返回已复核的稳定指纹。"""

    has_ungrouped_tn = "tn_envelope_violation" in cfg.rewards
    matches: list[DiscoveryRewardProfile] = []
    for reward_profile in DiscoveryRewardProfile:
        try:
            _assert_discovery_reward_contract(
                cfg,
                reward_profile,
                ungrouped=has_ungrouped_tn,
            )
        except RuntimeError:
            continue
        matches.append(reward_profile)
    if len(matches) != 1:
        return None
    reward_profile = matches[0]
    return reward_profile.value, _reward_contract_hash(cfg, reward_profile)


def _apply_actor_history(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """把 actor 观测换成五帧展平历史；critic 与其余训练契约保持不变。"""

    actor_cfg = cfg.observations["actor"]
    cfg.observations["actor"] = replace(
        actor_cfg,
        history_length=RECOVERY_DISCOVERY_HISTORY_LENGTH,
        flatten_history_dim=True,
    )
    return cfg


def env_cfg(
    play: bool = False,
    reward_profile: DiscoveryRewardProfile = DiscoveryRewardProfile.BASELINE,
) -> ManagerBasedRlEnvCfg:
    """标准姿态 Discovery 环境配置，奖励改革必须通过显式 profile 选择。"""
    cfg = recovery_env_cfg(play=play)
    if not play:
        cfg.scene.num_envs = _TRAIN_NUM_ENVS_PER_RANK

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.height_range = (0.24, 0.30)
    command_cfg.standing_height_range = (0.24, 0.30)
    command_cfg.deployment_ranges = dict(_DISCOVERY_DEPLOYMENT_RANGES)
    if reward_profile is DiscoveryRewardProfile.GENTLE:
        # 静止定价修正 Z4（2026-09-01）：双零静站命令份额 0.05 → 0.15。d9m1mi7p 实测
        # 静止漂移全程无改善，0.05 的静站预算过小；ungrouped 侧已有份额提升加速
        # 站立平衡学习的先例（reset 姿态份额 0.08 → 0.25）。
        command_cfg.standing_ratio = 0.15

    cfg.curriculum = {}
    inherited_events = dict(cfg.events)
    cfg.events = {
        "assign_env_groups": EventTermCfg(
            func=env_groups.AssignEnvGroups,
            mode="startup",
            params={"env_groups": _PLAY_ENV_GROUPS if play else _TRAIN_ENV_GROUPS},
        )
    }
    reset_scene_event = inherited_events.get("reset_scene_to_default")
    if reset_scene_event is not None:
        cfg.events["reset_scene_to_default"] = reset_scene_event

    root_common_params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "pos_xy_range": (-0.15, 0.15),
        "height_offset_range": (0.0, 0.02),
        "yaw_range": (-math.pi, math.pi),
        "roll_jitter_range": (-math.radians(5.0), math.radians(5.0)),
        "pitch_jitter_range": (-math.radians(5.0), math.radians(5.0)),
        "lin_vel_range": (0.0, 0.0),
        "ang_vel_range": (0.0, 0.0),
        "clearance_range": (0.001, 0.005),
        "standard_curriculum_stages": [
            {
                "iteration": 0,
                "roll_jitter_range": (-math.radians(5.0), math.radians(5.0)),
                "pitch_jitter_range": (-math.radians(5.0), math.radians(5.0)),
                "lin_vel_range": (0.0, 0.0),
                "ang_vel_range": (0.0, 0.0),
            },
            {
                "iteration": 300,
                "roll_jitter_range": (-math.radians(10.0), math.radians(10.0)),
                "pitch_jitter_range": (-math.radians(10.0), math.radians(10.0)),
                "lin_vel_range": (-0.03, 0.03),
                "ang_vel_range": (-0.10, 0.10),
            },
            {
                "iteration": 800,
                "roll_jitter_range": (-math.radians(15.0), math.radians(15.0)),
                "pitch_jitter_range": (-math.radians(15.0), math.radians(15.0)),
                "lin_vel_range": (-0.05, 0.05),
                "ang_vel_range": (-0.20, 0.20),
            },
            {
                "iteration": 1500,
                "roll_jitter_range": (-math.radians(20.0), math.radians(20.0)),
                "pitch_jitter_range": (-math.radians(20.0), math.radians(20.0)),
                "lin_vel_range": (-0.08, 0.08),
                "ang_vel_range": (-0.30, 0.30),
            },
        ],
        "use_iterations": True,
        "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
    }
    cfg.events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_recovery_discovery_mixed,
        mode="reset",
        params={
            **root_common_params,
            "pose_weights_by_group": {
                "loco": (1.0, 0.0, 0.0, 0.0, 0.0),
                "recover": (0.0, 0.20, 0.20, 0.30, 0.30),
            },
            "recovery_state_cache_path": str(_RECOVERY_STATE_CACHE_PATH),
            "recovery_state_cache_split": "train",
            "source_curriculum_stages_by_group": {
                "recover": [
                    {"iteration": 0, "cache_ratio": 0.0},
                    {"iteration": 300, "cache_ratio": 0.0},
                    {"iteration": 800, "cache_ratio": 0.0},
                    {"iteration": 1500, "cache_ratio": 0.10},
                    {"iteration": 2000, "cache_ratio": 0.25},
                    {"iteration": 2600, "cache_ratio": 0.45},
                    {"iteration": 3400, "cache_ratio": 0.60},
                    {"iteration": 4200, "cache_ratio": 0.70},
                ],
            },
        },
    )

    joint_common_params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_offset_range": 0.0,
        "joint_vel_range": (0.0, 0.0),
        "wheel_joint_vel_range": (0.0, 0.0),
        "joint_randomization_prob": 0.0,
        "align_root_height_to_wheels": True,
        "height_conditioned_default": True,
        "use_iterations": True,
        "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
    }
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params={
            **joint_common_params,
            "randomization_group_names": ("recover",),
            "curriculum_stages": [
                {
                    "iteration": 0,
                    "joint_offset_range": 0.0,
                    "joint_vel_range": (0.0, 0.0),
                    "joint_randomization_prob": 0.0,
                },
                {
                    "iteration": 300,
                    "joint_offset_range": 0.10,
                    "joint_vel_range": (-0.20, 0.20),
                    "joint_randomization_prob": 0.25,
                },
                {
                    "iteration": 800,
                    "joint_offset_range": 0.20,
                    "joint_vel_range": (-0.40, 0.40),
                    "joint_randomization_prob": 0.50,
                },
                {
                    "iteration": 1500,
                    "joint_offset_range": 0.25,
                    "joint_vel_range": (-0.50, 0.50),
                    "joint_randomization_prob": 0.75,
                },
            ],
        },
    )

    for event_name, event_cfg in inherited_events.items():
        if event_name not in {
            "reset_scene_to_default",
            "reset_root_state",
            "reset_joints",
            "push_robots",
        }:
            cfg.events[event_name] = event_cfg

    critic_cfg = cfg.observations["critic"]
    cfg.observations["critic"] = replace(
        critic_cfg,
        terms={
            **critic_cfg.terms,
            "group_id": ObservationTermCfg(func=env_groups.env_group_id_obs),
        },
    )

    cfg.terminations["loco_bad_orientation"] = TerminationTermCfg(
        func=env_groups.FilteredTerminationWrapper,
        time_out=False,
        params={
            "group_names": ("loco",),
            "wrapped_term": {
                "func": terminations.bad_orientation_delayed,
                "params": {"limit_angle": math.radians(30.0), "max_steps": 25},
            },
        },
    )
    cfg.terminations["loco_base_contact"] = TerminationTermCfg(
        func=env_groups.FilteredTerminationWrapper,
        time_out=False,
        params={
            "group_names": ("loco",),
            "wrapped_term": {
                "func": terminations.base_contact,
                "params": {"sensor_name": "collision_sensor", "force_threshold": 1.0},
            },
        },
    )
    # recover_stagnation 已删除（2026-09-01）：任务 16-19 实证其复活后把 recover episode
    # 一律截到 ~6-7s、起身学习死亡；4gs/7lx 学会起身的生态里它因自毁 bug 等效不存在。
    if not play:
        cfg.events["push_robots_loco"] = EventTermCfg(
            func=env_groups.FilteredEventWrapper,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "group_names": ("loco",),
                "wrapped_term": {
                    "func": mdp_events.push_robots,
                    "params": {
                        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                },
            },
        )
        cfg.curriculum = {
            "commands_vel": CurriculumTermCfg(
                func=curriculums.GroupedRewardVelocityCurriculum,
                params={
                    "command_name": "velocity_height",
                    "loco_group_name": "loco",
                    "recover_group_name": "recover",
                    "loco_init_level": 0.15,
                    "level_step": 0.05,
                    "max_lin_vel_x": _DISCOVERY_MAX_LIN_VEL_X,
                    "max_ang_vel_yaw": _DISCOVERY_MAX_ANG_VEL_YAW,
                    "evaluation_window_steps": 1000,
                    "required_consecutive_windows": 2,
                    "min_episodes_per_window": 32,
                    "min_tracking_samples": 256,
                    "lin_score_threshold": 0.80,
                    "yaw_score_threshold": 0.70,
                    "recover_ready_threshold": 0.60,
                },
            ),
            "commands_height": CurriculumTermCfg(
                func=curriculums.commands_height,
                params={
                    "command_name": "velocity_height",
                    "use_iterations": True,
                    "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                    "height_stages": [
                        {
                            "iteration": 0,
                            "height_range": (0.24, 0.30),
                        },
                        {
                            "iteration": 300,
                            "height_range": (0.23, 0.32),
                        },
                        {
                            "iteration": 800,
                            "height_range": (0.215, 0.36),
                        },
                        {
                            "iteration": 1500,
                            "height_range": _DISCOVERY_FINAL_HEIGHT_RANGE,
                        },
                    ],
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculums.push_disturbance,
                params={
                    "use_iterations": True,
                    "steps_per_policy_iter": _STEPS_PER_POLICY_ITER,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "iteration": 2600,
                            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        },
                        {
                            "iteration": 3400,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "iteration": 4200,
                            "velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)},
                        },
                    ],
                },
            ),
        }

    _configure_discovery_reward_contract(cfg, reward_profile)
    return cfg


def ungrouped_env_cfg(
    play: bool = False,
    reward_profile: DiscoveryRewardProfile = DiscoveryRewardProfile.BASELINE,
) -> ManagerBasedRlEnvCfg:
    """生成与分组实验同超参数、但使用旧混合 reset 的公平基线。

    actor 观测与分组任务一致，同样使用五帧展平历史，使唯一差异是 env 分组本身。
    标准任务固定使用 baseline；实验性记账只能由独立任务传入非默认 profile。
    """

    cfg = env_cfg(play=play, reward_profile=reward_profile)
    cfg.rewards["tn_envelope_violation"] = RewardTermCfg(
        func=rewards.NormalizedTnEnvelopeViolation,
        weight=_LOCO_TN_REWARD_WEIGHT,
        params={
            "safe_tn_ratio": _LOCO_TN_SAFE_RATIO,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JointGroup.POLICY_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )
    grouped_events = dict(cfg.events)

    root_params = dict(grouped_events["reset_root_state"].params)
    root_params.pop("pose_weights_by_group", None)
    source_stages_by_group = root_params.pop("source_curriculum_stages_by_group", {})
    root_params.update(
        {
            # 站立份额 0.08 -> 0.25（倒地四类按原比例缩到 0.75）：分组唯一被证实的贡献是
            # 50% 保底站立练习；0.08 时两条 Ungrouped run（yzau6pg5/ayre2wda）的站立平衡
            # 都拖到 iter 2400-2800 才学会（upright_gate 0.90->0.97 的爬升段）。
            "pose_weights": (0.25, 0.14, 0.14, 0.235, 0.235),
            "source_curriculum_stages": source_stages_by_group.get("recover"),
        }
    )
    if root_params["source_curriculum_stages"] is not None:
        root_params["source_curriculum_stages"] = [
            {
                **stage,
                "near_upright_ratio": 0.05 if int(stage["iteration"]) >= 3400 else 0.0,
            }
            for stage in root_params["source_curriculum_stages"]
        ]

    joint_params = dict(grouped_events["reset_joints"].params)
    joint_params.pop("randomization_group_names", None)

    cfg.events = {}
    reset_scene = grouped_events.get("reset_scene_to_default")
    if reset_scene is not None:
        cfg.events["reset_scene_to_default"] = reset_scene
    cfg.events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_recovery_discovery_mixed,
        mode="reset",
        params=root_params,
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params=joint_params,
    )
    for event_name in (
        "friction",
        "restitution",
        "base_mass",
        "inertia",
        "com",
        "pd_gains",
        "default_dof_pos",
        "snap_root_to_collision_clearance",
    ):
        event = grouped_events.get(event_name)
        if event is not None:
            cfg.events[event_name] = event
    grouped_push = grouped_events.get("push_robots_loco")
    if grouped_push is not None:
        wrapped_push = grouped_push.params["wrapped_term"]
        cfg.events["push_robots"] = EventTermCfg(
            func=wrapped_push["func"],
            mode="interval",
            interval_range_s=grouped_push.interval_range_s,
            params=dict(wrapped_push["params"]),
        )

    critic_cfg = cfg.observations["critic"]
    cfg.observations["critic"] = replace(
        critic_cfg,
        terms={name: term for name, term in critic_cfg.terms.items() if name != "group_id"},
    )
    cfg.terminations.pop("loco_bad_orientation", None)
    cfg.terminations.pop("loco_base_contact", None)
    if not play:
        # 武装式宽限翻倒终止：只约束「站稳过又翻倒且宽限内救不回」的 episode，
        # 倒地开局与自救成功均不受影响。参数由 yzau6pg5@4999 名义 plant 探测定标：
        # 起身 p99=1.34s/max=1.46s -> 宽限 4s；直立带 max 3.0° -> 30° 留足 DR 余量。
        # armed 后离开直立带立即计时，避免任务 8-11 的 33° 斜靠停车绕过 60° 旧触发角。
        cfg.terminations["graced_fall"] = TerminationTermCfg(
            func=terminations.graced_fall,
            time_out=False,
            params={
                "arm_angle_deg": 30.0,
                "arm_sustain_steps": 25,
                "leave_angle_deg": 30.0,
                "grace_steps": 200,
                "recover_sustain_steps": 10,
                # 站立位姿 reset 出生即武装：修复冷启动死锁（任务 5 实测 armed_rate
                # 全程 0.000，站立开局 ~1s 内翻倒、撑不过 0.5s 武装门槛），使站立
                # 练习以 ~5s 而非 20s 周期回收。
                "arm_at_reset": True,
            },
        )
        cfg.curriculum["commands_vel"] = _ungrouped_velocity_curriculum_cfg()

    _assert_discovery_reward_contract(cfg, reward_profile, ungrouped=True)
    return _apply_actor_history(cfg)


def history_env_cfg(
    play: bool = False,
    reward_profile: DiscoveryRewardProfile = DiscoveryRewardProfile.BASELINE,
) -> ManagerBasedRlEnvCfg:
    """生成五帧 actor 历史观测版本，critic 与其余训练契约保持不变。"""

    return _apply_actor_history(env_cfg(play=play, reward_profile=reward_profile))


def teacher_history_env_cfg(
    play: bool = False,
    reward_profile: DiscoveryRewardProfile = DiscoveryRewardProfile.BASELINE,
) -> ManagerBasedRlEnvCfg:
    """生成倒置起身 scripted teacher 的独立 History-MLP 任务。

    teacher 只在训练配置启用；play/eval 默认关闭，用于直接检验策略是否完成接棒。
    奖励、reset 与课程继承指定的 grouped profile，唯一额外行为是 opt-in 动作变换
    及其配套的 processed last-action 观测。
    """

    cfg = history_env_cfg(play=play, reward_profile=reward_profile)
    action_cfg = cfg.actions["delayed_action"]
    cfg.actions["delayed_action"] = replace(
        action_cfg,
        recovery_teacher=RecoveryTeacherCfg(
            enabled=not play,
            alpha_start=0.85,
            hold_iters=200,
            end_iter=500,
            alpha_end=0.0,
            steps_per_policy_iter=_STEPS_PER_POLICY_ITER,
            sweep_action_indices=(0, 2),
            hard_zero_wheel_indices=(4, 5),
        ),
    )

    for group_name in ("actor", "critic"):
        group_cfg = cfg.observations[group_name]
        terms = dict(group_cfg.terms)
        if "last_actions" not in terms:
            raise RuntimeError(f"Recovery teacher 的 {group_name} 观测缺少 last_actions")
        terms["last_actions"] = replace(
            terms["last_actions"],
            func=mdp_observations.processed_last_actions_obs,
            params={"action_name": "delayed_action"},
        )
        cfg.observations[group_name] = replace(group_cfg, terms=terms)
    return cfg


def torque_assist_history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """生成按当前姿态施加满幅外部扭矩的 GENTLE 任务。

    被采样到的 episode 只要倾角超过 30° 就施加 20 N·m，进入直立带立即撤力并
    清零本次辅助计时；再次跌出直立带重新获得完整 3 s 预算，而不是整个 episode
    只有一次 3 s 累计额度。退火只降 episode 采样概率（iter 0-199 全采样、
    200-499 线性降到 0），不降力矩幅值——原生扫频实测 <=18 N·m 完全翻不起来，
    阶梯降幅方案在首次降到 15 N·m 时即失效。critic 额外获得 3D 辅助状态观测，
    actor 的 34D 部署契约不变。play/eval 默认关闭辅助。
    """

    cfg = history_env_cfg(play=play, reward_profile=DiscoveryRewardProfile.GENTLE)
    action_cfg = cfg.actions["delayed_action"]
    cfg.actions["delayed_action"] = replace(
        action_cfg,
        recovery_torque_assist=RecoveryTorqueAssistCfg(
            enabled=not play,
            torque_nm=20.0,
            min_effective_torque_nm=19.0,
            body_name="base_link",
            body_axis=(0.0, 1.0, 0.0),
            exit_upright_angle_deg=30.0,
            max_assist_time_s=3.0,
            probability_start=1.0,
            hold_iters=200,
            end_iter=500,
            probability_end=0.0,
            steps_per_policy_iter=_STEPS_PER_POLICY_ITER,
        ),
    )

    critic_cfg = cfg.observations["critic"]
    cfg.observations["critic"] = replace(
        critic_cfg,
        terms={
            **critic_cfg.terms,
            "torque_assist_state": ObservationTermCfg(
                func=mdp_observations.recovery_torque_assist_state_obs,
                params={"action_name": "delayed_action"},
            ),
        },
    )
    return cfg
