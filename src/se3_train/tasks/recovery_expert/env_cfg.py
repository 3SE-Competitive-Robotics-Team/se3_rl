"""Recovery 专家单任务环境——从 Recover-Loco-Grouped 完全隔离的纯倒地自启线。

设计目标（专家管线 E1 的载体）：
- episode 分布锁定为纯倒地（标准倒地姿态 + 40k cache 课程），无 loco 群体
- 剔除全部分组机械（AssignEnvGroups / Filtered 包装 / group_id 特权观测）
  与 loco 专属件（速度课程、push、loco 终止、loco TN 分组项）
- 其余契约与 Grouped 任务逐项一致：34D actor 五帧历史、critic 特权包 v2、
  恢复 reset/关节课程、奖励权重（双组 TN 合并为单项，权重与组版相同）

本文件只做结构隔离；专家阶段的降额 plant / 约束 / 奖励瘦身在此基础上
作为独立提交演进。
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from se3_shared import JointGroup
from se3_train.mdp import events as mdp_events
from se3_train.mdp import terminations
from se3_train.tasks.recovery import rewards
from se3_train.tasks.recovery_discovery.env_cfg import (
    _DISCOVERY_REWARD_WEIGHTS,
    _RECOVER_TN_REWARD_WEIGHT,
    _RECOVER_TN_SAFE_RATIO,
    history_env_cfg,
)

# 纯倒地 episode 分布 = 组版 recover 组的姿态权重（standing 0%）。
_EXPERT_POSE_WEIGHTS = (0.0, 0.20, 0.20, 0.30, 0.30)

# 奖励契约：组版权重表去掉双组 TN、合并为单项（权重/阈值与组版相同）。
EXPERT_REWARD_WEIGHTS = {
    **{
        name: weight
        for name, weight in _DISCOVERY_REWARD_WEIGHTS.items()
        if name not in ("tn_envelope_violation_loco", "tn_envelope_violation_recover")
    },
    "tn_envelope_violation": _RECOVER_TN_REWARD_WEIGHT,
}


def _assert_expert_reward_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    actual = set(cfg.rewards)
    expected = set(EXPERT_REWARD_WEIGHTS)
    if actual != expected:
        raise RuntimeError(
            "Recovery-Expert reward contract drifted: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    bad_weights = {
        name: float(cfg.rewards[name].weight)
        for name, expected_weight in EXPERT_REWARD_WEIGHTS.items()
        if abs(float(cfg.rewards[name].weight) - float(expected_weight)) > 1.0e-12
    }
    if bad_weights:
        raise RuntimeError(f"Recovery-Expert reward weight drifted: {bad_weights}")


def expert_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """纯 recover 专家环境：五帧历史 actor + critic v2，episode 全部从倒地开始。"""
    cfg = history_env_cfg(play=play)

    # --- 奖励：双组 TN 合并为无过滤单项（群体即全部 env）---
    cfg.rewards.pop("tn_envelope_violation_loco")
    cfg.rewards.pop("tn_envelope_violation_recover")
    cfg.rewards["tn_envelope_violation"] = RewardTermCfg(
        func=rewards.NormalizedTnEnvelopeViolation,
        weight=_RECOVER_TN_REWARD_WEIGHT,
        params={
            "safe_tn_ratio": _RECOVER_TN_SAFE_RATIO,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JointGroup.POLICY_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    # --- 事件：解除分组包装，reset 分布锁纯倒地 ---
    events = dict(cfg.events)
    events.pop("assign_env_groups")
    events.pop("push_robots_loco", None)  # 组版里 recover 组从不被 push，保持行为对齐

    root_params = dict(events["reset_root_state"].params)
    root_params.pop("pose_weights_by_group")
    stages_by_group = root_params.pop("source_curriculum_stages_by_group")
    root_params["pose_weights"] = _EXPERT_POSE_WEIGHTS
    root_params["source_curriculum_stages"] = stages_by_group["recover"]
    events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_recovery_discovery_mixed,
        mode="reset",
        params=root_params,
    )

    joint_params = dict(events["reset_joints"].params)
    joint_params.pop("randomization_group_names")  # 全部 env 都是 recover 群体
    events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params=joint_params,
    )
    cfg.events = events

    # --- critic：去掉 group_id 特权（其余特权包 v2 原样保留）---
    critic_cfg = cfg.observations["critic"]
    cfg.observations["critic"] = replace(
        critic_cfg,
        terms={name: term for name, term in critic_cfg.terms.items() if name != "group_id"},
    )

    # --- 终止：去 loco 项；恢复停滞判定解除组过滤后全量生效 ---
    cfg.terminations.pop("loco_bad_orientation")
    cfg.terminations.pop("loco_base_contact")
    cfg.terminations["recover_stagnation"] = TerminationTermCfg(
        func=terminations.recovery_stagnation,
        time_out=False,
        params={"max_steps": 300, "min_delta": 0.02},
    )

    # --- 课程：只保留高度课程（速度恒零无课程可言，push 与组版 recover 对齐为无）---
    if not play:
        cfg.curriculum = {"commands_height": cfg.curriculum["commands_height"]}

    _assert_expert_reward_contract(cfg)
    return cfg
