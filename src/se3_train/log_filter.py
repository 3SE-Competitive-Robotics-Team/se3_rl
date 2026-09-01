"""训练日志常驻键过滤。

诊断项的计算与写入代码一律保留，这里只裁剪最终写给 logger 的键集合：
默认只保留奖励记账、终止、课程三个完整命名空间，加上一份显式的
``Recovery/`` 与 ``Reset/`` 常驻清单。设 ``SE3_VERBOSE_DIAG=1`` 恢复全量。

背景：job21 基线单轮写出 371 个键，其中 260 个在 ``Recovery/`` 命名空间下、
115 个是同一指标的分箱展开（23 个族 × 3-6 档），且全仓库没有任何脚本消费它们。
"""

from __future__ import annotations

import os

VERBOSE_DIAGNOSTICS_ENV = "SE3_VERBOSE_DIAG"

# 整个命名空间常驻：奖励记账、终止统计、课程状态都是判读训练的必需项。
_KEEP_NAMESPACES = (
    "Episode_Reward/",
    "Episode_Termination/",
    "Curriculum/",
)

_KEEP_EXACT = frozenset(
    {
        # 起身姿态与失败率的顶层量
        "Recovery/bad_orientation_raw_rate",
        "Recovery/bad_orientation_counted_rate",
        "Recovery/bad_orientation_grace_rate",
        "Recovery/bad_orientation_termination_rate",
        "Recovery/diag_tilt_deg_mean",
        "Recovery/diag_upright_15deg_rate",
        "Recovery/diag_upright_30deg_rate",
        "Recovery/diag_height_error_abs_m",
        "Recovery/diag_action_delta_norm",
        "Recovery/diag_active_target_clamp_rate",
        "Recovery/diag_leg_contact_rate",
        "Recovery/wheel_air_ratio",
        "Recovery/wheel_air_velocity_penalty",
        # reset 构成：起身样本从哪来、跌倒姿态怎么分布
        "Reset/source_cache_ratio",
        "Reset/source_standard_ratio",
        "Reset/source_curriculum_progress",
        "Reset/standard_pose_standing_ratio",
        "Reset/standard_pose_left_side_ratio",
        "Reset/standard_pose_right_side_ratio",
        "Reset/standard_pose_prone_ratio",
        "Reset/standard_pose_supine_ratio",
        "Reset/standard_pose_mean_init_tilt_deg",
        "Reset/joint_randomization_prob",
    }
)

_KEEP_PREFIXES = (
    # 外部扭矩引导：只在 TorqueAssist 任务出现，是该任务的判读依据。
    "Recovery/diag_torque_assist_",
    # 唯一按跌倒姿态（pitch_inverted / roll_side / mixed_full）拆开的族，
    # 是区分「倒置起不来」和「侧卧起不来」的唯一现成口子，连同样本数一起留。
    "Recovery/diag_upright_15deg_rate_by_reset_pose/",
    "Recovery/diag_upright_30deg_rate_by_reset_pose/",
    "Recovery/diag_sample_rate_by_reset_pose/",
)


def verbose_diagnostics_enabled() -> bool:
    """是否保留全部诊断键。"""

    return os.environ.get(VERBOSE_DIAGNOSTICS_ENV, "0") == "1"


def keep_log_key(key: str) -> bool:
    """判断一个日志键是否常驻。"""

    if key.startswith(_KEEP_NAMESPACES):
        return True
    if key in _KEEP_EXACT:
        return True
    return key.startswith(_KEEP_PREFIXES)


def filter_extras_log(extras: object) -> tuple[int, int]:
    """就地裁剪 ``extras["log"]``，返回 ``(保留数, 丢弃数)``。

    只删字典条目，不触碰 ``extras`` 的其余内容（``time_outs`` 等 PPO 依赖项）。
    """

    if verbose_diagnostics_enabled():
        return (0, 0)
    if not isinstance(extras, dict):
        return (0, 0)
    log = extras.get("log")
    if not isinstance(log, dict):
        return (0, 0)
    dropped = [key for key in log if not keep_log_key(key)]
    for key in dropped:
        del log[key]
    return (len(log), len(dropped))


__all__ = [
    "VERBOSE_DIAGNOSTICS_ENV",
    "filter_extras_log",
    "keep_log_key",
    "verbose_diagnostics_enabled",
]
