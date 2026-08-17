"""训练端按名称解析关节和执行器索引。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_SHARED_ROBOT = SharedRobotConfig()


def is_closedchain_model(entity: Any) -> bool:
    """判断当前 MJCF 是否包含闭链主动驱动杆。"""
    joint_names = set(_entity_names(entity, "joint_names"))
    return all(name in joint_names for name in ("l_drive_bar_Joint", "r_drive_bar_Joint"))


def policy_leg_joint_ids(entity: Any) -> tuple[int, ...]:
    """返回 policy 腿部关节索引，闭链为主动杆，开链回退为 lf1/rf1。"""
    return joint_ids(
        entity,
        JointGroup.POLICY_LEG_NAMES,
        fallback_names=JointGroup.OPENCHAIN_LEG_NAMES,
    )


def wheel_joint_ids(entity: Any) -> tuple[int, ...]:
    """返回左右轮关节索引。"""
    return joint_ids(entity, JointGroup.WHEEL_NAMES)


def policy_joint_ids(entity: Any) -> tuple[int, ...]:
    """返回 6 维 policy 关节索引。"""
    return (*policy_leg_joint_ids(entity), *wheel_joint_ids(entity))


def active_rod_joint_pairs(entity: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    """返回同侧两主动杆的前杆/后杆索引对。"""
    leg_ids = policy_leg_joint_ids(entity)
    return ((leg_ids[0], leg_ids[1]), (leg_ids[2], leg_ids[3]))


def active_rod_angle_terms(entity: Any) -> tuple[tuple[int, int, float, float], ...]:
    """返回主动杆夹角表达式：cf * q_front + cb * q_back。"""
    pairs = active_rod_joint_pairs(entity)
    return tuple(
        (front_id, back_id, float(front_coef), float(back_coef))
        for (front_id, back_id), (front_coef, back_coef) in zip(
            pairs,
            _SHARED_ROBOT.active_rod_angle_coeffs,
            strict=True,
        )
    )


def active_leg_mirror_diffs(
    entity: Any,
    joint_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 policy 主动杆坐标下的左右镜像误差。"""
    lf0_id, lb_id, rf0_id, rb_id = policy_leg_joint_ids(entity)
    if is_closedchain_model(entity):
        return joint_pos[:, lf0_id] + joint_pos[:, rf0_id], joint_pos[:, lb_id] + joint_pos[
            :, rb_id
        ]
    return joint_pos[:, lf0_id] - joint_pos[:, rf0_id], joint_pos[:, lb_id] - joint_pos[:, rb_id]


def output_leg_mirror_diffs(
    entity: Any,
    joint_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回输出腿坐标下的左右镜像误差。"""
    lf0_id, lf1_id, rf0_id, rf1_id = output_leg_joint_ids(entity)
    if is_closedchain_model(entity):
        return joint_pos[:, lf0_id] + joint_pos[:, rf0_id], joint_pos[:, lf1_id] + joint_pos[
            :, rf1_id
        ]
    return joint_pos[:, lf0_id] - joint_pos[:, rf0_id], joint_pos[:, lf1_id] - joint_pos[:, rf1_id]


def output_leg_joint_ids(entity: Any) -> tuple[int, ...]:
    """返回输出机构腿部关节索引。"""
    return joint_ids(entity, JointGroup.OUTPUT_LEG_NAMES)


def output_knee_joint_ids(entity: Any) -> tuple[int, ...]:
    """返回被动输出小腿关节索引。"""
    return joint_ids(entity, JointGroup.OUTPUT_KNEE_NAMES)


def leg_actuator_ids(entity: Any) -> tuple[int, ...]:
    """返回 policy 腿部电机 actuator 索引，排除气弹簧 actuator。"""
    return actuator_ids(
        entity,
        JointGroup.POLICY_LEG_NAMES,
        fallback_names=JointGroup.OPENCHAIN_LEG_NAMES,
    )


def wheel_actuator_ids(entity: Any) -> tuple[int, ...]:
    """返回左右轮电机 actuator 索引。"""
    return actuator_ids(entity, JointGroup.WHEEL_NAMES)


def tensor_ids(ids: Sequence[int], *, device: torch.device | str) -> torch.Tensor:
    """把索引转成当前设备上的 long tensor。"""
    return torch.tensor(tuple(ids), device=device, dtype=torch.long)


def joint_ids(
    entity: Any,
    names: Sequence[str],
    *,
    fallback_names: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """按关节名返回 entity-local 索引。"""
    return _ids_from_names(entity, "joint_names", names, fallback_names=fallback_names)


def actuator_ids(
    entity: Any,
    names: Sequence[str],
    *,
    fallback_names: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """按 actuator 名返回 entity-local 索引。"""
    return _ids_from_names(entity, "actuator_names", names, fallback_names=fallback_names)


def _ids_from_names(
    entity: Any,
    attr: str,
    names: Sequence[str],
    *,
    fallback_names: Sequence[str] | None,
) -> tuple[int, ...]:
    available = _entity_names(entity, attr)
    selected = _select_names(available, names, fallback_names=fallback_names)
    name_to_idx = {name: idx for idx, name in enumerate(available)}
    return tuple(name_to_idx[name] for name in selected)


def _select_names(
    available: Sequence[str],
    names: Sequence[str],
    *,
    fallback_names: Sequence[str] | None,
) -> tuple[str, ...]:
    available_set = set(available)
    primary = tuple(names)
    if all(name in available_set for name in primary):
        return primary
    fallback = tuple(fallback_names or ())
    if fallback and all(name in available_set for name in fallback):
        return fallback
    missing = [name for name in primary if name not in available_set]
    raise ValueError(f"模型缺少关节/执行器: {missing}; 可用名称: {tuple(available)}")


def _entity_names(entity: Any, attr: str) -> tuple[str, ...]:
    names = getattr(entity, attr)
    return tuple(str(name) for name in names)
