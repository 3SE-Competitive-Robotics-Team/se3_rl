"""AMP 判别器输入：契约 19 维运动状态单帧（docs/amp_input.md、se3_shared.amp）按 AMP_DISCRIMINATOR_FIELDS 切列。

单帧 = [机身系重力(3), 机身角速度(3), 机身 link 原点线速度(3),
        左右轮心相对髋轴的 xz 位置(4), 其随体时间导数(4), 左右轮自转(2)]，
物理左轮在前，机身坐标右手「前、左、上」，全部原始 SI 单位；判别器自带共享的归一化。

本文件只负责从 mjlab 的 link 运动学取世界系量，几何换算交给 `se3_shared.amp.amp_frame_from_world`，
与复旦数据导出脚本（scripts/export_fudan_amp_features.py）走同一个函数，保证两边口径一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from se3_shared.amp import AMP_FEATURE_NAMES, amp_frame_from_world
from se3_train.mdp.joint_indices import wheel_joint_ids
from se3_train.mdp.observations import _finite_clamp

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 物理左、右：SerialLeg 左轮轴为 +Y（l_wheel_Link），右轮轴为 −Y（r_wheel_Link）。
AMP_WHEEL_BODY_SUFFIXES = ("l_wheel_Link", "r_wheel_Link")
AMP_HIP_BODY_SUFFIXES = ("lf0_Link", "rf0_Link")
# 关节轮速乘以该符号后，正值统一表示向前滚动（2026-09-08 用 R3 model_500 前进回放实测：
# 左 +8.1 rad/s、右 −7.9 rad/s 对应 vx 0.47 m/s）。
AMP_WHEEL_SPIN_SIGNS = (1.0, -1.0)
# 判别器实际看的字段（2026-09-08 用户定，A2）：去掉左右轮自转——专家数据里轮速是打滑/悬空时的读数
# （22±41 rad/s，策略 0.8±4.5），判别器仅凭它就能分开两边、风格信号无梯度；几何/速度字段两车同尺寸可直接比。
AMP_DISCRIMINATOR_FIELDS: tuple[str, ...] = tuple(n for n in AMP_FEATURE_NAMES if not n.endswith("_spin"))
_IDS_ATTR = "_se3_amp_body_ids"
_FIELD_IDX_ATTR = "_se3_amp_field_idx"


def amp_field_indices(fields: tuple[str, ...] | list[str]) -> list[int]:
    """契约字段名 → 19 维帧内下标；字段必须存在、非空且不重复。"""
    names = list(AMP_FEATURE_NAMES)
    unknown = [f for f in fields if f not in names]
    if unknown or not fields or len(set(fields)) != len(fields):
        raise ValueError(f"AMP 字段非法：未知 {unknown}，请求 {list(fields)}")
    return [names.index(f) for f in fields]


def _body_ids(env: ManagerBasedRlEnv, suffixes: tuple[str, ...], names: list[str]) -> list[int]:
    found = []
    for suffix in suffixes:
        matches = [i for i, name in enumerate(names) if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"AMP body '{suffix}' 匹配到 {len(matches)} 个：{names}")
        found.append(matches[0])
    return found


def _amp_ids(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(轮 body ids[2], 髋 body ids[2], 轮关节 ids[2], 轮速符号[2])，按 env 缓存。"""
    cached = getattr(env, _IDS_ATTR, None)
    if cached is None:
        robot = env.scene["robot"]
        names = list(robot.body_names)
        wheels = torch.tensor(_body_ids(env, AMP_WHEEL_BODY_SUFFIXES, names), device=env.device, dtype=torch.long)
        hips = torch.tensor(_body_ids(env, AMP_HIP_BODY_SUFFIXES, names), device=env.device, dtype=torch.long)
        joint_ids = list(wheel_joint_ids(robot))
        joint_names = [robot.joint_names[i] for i in joint_ids]
        if joint_names != ["l_wheel_Joint", "r_wheel_Joint"]:
            raise ValueError(f"轮关节顺序应为 [l_wheel_Joint, r_wheel_Joint]，实际 {joint_names}")
        joints = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
        signs = torch.tensor(AMP_WHEEL_SPIN_SIGNS, device=env.device, dtype=torch.float32)
        cached = (wheels, hips, joints, signs)
        setattr(env, _IDS_ATTR, cached)
    return cached


def amp_motion_frame(
    env: ManagerBasedRlEnv, fields: tuple[str, ...] = AMP_DISCRIMINATOR_FIELDS
) -> torch.Tensor:
    """AMP 运动状态单帧 [B, len(fields)]：先按契约算 19 维（AMP_FEATURE_NAMES），再按 fields 切列。"""
    robot = env.scene["robot"]
    wheels, hips, joints, signs = _amp_ids(env)
    data = robot.data
    frame = amp_frame_from_world(
        rotation_world_from_body=matrix_from_quat(data.root_link_quat_w),
        base_lin_vel_world=data.root_link_lin_vel_w,
        base_ang_vel_world=data.root_link_ang_vel_w,
        wheel_pos_world=data.body_link_pos_w[:, wheels],
        hip_pos_world=data.body_link_pos_w[:, hips],
        wheel_lin_vel_world=data.body_link_lin_vel_w[:, wheels],
        hip_lin_vel_world=data.body_link_lin_vel_w[:, hips],
        wheel_spin_forward=data.joint_vel[:, joints] * signs,
    )
    frame = _finite_clamp(frame)
    key = tuple(fields)
    if key == tuple(AMP_FEATURE_NAMES):
        return frame
    cache: dict = getattr(env, _FIELD_IDX_ATTR, None) or {}
    idx = cache.get(key)
    if idx is None:
        idx = torch.tensor(amp_field_indices(key), device=env.device, dtype=torch.long)
        cache[key] = idx
        setattr(env, _FIELD_IDX_ATTR, cache)
    return frame[:, idx]


def amp_terrain_mask(env: ManagerBasedRlEnv, terrain_type_names: tuple[str, ...] = ("stairs_up",)) -> torch.Tensor:
    """[B, 1]：env 是否在允许 AMP 生效的子地形列上（1/0）。非课程地形（无分列）时全 1。

    照 kyber fork 的 enabled_group_mask：只有这些 env 拿风格奖励、进判别器的策略窗口。
    平地热身期所有 env 都在平地列，掩码全 0，判别器不更新、预热计数不走，换列后才开始。
    """
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if not terrain_type_names or generator is None or terrain_types is None or not generator.curriculum:
        return torch.ones(env.num_envs, 1, device=env.device)
    names = list(generator.sub_terrains.keys())
    allowed = [names.index(n) for n in terrain_type_names if n in names]
    types = terrain_types.to(device=env.device, dtype=torch.long)
    active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for col in allowed:
        active |= types == col
    return active.to(dtype=torch.float32).unsqueeze(-1)


def amp_obs_dim(fields: tuple[str, ...] = AMP_DISCRIMINATOR_FIELDS) -> int:
    return len(amp_field_indices(tuple(fields)))


def build_amp_obs_terms(fields: tuple[str, ...] = AMP_DISCRIMINATOR_FIELDS) -> dict[str, ObservationTermCfg]:
    """AMP 观测组唯一一项：按 fields 切列的运动帧（无噪声、无缩放）；数据集侧必须传同一个 fields。"""
    amp_field_indices(tuple(fields))
    return {"motion_frame": ObservationTermCfg(func=amp_motion_frame, params={"fields": tuple(fields)})}


def build_amp_mask_terms(terrain_type_names: tuple[str, ...]) -> dict[str, ObservationTermCfg]:
    """AMP 掩码观测组唯一一项：所在地形列是否启用 AMP。"""
    return {
        "terrain": ObservationTermCfg(
            func=amp_terrain_mask, params={"terrain_type_names": tuple(terrain_type_names)}
        )
    }


__all__ = [
    "AMP_DISCRIMINATOR_FIELDS",
    "AMP_HIP_BODY_SUFFIXES",
    "AMP_WHEEL_BODY_SUFFIXES",
    "AMP_WHEEL_SPIN_SIGNS",
    "amp_field_indices",
    "amp_motion_frame",
    "amp_obs_dim",
    "amp_terrain_mask",
    "build_amp_mask_terms",
    "build_amp_obs_terms",
]
