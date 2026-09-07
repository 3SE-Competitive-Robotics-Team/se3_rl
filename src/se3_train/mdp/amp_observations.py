"""AMP 判别器输入：19 维运动状态单帧，契约见 docs/amp_input.md 与 se3_shared.amp。

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

from se3_shared.amp import AMP_FRAME_DIM, amp_frame_from_world
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
_IDS_ATTR = "_se3_amp_body_ids"


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


def amp_motion_frame(env: ManagerBasedRlEnv) -> torch.Tensor:
    """19 维 AMP 运动状态单帧 [B, 19]，见 se3_shared.amp.AMP_FEATURE_NAMES。"""
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
    return _finite_clamp(frame)


def amp_obs_dim() -> int:
    return AMP_FRAME_DIM


def build_amp_obs_terms() -> dict[str, ObservationTermCfg]:
    """AMP 观测组唯一一项：19 维运动帧（无噪声、无缩放）。"""
    return {"motion_frame": ObservationTermCfg(func=amp_motion_frame)}


__all__ = [
    "AMP_HIP_BODY_SUFFIXES",
    "AMP_WHEEL_BODY_SUFFIXES",
    "AMP_WHEEL_SPIN_SIGNS",
    "amp_motion_frame",
    "amp_obs_dim",
    "build_amp_obs_terms",
]
