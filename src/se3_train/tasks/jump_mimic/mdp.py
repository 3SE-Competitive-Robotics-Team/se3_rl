"""跳跃 mimic 任务专属 MDP 项：模仿奖励、偏离终止、站姿 reset。

模仿对象是每条腿的虚拟腿长（轮心到髋轴距离）、机身高度、机身竖直速度与轮子接触，不逐关节跟踪角度：
参考里轮心 x 固定在默认站姿，若逐关节跟踪等于把轮子前后位置钉死，会和轮腿的前后平衡冲突；腿的摆角留给平衡。
参考只约束局部竖直动作，前进速度、轮速等交给奖励塑形（用户定，2026-10-01）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from se3_shared.fourbar import output_leg_wheel_xz_torch, policy_to_output_pos_torch
from se3_train.mdp import rewards as flat_rewards
from se3_train.mdp.events import _apply_policy_leg_reset
from se3_train.mdp.joint_indices import policy_leg_joint_ids, tensor_ids, wheel_joint_ids

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from .commands import JumpMimicCommandTerm

_ASSET = SceneEntityCfg("robot")
DEFAULT_OFFSETS_STEPS: tuple[int, ...] = (0, 2, 5, 10)
REFERENCE_SCALES = {"leg_len_err": 5.0, "base_z_rel": 5.0, "base_vz": 0.5, "contact": 1.0}
"""部署契约 se3.jump_ref.v1 的 offsets_steps / scales 字段。当前策略不看参考帧（J10 起用一维相位），两项只为契约格式
保留：runtime 解析器要求这两个字段，已部署的 RJ1 artifact 也带着同样的值。"""


def _term(env: ManagerBasedRlEnv, command_name: str) -> JumpMimicCommandTerm:
    return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def leg_lengths(env: ManagerBasedRlEnv) -> torch.Tensor:
    """左右虚拟腿长 [N, 2]（轮心到髋轴距离，m）。"""
    robot = env.scene[_ASSET.name]
    ids = getattr(env, "_jump_mimic_leg_ids", None)
    if ids is None:
        ids = tensor_ids(policy_leg_joint_ids(robot), device=env.device)
        env._jump_mimic_leg_ids = ids
    xz = output_leg_wheel_xz_torch(policy_to_output_pos_torch(robot.data.joint_pos[:, ids]))
    return torch.linalg.norm(xz, dim=-1)


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """机身离地高度（平地：根节点 z − env 原点 z）。"""
    robot = env.scene[_ASSET.name]
    return robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


# ---------------------------------------------------------------- 奖励
def mimic_leg_length(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", sigma: float = 0.03
) -> torch.Tensor:
    """两腿腿长跟踪 exp(−Σ ΔL² / σ²)。"""
    ref = _term(env, command_name).reference()
    err = (ref.leg_len - leg_lengths(env)).square().sum(dim=-1)
    return torch.exp(-err / sigma**2)


def mimic_base_height(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", sigma: float = 0.05
) -> torch.Tensor:
    """机身高度跟踪 exp(−Δz² / σ²)。"""
    term = _term(env, command_name)
    err = term.reference().base_z - base_height(env)
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        active = term.active.float()
        log["Jump/height_err_abs_active"] = (err.abs() * active).sum() / active.sum().clamp(min=1.0)
    return torch.exp(-err.square() / sigma**2)


def mimic_base_vz(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", sigma: float = 0.5
) -> torch.Tensor:
    """机身竖直速度跟踪 exp(−Δvz² / σ²)。"""
    robot = env.scene[_ASSET.name]
    err = _term(env, command_name).reference().base_vz - robot.data.root_link_lin_vel_w[:, 2]
    return torch.exp(-err.square() / sigma**2)


def mimic_contact(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    sensor_name: str = "wheel_sensor",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """两轮接触与参考一致的比例（0/0.5/1）。"""
    force = env.scene[sensor_name].data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact = (
        torch.linalg.norm(torch.nan_to_num(force), dim=-1) > force_threshold
    ).float()  # [N, 2]
    ref = _term(env, command_name).reference().contact.unsqueeze(-1)
    return (contact == ref).float().mean(dim=-1)


def tracking_lin_vel_jump(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height", **kwargs
) -> torch.Tensor:
    """Flat 速度跟踪，跳跃期间去掉核里的 vz 项。

    Flat 核 exp(−(Δvx² + w·vz²)/σ) 在腾空时 vz≈2 m/s，整项归零，跳跃中前进速度没有任何塑形；
    跳跃期间竖直运动由模仿项管，速度跟踪只管 vx（前进跳，用户定 2026-10-01）。
    """
    vz_weight = float(kwargs.pop("vz_weight", 2.0))
    active = _term(env, command_name).active
    weight = torch.where(active, torch.zeros_like(active, dtype=torch.float), vz_weight)
    return flat_rewards.tracking_lin_vel(env, command_name, vz_weight=weight, **kwargs)


def not_jumping(
    env: ManagerBasedRlEnv, inner, params: dict, command_name: str = "velocity_height"
) -> torch.Tensor:
    """任意奖励项在跳跃期间置零（静站罚、轮子大接触力罚等与跳跃动作冲突的项）。"""
    return inner(env, **params) * (~_term(env, command_name).active).float()


# ---------------------------------------------------------------- 终止
def mimic_deviation(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    max_height_error: float = 0.25,
    max_leg_error: float = 0.12,
) -> torch.Tensor:
    """跳跃期间机身高度或任一腿长偏离参考过大 → 提前终止（DeepMimic early termination）。"""
    term = _term(env, command_name)
    ref = term.reference()
    height_bad = (ref.base_z - base_height(env)).abs() > max_height_error
    leg_bad = (ref.leg_len - leg_lengths(env)).abs().amax(dim=-1) > max_leg_error
    return term.active & (height_bad | leg_bad)


# ---------------------------------------------------------------- reset
def reset_jump_mimic(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "velocity_height",
) -> None:
    """按参考站姿（第 0 帧）写机器人状态，静止起步、朝向随机（J9 起不做 RSI）。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    term = _term(env, command_name)
    term.pre_resample_for_reset(env_ids)
    n = len(env_ids)
    zeros_long = torch.zeros(n, dtype=torch.long, device=env.device)
    frame = term.library.frame(zeros_long, torch.zeros(n, device=env.device))

    robot = env.scene[_ASSET.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, tensor_ids(wheel_joint_ids(robot), device=env.device)] = 0.0
    _apply_policy_leg_reset(
        robot,
        joint_pos,
        joint_vel,
        tensor_ids(policy_leg_joint_ids(robot), device=env.device),
        frame.leg_pos.clone(),
        torch.zeros_like(frame.leg_vel),
    )
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    pos = env.scene.env_origins[env_ids].clone()
    pos[:, 2] += frame.base_z
    yaw = (torch.rand(n, device=env.device) * 2.0 - 1.0) * math.pi
    zeros = torch.zeros_like(yaw)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


__all__ = [
    "DEFAULT_OFFSETS_STEPS",
    "REFERENCE_SCALES",
    "mimic_base_height",
    "mimic_base_vz",
    "mimic_contact",
    "mimic_deviation",
    "mimic_leg_length",
    "not_jumping",
    "reset_jump_mimic",
    "tracking_lin_vel_jump",
]
