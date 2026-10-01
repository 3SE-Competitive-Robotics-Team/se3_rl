"""跳跃 mimic 任务专属 MDP 项：参考帧观测、模仿奖励、偏离终止、参考状态初始化。

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
from se3_train.mdp.events import _apply_policy_leg_reset
from se3_train.mdp.joint_indices import policy_leg_joint_ids, tensor_ids, wheel_joint_ids

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from .commands import JumpMimicCommandTerm

_ASSET = SceneEntityCfg("robot")
DEFAULT_OFFSETS_STEPS: tuple[int, ...] = (0, 2, 5, 10)
"""参考帧偏移（policy step）：当前、+40 ms、+100 ms、+200 ms。"""
FEATURES_PER_FRAME = 5
REFERENCE_SCALES = {"leg_len_err": 5.0, "base_z_rel": 5.0, "base_vz": 0.5, "contact": 1.0}
"""参考帧各特征缩放；部署契约 jump_reference.scales 与 runtime 播放器同式。"""


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


# ---------------------------------------------------------------- 观测
def jump_reference_obs(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    offsets_steps: tuple[int, ...] = DEFAULT_OFFSETS_STEPS,
) -> torch.Tensor:
    """参考帧观测，每帧 5 维 × len(offsets_steps)：
    [左腿长差×5, 右腿长差×5, (参考机身高度−站姿)×5, 参考竖直速度×0.5, 参考接触]，腿长差 = 参考 − 当前。
    """
    term = _term(env, command_name)
    current = leg_lengths(env)
    stand = term.library.stand_height
    feats = []
    for off in offsets_steps:
        ref = term.reference(off * float(env.step_dt))
        feats.append(
            torch.cat(
                [
                    (ref.leg_len - current) * REFERENCE_SCALES["leg_len_err"],
                    ((ref.base_z - stand) * REFERENCE_SCALES["base_z_rel"]).unsqueeze(-1),
                    (ref.base_vz * REFERENCE_SCALES["base_vz"]).unsqueeze(-1),
                    (ref.contact * REFERENCE_SCALES["contact"]).unsqueeze(-1),
                ],
                dim=-1,
            )
        )
    return torch.cat(feats, dim=-1)


def jump_reference_state_obs(
    env: ManagerBasedRlEnv, command_name: str = "velocity_height"
) -> torch.Tensor:
    """critic 特权：[正在跳, 参考进度 t/时长, 参考编号 one-hot]。"""
    term = _term(env, command_name)
    progress = term.ref_t / term.library.duration[term.ref_id]
    one_hot = torch.nn.functional.one_hot(term.ref_id, term.library.num_refs).float()
    active = term.active.float()
    return torch.cat(
        [active.unsqueeze(-1), (progress * active).unsqueeze(-1), one_hot * active.unsqueeze(-1)],
        -1,
    )


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
    rsi_prob: float = 0.5,
) -> None:
    """按参考写机器人状态：rsi_prob 的 env 从随机参考的随机时刻开始（RSI），其余从站姿开始；朝向随机。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    term = _term(env, command_name)
    term.pre_resample_for_reset(env_ids)
    lib = term.library
    n = len(env_ids)
    rsi = torch.rand(n, device=env.device) < float(rsi_prob)
    ref_id = torch.randint(0, lib.num_refs, (n,), device=env.device)
    t = torch.rand(n, device=env.device) * lib.duration[ref_id] * rsi.float()
    if bool(rsi.any()):
        term.start_reference(env_ids[rsi], ref_id[rsi], t[rsi])
    frame = lib.frame(ref_id, t)

    robot = env.scene[_ASSET.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, tensor_ids(wheel_joint_ids(robot), device=env.device)] = 0.0
    leg_vel = frame.leg_vel * rsi.float().unsqueeze(-1)
    _apply_policy_leg_reset(
        robot,
        joint_pos,
        joint_vel,
        tensor_ids(policy_leg_joint_ids(robot), device=env.device),
        frame.leg_pos.clone(),
        leg_vel,
    )
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    pos = env.scene.env_origins[env_ids].clone()
    pos[:, 2] += frame.base_z
    yaw = (torch.rand(n, device=env.device) * 2.0 - 1.0) * math.pi
    zeros = torch.zeros_like(yaw)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 2] = frame.base_vz * rsi.float()
    robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


__all__ = [
    "DEFAULT_OFFSETS_STEPS",
    "FEATURES_PER_FRAME",
    "REFERENCE_SCALES",
    "jump_reference_obs",
    "jump_reference_state_obs",
    "mimic_base_height",
    "mimic_base_vz",
    "mimic_contact",
    "mimic_deviation",
    "mimic_leg_length",
    "not_jumping",
    "reset_jump_mimic",
]
