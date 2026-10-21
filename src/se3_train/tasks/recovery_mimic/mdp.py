"""起身 mimic 任务专属 MDP 项：参考观测、跟踪奖励、姿态偏离终止、从倒地状态库出生的 reset。

参考时钟就是 episode 时间（t = episode_length_buf × step_dt），每个 episode 从参考第 0 帧（倒地）开始；
起点从状态库（se3_tools.fall_state_cache）里与参考同一家族的状态抽，朝向随机。
跟踪对象：机身重力投影（全姿态）、四个腿电机角（policy 主动杆语义，大腿可转整圈，误差按 2π 取最近）、机身高度、
腿速、机身角速度。轮速不跟踪（参考里轮速目标 0，留给正则）。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

from se3_shared import JointGroup
from se3_train.mdp.joint_indices import policy_leg_joint_ids, tensor_ids
from se3_train.robot_cfg import _MJCF_PATH

from .reference import RecoveryReferenceLibrary

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ASSET = SceneEntityCfg("robot")
_STATE_ATTR = "_se3_recovery_mimic_state"


class _MimicState:
    """参考库、每个 env 的参考编号、腿关节索引。"""

    def __init__(self, env: ManagerBasedRlEnv, reference_paths: tuple[str, ...]):
        self.library = RecoveryReferenceLibrary(tuple(reference_paths), env.device)
        self.ref_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.leg_ids = tensor_ids(policy_leg_joint_ids(env.scene[_ASSET.name]), device=env.device)
        self.starts: dict | None = None


def _state(env: ManagerBasedRlEnv, reference_paths: tuple[str, ...]) -> _MimicState:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = _MimicState(env, reference_paths)
        setattr(env, _STATE_ATTR, state)
    return state


def _time(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf.float() * float(env.step_dt)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _reference(env: ManagerBasedRlEnv, reference_paths: tuple[str, ...]):
    state = _state(env, reference_paths)
    return state, state.library.frame(state.ref_id, _time(env))


def _leg_error(env: ManagerBasedRlEnv, state: _MimicState, ref_leg: torch.Tensor) -> torch.Tensor:
    """参考腿角 − 当前腿角，按 2π 取最近（大腿可转整圈，参考里转了约 200°）。"""
    return _wrap(ref_leg - env.scene[_ASSET.name].data.joint_pos[:, state.leg_ids])


# ---------------------------------------------------------------- 观测
def recovery_reference_obs(
    env: ManagerBasedRlEnv, reference_paths: tuple[str, ...]
) -> torch.Tensor:
    """参考信息 [N, 9]：相位（t / 参考时长，夹到 1）、参考重力投影 3、腿角误差 4、参考机身高度 1。教师专用特权观测。"""
    state, ref = _reference(env, reference_paths)
    phase = torch.clamp(_time(env) / state.library.duration[state.ref_id], max=1.0)
    return torch.cat(
        (
            phase.unsqueeze(1),
            ref.projected_gravity,
            _leg_error(env, state, ref.leg_pos),
            ref.base_z.unsqueeze(1),
        ),
        dim=1,
    )


# ---------------------------------------------------------------- 奖励
def mimic_orientation(
    env: ManagerBasedRlEnv, reference_paths: tuple[str, ...], sigma: float = 0.4
) -> torch.Tensor:
    """机身姿态跟踪 exp(−‖g_b − g_ref‖² / σ²)（全角度，倒置也连续）；顺带记姿态误差与参考末段的直立率。"""
    state, ref = _reference(env, reference_paths)
    pg = env.scene[_ASSET.name].data.projected_gravity_b
    err = (pg - ref.projected_gravity).square().sum(dim=1)
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        cos = torch.clamp((pg * ref.projected_gravity).sum(dim=1), -1.0, 1.0)
        log["RecoveryMimic/orientation_err_deg"] = torch.rad2deg(torch.acos(cos)).mean()
        log["RecoveryMimic/leg_err_rad"] = _leg_error(env, state, ref.leg_pos).abs().mean()
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
        late = (_time(env) >= state.library.duration[state.ref_id] - 1.5).float()
        log["RecoveryMimic/upright_rate_late"] = (
            (tilt < 15.0).float() * late
        ).sum() / late.sum().clamp(min=1.0)
    return torch.exp(-err / sigma**2)


def mimic_leg_pos(
    env: ManagerBasedRlEnv, reference_paths: tuple[str, ...], sigma: float = 0.5
) -> torch.Tensor:
    """四个腿电机角跟踪 exp(−Σ Δq² / σ²)。"""
    state, ref = _reference(env, reference_paths)
    return torch.exp(-_leg_error(env, state, ref.leg_pos).square().sum(dim=1) / sigma**2)


def mimic_base_height(
    env: ManagerBasedRlEnv, reference_paths: tuple[str, ...], sigma: float = 0.05
) -> torch.Tensor:
    """机身离地高度跟踪 exp(−Δz² / σ²)（平地：根节点 z − env 原点 z）。"""
    _, ref = _reference(env, reference_paths)
    z = env.scene[_ASSET.name].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-(ref.base_z - z).square() / sigma**2)


def mimic_leg_vel(
    env: ManagerBasedRlEnv, reference_paths: tuple[str, ...], sigma: float = 3.0
) -> torch.Tensor:
    """腿电机速度跟踪 exp(−Σ Δq̇² / σ²)。"""
    state, ref = _reference(env, reference_paths)
    vel = env.scene[_ASSET.name].data.joint_vel[:, state.leg_ids]
    return torch.exp(-(ref.leg_vel - vel).square().sum(dim=1) / sigma**2)


def mimic_body_ang_vel(
    env: ManagerBasedRlEnv, reference_paths: tuple[str, ...], sigma: float = 2.0
) -> torch.Tensor:
    """机身角速度（机身系）跟踪 exp(−‖Δω‖² / σ²)。"""
    _, ref = _reference(env, reference_paths)
    omega = env.scene[_ASSET.name].data.root_link_ang_vel_b
    return torch.exp(-(ref.base_ang_vel_b - omega).square().sum(dim=1) / sigma**2)


# ---------------------------------------------------------------- 终止
def orientation_deviation(
    env: ManagerBasedRlEnv,
    reference_paths: tuple[str, ...],
    max_error_deg: float = 60.0,
    grace_s: float = 0.3,
) -> torch.Tensor:
    """机身姿态与参考夹角超过 max_error_deg → 提前终止（DeepMimic early termination）；出生后 grace_s 内不判。"""
    _, ref = _reference(env, reference_paths)
    pg = env.scene[_ASSET.name].data.projected_gravity_b
    cos = torch.clamp((pg * ref.projected_gravity).sum(dim=1), -1.0, 1.0)
    return (cos < math.cos(math.radians(max_error_deg))) & (_time(env) > grace_s)


# ---------------------------------------------------------------- reset
def _load_starts(
    env: ManagerBasedRlEnv,
    cache_path: str,
    family: tuple[int, float, bool],
    reference_leg_pos: np.ndarray,
    max_leg_error_rad: float,
) -> dict:
    """从倒地状态库取与参考同一家族、且腿姿接近参考起点的状态，关节列按实体顺序重排。

    家族 = 重力投影主轴、符号、腿是否折叠。同家族里约一半样本的大腿朝向与参考起点相反（差约 π），
    从那里出发要先把腿转半圈才接得上参考，需要另一条反向参考；这里按四个腿电机角与参考第 0 帧的最大误差
    （按 2π 取最近）≤ max_leg_error_rad 过滤。"""
    data = np.load(cache_path)
    meta = json.loads(str(data["metadata_json"]))
    if meta.get("mjcf_sha256") != hashlib.sha256(_MJCF_PATH.read_bytes()).hexdigest():
        raise ValueError(f"倒地状态库 {Path(cache_path).name} 与当前 MJCF 不一致，需重新生成")
    names = [str(n) for n in data["joint_names"]]
    cols = [names.index(n) for n in JointGroup.POLICY_LEG_NAMES]
    jp, pg = data["joint_pos"], data["projected_gravity"]
    alpha = np.maximum(jp[:, cols[0]] - jp[:, cols[1]], jp[:, cols[3]] - jp[:, cols[2]])
    axis = np.argmax(np.abs(pg), 1)
    sign = np.sign(pg[np.arange(len(pg)), axis])
    ax, sg, folded = family
    mask = (axis == int(ax)) & (sign == float(sg)) & ((alpha < 0.15) if folded else (alpha >= 0.5))
    diff = jp[:, cols] - np.asarray(reference_leg_pos)[None, :]
    mask &= np.abs(np.arctan2(np.sin(diff), np.cos(diff))).max(axis=1) <= float(max_leg_error_rad)
    if not mask.any():
        raise ValueError(f"状态库里没有家族 {family}、腿姿接近参考起点的样本")
    entity_names = tuple(env.scene[_ASSET.name].joint_names)
    order = [names.index(n) for n in entity_names]

    def tensor(a):
        return torch.as_tensor(a[mask], dtype=torch.float32, device=env.device)

    return {
        "root_z": tensor(data["root_pos"])[:, 2],
        "root_quat": tensor(data["root_quat"]),
        "joint_pos": tensor(jp)[:, order],
        "size": int(mask.sum()),
    }


def reset_recovery_mimic(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    reference_paths: tuple[str, ...],
    cache_path: str,
    family: tuple[int, float, bool],
    max_start_leg_error_rad: float = 0.5,
    xy_offset_m: float = 0.1,
) -> None:
    """从状态库同家族、腿姿接近参考起点的状态出生：绕世界 z 随机 yaw、xy ±xy_offset_m，静止；参考编号随机选（目前只有一条）。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env_ids.to(device=env.device, dtype=torch.long)
    state = _state(env, reference_paths)
    if state.starts is None:
        ref0 = state.library.leg_pos[0, 0].detach().cpu().numpy()
        state.starts = _load_starts(env, cache_path, family, ref0, max_start_leg_error_rad)
    starts = state.starts
    n = ids.numel()
    k = torch.randint(starts["size"], (n,), device=env.device)
    state.ref_id[ids] = torch.randint(state.library.num_refs, (n,), device=env.device)

    yaw = (torch.rand(n, device=env.device) * 2.0 - 1.0) * math.pi
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
    quat = quat_mul(quat_from_angle_axis(yaw, z_axis), starts["root_quat"][k])
    pos = env.scene.env_origins[ids].clone()
    pos[:, :2] += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * float(xy_offset_m)
    pos[:, 2] += starts["root_z"][k]
    robot = env.scene[_ASSET.name]
    robot.write_root_link_pose_to_sim(torch.cat((pos, quat), dim=1), env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
    robot.write_joint_state_to_sim(
        starts["joint_pos"][k], torch.zeros_like(starts["joint_pos"][k]), env_ids=ids
    )


__all__ = [
    "mimic_base_height",
    "mimic_body_ang_vel",
    "mimic_leg_pos",
    "mimic_leg_vel",
    "mimic_orientation",
    "orientation_deviation",
    "recovery_reference_obs",
    "reset_recovery_mimic",
]
