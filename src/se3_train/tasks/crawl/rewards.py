"""趴地移动任务专属奖励：导轮承重与左右主轮对称。

两项都针对第一版（W&B 5rt7ujl1，model_3499）的回放结论：
- 机身离地 0.152 m、导轮只承担 3–4% 体重，实为低姿态两轮平衡；高度罚对抬高 1–2 cm 几乎不收钱。
- 两腿主动杆夹角都顶在上限，左轮在机身下 x +0.06、右轮在后方 x −0.12，左右腿长差 Δz 5.6 cm，机身横滚约 8°。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared.torch_constants import device_index
from se3_train.mdp.contact_utils import finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_WEIGHT_ATTR = "_crawl_robot_weight_n"
_MAIN_WHEEL_IDS_ATTR = "_crawl_main_wheel_body_ids"


def _robot_weight_n(env: ManagerBasedRlEnv, asset_name: str) -> float:
    """名义整机重力（N），按 MJCF 里该实体全部 body 的质量求和并缓存；质量 DR 不回写，只作归一化尺度。"""
    cached = getattr(env, _ROBOT_WEIGHT_ATTR, None)
    if isinstance(cached, float):
        return cached
    mj_model = env.sim.mj_model
    prefix = f"{asset_name}/"
    mass = 0.0
    for body_id in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith(prefix):
            mass += float(mj_model.body_mass[body_id])
    if mass <= 0.0:
        raise RuntimeError(f"找不到实体 {asset_name!r} 的 body 质量")
    weight = mass * float(abs(mj_model.opt.gravity[2]))
    setattr(env, _ROBOT_WEIGHT_ATTR, weight)
    return weight


def guide_wheel_support(
    env: ManagerBasedRlEnv,
    left_sensor_name: str,
    right_sensor_name: str,
    side_share_target: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """导轮承重不足罚，∈ [0, 1]。

    每侧三个导轮的接触合力之和除以整机重力得到该侧承重占比，低于 side_share_target 的缺口按目标归一化，
    左右取平均。按侧计价而非按总量，单侧抬起（横滚）同样扣分。目标 0.25/侧（合计 50%）：名义趴姿 + 腿 PD
    在原生 MuJoCo 里导轮承担 81–89%，留足余量；只看接触与否抓不到“导轮轻擦地、主轮扛全重”的状态。
    """
    weight = _robot_weight_n(env, asset_cfg.name)
    target = float(side_share_target)
    shares = []
    for sensor_name in (left_sensor_name, right_sensor_name):
        force = env.scene[sensor_name].data.force
        if force is None:
            shares.append(torch.zeros(env.num_envs, device=env.device))
            continue
        shares.append(finite_contact_force_norm(force).sum(dim=1) / weight)
    share = torch.stack(shares, dim=1)
    deficit = torch.clamp(target - share, min=0.0) / target
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        log["Crawl/guide_share_left"] = share[:, 0].mean()
        log["Crawl/guide_share_right"] = share[:, 1].mean()
        log["Crawl/guide_share_total"] = share.sum(dim=1).mean()
    return deficit.mean(dim=1)


def wheel_symmetry_offset(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """左右主轮心在机身系里的前后错位 Δx 与高度差 Δz 的平方和（m²）。

    与 Rough 的 wheel_fore_aft_offset（M49）同为机身系几何罚，但同时罚竖直分量：趴地任务没有 roll 指令，
    左右腿长差只会把机身撑歪。关节空间的 joint_mirror（−0.179）在 Rough 上已证实压不住跨立。
    """
    robot = env.scene[asset_cfg.name]
    body_ids = getattr(env, _MAIN_WHEEL_IDS_ATTR, None)
    if body_ids is None:
        body_ids, names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
        if len(body_ids) != 2:
            raise RuntimeError(f"必须找到左右主轮 body，实际找到: {names}")
        setattr(env, _MAIN_WHEEL_IDS_ATTR, body_ids)
    index = device_index(tuple(body_ids), device=env.device)
    delta_w = robot.data.body_link_pos_w[:, index, :] - robot.data.root_link_pos_w[:, None, :]
    quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    wheel_b = quat_apply_inverse(quat.reshape(-1, 4), delta_w.reshape(-1, 3)).reshape(-1, 2, 3)
    dx = wheel_b[:, 0, 0] - wheel_b[:, 1, 0]
    dz = wheel_b[:, 0, 2] - wheel_b[:, 1, 2]
    log = env.extras.setdefault("log", {}) if hasattr(env, "extras") else None
    if isinstance(log, dict):
        log["Crawl/wheel_dx_abs"] = dx.abs().mean()
        log["Crawl/wheel_dz_body_abs"] = dz.abs().mean()
    return dx.square() + dz.square()


__all__ = ["guide_wheel_support", "wheel_symmetry_offset"]
