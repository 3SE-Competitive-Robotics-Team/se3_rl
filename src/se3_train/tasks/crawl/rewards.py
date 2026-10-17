"""趴地移动任务专属奖励：导轮承重与左右主轮对称。

两项都针对第一版（W&B 5rt7ujl1，model_3499）的回放结论：
- 机身离地 0.152 m、导轮只承担 3–4% 体重，实为低姿态两轮平衡；高度罚对抬高 1–2 cm 几乎不收钱。
- 两腿主动杆夹角都顶在上限，左轮在机身下 x +0.06、右轮在后方 x −0.12，左右腿长差 Δz 5.6 cm，机身横滚约 8°。
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared import DM8009P, JointGroup, RobotConfig
from se3_shared.fourbar import policy_to_closedchain_passive_pos_np
from se3_shared.torch_constants import device_constant, device_index
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.diagnostic_logging import should_log_diagnostics
from se3_train.mdp.joint_indices import policy_leg_joint_ids
from se3_train.robot_cfg import GUIDE_WHEEL_MJCF_PATH

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_WEIGHT_ATTR = "_crawl_robot_weight_n"
_MAIN_WHEEL_IDS_ATTR = "_crawl_main_wheel_body_ids"
# 与 mdp.rewards 的诊断日志同一节流：每 64 个 policy step 写一次，速度课程门控也按这个节奏取样。
_LOG_INTERVAL_STEPS = 64
_LOG_INTERVAL_ATTR = "_se3_reward_log_interval_steps"


def _crawl_log(env: ManagerBasedRlEnv) -> dict | None:
    """本步需要写诊断标量时返回 env.extras["log"]，否则 None。"""
    if not hasattr(env, "extras") or not should_log_diagnostics(
        env, _LOG_INTERVAL_STEPS, attr_name=_LOG_INTERVAL_ATTR
    ):
        return None
    return env.extras.setdefault("log", {})


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
    log = _crawl_log(env)
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
    log = _crawl_log(env)
    if isinstance(log, dict):
        log["Crawl/wheel_dx_abs"] = dx.abs().mean()
        log["Crawl/wheel_dz_body_abs"] = dz.abs().mean()
    return dx.square() + dz.square()


@functools.lru_cache(maxsize=4)
def crawl_spring_sag(
    target: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """名义腿 PD 以 target 为目标、零动作静置 3 s 后，policy 腿关节相对 target 的稳态偏移（rad）。

    偏移来自膝气弹簧（300 N）把折叠的腿撑开、kp 60 的 PD 被顶开约 0.3 rad/关节（2026-10-17 实测）。
    在带导轮 MJCF 上用原生 MuJoCo 实算一次并缓存：名义 kp/kd、额定扭矩限幅、无弹簧补偿前馈，
    与训练默认一致；MJCF 或弹簧改了自动跟着变，不写死常数。
    """
    rc = RobotConfig()
    model = mujoco.MjModel.from_xml_path(str(GUIDE_WHEEL_MJCF_PATH))
    model.opt.timestep = rc.sim_dt
    data = mujoco.MjData(model)
    q = np.asarray(target, dtype=np.float64)
    leg = [model.joint(name).id for name in JointGroup.POLICY_LEG_NAMES]
    qadr = [int(model.jnt_qposadr[j]) for j in leg]
    dadr = [int(model.jnt_dofadr[j]) for j in leg]
    passive = [int(model.joint(n).qposadr[0]) for n in JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES]
    data.qpos[3] = 1.0
    data.qpos[qadr] = q
    data.qpos[passive] = policy_to_closedchain_passive_pos_np(q)
    mujoco.mj_forward(model, data)
    data.qpos[2] -= _lowest_support_z(model, data) - 0.002
    limit = float(DM8009P.rated_torque)
    for _ in range(int(3.0 / model.opt.timestep)):
        tau = rc.leg_kp * (q - data.qpos[qadr]) - rc.leg_kd * data.qvel[dadr]
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dadr] = np.clip(tau, -limit, limit)
        mujoco.mj_step(model, data)
    return tuple(float(v) for v in data.qpos[qadr] - q)


def _lowest_support_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """主轮底与导轮底的最低世界高度，用于把静置仿真的初始机身放到地面上。"""
    lows = [data.xpos[model.body(n).id][2] - 0.060 for n in ("l_wheel_Link", "r_wheel_Link")]
    lows += [
        data.xpos[model.body(f"{side}_guide_wheel_{i}_Link").id][2] - 0.010
        for side in "lr"
        for i in range(3)
    ]
    return float(min(lows))


def crawl_pose_deviation(
    env: ManagerBasedRlEnv,
    nominal_policy_leg_pos: tuple[float, float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """四个 policy 腿关节偏离“零动作沉降趴姿”的 L2 范数（rad），始终生效。

    参考 = 实体默认姿态（v4 起即名义趴姿，含 startup DR 的 ±0.05 rad）+ 气弹簧沉降偏移 crawl_spring_sag。
    这样零动作就是本项的最优点：策略不必一直出力顶弹簧（约 8.6 N·m/关节）去追名义角，真机电机不长期带载；
    偏离零动作姿态才扣分。弹簧力与 PD 增益的 DR（各 ±10%）只留下约 ±0.03 rad 的残差。
    不复用 mdp.rewards.joint_pos_penalty：它按高度指令查站姿默认表，0.136 m 查出来的不是趴姿。
    """
    robot = env.scene[asset_cfg.name]
    leg_ids = device_index(policy_leg_joint_ids(robot), device=env.device)
    sag = device_constant(crawl_spring_sag(tuple(nominal_policy_leg_pos)), device=env.device)
    reference = robot.data.default_joint_pos[:, leg_ids] + sag
    deviation = torch.linalg.norm(robot.data.joint_pos[:, leg_ids] - reference, dim=1)
    log = _crawl_log(env)
    if isinstance(log, dict):
        log["Crawl/pose_deviation_rad"] = deviation.mean()
    return deviation


__all__ = [
    "crawl_pose_deviation",
    "crawl_spring_sag",
    "guide_wheel_support",
    "wheel_symmetry_offset",
]
