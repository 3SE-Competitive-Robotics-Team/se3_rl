"""复旦 v3 奖励集（yly-true/fudan_rl_wheel_leg `8204e853` 上台阶 v3 快照），用于全地形统一奖励的对照实验。

复旦的奖励不分地形列：每一项乘权重、乘 dt 后逐项限幅到 [−dt, +dt]（`clip_single_reward=1`），即每项每秒贡献
最多 ±1，总和允许为负，没有存活奖励与终止罚。mjlab 的 RewardManager 只做"值 × 权重 × dt"、不支持逐项限幅，
所以这里每个函数自己完成"乘复旦权重 → 限幅到 ±clip"，注册时 RewardTermCfg 权重固定为 1：
每步贡献 dt·clip(权重·原值, ±1)，与复旦 `clip(原值·权重·dt, ±dt)` 逐项等价，`Episode_Reward/*` 就是限幅后的每秒贡献。

与复旦的已知口径差异（机器人与控制频率不同，按复旦原权重照搬、不做换算）：
- 控制频率 50 Hz（复旦 100 Hz）、腿动作缩放 0.25 rad、轮 45 rad/s（复旦 0.5 rad、10 rad/s），
  action_rate / action_smooth 的物理含义随之不同；
- 虚拟腿角 θ0 用髋关节（lf0/rf0_Link 原点）到轮心的连线在机身系里相对竖直的夹角，是闭链四连杆的等效
  虚拟腿，复旦是两连杆 FK；
- 高度指令沿用本仓库的 0.20–0.38 m（台阶列另有随等级抬高的下限），复旦是 0.09–0.33 m；
- 复旦的 collision 项罚的接触体列表为空、恒为 0，这里不加。

高度参考（M29 起）逐步复刻复旦 `_get_heights`：机身周围 11×7 个点（只随偏航转）落到 0.1 m 世界格点上取整，
每点取本格与 +x、+y 两个相邻格点里最低的高度，77 点全部求均值。复旦地形本来就建在这张格点上，
立面上的"取最低"恰好对应它把低处顶点挪到立面下沿的三角网格，读到的是真实表面；斜坡上则比真实表面
平均低半格的坡升（坡度 0.4 时约 2 cm）。我们的地形不在格点上建，所以开训时把整张地形在同一套 0.1 m
格点上的表面高度离线算一遍（box 直接栅格化，hfield 用 mj_ray 从上往下打），每步只查表。
M28 用的是 critic 的 77 条射线打到的真实表面（打空或打到背面的射线丢掉），与此不同。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared.action_history import periodic_policy_action_second_difference_torch
from se3_train.mdp.action_period import front_action_periods_from_env
from se3_train.mdp.joint_indices import wheel_actuator_ids, wheel_joint_ids
from se3_train.mdp.rewards import (
    _DEFAULT_ASSET_CFG,
    _policy_leg_acc,
    _policy_leg_torque_and_vel,
    dof_pos_limits,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg

# 复旦 v3 的 `tracking_sigma`；enhance 项用它的 10 倍作宽核。
FUDAN_TRACKING_SIGMA = 0.25
# 复旦 terrain.horizontal_scale 与 measured_points_x / measured_points_y（机身系，只随偏航转）。
FUDAN_HEIGHT_LATTICE_M = 0.1
FUDAN_MEASURED_POINTS_X: tuple[float, ...] = tuple(round(-0.5 + 0.1 * i, 1) for i in range(11))
FUDAN_MEASURED_POINTS_Y: tuple[float, ...] = tuple(round(-0.3 + 0.1 * i, 1) for i in range(7))
# rough 把机器人碰撞体挪到 group 3（ROUGH_ROBOT_COLLISION_GEOM_GROUP），group 0 只剩地形。
_FUDAN_TERRAIN_GEOM_GROUP = 0
_EDGE_EPS_M = 1e-4
_HEIGHT_LATTICE_ATTR = "_se3_fudan_height_lattice"


def _scaled_clip(raw: torch.Tensor, scale: float, clip: float) -> torch.Tensor:
    """复旦的逐项限幅：权重乘原值后夹到 ±clip（每秒量）。"""
    return torch.clamp(float(scale) * raw, -float(clip), float(clip))


def _vx_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, 0] - env.scene["robot"].data.root_link_lin_vel_b[:, 0]


def tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
    sigma: float = FUDAN_TRACKING_SIGMA,
    shadow_func=None,
    shadow_params: dict | None = None,
) -> torch.Tensor:
    """1.3·exp(−e²/σ)，e 为 vx 跟踪误差。

    `shadow_func` 只为保留日志：原 tracking_lin_vel 会写速度课程读的
    `Locomotion/tracking_lin_vel_reward_curriculum` 与 `Rough/*` 诊断，删掉它速度课程就永远不推进；
    这里调用一次、丢弃返回值，不计入奖励。
    """
    if shadow_func is not None:
        shadow_func(env, **(shadow_params or {}))
    raw = 1.3 * torch.exp(-_vx_error(env, command_name).square() / float(sigma))
    return _scaled_clip(raw, scale, clip)


def tracking_lin_vel_enhance(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
    sigma: float = FUDAN_TRACKING_SIGMA,
) -> torch.Tensor:
    """1.3·(exp(−e²/(10σ)) − 1)：大误差段仍有梯度的有界负项。"""
    raw = 1.3 * (torch.exp(-_vx_error(env, command_name).square() / (10.0 * float(sigma))) - 1.0)
    return _scaled_clip(raw, scale, clip)


def tracking_ang_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
    sigma: float = FUDAN_TRACKING_SIGMA,
) -> torch.Tensor:
    """exp(−e²/σ)，e 为 yaw 角速度跟踪误差。"""
    cmd = env.command_manager.get_command(command_name)
    error = cmd[:, 1] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return _scaled_clip(torch.exp(-error.square() / float(sigma)), scale, clip)


@dataclass(frozen=True)
class _HeightLattice:
    """整张地形在 0.1 m 世界格点上的表面高度，`heights[i, j]` 对应世界坐标 (x0 + i·h, y0 + j·h)。"""

    heights: torch.Tensor
    x0: float
    y0: float
    points: torch.Tensor


def _terrain_geom_ids(m: mujoco.MjModel) -> list[int]:
    """地形 geom：焊在 world 上且在 group 0；确认机器人没有 geom 留在 group 0（否则射线会打到自己）。"""
    static = m.body_weldid[m.geom_bodyid] == 0
    in_group = m.geom_group == _FUDAN_TERRAIN_GEOM_GROUP
    if np.any(in_group & ~static):
        raise RuntimeError(
            f"机器人仍有 geom 在 group {_FUDAN_TERRAIN_GEOM_GROUP}，复旦高度格点会打到机身；"
            "rough 应通过 collision_geom_group 把碰撞体挪出该 group"
        )
    return np.flatnonzero(in_group & static).tolist()


def _build_height_lattice(env: ManagerBasedRlEnv) -> _HeightLattice:
    """把地形离线栅格化到复旦 heightfield 的 0.1 m 格点上（格点对齐世界原点，地块边与立面都落在格线上）。

    落在 box 边上的格点取高的一侧（边界含 1e-4 m 容差），与复旦"立面所在格点存高值"的约定一致；
    hfield 覆盖的格点用 mj_ray 从上方竖直打下，取第一个命中（即表面）。任何格点没被覆盖都直接报错。
    """
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    h = FUDAN_HEIGHT_LATTICE_M
    geoms = _terrain_geom_ids(m)

    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    hfields: list[tuple[np.ndarray, np.ndarray]] = []
    for g in geoms:
        rot = d.geom_xmat[g].reshape(3, 3)
        if not np.allclose(np.abs(rot), np.round(np.abs(rot)), atol=1e-6):
            raise NotImplementedError(f"地形 geom {g} 不是轴对齐的，复旦高度格点不支持")
        center = d.geom_xpos[g].copy()
        geom_type = int(m.geom_type[g])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            boxes.append((center, np.abs(rot) @ m.geom_size[g]))
        elif geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
            size = m.hfield_size[m.geom_dataid[g]]
            hfields.append((center, np.array([size[0], size[1], size[2]])))
        else:
            raise NotImplementedError(f"地形 geom 类型 {mujoco.mjtGeom(geom_type).name} 未支持")

    extents = [(c - s, c + s) for c, s in boxes + hfields]
    lo = np.min([e[0] for e in extents], axis=0)
    hi = np.max([e[1] for e in extents], axis=0)
    x0 = math.floor(lo[0] / h + 1e-6) * h
    y0 = math.floor(lo[1] / h + 1e-6) * h
    nx = math.floor((hi[0] - x0) / h + 1e-6) + 1
    ny = math.floor((hi[1] - y0) / h + 1e-6) + 1
    heights = np.full((nx, ny), -np.inf)

    def index_range(a: float, b: float, origin: float, n: int) -> slice:
        start = max(math.ceil((a - _EDGE_EPS_M - origin) / h), 0)
        stop = min(math.floor((b + _EDGE_EPS_M - origin) / h), n - 1)
        return slice(start, stop + 1)

    for center, half in boxes:
        ix = index_range(center[0] - half[0], center[0] + half[0], x0, nx)
        iy = index_range(center[1] - half[1], center[1] + half[1], y0, ny)
        np.maximum(heights[ix, iy], center[2] + half[2], out=heights[ix, iy])

    group_mask = np.zeros(6, dtype=np.uint8)
    group_mask[_FUDAN_TERRAIN_GEOM_GROUP] = 1
    top = float(hi[2]) + 1.0
    down = np.array([0.0, 0.0, -1.0])
    hit_geom = np.zeros(1, dtype=np.int32)
    for center, half in hfields:
        ix = index_range(center[0] - half[0], center[0] + half[0], x0, nx)
        iy = index_range(center[1] - half[1], center[1] + half[1], y0, ny)
        for i in range(ix.start, ix.stop):
            for j in range(iy.start, iy.stop):
                origin = np.array([x0 + i * h, y0 + j * h, top])
                dist = mujoco.mj_ray(m, d, origin, down, group_mask, True, -1, hit_geom)
                if dist >= 0.0:
                    heights[i, j] = max(heights[i, j], top - dist)

    if not np.all(np.isfinite(heights)):
        raise RuntimeError(f"复旦高度格点有 {int((~np.isfinite(heights)).sum())} 个没被地形覆盖")
    grid_x, grid_y = np.meshgrid(FUDAN_MEASURED_POINTS_X, FUDAN_MEASURED_POINTS_Y, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return _HeightLattice(
        heights=torch.as_tensor(heights, dtype=torch.float32, device=env.device),
        x0=float(x0),
        y0=float(y0),
        points=torch.as_tensor(points, dtype=torch.float32, device=env.device),
    )


def fudan_ground_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """复旦 `_get_heights` 的 77 点地面高度均值，形状 [B]。格点表首次调用时建好并缓存在 env 上。"""
    lattice = getattr(env, _HEIGHT_LATTICE_ATTR, None)
    if lattice is None:
        lattice = _build_height_lattice(env)
        setattr(env, _HEIGHT_LATTICE_ATTR, lattice)
    data = env.scene["robot"].data
    pos = data.root_link_pos_w
    quat = data.root_link_quat_w
    # 复旦 quat_apply_yaw：把四元数的 x、y 分量置零再归一化，即绕 z 的 twist 角。
    yaw = 2.0 * torch.atan2(quat[:, 3], quat[:, 0])
    cos, sin = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    px, py = lattice.points[:, 0], lattice.points[:, 1]
    world_x = pos[:, :1] + cos * px - sin * py
    world_y = pos[:, 1:2] + sin * px + cos * py
    h = FUDAN_HEIGHT_LATTICE_M
    nx, ny = lattice.heights.shape
    ix = torch.floor((world_x - lattice.x0) / h).long().clamp(0, nx - 2)
    iy = torch.floor((world_y - lattice.y0) / h).long().clamp(0, ny - 2)
    grid = lattice.heights
    sampled = torch.minimum(torch.minimum(grid[ix, iy], grid[ix + 1, iy]), grid[ix, iy + 1])
    return sampled.mean(dim=1)


def _height_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """机身 z − 复旦口径的 77 点地面均值 − 高度指令。"""
    cmd = env.command_manager.get_command(command_name)
    base_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return base_z - fudan_ground_height(env) - cmd[:, 4]


def base_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
) -> torch.Tensor:
    """1.5·exp(−1000·e²)：高度窄核正奖励（约 ±3 cm）。"""
    error = _height_error(env, command_name)
    return _scaled_clip(1.5 * torch.exp(-1000.0 * error.square()), scale, clip)


def base_height_enhance(
    env: ManagerBasedRlEnv,
    command_name: str,
    scale: float,
    clip: float = 1.0,
) -> torch.Tensor:
    """exp(−e²/0.01) − 1：高度宽核有界负项（约 ±10 cm）。"""
    error = _height_error(env, command_name)
    return _scaled_clip(torch.exp(-error.square() / 0.01) - 1.0, scale, clip)


def orientation(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """投影重力的水平分量平方和。"""
    gravity = env.scene["robot"].data.projected_gravity_b[:, :2]
    return _scaled_clip(gravity.square().sum(dim=1), scale, clip)


def ang_vel_xy(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """roll/pitch 角速度平方和。"""
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b[:, :2]
    return _scaled_clip(ang_vel.square().sum(dim=1), scale, clip)


def lin_vel_z(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """机身竖直线速度平方。"""
    return _scaled_clip(env.scene["robot"].data.root_link_lin_vel_b[:, 2].square(), scale, clip)


def _body_ids(env: ManagerBasedRlEnv, names: tuple[str, str]) -> list[int]:
    attr = "_se3_fudan_body_ids_" + "_".join(names)
    cached = getattr(env, attr, None)
    if isinstance(cached, list):
        return cached
    ids, found = env.scene["robot"].find_bodies(names, preserve_order=True)
    if len(ids) != len(names):
        raise RuntimeError(f"必须找到 {names}，实际找到 {found}")
    setattr(env, attr, list(ids))
    return list(ids)


def virtual_leg_angles(env: ManagerBasedRlEnv) -> torch.Tensor:
    """左右虚拟腿角 θ0，形状 [B, 2]：机身系里髋关节 → 轮心连线相对竖直向下的夹角，前摆为正。"""
    robot = env.scene["robot"]
    hip = _body_ids(env, ("lf0_Link", "rf0_Link"))
    wheel = _body_ids(env, ("l_wheel_Link", "r_wheel_Link"))
    delta_w = robot.data.body_link_pos_w[:, wheel, :] - robot.data.body_link_pos_w[:, hip, :]
    quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    delta_b = quat_apply_inverse(quat, delta_w.reshape(-1, 3)).reshape(env.num_envs, 2, 3)
    return torch.atan2(delta_b[..., 0], -delta_b[..., 2])


def nominal_state(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """左右虚拟腿角之差的平方（复旦 nominal_state 的负权重分支）。"""
    theta0 = virtual_leg_angles(env)
    return _scaled_clip((theta0[:, 0] - theta0[:, 1]).square(), scale, clip)


def dof_vel(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """四个腿部主动杆关节速度平方和（复旦只罚腿，不含轮）。"""
    _, vel = _policy_leg_torque_and_vel(env, env.scene["robot"])
    return _scaled_clip(vel.square().sum(dim=1), scale, clip)


def dof_acc(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """六个受控关节加速度平方和（含轮）。"""
    robot = env.scene["robot"]
    acc = torch.cat(
        (_policy_leg_acc(env, robot), robot.data.joint_acc[:, wheel_joint_ids(robot)]), dim=1
    )
    return _scaled_clip(acc.square().sum(dim=1), scale, clip)


def torques(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """六个执行器力矩平方和（含轮）。"""
    robot = env.scene["robot"]
    leg, _ = _policy_leg_torque_and_vel(env, robot)
    wheel = robot.data.actuator_force[:, wheel_actuator_ids(robot)]
    return _scaled_clip(torch.cat((leg, wheel), dim=1).square().sum(dim=1), scale, clip)


def action_rate(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """六维动作一阶差分平方和。"""
    delta = env.action_manager.action - env.action_manager.prev_action
    return _scaled_clip(delta.square().sum(dim=1), scale, clip)


class ActionSmooth(ManagerTermBase):
    """腿部四维动作二阶差分平方和；前两步动作按 env 缓存，reset 时清零（与复旦重置 last_actions 一致）。

    二阶差分沿用本仓库的周期处理（前关节动作按周期取最短差），与现有 action_smoothness 的动作语义一致。
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        del cfg
        shape = (env.num_envs, env.action_manager.total_action_dim)
        self._prev = torch.zeros(shape, device=env.device)
        self._prev_prev = torch.zeros(shape, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._prev[ids] = 0.0
        self._prev_prev[ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
        action = env.action_manager.action
        second = periodic_policy_action_second_difference_torch(
            action,
            self._prev,
            self._prev_prev,
            front_action_period=front_action_periods_from_env(env),
        )
        self._prev_prev.copy_(self._prev)
        self._prev.copy_(action)
        return _scaled_clip(second[:, :4].square().sum(dim=1), scale, clip)


def dof_pos_limits_clipped(env: ManagerBasedRlEnv, scale: float, clip: float = 1.0) -> torch.Tensor:
    """腿部软限位越界量（沿用闭链的同侧两主动杆夹角口径）。"""
    return _scaled_clip(dof_pos_limits(env, _DEFAULT_ASSET_CFG), scale, clip)


# 复旦 v3 的非零权重（legged_robot_config.py 的 rewards.scales，collision 恒 0 不计）。
FUDAN_V3_SCALES: dict[str, float] = {
    "tracking_lin_vel": 1.0,
    "tracking_lin_vel_enhance": 1.0,
    "tracking_ang_vel": 1.0,
    "base_height": 1.0,
    "base_height_enhance": 1.0,
    "nominal_state": -1.0,
    "lin_vel_z": -0.1,
    "ang_vel_xy": -0.07,
    "orientation": -20.0,
    "dof_vel": -5e-5,
    "dof_acc": -2.5e-7,
    "torques": -1e-4,
    "action_rate": -0.05,
    "action_smooth": -0.05,
    "dof_pos_limits": -1.0,
}

__all__ = [
    "FUDAN_HEIGHT_LATTICE_M",
    "FUDAN_MEASURED_POINTS_X",
    "FUDAN_MEASURED_POINTS_Y",
    "FUDAN_TRACKING_SIGMA",
    "FUDAN_V3_SCALES",
    "ActionSmooth",
    "action_rate",
    "ang_vel_xy",
    "base_height",
    "base_height_enhance",
    "dof_acc",
    "dof_pos_limits_clipped",
    "dof_vel",
    "fudan_ground_height",
    "lin_vel_z",
    "nominal_state",
    "orientation",
    "torques",
    "tracking_ang_vel",
    "tracking_lin_vel",
    "tracking_lin_vel_enhance",
    "virtual_leg_angles",
]
