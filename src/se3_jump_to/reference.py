"""原地跳跃运动学参考轨迹生成器（se3.jump_ref.v1，2026-10-01）。

直接在 policy 主动杆坐标下工作：每帧给定两轮轮心相对髋轴的 (x, z)，经 se3_shared.fourbar 逆解得到主动杆角。
只保证运动学（关节在行程内、接地段轮子贴地不滑、机身高度与速度连续、命中目标离地间隙），不检查力矩；
动力学可行性由后续训练 rollout 迭代数据集解决。机身竖直运动分段：
  stand → crouch（smoothstep 下蹲）→ push（恒加速蹬伸到起跳速度）→ flight（整机质心抛体；上升收腿、下降伸腿）
  → cushion（恒减速缓冲到 0）→ recover（smoothstep 回站姿）→ hold
接地段轮心 x 固定为默认站姿值；腾空段腿长按 smoothstep 变化，机身高度由"质心走抛物线"反解（腿约 2 kg，
收腿会把机身相对质心拉低）。目标高度 = 腾空最高点轮底离地间隙。

旧 `se3_jump_to.cli` 的插值生成器在当前闭链模型上失效（旧输出坐标的膝角边界把膝关节钉死在上限、起点高度与
v2 默认站姿不符），已由本模块取代。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from se3_shared import JointGroup, RobotConfig
from se3_shared.fourbar import (
    output_leg_wheel_xz_np,
    output_to_policy_pos_np,
    policy_to_closedchain_passive_pos_np,
    policy_to_output_pos_np,
    wheel_xz_to_output_pos_np,
)
from se3_shared.grounded_pose import find_wheel_collision_radius

MJCF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_closed_chain_v3_train_obb_trim.xml"
)
REFERENCE_FORMAT = "se3.jump_ref.v1"
PHASE_NAMES = ("stand", "crouch", "push", "flight", "cushion", "recover", "hold")
G = 9.81


@dataclass
class JumpReferenceParams:
    """一条参考轨迹的设计参数（长度 m，时间 s）。"""

    target_clearance: float
    stand_height: float = 0.28
    """起止站姿机身高度。"""
    crouch_leg: float = 0.150
    """下蹲到底的腿长（行程下限 0.135）。"""
    takeoff_leg: float = 0.315
    """起跳瞬间腿长（行程上限 0.331）。"""
    land_leg: float = 0.300
    """触地瞬间腿长。"""
    cushion_leg: float = 0.160
    """缓冲到底的腿长。"""
    tuck_leg_min: float = 0.150
    """空中最短腿长。"""
    min_flight_time: float = 0.30
    """腾空时长下限：低目标高度用部分收腿而不是贴着地蹦。"""
    t_hold0: float = 0.20
    t_crouch: float = 0.35
    t_recover: float = 0.40
    t_hold1: float = 0.30
    dt: float = 0.005


class ReferenceKinematics:
    """MuJoCo 闭链 FK：policy 主动杆角、腿长、质心偏移之间的换算。"""

    def __init__(self, mjcf_path: Path = MJCF_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.wheel_radius = find_wheel_collision_radius(self.model, "l_wheel_Link")
        self._policy_adr = [
            self.model.jnt_qposadr[self.model.joint(n).id] for n in JointGroup.POLICY_LEG_NAMES
        ]
        self._passive_adr = [
            self.model.jnt_qposadr[self.model.joint(n).id]
            for n in JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES
        ]
        default_policy = np.asarray(RobotConfig().default_dof_pos[:4], dtype=np.float64)
        self.wheel_x = float(output_leg_wheel_xz_np(policy_to_output_pos_np(default_policy))[0, 0])

    def leg_pos(self, leg_len: float) -> np.ndarray:
        """腿长 → policy 4D 主动杆角（左右对称，轮心 x 固定为默认站姿）。"""
        z = -np.sqrt(max(leg_len**2 - self.wheel_x**2, 1e-9))
        xz = np.array([[self.wheel_x, z], [self.wheel_x, z]])
        return output_to_policy_pos_np(wheel_xz_to_output_pos_np(xz))

    def com_offset(self, policy_leg: np.ndarray) -> float:
        """质心 z − 机身 z（机身直立）。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._policy_adr] = policy_leg
        self.data.qpos[self._passive_adr] = policy_to_closedchain_passive_pos_np(policy_leg)
        self.data.qpos[0:3] = [0.0, 0.0, 1.0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        return float(self.data.subtree_com[0][2] - 1.0)

    def grounded_base_z(self, leg_len: float) -> float:
        return self.wheel_radius + float(np.sqrt(max(leg_len**2 - self.wheel_x**2, 1e-9)))

    def leg_len_for_base_z(self, base_z: float) -> float:
        return float(np.hypot(self.wheel_x, base_z - self.wheel_radius))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_reference(p: JumpReferenceParams, kin: ReferenceKinematics) -> dict[str, np.ndarray]:
    """生成一条参考轨迹，返回可直接 np.savez 的字段字典。"""
    dt = p.dt
    z_s = p.stand_height
    z_c = kin.grounded_base_z(p.crouch_leg)
    z_to = kin.grounded_base_z(p.takeoff_leg)
    com0 = z_to + kin.com_offset(kin.leg_pos(p.takeoff_leg))
    # 初值：全收腿所需机身上升（按机身≈质心估），与腾空时长下限取大
    tuck_z_rel_min = -np.sqrt(p.tuck_leg_min**2 - kin.wheel_x**2)
    rise = max(
        p.target_clearance + kin.wheel_radius - tuck_z_rel_min - z_to,
        G * (p.min_flight_time / 2) ** 2 / 2,
    )
    v_to = float(np.sqrt(2 * G * rise))

    def flight(l_tuck: float, v: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        """质心抛体；上升段 smoothstep 收到 l_tuck，下降段 smoothstep 伸到 land_leg，积分到轮底触地。"""
        t_apex = v / G
        zs: list[float] = []
        legs: list[float] = []
        k = 0
        while True:
            k += 1
            t = k * dt
            com = com0 + v * t - 0.5 * G * t**2
            if t <= t_apex:
                s = _smoothstep(np.array(t / (0.8 * t_apex)))
                leg = p.takeoff_leg + (l_tuck - p.takeoff_leg) * s
            else:
                s = _smoothstep(np.array((t - t_apex) / (0.7 * t_apex)))
                leg = l_tuck + (p.land_leg - l_tuck) * s
            leg = float(leg)
            base = com - kin.com_offset(kin.leg_pos(leg))
            wheel_bottom = base - np.sqrt(leg**2 - kin.wheel_x**2) - kin.wheel_radius
            if t > t_apex and wheel_bottom <= 0.0:
                break
            zs.append(base)
            legs.append(leg)
            if k > 2000:
                raise RuntimeError("腾空段未落地")
        z_arr, leg_arr = np.array(zs), np.array(legs)
        clearance = z_arr - np.sqrt(leg_arr**2 - kin.wheel_x**2) - kin.wheel_radius
        return z_arr, leg_arr, float(clearance.max()), (k - 1) * dt + dt

    if flight(p.tuck_leg_min, v_to)[2] < p.target_clearance:
        # 全收腿仍不够高（机身≠质心、收腿把机身拉低）：二分加大机身抬升，空中收到最短腿
        lo, hi = rise, rise + 0.4
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if flight(p.tuck_leg_min, float(np.sqrt(2 * G * mid)))[2] < p.target_clearance:
                lo = mid
            else:
                hi = mid
        v_to = float(np.sqrt(2 * G * hi))
        l_tuck = p.tuck_leg_min
    else:
        lo, hi = p.tuck_leg_min, p.takeoff_leg
        for _ in range(40):  # 腿长越短间隙越大：二分到间隙 = 目标
            mid = 0.5 * (lo + hi)
            if flight(mid, v_to)[2] > p.target_clearance:
                lo = mid
            else:
                hi = mid
        l_tuck = 0.5 * (lo + hi)
    z_f, leg_f, clearance_peak, t_flight = flight(l_tuck, v_to)
    a_push = v_to**2 / (2 * (z_to - z_c))
    t_push = v_to / a_push

    segs: list[tuple[str, np.ndarray, np.ndarray]] = []

    def add_grounded(phase: str, z: np.ndarray) -> None:
        segs.append((phase, z, np.array([kin.leg_len_for_base_z(float(v)) for v in z])))

    n = round(p.t_hold0 / dt)
    add_grounded("stand", np.full(n, z_s))
    n = round(p.t_crouch / dt)
    add_grounded("crouch", z_s + (z_c - z_s) * _smoothstep(np.arange(1, n + 1) / n))
    n = max(2, round(t_push / dt))
    t = np.arange(1, n + 1) * dt * (t_push / (n * dt))
    add_grounded("push", z_c + 0.5 * a_push * t**2)
    segs.append(("flight", z_f, leg_f))
    z_td = kin.grounded_base_z(p.land_leg)
    v_td = min((z_td - z_f[-1]) / dt, v_to - G * t_flight)
    z_cu = kin.grounded_base_z(p.cushion_leg)
    a_cu = v_td**2 / (2 * (z_td - z_cu))
    t_cu = -v_td / a_cu
    n = max(2, round(t_cu / dt))
    t = np.arange(0, n) * dt * (t_cu / max((n - 1) * dt, dt))
    add_grounded("cushion", z_td + v_td * t + 0.5 * a_cu * t**2)
    n = round(p.t_recover / dt)
    add_grounded("recover", z_cu + (z_s - z_cu) * _smoothstep(np.arange(1, n + 1) / n))
    n = round(p.t_hold1 / dt)
    add_grounded("hold", np.full(n, z_s))

    base_z = np.concatenate([s[1] for s in segs])
    leg = np.concatenate([s[2] for s in segs])
    phase = np.concatenate([np.full(len(s[1]), PHASE_NAMES.index(s[0])) for s in segs])
    policy_leg = np.stack([kin.leg_pos(float(v)) for v in leg])
    com = base_z + np.array([kin.com_offset(q) for q in policy_leg])
    base_pos = np.zeros((len(base_z), 3))
    base_pos[:, 2] = base_z
    base_vel = np.zeros_like(base_pos)
    base_vel[:, 2] = np.gradient(base_z, dt)
    meta = dict(
        asdict(p),
        v_takeoff=v_to,
        a_push=a_push,
        t_push=t_push,
        l_tuck=l_tuck,
        flight_time=t_flight,
        clearance_peak=clearance_peak,
        v_touchdown=v_td,
        a_cushion=a_cu,
        t_cushion=t_cu,
        z_crouch=z_c,
        z_takeoff=z_to,
        z_touchdown=z_td,
        z_cushion=z_cu,
        phase_names=list(PHASE_NAMES),
        leg_order=list(JointGroup.POLICY_LEG_NAMES),
        format=REFERENCE_FORMAT,
    )
    return {
        "dt": np.float64(dt),
        "base_pos": base_pos,
        "base_vel": base_vel,
        "com_z": com,
        "leg_pos": policy_leg,
        "leg_vel": np.gradient(policy_leg, dt, axis=0),
        "leg_len": leg,
        "contact": phase != PHASE_NAMES.index("flight"),
        "phase": phase,
        "wheel_clearance": base_z - np.sqrt(leg**2 - kin.wheel_x**2) - kin.wheel_radius,
        "meta": np.array(json.dumps(meta, ensure_ascii=False)),
    }


__all__ = [
    "PHASE_NAMES",
    "REFERENCE_FORMAT",
    "JumpReferenceParams",
    "ReferenceKinematics",
    "generate_reference",
]
