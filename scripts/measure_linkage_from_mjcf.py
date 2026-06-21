"""从 MJCF 直接量取四连杆杆长 + 气弹簧挂点（左腿），control=0 姿态。

只读模型，不依赖 planar5rod.c 的任何参数。
所有关键点用 MuJoCo FK 在世界坐标下取出，再投影到 x-z 平面计算距离。
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

_MJCF = Path(__file__).resolve().parent.parent / (
    "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    # base 直立放原点；所有受控关节 = 0（control=0 姿态）
    data.qpos[0:7] = [0, 0, 0.22, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_tendon(model, data)

    def site(name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return data.site_xpos[sid].copy()

    def bodypos(name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid].copy()

    def jntpos(name: str) -> np.ndarray:
        """关节铰点世界坐标（body xpos + 局部 jnt_pos 旋转后偏移）。"""
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = model.jnt_bodyid[jid]
        # 世界位置 = body 原点 + R_body * jnt_pos_local
        R = data.xmat[bid].reshape(3, 3)
        return data.xpos[bid] + R @ model.jnt_pos[jid]

    def jntaxis(name: str) -> np.ndarray:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = model.jnt_bodyid[jid]
        R = data.xmat[bid].reshape(3, 3)
        return R @ model.jnt_axis[jid]

    def d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def dxz(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.array([a[0], a[2]])
        bb = np.array([b[0], b[2]])
        return float(np.linalg.norm(aa - bb))

    print("=" * 90)
    print("从 MJCF 量取（左腿，control=0，世界坐标，单位 mm；x-z 平面距离）")
    print("=" * 90)

    # ---- 关键铰点 / 端点 ----
    O_lf0 = jntpos("lf0_Joint")  # 前主动杆根铰（共轴）
    O_drive = jntpos("l_drive_bar_Joint")  # 后驱动杆根铰（共轴）
    K = jntpos("lf1_Joint")  # 膝铰（小腿上段绕此转），lf1 body 内 pos
    P = jntpos("l_coupler_Joint")  # 连杆根铰（在 drive_bar 末端）
    C = site("lf_coupler_closure")  # 小腿上段闭链点（connect 目标）
    D = site("l_coupler_end")  # 连杆末端（connect 源）
    wheel = bodypos("l_wheel_Link")  # 轮轴中心
    thigh_end = site("lf_thigh_end")  # thigh 末端 site
    calf_closure = site("lf_calf_closure")  # 小腿上的膝铰 site

    print("\n[关键点世界坐标 (x, y, z) mm]")
    for name, p in [
        ("O lf0_Joint (前杆根铰)", O_lf0),
        ("O l_drive_bar_Joint(后杆根铰)", O_drive),
        ("K lf1_Joint (膝铰)", K),
        ("P l_coupler_Joint(连杆根铰)", P),
        ("C lf_coupler_closure(小腿闭链点)", C),
        ("D l_coupler_end (连杆末端)", D),
        ("wheel 轮轴中心", wheel),
        ("lf_thigh_end site", thigh_end),
        ("lf_calf_closure site", calf_closure),
    ]:
        print(f"  {name:34s}: ({p[0] * 1000:8.2f}, {p[1] * 1000:8.2f}, {p[2] * 1000:8.2f})")

    print("\n[两根主动杆共轴检验]")
    print(f"  O_lf0 与 O_drive 距离 = {d(O_lf0, O_drive) * 1000:.3f} mm (应≈0)")
    print(f"  connect 残差 |D-C|    = {d(D, C) * 1000:.4f} mm (应≈0)")

    print("\n[四连杆杆长 (x-z 平面)]")
    l_thigh_OK = dxz(O_lf0, K)  # 前主动杆 O->K
    l_calf_upper_KC = dxz(K, C)  # 小腿上段 K->C
    l_drive_OP = dxz(O_drive, P)  # 后驱动杆 O->P
    l_coupler_PD = dxz(P, D)  # 连杆 P->D
    l_calf_lower_Kwheel = dxz(K, wheel)  # 小腿下段 K->轮轴
    l_leg_Owheel = dxz(O_lf0, wheel)  # 整腿 O->轮轴
    print(f"  前主动杆  O->K  (thigh)        = {l_thigh_OK * 1000:8.2f} mm")
    print(f"  后驱动杆  O->P  (drive_bar)    = {l_drive_OP * 1000:8.2f} mm")
    print(f"  连杆      P->D  (coupler)      = {l_coupler_PD * 1000:8.2f} mm")
    print(f"  小腿上段  K->C  (膝->闭链点)   = {l_calf_upper_KC * 1000:8.2f} mm")
    print(f"  小腿下段  K->轮 (膝->轮轴)      = {l_calf_lower_Kwheel * 1000:8.2f} mm")
    print(f"  整腿      O->轮 (腿长)          = {l_leg_Owheel * 1000:8.2f} mm")

    print("\n[各杆与世界 x 轴夹角 (control=0, 度)]")

    def ang_xz(a: np.ndarray, b: np.ndarray) -> float:
        v = np.array([b[0] - a[0], b[2] - a[2]])
        return math.degrees(math.atan2(v[1], v[0]))

    print(f"  前主动杆 O->K  = {ang_xz(O_lf0, K):8.3f} deg")
    print(f"  后驱动杆 O->P  = {ang_xz(O_drive, P):8.3f} deg")
    print(f"  连杆     P->D  = {ang_xz(P, D):8.3f} deg")
    print(f"  小腿     K->轮 = {ang_xz(K, wheel):8.3f} deg")

    # ---- 气弹簧挂点 ----
    p1 = site("l_spring_p1")  # thigh 上
    p2 = site("l_spring_p2")  # 小腿上
    ten = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "l_knee_spring")
    l_spring = float(data.ten_length[ten])
    print("\n[气弹簧挂点 (世界坐标 mm)]")
    print(
        f"  P1 l_spring_p1 (thigh 上): ({p1[0] * 1000:8.2f}, {p1[1] * 1000:8.2f}, {p1[2] * 1000:8.2f})"
    )
    print(
        f"  P2 l_spring_p2 (小腿上)  : ({p2[0] * 1000:8.2f}, {p2[1] * 1000:8.2f}, {p2[2] * 1000:8.2f})"
    )
    print(f"  气弹簧自由长度 |P1-P2|    = {d(p1, p2) * 1000:.2f} mm  (3D)")
    print(f"  气弹簧 tendon 长度        = {l_spring * 1000:.2f} mm  (MuJoCo)")
    print(f"  P1 相对 O_lf0 距离        = {d(p1, O_lf0) * 1000:.2f} mm")
    print(f"  P1 相对 K(膝铰) 距离      = {d(p1, K) * 1000:.2f} mm")
    print(f"  P2 相对 K(膝铰) 距离      = {d(p2, K) * 1000:.2f} mm")

    # ---- 气弹簧挂点在各自杆上的局部坐标（相对膝铰，便于核对力臂）----
    print("\n[气弹簧挂点相对膝铰 K 的位置 (世界 x-z, mm)]")
    print(f"  P1-K : dx={(p1[0] - K[0]) * 1000:7.2f}, dz={(p1[2] - K[2]) * 1000:7.2f}")
    print(f"  P2-K : dx={(p2[0] - K[0]) * 1000:7.2f}, dz={(p2[2] - K[2]) * 1000:7.2f}")

    print("\n[关节轴方向 (世界)]")
    for jn in ["lf0_Joint", "l_drive_bar_Joint", "lf1_Joint", "l_coupler_Joint"]:
        ax = jntaxis(jn)
        print(f"  {jn:22s}: ({ax[0]:+.3f}, {ax[1]:+.3f}, {ax[2]:+.3f})")


if __name__ == "__main__":
    main()
