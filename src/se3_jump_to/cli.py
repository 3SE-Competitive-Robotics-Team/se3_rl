"""SerialLeg 跳跃 NPC 参考轨迹生成命令行入口。

用法：
    uv run se3-jump-to
    uv run se3-jump-to --height 0.4 --output assets/trajectories/jump_0.4m.npz
    uv run se3-jump-to --output assets/trajectories/jump_0.6m.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 轨迹参数常量（默认对应 0.6m wheel clearance 跳跃）
# ──────────────────────────────────────────────────────────────────────────────
TARGET_WHEEL_CLEARANCE = 0.6
DT = 0.01
N_STANCE = 40  # 站立段（蹲下 + 蹬地）
N_DESCEND = 18  # 站立段中蹲下阶段帧数
N_CUSHION = 16  # 着陆缓冲段
N_RECOVER = 10  # 恢复站立段
DEFAULT_OUTPUT = Path("assets/trajectories/jump_0.6m.npz")
G = 9.81


# ──────────────────────────────────────────────────────────────────────────────
# 插值工具函数
# ──────────────────────────────────────────────────────────────────────────────


def _smoothstep(x: float) -> float:
    """三次 smoothstep，端点速度为 0。"""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _hermite(p0: float, v0: float, p1: float, v1: float, t: float, duration: float) -> float:
    """三次 Hermite 插值，显式约束端点位置和速度。"""
    s = float(np.clip(t / duration, 0.0, 1.0))
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1


def _blend_pose(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """按 smoothstep 平滑混合两组关节角。"""
    beta = _smoothstep(alpha)
    return (1.0 - beta) * q0 + beta * q1


def _grounded_q_and_z(
    fk, base_z: float, wheel_x_fixed: float, q_prev: np.ndarray
) -> tuple[np.ndarray, float]:
    """求接地 IK，并把 base_z 修正到轮子刚好贴地。"""
    q = fk.ik_grounded(base_z, wheel_x_fixed, q_prev[:2])
    result = fk.fk([0.0, 0.0, base_z], q)
    corrected_z = base_z + (fk.wheel_radius - float(result["l_wheel"][2]))
    return q, corrected_z


def _build_grounded_segment(
    fk,
    z_values: np.ndarray,
    wheel_x_fixed: float,
    q_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按高度序列生成接地段 q_ref，并逐帧消除轮子穿地误差。"""
    q_values = np.zeros((len(z_values), 6))
    z_corrected = np.zeros(len(z_values))
    q_prev = q_start.copy()
    for i, z in enumerate(z_values):
        q, z_fix = _grounded_q_and_z(fk, float(z), wheel_x_fixed, q_prev)
        q_values[i] = q
        z_corrected[i] = z_fix
        q_prev = q
    return q_values, z_corrected


def _finite_difference_velocity(z: np.ndarray) -> np.ndarray:
    """用离散高度生成与轨迹帧一致的参考速度。"""
    vz = np.zeros_like(z)
    vz[1:-1] = (z[2:] - z[:-2]) / (2.0 * DT)
    vz[0] = (z[1] - z[0]) / DT
    vz[-1] = (z[-1] - z[-2]) / DT
    return vz


def _finite_difference_q_vel(q: np.ndarray, dt: float) -> np.ndarray:
    """用中心差分从关节角序列估算各帧关节角速度。

    q: (N, 6)，返回 (N, 6)，端点用单侧差分。
    """
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    dq[0] = (q[1] - q[0]) / dt
    dq[-1] = (q[-1] - q[-2]) / dt
    return dq


def _ballistic_flight_steps(target_height: float, dt: float = DT) -> int:
    """按标准重力计算飞行段帧数。

    target_height 表示轮子离地目标高度，也作为 base 的抛体上升高度。
    """
    h = max(float(target_height), 0.01)
    flight_time = 2.0 * np.sqrt(2.0 * h / G)
    return max(2, round(flight_time / dt))


def _ballistic_takeoff_velocity(flight_steps: int, dt: float = DT) -> float:
    """由离散飞行时长反算起跳速度，保证落地时回到起跳高度。"""
    flight_time = max(float(flight_steps) * dt, dt)
    return 0.5 * G * flight_time


# ──────────────────────────────────────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────────────────────────────────────


def _validate(path: Path, target_wheel_clearance: float) -> None:
    """打印轨迹核心体检指标。"""
    from se3_jump_to.kinematics import get_fk

    fk = get_fk()
    d = np.load(path)
    base_pos = d["base_pos"]
    base_vel = d["base_vel"]
    q_ref = d["q_ref"]
    dt = float(d["dt"])
    t_stance = float(d["t_stance"])
    t_land = float(d["t_land"])
    i_stance = round(t_stance / dt)
    i_land = round(t_land / dt)

    lz = []
    rz = []
    lx = []
    for pos, q in zip(base_pos, q_ref, strict=True):
        result = fk.fk(pos, q)
        lz.append(float(result["l_wheel"][2]))
        rz.append(float(result["r_wheel"][2]))
        lx.append(float(result["l_wheel"][0]))
    clearance = np.minimum(np.array(lz), np.array(rz)) - fk.wheel_radius
    q_step = np.max(np.abs(np.diff(q_ref, axis=0)), axis=1)

    print(f"[NPC] 保存: {path}")
    print(f"[NPC] 步数={len(base_pos)}  时长={len(base_pos) * dt:.2f}s")
    print(f"[NPC] 起跳={t_stance:.2f}s  触地={t_land:.2f}s")
    print(
        f"[NPC] wheel clearance 峰值={clearance.max():.4f}m  "
        f"目标={target_wheel_clearance:.4f}m  最小={clearance.min():+.6f}m"
    )
    print(
        f"[NPC] 接地段最小 clearance: stance={clearance[:i_stance].min():+.6f}m  "
        f"landing={clearance[i_land:].min():+.6f}m"
    )
    print(
        f"[NPC] 边界: 起跳 q_step={q_step[i_stance - 1]:.4f}rad  "
        f"触地 q_step={q_step[i_land - 1]:.4f}rad"
    )
    print(
        f"[NPC] 边界速度: 起跳 {base_vel[i_stance - 1, 2]:+.3f} -> "
        f"{base_vel[i_stance, 2]:+.3f} m/s, 触地 {base_vel[i_land - 1, 2]:+.3f} -> "
        f"{base_vel[i_land, 2]:+.3f} m/s"
    )
    print(
        f"[NPC] 最大关节单步={q_step.max():.4f}rad  "
        f"站立轮子 x 漂移={np.ptp(np.array(lx[:i_stance])):.6f}m"
    )
    # 验证 q_vel
    if "q_vel" in d:
        q_vel = d["q_vel"]
        print(f"[NPC] q_vel 范围: min={q_vel.min():.3f}  max={q_vel.max():.3f} rad/s")
        print(
            f"[NPC] stance 末帧 q_vel (frame {i_stance - 1}): "
            f"{q_vel[i_stance - 1].round(2).tolist()}"
        )
    else:
        print("[NPC] 警告: npz 中没有 q_vel 字段，请重新生成轨迹")
        q_vel = None


# ──────────────────────────────────────────────────────────────────────────────
# 生成逻辑
# ──────────────────────────────────────────────────────────────────────────────


def generate(output: Path, target_wheel_clearance: float = TARGET_WHEEL_CLEARANCE) -> None:
    """生成 NPC 参考轨迹并保存到 output。"""
    from se3_jump_to.kinematics import get_fk

    fk = get_fk()
    q_home = np.array(fk.default_qpos)
    q_tuck = fk.optimal_tuck_pose()

    wheel_x_fixed = float(fk.fk([0.0, 0.0, fk.default_base_height], q_home)["l_wheel"][0])

    z_home = float(fk.default_base_height)
    z_squat = float(fk.wheel_radius + fk.leg_length(q_tuck[0], q_tuck[1]))

    # 起跳/飞行姿态使用接近默认站立的腿长，让 base_link 的上升量与
    # wheel clearance 保持自然对应。旧版固定伸满腿会让 0.1m 轮子离地
    # 对应约 0.23m base 上升，动作看起来突兀且能量不经济。
    q_takeoff, z_takeoff = _grounded_q_and_z(fk, z_home, wheel_x_fixed, q_home)

    n_flight = _ballistic_flight_steps(target_wheel_clearance)
    flight_edge_speed = _ballistic_takeoff_velocity(n_flight)

    # 站立段：蹲下（smoothstep）+ 蹬地（Hermite）
    z_stance = np.zeros(N_STANCE)
    for i in range(N_DESCEND):
        alpha = i / max(N_DESCEND - 1, 1)
        z_stance[i] = z_home + (z_squat - z_home) * _smoothstep(alpha)

    n_extend = N_STANCE - N_DESCEND
    duration_extend = (n_extend - 1) * DT
    for j in range(n_extend):
        idx = N_DESCEND + j
        z_stance[idx] = _hermite(
            z_squat,
            0.0,
            z_takeoff,
            flight_edge_speed,
            j * DT,
            duration_extend,
        )

    q_stance, z_stance = _build_grounded_segment(fk, z_stance, wheel_x_fixed, q_home)
    q_stance[-1] = q_takeoff
    z_stance[-1] = z_takeoff

    # 飞行段：标准重力抛体。保持伸腿姿态，让 wheel clearance 直接等于 base 抛体高度，
    # 避免低高度目标被收腿姿态伪装成“悬停”。
    z_flight = np.zeros(n_flight)
    q_flight = np.zeros((n_flight, 6))
    for k in range(n_flight):
        t = (k + 1) * DT
        z_flight[k] = z_takeoff + flight_edge_speed * t - 0.5 * G * t**2
        q_flight[k] = q_takeoff

    # 着陆缓冲段（Hermite）+ 恢复站立（smoothstep）
    z_cushion = np.zeros(N_CUSHION)
    duration_cushion = N_CUSHION * DT
    for k in range(N_CUSHION):
        z_cushion[k] = _hermite(
            z_takeoff,
            -flight_edge_speed,
            z_squat,
            0.0,
            (k + 1) * DT,
            duration_cushion,
        )

    z_recover = np.zeros(N_RECOVER)
    for k in range(N_RECOVER):
        alpha = (k + 1) / N_RECOVER
        z_recover[k] = z_squat + (z_home - z_squat) * _smoothstep(alpha)

    q_landing, z_landing = _build_grounded_segment(
        fk,
        np.concatenate([z_cushion, z_recover]),
        wheel_x_fixed,
        q_takeoff,
    )

    # 拼接完整轨迹
    z_all = np.concatenate([z_stance, z_flight, z_landing])
    q_ref = np.vstack([q_stance, q_flight, q_landing])

    base_pos = np.zeros((len(z_all), 3))
    base_vel = np.zeros((len(z_all), 3))
    base_pos[:, 2] = z_all
    base_vel[:, 2] = _finite_difference_velocity(z_all)

    # 关节角速度：用中心差分从 q_ref 估算
    # q_vel shape: (N, 6)，与 q_ref 一一对应，供 RSI 初始化时注入关节速度
    q_vel = _finite_difference_q_vel(q_ref, DT)

    # GRF（由加速度反算）
    acc_z = np.gradient(base_vel[:, 2], DT)
    stance_force = np.maximum(fk.total_mass * (acc_z[:N_STANCE] + fk.g), 0.0)
    grf_left = np.zeros((N_STANCE, 3))
    grf_right = np.zeros((N_STANCE, 3))
    grf_left[:, 2] = 0.5 * stance_force
    grf_right[:, 2] = 0.5 * stance_force

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        base_pos=base_pos,
        base_vel=base_vel,
        q_ref=q_ref,
        q_vel=q_vel,
        grf_left=grf_left,
        grf_right=grf_right,
        t_stance=N_STANCE * DT,
        t_flight=n_flight * DT,
        t_land=(N_STANCE + n_flight) * DT,
        dt=DT,
    )
    _validate(output, target_wheel_clearance)


# ──────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SerialLeg 跳跃 NPC 参考轨迹生成")
    parser.add_argument(
        "--height",
        type=float,
        default=TARGET_WHEEL_CLEARANCE,
        help=f"目标轮子离地高度（m），默认 {TARGET_WHEEL_CLEARANCE}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"保存路径（.npz），默认 {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    print(f"[NPC] 生成 {args.height:.2f}m 跳跃参考轨迹 -> {output}")
    generate(output, target_wheel_clearance=args.height)


if __name__ == "__main__":
    main()
