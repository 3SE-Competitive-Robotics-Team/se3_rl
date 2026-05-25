"""NPC 参考轨迹的完整 3D MuJoCo Rerun 重放。

复用 se3_sim2sim 的 RerunViewer，渲染完整的机器人 3D 模型。

用法：
    uv run se3-jump-to-replay --traj assets/trajectories/jump_0.6m.npz
    uv run se3-jump-to-replay --traj assets/trajectories/jump_0.6m.npz --speed 0.3
    uv run se3-jump-to-replay --traj assets/trajectories/jump_0.6m.npz --loop
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="NPC 轨迹 3D Rerun 重放")
    parser.add_argument(
        "--traj",
        type=str,
        default="assets/trajectories/jump_0.6m.npz",
        help="轨迹文件路径",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="播放速度倍数（1.0=实时，0.3=慢放），默认 0.3",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环播放",
    )
    args = parser.parse_args()

    # 加载轨迹
    traj_path = Path(args.traj)
    if not traj_path.exists():
        print(f"[Replay] 轨迹文件不存在: {traj_path}")
        return

    d = np.load(traj_path)
    base_pos: np.ndarray = d["base_pos"]  # (N, 3)
    base_vel: np.ndarray = d["base_vel"]  # (N, 3)
    q_ref: np.ndarray = d["q_ref"]  # (N, 6)
    grf_left: np.ndarray = d["grf_left"]  # (N_stance, 3)
    dt = float(d["dt"])
    t_stance = float(d["t_stance"])
    t_flight = float(d["t_flight"])
    t_land = float(d["t_land"])
    n_steps = len(base_pos)
    total_time = n_steps * dt

    print(f"[Replay] 加载轨迹: {traj_path.name}")
    print(f"         总步数={n_steps}  时长={total_time:.2f}s  dt={dt}s")
    print(f"         站立={t_stance:.2f}s | 飞行={t_flight:.2f}s | 着陆={t_land:.2f}s")

    # 加载 MuJoCo 模型
    mjcf_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robots"
        / "serialleg"
        / "mjcf"
        / "serialleg_fidelity_cylinder_wheels.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # 受控关节 qpos 索引
    ctrl_joint_names = (
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    )
    ctrl_qpos_idx = []
    for name in ctrl_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ctrl_qpos_idx.append(int(model.jnt_qposadr[jid]))

    lw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_Link")
    rw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_Link")

    # 初始化 RerunViewer（完整 3D 机器人渲染）
    from se3_sim2sim.rerun_viewer import RerunViewer

    viewer = RerunViewer(app_id="se3_jump_to_replay", spawn=True, follow_body="base_link")
    viewer.log_model(model)
    print("[Replay] Rerun 已启动，完整 3D 机器人渲染中...")
    print("[Replay] 请在 Rerun 窗口中查看 3D 动作序列")

    # 播放循环
    play_count = 0
    while True:
        play_count += 1
        if play_count > 1 and not args.loop:
            break
        if play_count > 1:
            print(f"\n[Replay] 第 {play_count} 次循环...")

        for step in range(n_steps):
            t = step * dt
            pos = base_pos[step]
            vel = base_vel[step]
            q = q_ref[step]

            # 设置 MuJoCo 状态
            mujoco.mj_resetData(model, data)
            data.qpos[0] = pos[0]
            data.qpos[1] = pos[1]
            data.qpos[2] = pos[2]
            data.qpos[3] = 1.0  # quaternion w
            data.qpos[4] = 0.0
            data.qpos[5] = 0.0
            data.qpos[6] = 0.0
            for i, idx in enumerate(ctrl_qpos_idx):
                data.qpos[idx] = float(q[i])
            data.time = t
            mujoco.mj_forward(model, data)

            # 阶段
            if t < t_stance:
                phase = "STANCE"
            elif t < t_land:
                phase = "FLIGHT"
            else:
                phase = "LANDING"

            lw_z = float(data.xpos[lw_id, 2])
            rw_z = float(data.xpos[rw_id, 2])
            wheel_clr = max(lw_z, rw_z) - 0.059

            grf_z = float(grf_left[step, 2]) if step < len(grf_left) else 0.0

            # 用 RerunViewer 渲染完整 3D 机器人
            telemetry = {
                "height": float(pos[2]),
                "wheel_clearance": wheel_clr,
                "tilt_deg": 0.0,
                "fail_tilt_deg": 80.0,
                "reward": grf_z / 300.0,  # 用 GRF 归一化值作为 reward 显示
                "base_ang_vel_body": [0.0, 0.0, 0.0],
                "base_ang_vel_world": [0.0, 0.0, 0.0],
                "projected_gravity": [0.0, 0.0, -1.0],
                "dof_pos": list(q),
                "dof_vel": [0.0] * 6,
                "policy_action_raw": [0.0] * 6,
                "policy_action_clipped": [0.0] * 6,
                "last_action": list(q),
                "last_ctrl": [0.0] * 6,
                "yaw_pid": {
                    "current_yaw": 0.0,
                    "target_yaw": 0.0,
                    "error": 0.0,
                    "command": 0.0,
                },
            }
            viewer.log_state(model, data, step=step, telemetry=telemetry)

            # 控制台输出（每 10 步）
            if step % 10 == 0:
                print(
                    f"t={t:.2f}s [{phase:7s}] "
                    f"base_z={pos[2]:.3f}m  "
                    f"vz={vel[2]:+.2f}m/s  "
                    f"wheel_clr={wheel_clr:+.3f}m  "
                    f"lf0={q[0]:.2f}  lf1={q[1]:.2f}"
                )

            time.sleep(dt / args.speed)

        print(f"\n[Replay] 播放完成  时长={total_time:.2f}s")
        if not args.loop:
            break

    viewer.close()
    print("[Replay] 结束")


if __name__ == "__main__":
    main()
