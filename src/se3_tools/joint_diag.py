"""终端交互式关节诊断工具。

无需 GUI,通过终端命令逐个测试关节方向、轮子方向、VMC 响应。
输出纯文本结果,适合 AI agent 和 SSH 远程调试。

用法:
    uv run se3-joint-diag
    uv run se3-joint-diag --mode sweep       # 自动扫描所有关节
    uv run se3-joint-diag --mode interactive  # 手动输入力矩测试
    uv run se3-joint-diag --mode vmc          # 验证 VMC 控制器
"""

import argparse

import mujoco
import numpy as np

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml"

ACTUATOR_NAMES = ["lf0_act", "lf1_act", "l_wheel_act", "rf0_act", "rf1_act", "r_wheel_act"]


def load_model(mjcf_path: str):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    return model, data


def reset_standing(model, data, height: float = 0.30):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = height
    mujoco.mj_forward(model, data)


def get_base_state(model, data) -> dict:
    return {
        "x": data.qpos[0],
        "y": data.qpos[1],
        "z": data.xpos[1, 2],
        "quat": data.qpos[3:7].tolist(),
    }


def mode_sweep(model, data, args):
    """自动扫描每个 actuator 的正/负方向响应。"""
    print("=" * 60)
    print("SWEEP: 逐个 actuator 施力,观察响应")
    print("=" * 60)
    print(f"{'actuator':15s} | {'ctrl':>5s} | {'qvel':>10s} | {'base_dx':>8s} | {'base_dz':>8s}")
    print("-" * 60)

    for act_id in range(model.nu):
        for ctrl_val in [+5.0, -5.0]:
            reset_standing(model, data, args.height)
            x0, z0 = data.qpos[0], data.xpos[1, 2]

            data.ctrl[act_id] = ctrl_val
            for _ in range(args.steps):
                mujoco.mj_step(model, data)

            jnt_id = model.actuator(act_id).trnid[0]
            dof_adr = model.jnt_dofadr[jnt_id]
            qvel = data.qvel[dof_adr]
            dx = data.qpos[0] - x0
            dz = data.xpos[1, 2] - z0

            print(
                f"{ACTUATOR_NAMES[act_id]:15s} | {ctrl_val:+5.1f} | "
                f"{qvel:+10.4f} | {dx:+8.4f} | {dz:+8.4f}"
            )

    print()
    print("解读:")
    print("  qvel > 0: 关节正方向转动")
    print("  base_dx > 0: 机器人向 +X 方向移动")
    print("  base_dz > 0: 机器人升高")


def mode_interactive(model, data, args):
    """手动输入 ctrl 值测试。"""
    print("=" * 60)
    print("INTERACTIVE: 手动输入 ctrl 值")
    print("=" * 60)
    print("Actuator 列表:")
    for i, name in enumerate(ACTUATOR_NAMES):
        r = model.actuator(i).ctrlrange
        print(f"  [{i}] {name:15s} range=[{r[0]:.1f}, {r[1]:.1f}]")
    print()
    print("输入格式: <actuator_id> <ctrl_value> <steps>")
    print("输入 'q' 退出, 'r' 重置, 's' 显示状态, 'all <v> <steps>' 给所有施力")
    print()

    reset_standing(model, data, args.height)

    while True:
        state = get_base_state(model, data)
        try:
            cmd = input(f"[z={state['z']:.4f} x={state['x']:.4f}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "r":
            reset_standing(model, data, args.height)
            print("  -> 重置完成")
            continue
        elif cmd == "s":
            print(f"  base: {state}")
            for i in range(model.nu):
                jnt_id = model.actuator(i).trnid[0]
                qpos_adr = model.jnt_qposadr[jnt_id]
                dof_adr = model.jnt_dofadr[jnt_id]
                print(
                    f"  {ACTUATOR_NAMES[i]:15s}: "
                    f"pos={data.qpos[qpos_adr]:+.4f} "
                    f"vel={data.qvel[dof_adr]:+.4f} "
                    f"ctrl={data.ctrl[i]:+.4f}"
                )
            continue
        elif cmd.startswith("all"):
            parts = cmd.split()
            val = float(parts[1]) if len(parts) > 1 else 5.0
            steps = int(parts[2]) if len(parts) > 2 else 50
            data.ctrl[:] = val
            for _ in range(steps):
                mujoco.mj_step(model, data)
            state_after = get_base_state(model, data)
            print(
                f"  -> all ctrl={val}, {steps} steps: z={state_after['z']:.4f} x={state_after['x']:.4f}"
            )
            continue

        parts = cmd.split()
        if len(parts) < 2:
            print("  格式: <id> <ctrl> [steps]")
            continue

        act_id = int(parts[0])
        ctrl_val = float(parts[1])
        steps = int(parts[2]) if len(parts) > 2 else 50

        data.ctrl[:] = 0
        data.ctrl[act_id] = ctrl_val
        for _ in range(steps):
            mujoco.mj_step(model, data)

        state_after = get_base_state(model, data)
        jnt_id = model.actuator(act_id).trnid[0]
        dof_adr = model.jnt_dofadr[jnt_id]
        print(
            f"  -> {ACTUATOR_NAMES[act_id]} ctrl={ctrl_val:+.1f}, {steps} steps: "
            f"z={state_after['z']:.4f} x={state_after['x']:.4f} "
            f"jvel={data.qvel[dof_adr]:+.4f}"
        )


def mode_vmc(model, data, args):
    """验证 VMC 控制器响应。"""
    print("=" * 60)
    print("VMC: 验证虚拟模型控制器")
    print("=" * 60)

    L1, L2 = 0.180, 0.200

    def fk(th1, th2):
        end_x = L1 * np.cos(th1) - L2 * np.sin(th1 + th2)
        end_y = L1 * np.sin(th1) + L2 * np.cos(th1 + th2)
        L0 = np.sqrt(end_x**2 + end_y**2)
        theta0 = np.arctan2(end_x, end_y)
        return L0, theta0

    # 默认关节角下的 VMC 状态
    reset_standing(model, data, args.height)
    print("\n默认关节角 (全零):")
    L0, theta0 = fk(0.0, 0.0)
    print(f"  L0 = {L0:.4f} m (目标约 0.22~0.28)")
    print(f"  theta0 = {theta0:.4f} rad ({np.degrees(theta0):.1f} deg)")
    print(f"  end_x = {L1 * np.cos(0) - L2 * np.sin(0):.4f} m")
    print(f"  end_y = {L1 * np.sin(0) + L2 * np.cos(0):.4f} m")

    # 扫描 f1 关节角,看 L0 变化
    print("\n扫描 lf1_Joint 角度,观察 L0 变化:")
    print(f"  {'f1 (rad)':>10s} | {'L0 (m)':>8s} | {'theta0 (deg)':>12s}")
    print("  " + "-" * 40)
    for f1 in np.linspace(-0.6, 0.8, 8):
        L0, theta0 = fk(0.0, f1)
        print(f"  {f1:+10.3f} | {L0:8.4f} | {np.degrees(theta0):+12.1f}")

    # 验证轮子方向
    print("\n轮子方向测试:")
    for act_id, name in [(2, "l_wheel"), (5, "r_wheel")]:
        reset_standing(model, data, args.height)
        x0 = data.qpos[0]
        data.ctrl[act_id] = 3.0
        for _ in range(200):
            mujoco.mj_step(model, data)
        dx = data.qpos[0] - x0
        print(f"  {name} ctrl=+3.0: dx={dx:+.4f} ({'前进' if dx > 0 else '后退'})")

    # 验证伸腿方向
    print("\n伸腿方向测试 (f1 ctrl -> 高度变化):")
    for ctrl_val in [-20, -10, +10, +20]:
        reset_standing(model, data, args.height)
        z0 = data.xpos[1, 2]
        data.ctrl[1] = ctrl_val  # lf1
        data.ctrl[4] = ctrl_val  # rf1
        for _ in range(100):
            mujoco.mj_step(model, data)
        dz = data.xpos[1, 2] - z0
        print(f"  f1 ctrl={ctrl_val:+3d}: dz={dz:+.4f} ({'升高' if dz > 0 else '降低'})")

    # 验证 f0 对 theta0 的影响
    print("\n大腿方向测试 (f0 ctrl -> 倾斜方向):")
    for ctrl_val in [-10, +10]:
        reset_standing(model, data, args.height)
        x0 = data.qpos[0]
        data.ctrl[0] = ctrl_val  # lf0
        data.ctrl[3] = ctrl_val  # rf0
        for _ in range(100):
            mujoco.mj_step(model, data)
        dx = data.qpos[0] - x0
        print(f"  f0 ctrl={ctrl_val:+3d}: dx={dx:+.4f} ({'前倾' if dx > 0 else '后倾'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="终端交互式关节诊断工具")
    parser.add_argument("--mjcf", default=MJCF_PATH, help="MJCF 模型路径")
    parser.add_argument("--height", type=float, default=0.30, help="初始基座高度")
    parser.add_argument("--steps", type=int, default=50, help="sweep 模式每次仿真步数")
    parser.add_argument(
        "--mode",
        choices=["sweep", "interactive", "vmc"],
        default="vmc",
        help="运行模式",
    )
    args = parser.parse_args()

    model, data = load_model(args.mjcf)

    if args.mode == "sweep":
        mode_sweep(model, data, args)
    elif args.mode == "interactive":
        mode_interactive(model, data, args)
    elif args.mode == "vmc":
        mode_vmc(model, data, args)


if __name__ == "__main__":
    main()
